r"""Stereographic (product-manifold) transformer predictor.

`KappaTransformer` is a transductive, single-graph node classifier/regressor built from a stack of
:class:`~manify.predictors.nn.layers.StereographicTransformer` blocks followed by a
:class:`~manify.predictors.nn.layers.StereographicLogits` head. It mirrors
:class:`~manify.predictors.kappa_gcn.KappaGCN`: it operates on a single graph of ``[n_nodes, dim]``
points on a stereographic ``ProductManifold`` (no batch dimension), and implements the
``BasePredictor`` ``fit`` / ``predict`` / ``predict_proba`` API.

The optional adjacency matrix ``A`` is normalized to ``A_hat`` (via
:func:`~manify.predictors.kappa_gcn.get_A_hat`) and used as the attention mask inside every block:
``A`` therefore restricts which nodes may attend to which. If ``A`` is ``None`` the transformer uses
full (all-to-all) self-attention, i.e. an attention-based set encoder over the nodes.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import geoopt
import torch

if TYPE_CHECKING:
    from beartype.typing import Literal
    from jaxtyping import Float, Real

from ..manifolds import ProductManifold
from ._base import BasePredictor
from .kappa_gcn import get_A_hat
from .nn import StereographicLogits, StereographicTransformer

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class KappaTransformer(BasePredictor, torch.nn.Module):
    r"""Stereographic product-manifold transformer for node classification/regression.

    Stacks ``num_layers`` stereographic transformer blocks over a single graph of ``[n_nodes, dim]``
    points on a stereographic ``ProductManifold``, then applies a stereographic logits head. The
    adjacency matrix (if provided) is used as the attention mask in every block.

    Args:
        pm: Stereographic ProductManifold defining the geometry.
        output_dim: Number of output features (classes for classification, target dim for regression).
        num_layers: Number of stereographic transformer blocks.
        num_heads: Number of attention heads per block.
        head_dim: Dimension of each attention head. Defaults to ``max(1, pm.dim // num_heads)``.
        use_layer_norm: Whether to use (tangent-space) layer normalization in each block.
        task: Task type, one of ["classification", "regression"].
        random_state: Random seed for reproducibility.
        device: Device to run the model on (default: None, uses ``pm.device``).

    Attributes:
        pm: The stereographic ProductManifold.
        blocks: ``torch.nn.ModuleList`` of stereographic transformer blocks.
        output_layer: Stereographic logits head.
        is_fitted_: Whether the model has been fitted.
        loss_history_: History of loss values during training.

    Raises:
        ValueError: If the ProductManifold is not stereographic, or task is unsupported.
    """

    def __init__(
        self,
        pm: ProductManifold,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        head_dim: int | None = None,
        use_layer_norm: bool = True,
        task: Literal["classification", "regression"] = "classification",
        random_state: int | None = None,
        device: str | None = None,
    ):
        BasePredictor.__init__(self, pm=pm, task=task, random_state=random_state, device=device)
        torch.nn.Module.__init__(self)

        if not pm.is_stereographic:
            raise ValueError(
                "ProductManifold must be stereographic for KappaTransformer to work. "
                "Please use pm.stereographic() to convert."
            )
        if task not in ("classification", "regression"):
            raise ValueError(f"KappaTransformer supports 'classification' and 'regression', got {task!r}.")

        if random_state is not None:
            torch.manual_seed(random_state)

        self.pm = pm
        self.task = task
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else max(1, pm.dim // num_heads)
        self.use_layer_norm = use_layer_norm

        self.blocks = torch.nn.ModuleList(
            [
                StereographicTransformer(
                    manifold=pm,
                    num_heads=num_heads,
                    dim=pm.dim,
                    head_dim=self.head_dim,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Softmax is applied in the loss (classification) / not at all (regression), so keep it off here.
        self.output_layer = StereographicLogits(output_dim, pm, apply_softmax=False)

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
        softmax: bool = False,
    ) -> Float[torch.Tensor, "n_nodes n_classes"] | Float[torch.Tensor, "n_nodes"]:
        """Forward pass through the transformer blocks and logits head.

        Args:
            X: Node features as points on the manifold, shape ``[n_nodes, dim]``.
            A_hat: Optional normalized adjacency matrix used as the attention mask. ``None`` means
                full attention.
            softmax: Whether to apply softmax to the output logits.

        Returns:
            Logits/values of shape ``[n_nodes, n_classes]`` (squeezed for regression).
        """
        H = X
        for block in self.blocks:
            H = block(H, A_hat)

        # Aggregate logits along edges only when an adjacency is provided (matches KappaGCN behavior).
        logits = self.output_layer(H, A_hat, aggregate_logits=A_hat is not None)

        if softmax:
            logits = torch.softmax(logits, dim=-1)

        return logits.squeeze()

    def fit(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        y: Real[torch.Tensor, "n_nodes"],
        A: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
        epochs: int = 2_000,
        lr: float = 1e-2,
        use_tqdm: bool = True,
        tqdm_prefix: str | None = None,
    ) -> KappaTransformer:
        """Fit the KappaTransformer model.

        Args:
            X: Feature matrix of points on the manifold, shape ``[n_nodes, dim]``.
            y: Labels for the nodes.
            A: Optional adjacency or distance matrix; normalized to ``A_hat`` and used as the
                attention mask. If ``None``, full attention is used.
            epochs: Number of training epochs.
            lr: Learning rate.
            use_tqdm: Whether to show a tqdm progress bar.
            tqdm_prefix: Prefix for the tqdm progress bar.

        Returns:
            self: The fitted predictor.
        """
        X = X.clone()
        y = y.clone()
        A = A.clone() if A is not None else None
        A_hat = get_A_hat(A, make_symmetric=True, add_self_loops=True) if A is not None else None

        # Euclidean parameters (everything except the manifold-valued bias points of the head).
        euclidean_params = [p for n, p in self.named_parameters() if not isinstance(p, geoopt.ManifoldParameter)]
        riemannian_params = [self.output_layer.p_ks]

        opt = torch.optim.Adam(euclidean_params, lr=lr)
        ropt = geoopt.optim.RiemannianAdam(riemannian_params, lr=lr)

        if self.task == "classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            y = y.long()
        else:  # regression
            loss_fn = torch.nn.MSELoss()
            y = y.to(self.pm.dtype)

        self.train()
        my_tqdm = tqdm(total=epochs, desc=tqdm_prefix) if use_tqdm else None

        losses = []
        for i in range(epochs):
            opt.zero_grad()
            ropt.zero_grad()
            y_pred = self(X, A_hat)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            ropt.step()

            if my_tqdm is not None:
                my_tqdm.update(1)
                my_tqdm.set_description(f"Epoch {i + 1}/{epochs}, Loss: {loss.item():.4f}")

            if torch.isnan(loss):
                print("Loss is NaN, stopping training.")
                break
            losses.append(loss.item())

        if my_tqdm is not None:
            my_tqdm.close()

        self.is_fitted_ = True
        self.loss_history_["train"] = losses
        return self

    def predict_proba(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Real[torch.Tensor, "n_nodes n_classes"] | Real[torch.Tensor, "n_nodes"]:
        """Predict class probabilities (classification) or values (regression).

        Args:
            X: Feature matrix of points on the manifold, shape ``[n_nodes, dim]``.
            A: Optional adjacency or distance matrix used as the attention mask.

        Returns:
            Predicted class probabilities / regression targets.
        """
        X = X.clone()
        A = A.clone() if A is not None else None
        A_hat = get_A_hat(A, make_symmetric=True, add_self_loops=True) if A is not None else None

        self.eval()
        with torch.no_grad():
            return self(X, A_hat, softmax=self.task == "classification")

    def __repr__(self) -> str:
        """String representation of the `KappaTransformer`."""
        return (
            f"{self.__class__.__name__}(\n"
            + f"  num_layers={self.num_layers},\n"
            + f"  num_heads={self.num_heads},\n"
            + f"  head_dim={self.head_dim},\n"
            + f"  output_layer={repr(self.output_layer)},\n"
            + f"  task='{self.task}',\n"
            + f"  output_dim={self.output_dim}\n)"
        )
