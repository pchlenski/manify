r"""$\kappa$-GCN implementation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import geoopt
import torch

if TYPE_CHECKING:
    from beartype.typing import Callable, Literal
    from jaxtyping import Float, Real

from ..manifolds import ProductManifold
from ._base import BasePredictor
from .nn import FermiDiracDecoder, KappaGCNLayer, KappaSequential, StereographicLogits

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def get_A_hat(
    A: Float[torch.Tensor, "n_nodes n_nodes"],
    make_symmetric: bool = True,
    add_self_loops: bool = True,
) -> Float[torch.Tensor, "n_nodes n_nodes"]:
    """Normalize adjacency matrix.

    Args:
        A: Adjacency matrix.
        make_symmetric: Whether to make the adjacency matrix symmetric.
        add_self_loops: Whether to add self-loops to the adjacency matrix.

    Returns:
        A_hat: Normalized adjacency matrix.
    """
    # Fix nans (without mutating the caller's tensor)
    A = torch.where(torch.isnan(A), torch.zeros_like(A), A)

    # Optional steps to make symmetric and add self-loops
    if make_symmetric and not torch.allclose(A, A.T):
        A = A + A.T
    if add_self_loops and not torch.allclose(torch.diag(A), torch.ones(A.shape[0], dtype=A.dtype, device=A.device)):
        A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    # Get degree matrix
    D = torch.diag(torch.sum(A, axis=1))

    # Compute D^(-1/2)
    D_inv_sqrt = torch.inverse(torch.sqrt(D))

    # Normalize adjacency matrix
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat.detach()


class KappaGCN(BasePredictor, torch.nn.Module):
    """Implementation for the Kappa GCN.

    Attributes:
        pm: ProductManifold object for the Kappa GCN.
        output_dim: Number of output features.
        num_hidden: Number of hidden layers.
        nonlinearity: Function for nonlinear activation.
        task: Task type, one of ["classification", "regression", "link_prediction"]
        random_state: Random seed for reproducibility.
        device: Device to run the model on (default: None, uses current device).
        is_fitted_: Whether the model has been fitted.
        loss_history_: History of loss values during training.

    Args:
        pm: ProductManifold object for the Kappa GCN
        output_dim: Number of output features
        num_hidden: Number of hidden layers.
        nonlinearity: Function for nonlinear activation.
        task: Task type, one of ["classification", "regression", "link_prediction"].
        random_state: Random seed for reproducibility.
        device: Device to run the model on (default: None, uses current device).

    Raises:
        ValueError: If the ProductManifold is not stereographic.
    """

    def __init__(
        self,
        pm: ProductManifold,
        output_dim: int,
        num_hidden: int = 2,
        nonlinearity: Callable = torch.relu,
        task: Literal["classification", "regression", "link_prediction"] = "classification",
        random_state: int | None = None,
        device: str | None = None,
    ):
        BasePredictor.__init__(self, pm=pm, task=task, random_state=random_state, device=device)
        torch.nn.Module.__init__(self)

        self.pm = pm
        self.task = task
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.nonlinearity = nonlinearity

        # Ensure pm is stereographic
        if not pm.is_stereographic:
            raise ValueError(
                "ProductManifold must be stereographic for KappaGCN to work.Please use pm.stereographic() to convert."
            )

        # Build layer dimensions
        dims = [pm.dim] + [pm.dim] * num_hidden

        # Build the main GCN layers using Sequential
        gcn_layers = []
        for i in range(len(dims) - 1):
            gcn_layers.append(KappaGCNLayer(dims[i], dims[i + 1], pm, nonlinearity))

        self.gcn_layers = KappaSequential(*gcn_layers)

        # Task-specific output layers - much cleaner now!
        if task == "link_prediction":
            self.output_layer = FermiDiracDecoder(pm, learnable_params=True)
        else:
            # This is the same for classification/regression since we apply softmax in the loss function, not here
            self.output_layer = StereographicLogits(output_dim, pm, apply_softmax=False)

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
        aggregate_logits: bool = True,
        softmax: bool = False,
    ) -> (
        Float[torch.Tensor, "n_nodes n_classes"]
        | Float[torch.Tensor, "n_nodes"]
        | Float[torch.Tensor, "n_nodes n_nodes"]
    ):
        """Forward pass through the GCN layers and output head."""
        # Pass through main GCN layers
        H = self.gcn_layers(X, A_hat)

        # Task-specific output using the specialized layers
        if self.task == "link_prediction":
            return self.output_layer(H)  # Flattened for link prediction
        else:
            # For classification/regression, use stereographic logits
            logits = self.output_layer(H, A_hat, aggregate_logits=aggregate_logits)

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
    ) -> KappaGCN:
        """Fit the Kappa GCN model.

        Args:
            X (torch.Tensor): Feature matrix.
            y (torch.Tensor): Labels for training nodes.
            A (torch.Tensor): Adjacency or distance matrix.
            epochs: Number of training epochs (default=200).
            lr: Learning rate (default=1e-2).
            use_tqdm: Whether to use tqdm for progress bar.
            tqdm_prefix: Prefix for tqdm progress bar.
        """
        # Copy everything
        X = X.clone()
        y = y.clone()
        A = A.clone() if A is not None else None

        # Convert A to A_hat
        A_hat = get_A_hat(A, make_symmetric=True, add_self_loops=True) if A is not None else None

        # Collect all paramters
        euclidean_params = []
        riemannian_params = []
        for layer in self.gcn_layers.layers:
            euclidean_params.append(layer.W)
        if self.task == "link_prediction":
            euclidean_params += [self.output_layer.temperature, self.output_layer.bias]
        else:
            euclidean_params += [self.output_layer.W]
            riemannian_params += [self.output_layer.p_ks]

        # Optimizers
        opt = torch.optim.Adam(euclidean_params, lr=lr)
        ropt = geoopt.optim.RiemannianAdam(riemannian_params, lr=lr) if riemannian_params else None

        if self.task == "classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            y = y.long()
        elif self.task == "regression":
            loss_fn = torch.nn.MSELoss()
            y = y.float()
        elif self.task == "link_prediction":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            # y = y.flatten().float()
            y = y.float()
        else:
            raise ValueError("Invalid task!")

        self.train()
        if use_tqdm:
            my_tqdm = tqdm(total=epochs, desc=tqdm_prefix)

        losses = []
        for i in range(epochs):
            opt.zero_grad()
            if riemannian_params:
                ropt.zero_grad()  # type: ignore
            y_pred = self(X, A_hat)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            if riemannian_params:
                ropt.step()  # type: ignore

            # Progress bar
            if use_tqdm:
                my_tqdm.update(1)
                my_tqdm.set_description(f"Epoch {i + 1}/{epochs}, Loss: {loss.item():.4f}")

            # Early termination for nan loss
            if torch.isnan(loss):
                print("Loss is NaN, stopping training.")
                break
            losses.append(loss.item())

        if use_tqdm:
            my_tqdm.close()

        self.is_fitted_ = True
        self.loss_history_["train"] = losses
        return self

    def predict_proba(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Real[torch.Tensor, "n_nodes n_classes"] | Real[torch.Tensor, "n_nodes"]:
        """Predict class probabilities using the trained Kappa GCN.

        Args:
            X (torch.Tensor): Feature matrix (NxD).
            A (torch.Tensor): Adjacency or distance matrix (NxN).

        Returns:
            torch.Tensor: Predicted class probabilities / regression targets.
        """
        # Copy everything
        X = X.clone()
        A = A.clone() if A is not None else None
        A_hat = get_A_hat(A, make_symmetric=True, add_self_loops=True) if A is not None else None

        # Get edges for test set
        self.eval()
        y_pred = self(X, A_hat)
        return y_pred

    def __repr__(self) -> str:
        """String representation of the `KappaGCN`.

        The purpose of this method is to make `KappaGCN` instances more closely resemble `nn.Sequential`,
        making for more readable output and informative debugging.

        Returns:
            str: String representation of the KappaGCN instance.
        """
        return (
            f"{self.__class__.__name__}(\n"
            + f"  gcn_layers={repr(self.gcn_layers)},\n"
            + f"  output_layer={repr(self.output_layer)},\n"
            + f"  task='{self.task}',\n"
            + f"  output_dim={self.output_dim}\n)"
        )
