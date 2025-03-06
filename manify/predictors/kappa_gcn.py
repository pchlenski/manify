"""
Kappa GCN implementation
"""

import sys
from typing import List, Optional, Literal, Union, Tuple, Callable
from jaxtyping import Float

import torch
import geoopt

from ..manifolds import Manifold, ProductManifold

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def get_A_hat(
    A: Float[torch.Tensor, "n_nodes n_nodes"], make_symmetric: bool = True, add_self_loops: bool = True
) -> Float[torch.Tensor, "n_nodes n_nodes"]:
    """
    Normalize adjacency matrix.

    Args:
        A (torch.Tensor): Adjacency matrix.
        make_symmetric (bool): Whether to make the adjacency matrix symmetric.
        add_self_loops (bool): Whether to add self-loops to the adjacency matrix.

    Returns:
        torch.Tensor: Normalized adjacency matrix.
    """

    # Fix nans
    A[torch.isnan(A)] = 0

    # Optional steps to make symmetric and add self-loops
    if make_symmetric and not torch.allclose(A, A.T):
        A = A + A.T
    if add_self_loops and not torch.allclose(torch.diag(A), torch.ones(A.shape[0], dtype=A.dtype, device=A.device)):
        A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    # Get degree matrix
    D = torch.diag(torch.sum(A, axis=1))  # type: ignore

    # Compute D^(-1/2)
    D_inv_sqrt = torch.inverse(torch.sqrt(D))

    # Normalize adjacency matrix
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat.detach()


# A kappa-GCN layer
class KappaGCNLayer(torch.nn.Module):
    """
    Implementation for the Kappa GCN layer

    Parameters
    ----------
    in_features: Number of input features
    out_features: Number of output features
    manifold: Manifold object for the Kappa GCN
    nonlinearity: Function for nonlinear activation.
    """

    def __init__(
        self, in_features: int, out_features: int, manifold: Manifold, nonlinearity: Optional[Callable] = torch.relu
    ):
        super().__init__()

        # Parameters are Euclidean, straightforardly
        # self.W = torch.rand(in_features, out_features)
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        # self.b = torch.nn.Parameter(torch.rand(out_features))

        # Noninearity must be applied via the manifold
        if nonlinearity is None:
            self.sigma = lambda x: x
        else:
            self.sigma = lambda x: manifold.expmap(nonlinearity(manifold.logmap(x)))

        # Also store manifold
        self.manifold = manifold

    def _left_multiply(
        self, A: Float[torch.Tensor, "n_nodes n_nodes"], X: Float[torch.Tensor, "n_nodes dim"], M: Manifold
    ):
        """
        Implementation for Kappa left matrix multiplication for message passing in product space

        Args:
            A: Adjacency matrix of the graph
            X: Embedding matrix of the graph.
            M: Manifold object for the Kappa GCN - need to specify in case we're going by component

        Returns:
            out: result of the Kappa left matrix multiplication.
        """

        # Vectorized version:
        return M.manifold.weighted_midpoint(
            xs=X.unsqueeze(0),  # (1, N, D)
            weights=A,  # (N, N)
            reducedim=[1],  # Sum over the N points dimension (dim 1)
            dim=-1,  # Compute conformal factors along the points dimension
            keepdim=False,  # Squeeze the batch dimension out
            lincomb=True,  # Scale by sum of weights (A.sum(dim=1))
            posweight=False,
        )

    def forward(
        self, X: Float[torch.Tensor, "n_nodes dim"], A_hat: Optional[Float[torch.Tensor, "n_nodes n_nodes"]] = None
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """
        Forward pass for the Kappa GCN layer.

        Args:
            X: Embedding matrix
            A_hat: Normalized adjacency matrix

        Returns:
            AXW: Transformed node features after message passing and nonlinear activation.
        """
        # 1. right-multiply X by W - mobius_matvec broadcasts correctly (verified)
        XW = self.manifold.manifold.mobius_matvec(m=self.W, x=X)

        # 2. left-multiply (X @ W) by A_hat - we need our own implementation for this
        if A_hat is None:
            AXW = XW
        elif isinstance(self.manifold, ProductManifold):
            XWs = self.manifold.factorize(XW)
            AXW = torch.hstack([self._left_multiply(A_hat, XW, M) for XW, M in zip(XWs, self.manifold.P)])
        else:
            AXW = self._left_multiply(A_hat, XW, self.manifold)

        # 3. Apply nonlinearity - note that sigma is wrapped with our manifold.apply decorator
        AXW = self.sigma(AXW)

        return AXW


class KappaGCN(torch.nn.Module):
    """
    Implementation for the Kappa GCN

    Parameters
    ----------
    pm: ProductManifold object for the Kappa GCN
    out_features: Number of output features
    nonlinearity: Function for nonlinear activation.
    """

    def __init__(
        self,
        pm: ProductManifold,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        nonlinearity: Callable = torch.relu,
        task: Literal["classification", "regression", "link_prediction"] = "classification",
    ):
        super().__init__()
        self.pm = pm
        self.task = task

        # Hidden layers
        if hidden_dims is None:
            dims = [pm.dim, pm.dim, pm.dim]  # 2 hidden layers
        elif not (all([M.curvature == 0] for M in pm.P) or all([d == pm.dim for d in hidden_dims])):
            raise ValueError("Only fully Euclidean manifolds can change hidden dimension size")
        else:
            dims = [pm.dim] + hidden_dims

        self.layers = torch.nn.ModuleList(
            [KappaGCNLayer(dims[i], dims[i + 1], pm, nonlinearity) for i in range(len(dims) - 1)]
        )

        # Final layer params
        if task == "link_prediction":
            self.fermi_dirac_temperature = torch.nn.Parameter(torch.tensor(1.0))
            self.fermi_dirac_bias = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.W_logits = torch.nn.Parameter(torch.randn(dims[-1], output_dim) * 0.01)
            self.p_ks = geoopt.ManifoldParameter(torch.zeros(output_dim, pm.dim), manifold=pm.manifold)  # type: ignore

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Optional[Float[torch.Tensor, "n_nodes n_nodes"]] = None,
        aggregate_logits: bool = True,
        softmax: bool = False,
    ):
        """
        Forward pass for the Kappa GCN.

        Args:
            X: Embedding matrix
            A_hat: Normalized adjacency matrix
            aggregate_logits: boolean for whether to aggregate logits
            softmax: boolean for whether to use softmax function

        Returns:
            logits_agg: output of Kappa GCN network
        """
        H = X

        # Pass through kappa-GCN layers
        for layer in self.layers:
            H = layer(H, A_hat)

        # Final layer is to get logits
        if self.task == "link_prediction":
            # Taken from https://arxiv.org/pdf/1910.12933
            return (-(self.pm.pdist2(H) - self.fermi_dirac_bias) / self.fermi_dirac_temperature).flatten()
        else:
            logits = self.get_logits(X=X, W=self.W_logits, b=self.p_ks)
            if A_hat is not None and aggregate_logits:
                logits = A_hat @ logits

            if softmax:
                logits = torch.softmax(logits, dim=1)

            return logits.squeeze()

    def _get_logits_single_manifold(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        W: Float[torch.Tensor, "dim n_classes"],
        b: Float[torch.Tensor, "n_classes"],
        M: Manifold,
        return_inner_products: bool = False,
    ) -> Union[
        Tuple[Float[torch.Tensor, "n_nodes n_classes"], Float[torch.Tensor, "n_nodes n_classes"]],
        Float[torch.Tensor, "n_nodes n_classes"],
    ]:
        """Helper function for get_logits"""

        # For convenience, get curvature and manifold
        kappa = torch.tensor(M.curvature, dtype=X.dtype, device=X.device)

        # # Need transposes because column vectors live in the tangent space
        # W = M.manifold.transp0(b, W.T).T  # (d, k)

        # Change shapes
        b = b[None, :]  # (1, k)
        X = X[:, None]  # (n, 1, d)

        # 1. Get z_k = -p_k \oplus_\kappa x (vectorized)
        # This works for the Euclidean case too - it becomes a simple subtraction
        z_ks = M.manifold.mobius_add(-b, X)  # (n, k, d)
        # z_ks = M.manifold.projx(z_ks)  # (n, k, d)

        # 2. Get norms for relevant terms
        z_k_norms = torch.norm(z_ks, dim=-1).clamp_min(1e-10)  # (n, k)
        a_k_norms = torch.norm(W, dim=0).clamp_min(1e-10)  # (k,)

        # 3. Get the distance to the hyperplane
        za = torch.einsum("nkd,dk->nk", z_ks, W)  # (n, k)

        # 4. Get the logits
        if abs(kappa) < 1e-4:
            # Euclidean case: it's just a dot product
            logits = 4 * za
        else:
            # Non-Euclidean case: need to do the arsinh
            dist = 2 * za / ((1 + kappa * z_k_norms**2) * a_k_norms)
            dist = geoopt.manifolds.stereographic.math.arsin_k(dist, kappa * abs(kappa))

            # Get the coefficients
            lambda_pks = M.manifold.lambda_x(b)  # (k,)
            coeffs = lambda_pks * a_k_norms  # / abs(kappa) ** 0.5
            logits = coeffs * dist

        if return_inner_products:
            return logits, za
        else:
            return logits

    def _get_logits_product_manifold(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        W: Float[torch.Tensor, "dims n_classes"],
        b: Float[torch.Tensor, "n_classes"],
        M: ProductManifold,
    ) -> Float[torch.Tensor, "n_nodes n_classes"]:
        """Helper function for get_logits"""

        # For convenience, get curvature and manifold
        # kappas = [man.curvature for manifold in M.P]
        Xs = M.factorize(X)
        bs = M.factorize(b)
        Ws = [w.T for w in M.factorize(W.T)]
        res = [
            self._get_logits_single_manifold(X_man, W_man, b_man, man, return_inner_products=True)
            for X_man, W_man, b_man, man in zip(Xs, Ws, bs, M.P)
        ]

        # Each result is (n, k) logits and (n, k) inner products
        logits, inner_products = zip(*res)

        # Final logits: l2 norm of logits * sign of inner product
        stacked_logits = torch.stack(logits, dim=2)  # (n, k, m)
        stack_products = torch.stack(inner_products, dim=2)  # (n, k, m)

        # Reduce
        logits = torch.norm(stacked_logits, dim=2)  # (n, k)
        signs = torch.sign(stack_products.sum(dim=2))  # (n, k)

        return logits * signs

    def get_logits(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        W: Optional[Float[torch.Tensor, "dims n_classes"]] = None,
        b: Optional[Float[torch.Tensor, "n_classes"]] = None,
    ) -> Float[torch.Tensor, "n_nodes n_classes"]:
        """
        Computes logits given the manifold.

        Credit to the Curve Your Attention paper for an implementation we referenced:
        https://openreview.net/forum?id=AN5uo4ByWH

        Args:
            X: Input points tensor of shape (n, d), where n is the number of points and d is the dimensionality.
            W: Weight tensor of shape (d, k), where k is the number of classes.
            b: Bias tensor of shape (k,)

        Returns:
            Logits: tensor of shape (n, k).
        """
        if W is None:
            W = self.W_logits
        if b is None:
            b = self.p_ks

        if isinstance(self.pm, ProductManifold):
            return self._get_logits_product_manifold(X, W, b, self.pm)
        elif isinstance(self.pm, Manifold):
            return self._get_logits_single_manifold(X, W, b, self.pm, return_inner_products=False)  # type: ignore
        else:
            raise ValueError("Manifold must be a Manifold or ProductManifold object.")

    def fit(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        y: Float[torch.Tensor, "n_nodes"],
        A: Optional[Float[torch.Tensor, "n_nodes n_nodes"]] = None,
        epochs: int = 2_000,
        lr: float = 1e-2,
        use_tqdm: bool = True,
        lp_indices: Optional[List[Tuple[int]]] = None,
    ) -> None:
        """
        Fit the Kappa GCN model.

        Args:
            X (torch.Tensor): Feature matrix.
            y (torch.Tensor): Labels for training nodes.
            adj (torch.Tensor): Adjacency or distance matrix.
            train_idx (torch.Tensor): Indices of nodes for training.
            epochs: Number of training epochs (default=200).
            lr: Learning rate (default=1e-2).
            use_tqdm: Whether to use tqdm for progress bar.
            lp_indices: Indices for link prediction task (e.g.: [(0, 1), (2, 3), ...]).
        """
        if lp_indices is None and self.task == "link_prediction":
            raise ValueError("Must provide indices for link prediction task!")

        # Copy everything
        X = X.clone()
        y = y.clone()
        A = A.clone() if A is not None else None

        # Standard fit
        opt_params = [layer.W for layer in self.layers]
        ropt_params = []
        if self.task == "link_prediction":
            opt_params += [self.fermi_dirac_temperature, self.fermi_dirac_bias]
        else:
            opt_params += [self.W_logits]
            ropt_params += [self.p_ks]
        opt = torch.optim.Adam(opt_params, lr=lr)
        if ropt_params:
            ropt = geoopt.optim.RiemannianAdam(ropt_params, lr=lr)

        if self.task == "classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            y = y.long()
        elif self.task == "regression":
            loss_fn = torch.nn.MSELoss()
            y = y.float()
        elif self.task == "link_prediction":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            y = y.flatten().float()
        else:
            raise ValueError("Invalid task!")

        self.train()
        if use_tqdm:
            my_tqdm = tqdm(total=epochs)

        for i in range(epochs):
            opt.zero_grad()
            if ropt_params:
                ropt.zero_grad()
            y_pred = self(X, A)
            if self.task == "link_prediction":
                y_pred = y_pred[lp_indices]
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            if ropt_params:
                ropt.step()

            # Progress bar
            if use_tqdm:
                my_tqdm.update(1)
                my_tqdm.set_description(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(
        self, X: Float[torch.Tensor, "n_nodes dim"], A: Optional[Float[torch.Tensor, "n_nodes n_nodes"]] = None
    ) -> Float[torch.Tensor, "n_nodes"]:
        """
        Make predictions using the trained Kappa GCN.

        Args:
            X (torch.Tensor): Feature matrix (NxD).
            A (torch.Tensor): Adjacency or distance matrix (NxN).

        Returns:
            torch.Tensor: Predicted labels or outputs.
        """
        # Copy everything
        X = X.clone()
        A = A.clone() if A is not None else None

        # Get edges for test set
        self.eval()
        y_pred = self(X, A)
        if self.task == "classification":
            return y_pred.argmax(dim=1).detach()
        elif self.task == "link_prediction":
            # Binarize the logits
            return (y_pred > 0).long().detach()
        else:
            return y_pred.detach()
