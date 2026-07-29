"""Implementation for Support Vector Machine in Product Manifolds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import torch

if TYPE_CHECKING:
    from beartype.typing import Any, Literal
    from jaxtyping import Float, Int

from ..manifolds import ProductManifold
from ._base import BasePredictor
from ._kernel import product_kernel


class ProductSpaceSVM(BasePredictor):
    """Product Space SVM class in a product manifold setting.

    Trains one-vs-rest SVMs with Euclidean, spherical, and hyperbolic constraints
    enforced via second-order-cone (SOC) formulations for convexity.

    Args:
        pm: A ProductManifold instance specifying component manifolds.
        weights: Optional per-manifold weights tensor.
        h_constraints: Whether to enforce hyperbolic constraints.
        e_constraints: Whether to enforce Euclidean constraints.
        s_constraints: Whether to enforce spherical constraints.
        task: Task type, either "classification" or "regression".
        epsilon: Slack parameter for SOC constraints.
        random_state: Random seed for reproducibility.
        device: Device for tensor computations.

    Attributes:
        pm: ProductManifold object associated with the predictor.
        weights: Per-manifold weights for kernel combination.
        h_constraints: Whether to enforce hyperbolic constraints.
        e_constraints: Whether to enforce Euclidean constraints.
        s_constraints: Whether to enforce spherical constraints.
        eps: Slack parameter for SOC constraints.
        beta: Dictionary storing SVM coefficients for each class.
        zeta: Dictionary storing slack variables for each class.
        epsilon: Dictionary storing epsilon values for each class.
        b: Dictionary storing bias terms for each class.
        X_train_: Training data points.
        is_fitted_: Boolean flag indicating if the predictor has been fitted.
    """

    def __init__(
        self,
        pm: ProductManifold,
        weights: Float[torch.Tensor, "n_manifolds"] | None = None,
        h_constraints: bool = True,
        e_constraints: bool = True,
        s_constraints: bool = True,
        task: Literal["classification", "regression"] = "classification",
        epsilon: float = 1e-5,
        random_state: int | None = None,
        device: str | None = None,
    ):
        """Initialize the ProductSpaceSVM.

        Args:
            pm: A ProductManifold instance specifying component manifolds.
            weights: Optional per-manifold weights tensor.
            h_constraints: Whether to enforce hyperbolic constraints.
            e_constraints: Whether to enforce Euclidean constraints.
            s_constraints: Whether to enforce spherical constraints.
            task: Task type, either "classification" or "regression".
            epsilon: Slack parameter for SOC constraints.
            random_state: Random seed for reproducibility.
            device: Device for tensor computations.
        """
        super().__init__(pm=pm, task=task, random_state=random_state, device=device)
        self.pm = pm
        self.h_constraints = h_constraints
        self.s_constraints = s_constraints
        self.e_constraints = e_constraints
        self.eps = epsilon
        self.task = task
        self.weights = torch.ones(len(pm.P), dtype=pm.dtype) if weights is None else weights
        assert len(self.weights) == len(pm.P), "Number of weights must match the number of manifolds."

    def fit(
        self,
        X: Float[torch.Tensor, "n_samples n_manifolds"],
        y: Int[torch.Tensor, "n_samples"],
    ) -> ProductSpaceSVM:
        """Fit one-vs-rest SVMs on the product manifold data.

        Args:
            X: Training points tensor.
            y: Integer class labels tensor.

        Returns:
            self: Fitted ProductSpaceSVM instance.
        """
        # unique classes
        self._store_classes(y)
        n = X.shape[0]

        # aggregated kernel
        Ks, _ = product_kernel(self.pm, X, None)
        K_sum = torch.ones((n, n), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights, strict=False):
            K_sum += w * K_m

        X_np = X.detach().cpu().numpy()
        K_np = K_sum.detach().cpu().numpy()

        def sqrtm_psd(P: np.ndarray) -> Any:
            w, V = np.linalg.eigh(P)
            w_s = np.sqrt(np.clip(w, 0, None))
            B = V @ np.diag(w_s) @ V.T
            return (B + B.T) * 0.5

        # containers
        self.beta = {}
        self.zeta = {}
        self.epsilon = {}
        self.b = {}

        for cls in self.classes_:
            cls_item = cls.item() if isinstance(cls, torch.Tensor) else cls
            # one-vs-rest labels: +1 for cls, -1 for others
            y_bin = torch.where(y == cls_item, 1, -1)
            Y = torch.diagflat(y_bin).detach().cpu().numpy()

            # variables
            beta_var = cp.Variable(n)
            zeta = cp.Variable(n, nonneg=True)
            eps_var = cp.Variable(1)
            b_var = cp.Variable(1)

            # base constraints
            constraints = [eps_var >= 0]
            constraints.append(Y @ (K_np @ beta_var + b_var) >= eps_var - zeta)

            # per-manifold SOC
            for M, K_comp in zip(self.pm.P, Ks, strict=False):
                P_np = K_comp.detach().cpu().numpy()
                if M.type == "E" and self.e_constraints:
                    B = sqrtm_psd(P_np)
                    constraints.append(cp.norm(B @ beta_var, 2) <= 1.0)
                elif M.type == "S" and self.s_constraints:
                    B = sqrtm_psd(P_np)
                    constraints.append(cp.norm(B @ beta_var, 2) <= np.sqrt(np.pi / 2))
                elif M.type == "H" and self.h_constraints:
                    # PSD split
                    eigvals, eigvecs = np.linalg.eigh(P_np)
                    plus = np.clip(eigvals, 0, None)
                    minus = np.clip(-eigvals, 0, None)
                    Kp = (eigvecs @ np.diag(plus) @ eigvecs.T + (eigvecs @ np.diag(plus) @ eigvecs.T).T) * 0.5
                    Km = (eigvecs @ np.diag(minus) @ eigvecs.T + (eigvecs @ np.diag(minus) @ eigvecs.T).T) * 0.5
                    Bp = sqrtm_psd(Kp)
                    Bm = sqrtm_psd(Km)

                    C_H = abs(M.curvature)
                    R = -M.scale
                    r_h = abs(np.arcsinh(-(R**2) * C_H))
                    r = self.eps

                    constraints.append(cp.norm(Bm @ beta_var, 2) <= np.sqrt(max(r, 0.0)))
                    constraints.append(cp.norm(Bp @ beta_var, 2) <= np.sqrt(max(r + r_h, 0.0)))

            # solve
            prob = cp.Problem(cp.Minimize(-eps_var + cp.sum(zeta)), constraints)
            prob.solve(solver="SCS")

            # save results. eps_var and b_var are length-1 cp.Variables, so their
            # .value is a 1-element ndarray; index it before float() (numpy 2.x no
            # longer coerces 1-element arrays to scalars implicitly).
            self.beta[cls_item] = np.ravel(beta_var.value)
            self.zeta[cls_item] = zeta.value
            self.epsilon[cls_item] = float(np.ravel(eps_var.value)[0])
            self.b[cls_item] = float(np.ravel(b_var.value)[0])

        # store training data
        self.X_train_ = torch.tensor(X_np, dtype=self.pm.dtype)
        self.is_fitted_ = True
        return self

    def predict_proba(
        self,
        X: Float[torch.Tensor, "n_samples n_manifolds"],
    ) -> Float[torch.Tensor, "n_samples n_classes"]:
        """Predict class probabilities using the fitted SVMs.

        Args:
            X: Test points tensor.

        Returns:
            class_probabilities: Class probabilities for each test sample.
        """
        X_tensor = torch.as_tensor(X, dtype=self.pm.dtype)
        X_tensor = X_tensor.to(self.X_train_.device)

        Ks_test, _ = product_kernel(self.pm, self.X_train_, X_tensor)
        Kt = torch.ones((self.X_train_.shape[0], X_tensor.shape[0]), dtype=X_tensor.dtype, device=X_tensor.device)
        for K_m, w in zip(Ks_test, self.weights, strict=False):
            Kt += w * K_m
        Kt_np = Kt.detach().cpu().numpy()

        n_test = X_tensor.shape[0]
        n_cls = len(self.classes_)
        dec = np.zeros((n_test, n_cls))
        for idx, cls in enumerate(self.classes_):
            cls_item = cls.item() if isinstance(cls, torch.Tensor) else cls
            beta_vec: np.ndarray = np.ravel(self.beta[cls_item])
            dec[:, idx] = Kt_np.T @ beta_vec + self.b[cls_item]

        exp_scores = np.exp(dec - dec.max(axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return torch.tensor(probs, dtype=torch.float32)
