"""Implementation for coordinate training and optimization"""

import sys
from typing import Tuple, List, Dict
from jaxtyping import Float, Int

import numpy as np
import torch
import geoopt

from .losses import distortion_loss, d_avg
from ..manifolds import ProductManifold

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def train_coords(
    pm: ProductManifold,
    dists: Float[torch.Tensor, "n_points n_points"],
    test_indices: Int[torch.Tensor, "n_test"] = torch.tensor([]),
    device: str = "cpu",
    burn_in_learning_rate: float = 1e-3,
    burn_in_iterations: int = 2_000,
    learning_rate: float = 1e-2,
    scale_factor_learning_rate: float = 0.0,  # Off by default
    training_iterations: int = 18_000,
    loss_window_size: int = 100,
    logging_interval: int = 10,
    scale=1.0,
) -> Tuple[Float[torch.Tensor, "n_points n_dim"], Dict[str, List[float]]]:
    """
    Coordinate training and optimization

    Args:
        pm: ProductManifold object that encapsulates the manifold and its signature.
        dists: (n_points, n_points) Tensor representing the pairwise distance matrix between points.
        test_indices: (n_test) Tensor representing the indices of the test points.
        device: Device for training (default: "cpu").
        burn_in_learning_rate: Learning rate during the burn-in phase (default: 1e-3).
        burn_in_iterations: Number of iterations during the burn-in phase (default: 2,000).
        learning_rate: Learning rate during the training phase (default: 1e-2).
        scale_factor_learning_rate: Learning rate for scale factor optimization (default: 0.0).
        training_iterations: Number of iterations for the training phase (default: 18,000).
        loss_window_size: Window size for computing the moving average of the loss (default: 100).
        logging_interval: Interval for logging the training progress (default: 10).

    Returns:
        pm.x_embed: Tensor of the final learned coordinates in the manifold.
        losses: List of loss values at each iteration during training.
    """
    # Move everything to the device
    n = dists.shape[0]
    covs = [torch.stack([torch.eye(M.dim) / pm.dim] * n).to(device) for M in pm.P]
    means = torch.stack([pm.mu0] * n).to(device)
    X, _ = pm.sample(z_mean=means, sigma_factorized=covs)
    dists = dists.to(device)

    # Get train and test indices set up
    use_test = len(test_indices) > 0
    test = torch.tensor([i in test_indices for i in range(len(dists))]).to(device)
    train = ~test

    # Initialize optimizer
    X = geoopt.ManifoldParameter(X, manifold=pm.manifold)
    ropt = geoopt.optim.RiemannianAdam(
        [
            {"params": [X], "lr": burn_in_learning_rate},
            {"params": [x._log_scale for x in pm.manifold.manifolds], "lr": 0},
        ]
    )

    # Init TQDM
    my_tqdm = tqdm(total=burn_in_iterations + training_iterations, leave=False)

    # Outer training loop - mostly setting optimizer learning rates up here
    losses = {"train_train": [], "test_test": [], "train_test": [], "total": []}

    # Actual training loop
    for i in range(burn_in_iterations + training_iterations):
        if i == burn_in_iterations:
            # Optimize curvature by changing lr
            # opt.lr = scale_factor_learning_rate
            # ropt.lr = learning_rate
            ropt.param_groups[0]["lr"] = learning_rate
            ropt.param_groups[1]["lr"] = scale_factor_learning_rate

        # Zero grad
        ropt.zero_grad()
        # opt.zero_grad()

        # 1. Train-train loss
        X_t = X[train]
        D_tt = pm.pdist(X_t)
        L_tt = distortion_loss(D_tt, dists[train][:, train], pairwise=True)
        L_tt.backward(retain_graph=True)
        losses["train_train"].append(L_tt.item())

        if use_test:
            # 2. Test-test loss
            X_q = X[test]
            D_qq = pm.pdist(X_q)
            L_qq = distortion_loss(D_qq, dists[test][:, test], pairwise=True)
            L_qq.backward(retain_graph=True)
            losses["test_test"].append(L_qq.item())

            # 3. Train-test loss
            X_t_detached = X[train].detach()
            D_tq = pm.dist(X_t_detached, X_q)  # Note 'dist' not 'pdist', as we're comparing different sets
            L_tq = distortion_loss(D_tq, dists[train][:, test], pairwise=False)
            L_tq.backward()
            losses["train_test"].append(L_tq.item())
        else:
            L_qq = 0
            L_tq = 0

        # Step
        # opt.step()
        ropt.step()
        L = L_tt + L_qq + L_tq
        losses["total"].append(L.item())

        # TQDM management
        my_tqdm.update(1)
        my_tqdm.set_description(f"Loss: {L.item():.3e}")

        # Logging
        if i % logging_interval == 0:
            d = {f"r{i}": f"{x._log_scale.item():.3f}" for i, x in enumerate(pm.manifold.manifolds)}
            d["D_avg"] = f"{d_avg(D_tt, dists[train][:, train], pairwise=True):.4f}"
            d["L_avg"] = f"{np.mean(losses['total'][-loss_window_size:]):.3e}"
            my_tqdm.set_postfix(d)

        # Early stopping for errors
        if torch.isnan(L):
            raise ValueError("Loss is NaN")

    return X.data.detach(), losses
