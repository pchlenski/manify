"""Implementation of different measurement metrics"""

from typing import List
from jaxtyping import Float

import torch
import networkx as nx

from ..manifolds import ProductManifold


def distortion_loss(
    D_est: Float[torch.Tensor, "n_points n_points"],
    D_true: Float[torch.Tensor, "n_points n_points"],
    pairwise: bool = False,
) -> Float[torch.Tensor, ""]:
    """
    Compute the distortion loss between estimated SQUARED distances and true SQUARED distances.
    Args:
        D_est (n_points, n_points): A tensor of estimated pairwise distances.
        D_true (n_points, n_points).: A tensor of true pairwise distances.
        pairwise (bool): A boolean indicating whether to return whether D_est and D_true are pairwise

    Returns:
        float: A float indicating the distortion loss, calculated as the sum of the squared relative
         errors between the estimated and true squared distances.
        
    See also: square_loss in HazyResearch hyperbolics repo:
    https://github.com/HazyResearch/hyperbolics/blob/master/pytorch/hyperbolic_models.py#L178
    """

    # Turn into flat vectors of pairwise distances. For pairwise distances, we only consider the upper triangle.
    if pairwise:
        n = D_true.shape[0]
        idx = torch.triu_indices(n, n, offset=1)
        D_true = D_true[idx[0], idx[1]]
        D_est = D_est[idx[0], idx[1]]
    else:
        D_true = D_true.flatten()
        D_est = D_est.flatten()

    # Mask out any infinite or nan values
    mask = torch.isfinite(D_true) & ~torch.isnan(D_true)
    D_true = D_true[mask]
    D_est = D_est[mask]

    return torch.sum(torch.abs((D_est / D_true) ** 2 - 1))


def d_avg(
    D_est: Float[torch.Tensor, "n_points n_points"],
    D_true: Float[torch.Tensor, "n_points n_points"],
    pairwise: bool = False,
) -> Float[torch.Tensor, ""]:
    """Average distance error D_av
    Args:
        D_est (n_points, n_points): A tensor of estimated pairwise distances.
        D_true (n_points, n_points).: A tensor of true pairwise distances.
        pairwise (bool): A boolean indicating whether to return whether D_est and D_true are pairwise

    Returns:
        float: A float indicating the average distance error D_avg, calculated as the
        mean relative error across all pairwise distances.
    """

    if pairwise:
        n = D_true.shape[0]
        idx = torch.triu_indices(n, n, offset=1)
        D_true = D_true[idx[0], idx[1]]
        D_est = D_est[idx[0], idx[1]]
    else:
        D_true = D_true.flatten()
        D_est = D_est.flatten()

    # Mask out any infinite or nan values
    mask = torch.isfinite(D_true) & ~torch.isnan(D_true)
    D_true = D_true[mask]
    D_est = D_est[mask]

    # Note that D_avg uses nonsquared distances:
    return torch.mean(torch.abs(D_est - D_true) / D_true)


def mean_average_precision(x_embed: Float[torch.Tensor, "n_points n_dim"], graph: nx.Graph) -> Float[torch.Tensor, ""]:
    """Mean averae precision (mAP) from the Gu et al paper."""
    raise NotImplementedError


def dist_component_by_manifold(pm: ProductManifold, x_embed: Float[torch.Tensor, "n_points n_dim"]) -> List[float]:
    """
    Compute the variance in pairwise distances explained by each manifold component.

    Args:
        pm: The product manifold.
        x_embed (n_points, n_dim): A tensor of embeddings.

    Returns:
        List[float]: A list of proportions, where each value represents the fraction
                     of total distance variance explained by the corresponding
                     manifold component.
    """
    sq_dists_by_manifold = [M.pdist2(x_embed[:, pm.man2dim[i]]) for i, M in enumerate(pm.P)]
    total_sq_dist = pm.pdist2(x_embed)

    return [
        torch.sum(D.triu(diagonal=1) / torch.sum(total_sq_dist.triu(diagonal=1))).item() for D in sq_dists_by_manifold
    ]
