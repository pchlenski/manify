import torch
import numpy as np


def delta_hyperbolicity(dists: torch.Tensor):
    """
    computes delta hyperbolicity value from distance matrix for a single value rather than the whole 4-tensor as done in image krukov paper & repo. 
    Switched to using torch rather than numpy
    """
    p = 0
    row = dists[p, :].unsqueeze(0)  # (1,N)
    col = dists[:, p].unsqueeze(1)  # (N,1)
    XY_p = 0.5 * (row + col - dists)

    XY_p_expanded_1 = XY_p.unsqueeze(2).expand(-1, -1, XY_p.size(0))
    XY_p_expanded_2 = XY_p.unsqueeze(0).expand(XY_p.size(0), -1, -1)

    minimum_vals = torch.minimum(XY_p_expanded_1, XY_p_expanded_2)

    maxmin = torch.max(minimum_vals, dim=1)[0]
    return torch.max(maxmin - XY_p)


def delta_full(dismat):
    # We are getting (y,z)_x = .5 (d(x,y) + d(x,z) - d(y,z))
    n = dismat.shape[0]
    p = 0  # Fix w_0
    d_xp = dismat[p, :].unsqueeze(0)  # (1,n)
    d_yp = dismat[:, p].unsqueeze(1)  # (n,1)
    # (n,n) matrix of all pairwise gromov products for fixed point
    XY_p = .5 * (d_xp + d_yp - dismat)

    XY_p_xy = XY_p.unsqueeze(2).expand(-1, -1, n)  # (n,n,n)
    XY_p_yz = XY_p.unsqueeze(0).expand(n, -1, -1)  # (n,n,n)
    XY_p_xz = XY_p.unsqueeze(1).expand(-1, n, -1)  # (n,n,n)

    # Return the 3-tensor of delta values before taking the max
    minmax = torch.minimum(XY_p_xy, XY_p_yz)
    return minmax - XY_p_xz
