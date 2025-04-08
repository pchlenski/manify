from jaxtyping import Float
import torch

def delta_hyperbolicity(dismat, sample_percent=1, relative=True, device='cpu', full=False):
    n = int(dismat.shape[0] * sample_percent)
    p = 0

    indices = torch.randperm(dismat.shape[0])[:n]

    if sample_percent == 1:
        sampled_dismat = dismat
    else:
        sampled_dismat = dismat[indices][:, indices]

    row = sampled_dismat[p, :].unsqueeze(0)  # (1,N)
    col = sampled_dismat[:, p].unsqueeze(1)  # (N,1)
    XY_p = 0.5 * (row + col - sampled_dismat)

    XY_p_xy = XY_p.unsqueeze(2).expand(-1, -1, n)  # (n,n,n)
    XY_p_yz = XY_p.unsqueeze(0).expand(n, -1, -1)  # (n,n,n)
    XY_p_xz = XY_p.unsqueeze(1).expand(-1, n, -1)  # (n,n,n)

    out = torch.minimum(XY_p_xy, XY_p_yz)

    if not full:
        delta = (out - XY_p_xz).max().item()
    else:
        delta = torch.abs(out - XY_p_xz)

    if relative:
        diam = torch.max(sampled_dismat).item()
        return 2 * delta / diam
    else:
        return delta
    

def old_delta_hyperbolicity(dismat: Float[torch.Tensor, "n_points n_points"]) -> Float[torch.Tensor, ""]:
    """
    Compute the delta hyperbolicity of a metric space given its distance matrix.
    
    The delta hyperbolicity is a measure of how close a metric space is to being a tree.
    
    Args:
        dismat (n_points, n_points): A tensor containing the pairwise distances between points in the metric space.
        
    Returns:
        float: The delta hyperbolicity of the metric space, with smaller values indicating a more tree-like space.
    """
    n = dismat.shape[0]
    p = 0
    row = dismat[p, :].unsqueeze(0)  # (1,N)
    col = dismat[:, p].unsqueeze(1)  # (N,1)
    XY_p = 0.5 * (row + col - dismat)

    XY_p_xy = XY_p.unsqueeze(2).expand(-1, -1, n)  # (n,n,n)
    XY_p_yz = XY_p.unsqueeze(0).expand(n, -1, -1)  # (n,n,n)
    XY_p_xz = XY_p.unsqueeze(1).expand(-1, n, -1)  # (n,n,n)

    out = torch.minimum(XY_p_xy, XY_p_yz)
    
    delta = (out - XY_p_xz).max()
    
    diam = torch.max(dismat)

    rel_delta = 2 * delta / diam

    return rel_delta.item()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    deltas = minmax - XY_p_xz
    return torch.abs(deltas)

def rel_deltas_full(dismat):
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
    
    deltas = minmax - XY_p_xz
    diam = torch.max(dismat)
    rel_deltas = 2 * deltas / diam
    return torch.abs(rel_deltas)

# def batched_delta_hyp(X: torch.Tensor, n_tries=10, batch_size=1500):
#     deltas = []

#     for i in tqdm(range(n_tries)):
#         idx = torch.randperm(len(X), device=device)[:batch_size]
#         X_batch = X[idx]
#         dists = torch.cdist(X_batch, X_batch, p=2)
#         diam = torch.max(dists)
#         delta_rel = 2 * delta_hyperbolicity(dists) / diam
#         deltas.append(delta_rel.item())
#     delta_tensor = torch.tensor(deltas, device=device)
#     return delta_tensor.mean().item(), delta_tensor.std().item()



