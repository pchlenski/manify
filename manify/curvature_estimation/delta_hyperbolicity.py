from jaxtyping import Float
import torch

def sampled_delta_hyperbolicity(dismat, n_samples=1000, reference_idx=0):
    n = dismat.shape[0]
    # Sample n_samples triplets of points randomly
    indices = torch.randint(0, n, (n_samples, 3))

    # Get gromov products
    # (j,k)_i = .5 (d(i,j) + d(i,k) - d(j,k))

    x,y,z = indices.T
    w = reference_idx # set reference point

    xy_w = .5 * (dismat[w,x] + dismat[w,y] - dismat[x,y])
    xz_w = .5 * (dismat[w,x] + dismat[w,z] - dismat[x,z])
    yz_w = .5 * (dismat[w,y] + dismat[w,z] - dismat[y,z])

    # delta(x,y,z) = min((x,y)_w,(y-z)_w) - (x,z)_w
    deltas = torch.minimum(xy_w,yz_w) - xz_w
    diam = torch.max(dismat)
    rel_deltas = 2 * deltas / diam

    return rel_deltas, indices

def iterative_delta_hyperbolicity(dismat):
    """delta(x,y,z) = min((x,y)_w,(y-z)_w) - (x,z)_w"""
    n = dismat.shape[0]
    w = 0
    gromov_products = torch.zeros((n,n))
    deltas = torch.zeros((n,n,n))

    # Get Gromov Products
    for x in range(n):
        for y in range(n):
            gromov_products[x,y] = gromov_product(w,x,y,dismat)

    # Get Deltas
    for x in range(n):
        for y in range(n):
            for z in range(n):
                xz_w = gromov_products[x,z]
                xy_w = gromov_products[x,y]
                yz_w = gromov_products[y,z]
                deltas[x,y,z] = torch.minimum(xy_w,yz_w) - xz_w
    
    diam = torch.max(dismat)
    rel_deltas = 2 * deltas / diam

    return rel_deltas, gromov_products


def gromov_product(i,j,k,dismat):
    """(j,k)_i = 0.5 (d(i,j) + d(i,k) - d(j,k))"""
    d_ij = dismat[i,j]
    d_ik = dismat[i,k]
    d_jk = dismat[j,k]
    return 0.5 * (d_ij + d_ik - d_jk)

def delta_hyperbolicity(dismat, relative=True, device='cpu', full=True):
    # Change: what happens if we sample specific indices from the entire thing (sample i,j,k indepedently) then use that to get the quantities from the distance matrix
    # Just sample triplets and calculate gromov products for those
    # We are always working with vectors of length n
    # For now, remove sampling 
    n = dismat.shape[0]
    p = 0

    row = dismat[p, :].unsqueeze(0)  # (1,N)
    col = dismat[:, p].unsqueeze(1)  # (N,1)
    XY_p = 0.5 * (row + col - dismat)

    XY_p_xy = XY_p.unsqueeze(2).expand(-1, -1, n)  # (n,n,n)
    XY_p_yz = XY_p.unsqueeze(0).expand(n, -1, -1)  # (n,n,n)
    XY_p_xz = XY_p.unsqueeze(1).expand(-1, n, -1)  # (n,n,n)

    out = torch.minimum(XY_p_xy, XY_p_yz)

    if not full:
        delta = (out - XY_p_xz).max().item()
    else:
        delta = out - XY_p_xz

    if relative:
        diam = torch.max(dismat).item()
        delta = 2 * delta / diam
    
    return delta

    
