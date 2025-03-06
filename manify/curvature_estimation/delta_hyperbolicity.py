import torch

def delta_hyperbolicity_iterative(distance_matrix):
    n = distance_matrix.shape[0]
    device = distance_matrix.device

    # Gromov Products
    gromov_products = torch.zeros((n, n, n), device=device)

    for x in range(n):
        for y in range(n):
            for z in range(n):
                # Gromov product formula: (y,z)_x = 1/2 * (d(x,y) + d(x,z) - d(y,z))
                gromov_products[x, y, z] = 0.5 * (
                    distance_matrix[x, y] +
                    distance_matrix[x, z] -
                    distance_matrix[y, z]
                )

    delta = 0.0

    for w in range(n):
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    # Four-point condition: (x,z)_w ≥ min((x,y)_w, (y,z)_w) - δ
                    # Solving for δ: δ ≥ min((x,y)_w, (y,z)_w) - (x,z)_w
                    current_delta = min(
                        gromov_products[w, x, y], gromov_products[w, y, z]) - gromov_products[w, x, z]
                    delta = max(delta, current_delta)

    # Calculate diameter for the relative delta
    diameter = torch.max(distance_matrix)
    delta_relative = 2 * delta / diameter

    return delta, delta_relative


def delta_hyperbolicity_parallel(distance_matrix):

    n = distance_matrix.shape[0]
    device = distance_matrix.device

    # Step 1: Calculate the matrix of Gromov products
    # We can compute this for all points at once

    # Expand dimensions to create (n, n, n) tensors
    d_xw = distance_matrix.unsqueeze(2).expand(n, n, n)  # d(x,w) for all w
    d_yw = distance_matrix.unsqueeze(1).expand(n, n, n)  # d(y,w) for all w
    d_xy = distance_matrix.unsqueeze(0).expand(n, n, n)  # d(x,y) for all pairs

    # Calculate Gromov products: (y,z)_x = 1/2 * (d(x,y) + d(x,z) - d(y,z))
    gromov_products = 0.5 * \
        (d_xw + d_yw.transpose(1, 2) - d_xy.transpose(0, 2))

    # Step 2: Compute the min-max matrix product for the 4-point condition
    delta_values = torch.zeros(n, device=device)

    for w in range(n):
        A = gromov_products[w]  # Matrix of (x,y)_w for fixed w

        # Compute min-max product (A ⊗ A)_ij = max_k min(A_ik, A_kj)
        A_min_max = torch.zeros((n, n), device=device)

        for i in range(n):
            for j in range(n):
                min_values = torch.minimum(A[i, :], A[:, j])
                A_min_max[i, j] = torch.max(min_values)

        # The delta for this w is the largest value in (A ⊗ A) - A
        delta_w = torch.max(A_min_max - A)
        delta_values[w] = delta_w

    # The overall delta is the maximum across all w
    delta = torch.max(delta_values)

    # Calculate diameter for the relative delta
    diameter = torch.max(distance_matrix)
    delta_relative = 2 * delta / diameter

    return delta, delta_relative


def min_max_matrix_product(A):
    """
    Compute the min-max matrix product
    """
    n = A.shape[0]
    device = A.device
    result = torch.zeros((n, n), device=device)

    # For each (i, j) entry in the result
    for i in range(n):
        for j in range(n):
            # Compute min(A_ik, A_kj) for all k, then take max
            min_values = torch.minimum(A[i, :], A[:, j])
            result[i, j] = torch.max(min_values)

    return result


def calculate_delta_hyperbolicity_optimized(distance_matrix):
    n = distance_matrix.shape[0]
    device = distance_matrix.device

    # According to the paper, we can fix w to any point (w=0)
    w = 0

    # Calculate Gromov products for fixed w
    A = torch.zeros((n, n), device=device)
    for x in range(n):
        for y in range(n):
            A[x, y] = 0.5 * (
                distance_matrix[w, x] +
                distance_matrix[w, y] -
                distance_matrix[x, y]
            )

    # Calculate min-max matrix product
    A_min_max = min_max_matrix_product(A)

    # Delta is the largest value in (A ⊗ A) - A
    delta = torch.max(A_min_max - A)

    # Calculate diameter for the relative delta
    diameter = torch.max(distance_matrix)
    delta_relative = 2 * delta / diameter

    return delta.item(), delta_relative.item()

def approx_delta_hyperbolicity(dists: torch.Tensor):
    """
    computes delta hyperbolicity value from distance matrix for a single value rather than the whole 4-tensor as done in image krukov paper & repo. 
    Switched to using torch rather than numpy
    """
    p = 0
    row = dists[p, :].unsqueeze(0) # (1,N)
    col = dists[:, p].unsqueeze(1) # (N,1)
    XY_p = 0.5 * (row + col - dists)

    # Create expanded tensors for minimum comparison
    XY_p_expanded_1 = XY_p.unsqueeze(2).expand(-1, -1, XY_p.size(0))
    XY_p_expanded_2 = XY_p.unsqueeze(0).expand(XY_p.size(0), -1, -1)

    # Calculate minimum along expanded dimension
    minimum_vals = torch.minimum(XY_p_expanded_1, XY_p_expanded_2)

    # Get max of minimum values along axis 1
    maxmin = torch.max(minimum_vals, dim=1)[0]

    # Calculate final result
    return torch.max(maxmin - XY_p)
