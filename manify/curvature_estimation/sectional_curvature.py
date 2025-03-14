import torch
from jaxtyping import Float

import random
import numpy as np



# The next couple functions are taken from this repo:
# https://github.com/HazyResearch/hyperbolics
# Paper: https://openreview.net/pdf?id=HJxeWnCcF7


def sectional_curvature(dismat, n_samples=100, k_neighbors=5):

    V = dismat.shape[0]
    curvatures = []
    
    for _ in range(n_samples):
        # Select a random node m
        m = torch.randint(0, V, (1,)).item()
        
        # Get distances from m to all other points
        distances = dismat[m].clone()
        distances[m] = float('inf')  # Exclude self
        
        # Find k_neighbors closest neighbors
        neighbor_values, neighbor_indices = torch.topk(distances, k_neighbors, largest=False)
        
        if torch.isinf(neighbor_values).any() or (neighbor_values == 0).any():
            continue
            
        # Select two random neighbors of m from the k closest
        if len(neighbor_indices) >= 2:
            perm = torch.randperm(len(neighbor_indices))
            b, c = neighbor_indices[perm[0]].item(), neighbor_indices[perm[1]].item()
            
            # Select another random node a (different from m, b, c)
            excluded = {m, b, c}
            available = [i for i in range(V) if i not in excluded]
            a = available[torch.randint(0, len(available), (1,)).item()]
            
            # Calculate sectional curvature
            k = (dismat[a, m] ** 2 + dismat[b, c] ** 2 / 4.0 - 
                 (dismat[a, b] ** 2 + dismat[a, c] ** 2) / 2.0)
            k /= (2 * dismat[a, m])
            
            curvatures.append(k)
    
    if not curvatures:
        return torch.tensor(0.0)
    
    # Return mean and also provide standard deviation for confidence
    curvature_tensor = torch.tensor(curvatures)
    mean_curvature = curvature_tensor.mean()
    
    return mean_curvature

def naive_sectional_curvature(dismat, n_samples=10000, threshold=.1):
    V = dismat.shape[0]
    curvatures = []
    skip=0
    
    for _ in range(n_samples):
        m = random.randint(0,V-1)

        neighorhood_mask = (dismat[m] < threshold)
        if neighorhood_mask.sum() < 2:
            skip += 1
            continue
        neighbor_indices = torch.where(neighorhood_mask)[0]

        bc_indices = torch.randperm(len(neighbor_indices))[:2]
        b, c = neighbor_indices[bc_indices[0]].item(), neighbor_indices[bc_indices[1]].item()
        
        # Select another random node a
        a = torch.randint(0, V, (1,)).item()
        
        # Calculate sectional curvature
        k = (dismat[a, m] ** 2 + dismat[b, c] ** 2 / 4.0 - (dismat[a, b] ** 2 + dismat[a, c] ** 2) / 2.0)
        k /= (2 * dismat[a, m])
        
        curvatures.append(k)
    print(skip)
    return torch.tensor(curvatures).mean() if curvatures else torch.tensor(0.0)

    

def Ka(D, m, b, c, a):
    if a == m:
        return 0.0
    k = D[a][m] ** 2 + D[b][c] ** 2 / 4.0 - (D[a][b] ** 2 + D[a][c] ** 2) / 2.0
    k /= 2 * D[a][m]
    return k


def K(D, n, m, b, c):
    ks = [Ka(D, m, b, c, a) for a in range(n)]
    return np.mean(ks)


def ref(D, size, n, m, b, c):
    ks = []
    for i in range(n):
        a = random.randint(0, size - 1)
        if a == b or a == c:
            continue
        else:
            ks.append(Ka(D, m, b, c, a))
    return np.mean(ks)


def estimate_curvature(G, D, n):
    for m in range(n):
        ks = []
        edges = list(G.edges(m))
        for i in range(len(edges)):
            for j in range(b, len(edges)):
                b = edges[i]
                c = edges[j]
                ks.append(K(D, n, b, c))
    return None


def sample(D, size, n_samples=100):
    samples = []
    _cnt = 0
    while _cnt < n_samples:
        a, b, c, m = random.sample(range(0, size), 4)
        k = Ka(D, m, b, c, a)
        samples.append(k)

        _cnt += 1

    return np.array(samples)


def estimate(D, size, n_samples):
    samples = sample(D, size, n_samples)
    m1 = np.mean(samples)
    m2 = np.mean(samples**2)
    return samples


def estimate_diff(D, size, n_sample, num):
    samples = []
    _cnt = 0
    while _cnt < n_sample:
        b, c, m = random.sample(range(0, size), 3)
        k = ref(D, size, num, m, b, c)
        # k=K(D, n, m, b, c)
        samples.append(k)
        _cnt += 1
    return np.array(samples)
