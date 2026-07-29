"""Oracle-free semantic tests for the stereographic transformer layers.

Points are always built via ``pm.stereographic(X)`` (never ``pm.sample`` on a stereographic
manifold, which crashes). Tests cover: smoke/shape, on-manifold validity, permutation equivariance,
masking (an edge that is masked out cannot let one node influence another), gradient finiteness, the
curvature -> 0 Euclidean limit, determinism under a fixed seed, and -- crucially -- that the
aggregation is the gyrovector Einstein midpoint (it differs from the Euclidean mean of the value
points at finite curvature and converges to it as curvature -> 0), guarding against a tangent-space
/ Euclidean-mean shortcut.
"""

import torch

from manify.manifolds import ProductManifold
from manify.predictors.nn.layers import (
    GeometricLinearizedAttention,
    StereographicAttention,
    StereographicLayerNorm,
    StereographicTransformer,
)

ATOL = 1e-5


def _make_points(signature, n_nodes=8, seed=0):
    """Build on-manifold stereographic points via the sample -> stereographic path."""
    torch.manual_seed(seed)
    pm = ProductManifold(signature=signature)
    X = pm.sample(n_nodes)
    pm_stereo, X_stereo = pm.stereographic(X)
    return pm_stereo, X_stereo


SIGNATURE = [(-1.0, 2), (0.0, 2), (1.0, 2)]


def test_layers_smoke_and_shape():
    """Each layer and the full block run and return [n_nodes, dim]."""
    pm, X = _make_points(SIGNATURE, n_nodes=7, seed=1)
    n, d = X.shape

    ln = StereographicLayerNorm(pm, d)
    assert ln(X).shape == (n, d)

    attn = StereographicAttention(pm, num_heads=2, dim=d, head_dim=3)
    assert attn(X).shape == (n, d)
    assert attn(X, torch.ones(n, n)).shape == (n, d)

    block = StereographicTransformer(pm, num_heads=2, dim=d, head_dim=3)
    assert block(X).shape == (n, d)
    assert block(X, torch.ones(n, n)).shape == (n, d)


def test_outputs_on_manifold():
    """Outputs are valid stereographic points (check_point + projection idempotence)."""
    pm, X = _make_points(SIGNATURE, n_nodes=9, seed=2)
    d = X.shape[1]

    block = StereographicTransformer(pm, num_heads=3, dim=d, head_dim=2).eval()
    with torch.no_grad():
        out = block(X)
        out_masked = block(X, torch.ones(X.shape[0], X.shape[0]))

    for o in (out, out_masked):
        assert pm.manifold.check_point(o), "Output must lie on the stereographic manifold"
        # Projection idempotence within tolerance.
        assert torch.allclose(pm.manifold.projx(o), o, atol=1e-4)


def test_permutation_equivariance_full_mask():
    """Permuting input nodes permutes outputs identically under full attention."""
    pm, X = _make_points(SIGNATURE, n_nodes=8, seed=3)
    n, d = X.shape
    block = StereographicTransformer(pm, num_heads=2, dim=d, head_dim=3).eval()

    perm = torch.randperm(n)
    with torch.no_grad():
        out = block(X)
        out_perm = block(X[perm])
    assert torch.allclose(out[perm], out_perm, atol=ATOL), "Block must be permutation equivariant (None mask)"

    # Same with an explicit all-ones mask (also permuted along both axes).
    mask = torch.ones(n, n)
    with torch.no_grad():
        out2 = block(X, mask)
        out2_perm = block(X[perm], mask[perm][:, perm])
    assert torch.allclose(out2[perm], out2_perm, atol=ATOL), "Block must be permutation equivariant (ones mask)"


def test_masking_blocks_influence():
    """A masked-out edge cannot let one node influence another's output."""
    pm, X = _make_points(SIGNATURE, n_nodes=6, seed=4)
    n, d = X.shape
    block = StereographicTransformer(pm, num_heads=2, dim=d, head_dim=3, use_layer_norm=False).eval()

    # Diagonal mask: each node attends only to itself, so node 0 cannot see node 1.
    mask = torch.eye(n)
    X_perturbed = X.clone()
    X_perturbed[1] = pm.manifold.projx(X[1] + 0.5 * torch.randn(d))

    with torch.no_grad():
        out = block(X, mask)
        out_perturbed = block(X_perturbed, mask)

    assert torch.allclose(out[0], out_perturbed[0], atol=1e-6), "Masked-out node 1 must not affect node 0"
    assert not torch.allclose(out[1], out_perturbed[1], atol=1e-6), "Node 1's own perturbation should change its output"


def test_gradient_finiteness():
    """Backward from a scalar loss yields finite grads for every parameter."""
    pm, X = _make_points(SIGNATURE, n_nodes=7, seed=5)
    d = X.shape[1]
    block = StereographicTransformer(pm, num_heads=2, dim=d, head_dim=3)

    out = block(X, torch.ones(X.shape[0], X.shape[0]))
    loss = out.pow(2).sum()
    loss.backward()

    n_grad = 0
    for name, p in block.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} received no gradient"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"
            n_grad += 1
    assert n_grad > 0, "Block should have trainable parameters"


def test_kappa_to_zero_euclidean_limit():
    """As curvature -> 0 (input pm AND per-head curvatures) the block approaches Euclidean.

    The faithful block reduces to a Euclidean linear-attention transformer only when BOTH the input
    manifold and the learnable per-head curvatures vanish; we drive both to zero together.
    """
    torch.manual_seed(6)
    X = torch.randn(6, 4, dtype=torch.float64) * 0.1

    # Euclidean reference: input curvature 0 and per-head curvatures 0.
    pm0 = ProductManifold(signature=[(0.0, 4)], stereographic=True)
    block0 = StereographicTransformer(
        pm0, num_heads=2, dim=4, head_dim=3, use_layer_norm=False, init_curvatures=[0.0, 0.0]
    ).eval()
    with torch.no_grad():
        out0 = block0(X)
    assert pm0.manifold.check_point(out0)

    # Small-curvature manifolds (and per-head curvatures) with identical weights should converge.
    prev = None
    for eps in (1e-1, 1e-2, 1e-3):
        pm_eps = ProductManifold(signature=[(-eps, 2), (eps, 2)], stereographic=True)
        block_eps = StereographicTransformer(
            pm_eps, num_heads=2, dim=4, head_dim=3, use_layer_norm=False, init_curvatures=[-eps, eps]
        ).eval()
        # Copy Euclidean weights, then set the per-head curvatures to +/- eps.
        block_eps.load_state_dict(block0.state_dict(), strict=False)
        with torch.no_grad():
            block_eps.mha.curvatures.copy_(torch.tensor([-eps, eps]))
            out_eps = block_eps(X)
        err = (out_eps - out0).abs().max().item()
        assert torch.isfinite(out_eps).all() and out_eps.shape == out0.shape
        if prev is not None:
            assert err < prev, "Output should approach the Euclidean limit as curvature shrinks"
        prev = err
    # At the smallest curvature the block is close to Euclidean (loose tolerance).
    assert prev < 1e-2, f"kappa->0 limit too far from Euclidean reference: {prev}"


def test_aggregation_is_einstein_midpoint_not_euclidean_mean():
    """The aggregation is the gyrovector Einstein midpoint, NOT the Euclidean mean of values.

    With zero queries/keys (uniform attention weights) the attention output is the Einstein midpoint
    of the value points. At finite curvature this must DIFFER from the arithmetic mean of the value
    points, and it must converge to that mean as curvature -> 0. This is the test that catches a
    tangent-space / Euclidean-mean shortcut.
    """
    from geoopt.manifolds.stereographic import math as smath

    torch.manual_seed(11)
    n_heads, n, head_dim = 1, 6, 3
    attn = GeometricLinearizedAttention(num_heads=n_heads, head_dim=head_dim)
    # Uniform weights: phi(elu(0)+1)=1 everywhere -> equal attention over all values.
    Q = torch.zeros(n_heads, n, head_dim)
    K = torch.zeros(n_heads, n, head_dim)

    prev_gap = None
    finite_curvature_gap = None
    for kv in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        k = torch.full((n_heads, 1, 1), float(kv))
        V = smath.project(torch.randn(n_heads, n, head_dim) * 0.3, k=k)
        with torch.no_grad():
            out = attn(Q, K, V, curvatures=k, mask=torch.ones(n, n))
        # Every query gets the same (uniform) midpoint; compare against the Euclidean mean.
        midpoint = out[0, 0]
        euclidean_mean = V[0].mean(dim=0)
        gap = (midpoint - euclidean_mean).norm().item()
        if kv == 1.0:
            finite_curvature_gap = gap
        if prev_gap is not None:
            assert gap < prev_gap, "Einstein midpoint should approach the Euclidean mean as kappa -> 0"
        prev_gap = gap

    assert finite_curvature_gap is not None and finite_curvature_gap > 1e-3, (
        "At finite curvature the Einstein midpoint must differ from the Euclidean mean "
        f"(got gap={finite_curvature_gap})"
    )
    assert prev_gap < 1e-3, f"Einstein midpoint should converge to the Euclidean mean as kappa->0 (got {prev_gap})"


def test_determinism_fixed_seed():
    """A fixed seed yields identical layer parameters and identical outputs."""
    pm, X = _make_points(SIGNATURE, n_nodes=8, seed=7)
    d = X.shape[1]

    torch.manual_seed(123)
    block_a = StereographicTransformer(pm, num_heads=2, dim=d, head_dim=3).eval()
    torch.manual_seed(123)
    block_b = StereographicTransformer(pm, num_heads=2, dim=d, head_dim=3).eval()

    with torch.no_grad():
        out_a = block_a(X)
        out_b = block_b(X)
    assert torch.allclose(out_a, out_b, atol=0.0), "Same seed must give identical outputs"


def test_geometric_linearized_attention_full_matches_masked():
    """The kernelized full-attention path equals the all-ones masked path exactly."""
    from geoopt.manifolds.stereographic import math as smath

    torch.manual_seed(8)
    n_heads, n, head_dim = 2, 5, 3
    k = torch.tensor([-1.0, 1.0]).view(n_heads, 1, 1)
    V = smath.project(torch.randn(n_heads, n, head_dim) * 0.2, k=k)
    Q = torch.randn(n_heads, n, head_dim) * 0.2
    K = torch.randn(n_heads, n, head_dim) * 0.2

    attn = GeometricLinearizedAttention(num_heads=n_heads, head_dim=head_dim)
    out_full = attn(Q, K, V, curvatures=k, mask=None)
    out_ones = attn(Q, K, V, curvatures=k, mask=torch.ones(n, n))
    assert torch.allclose(out_full, out_ones, atol=1e-5), "None mask should equal an all-ones mask"


def test_geometric_linearized_attention_matches_brute_force_midpoint():
    """The kernelized aggregation equals a brute-force Einstein-midpoint computation."""
    from geoopt.manifolds.stereographic import math as smath

    torch.manual_seed(9)
    n_heads, n, head_dim = 1, 4, 3
    k = torch.tensor([1.0]).view(n_heads, 1, 1)
    V = smath.project(torch.randn(n_heads, n, head_dim) * 0.2, k=k)
    Q = torch.randn(n_heads, n, head_dim) * 0.2
    K = torch.randn(n_heads, n, head_dim) * 0.2

    attn = GeometricLinearizedAttention(num_heads=n_heads, head_dim=head_dim)
    with torch.no_grad():
        out = attn(Q, K, V, curvatures=k, mask=torch.ones(n, n))

    # Brute-force Einstein midpoint (Eq 7).
    q_tilde = smath.parallel_transport0back(V, Q, k=k)
    k_tilde = smath.parallel_transport0back(V, K, k=k)
    phi_q = torch.nn.functional.elu(q_tilde) + 1
    phi_k = torch.nn.functional.elu(k_tilde) + 1
    alpha = torch.einsum("hnd,hmd->hnm", phi_q, phi_k)  # [H, N, N]
    lam = smath.lambda_x(x=V, k=k, keepdim=True, dim=-1)  # [H, N, 1]
    out_bf = torch.zeros(n_heads, n, head_dim)
    for i in range(n):
        denom = (alpha[0, i] * (lam[0, :, 0] - 1)).sum()
        coef = alpha[0, i] * lam[0, :, 0] / denom
        out_bf[0, i] = (coef[:, None] * V[0]).sum(0)
    out_bf = smath.project(out_bf, k=k)
    out_bf = smath.mobius_scalar_mul(torch.tensor(0.5), out_bf, k=k, dim=-1)
    out_bf = smath.project(out_bf, k=k)

    assert torch.allclose(out, out_bf, atol=1e-5), "Kernelized aggregation must match brute-force Einstein midpoint"
