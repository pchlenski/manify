"""Oracle-free semantic tests for the stereographic transformer layers.

Points are always built via ``pm.stereographic(X)`` (never ``pm.sample`` on a stereographic
manifold, which crashes). Tests cover: smoke/shape, on-manifold validity, permutation equivariance,
masking (an edge that is masked out cannot let one node influence another), gradient finiteness, the
curvature -> 0 Euclidean limit, and determinism under a fixed seed.
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
        assert pm.manifold.check_point(
            o
        ), "Output must lie on the stereographic manifold"
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
    assert torch.allclose(
        out[perm], out_perm, atol=ATOL
    ), "Block must be permutation equivariant (None mask)"

    # Same with an explicit all-ones mask (also permuted along both axes).
    mask = torch.ones(n, n)
    with torch.no_grad():
        out2 = block(X, mask)
        out2_perm = block(X[perm], mask[perm][:, perm])
    assert torch.allclose(
        out2[perm], out2_perm, atol=ATOL
    ), "Block must be permutation equivariant (ones mask)"


def test_masking_blocks_influence():
    """A masked-out edge cannot let one node influence another's output."""
    pm, X = _make_points(SIGNATURE, n_nodes=6, seed=4)
    n, d = X.shape
    block = StereographicTransformer(
        pm, num_heads=2, dim=d, head_dim=3, use_layer_norm=False
    ).eval()

    # Diagonal mask: each node attends only to itself, so node 0 cannot see node 1.
    mask = torch.eye(n)
    X_perturbed = X.clone()
    X_perturbed[1] = pm.manifold.projx(X[1] + 0.5 * torch.randn(d))

    with torch.no_grad():
        out = block(X, mask)
        out_perturbed = block(X_perturbed, mask)

    assert torch.allclose(
        out[0], out_perturbed[0], atol=1e-6
    ), "Masked-out node 1 must not affect node 0"
    assert not torch.allclose(
        out[1], out_perturbed[1], atol=1e-6
    ), "Node 1's own perturbation should change its output"


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
            assert torch.isfinite(
                p.grad
            ).all(), f"Parameter {name} has non-finite gradient"
            n_grad += 1
    assert n_grad > 0, "Block should have trainable parameters"


def test_kappa_to_zero_euclidean_limit():
    """As curvature -> 0 the block approaches a plain Euclidean linear-attention block."""
    torch.manual_seed(6)
    X = torch.randn(6, 4) * 0.1

    # Euclidean reference (curvature 0): logmap0/expmap0 are identities.
    pm0 = ProductManifold(signature=[(0.0, 4)], stereographic=True)
    block0 = StereographicTransformer(
        pm0, num_heads=2, dim=4, head_dim=3, use_layer_norm=False
    ).eval()
    with torch.no_grad():
        out0 = block0(X)
    assert pm0.manifold.check_point(out0)

    # Small-curvature manifolds with identical weights should approach the Euclidean output.
    prev = None
    for eps in (1e-1, 1e-2, 1e-3):
        pm_eps = ProductManifold(signature=[(-eps, 2), (eps, 2)], stereographic=True)
        block_eps = StereographicTransformer(
            pm_eps, num_heads=2, dim=4, head_dim=3, use_layer_norm=False
        ).eval()
        block_eps.load_state_dict(block0.state_dict())
        with torch.no_grad():
            out_eps = block_eps(X)
        err = (out_eps - out0).abs().max().item()
        assert torch.isfinite(out_eps).all() and out_eps.shape == out0.shape
        if prev is not None:
            assert (
                err < prev
            ), "Output should approach the Euclidean limit as curvature shrinks"
        prev = err
    # At the smallest curvature the block is close to Euclidean (loose tolerance).
    assert prev < 1e-2, f"kappa->0 limit too far from Euclidean reference: {prev}"


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
    assert torch.allclose(
        out_a, out_b, atol=0.0
    ), "Same seed must give identical outputs"


def test_geometric_linearized_attention_full_matches_masked():
    """Full-attention fast path equals the all-ones masked path (same kernel attention)."""
    torch.manual_seed(8)
    n_heads, n, head_dim = 2, 5, 3
    Q = torch.randn(n_heads, n, head_dim)
    K = torch.randn(n_heads, n, head_dim)
    V = torch.randn(n_heads, n, head_dim)

    attn = GeometricLinearizedAttention(num_heads=n_heads, head_dim=head_dim)
    out_full = attn(Q, K, V, None)
    out_ones = attn(Q, K, V, torch.ones(n, n))
    assert torch.allclose(
        out_full, out_ones, atol=1e-5
    ), "None mask should equal an all-ones mask"
