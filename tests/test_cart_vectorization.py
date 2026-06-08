"""Direct equivalence tests for the vectorized info-gain computation.

The live ``manify.predictors.decision_tree._get_info_gains_nobatch`` was
vectorized (sort + circular window + cumulative sums) in the PERFORMANCE step of
the CART refactor (see docs/design/cart_refactor.md, sections 3.3 and 8). These
tests assert the vectorized implementation reproduces a straightforward
reference double-loop ("for d: for j:") across random shapes and seeds.

* GINI: counts are integer-valued, so the vectorized ``ig`` must be *bit-exact*
  (``torch.equal``) to the reference loop. This is what keeps classification
  predictions bit-identical and the parity grid green.
* MSE: the vectorized path uses a cumulative-sum variance
  (``sumsq - sum**2/n``), which is algebraically equal to but not float-bit-
  identical to the loop's literal ``sum((x - mean)**2)`` (chosen option (b) in
  the task spec). We therefore assert ``allclose`` for MSE, and separately the
  MSE correctness is pinned against sklearn in test_cart_invariants.py.

The references below are frozen, self-contained copies of the pre-vectorization
double-loop implementation (GINI from tests/legacy/decision_tree_legacy.py; MSE
captured from the pre-edit live function, since legacy's MSE branch raised
NotImplementedError).
"""

import pytest
import torch

from manify.predictors.decision_tree import _angular_greater, _get_info_gains_nobatch


def _reference_info_gains(
    angles: torch.Tensor,
    labels: torch.Tensor,
    criterion: str = "gini",
    min_values_leaf: int = 1,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Frozen reference: the original O(n^2 * d) double-loop info-gain.

    GINI branch is the legacy nobatch oracle; MSE branch is the pre-edit live
    nobatch implementation (legacy nobatch raised NotImplementedError for MSE).
    """
    if criterion == "gini":
        pos_labels = torch.zeros((angles.shape[0], angles.shape[1], labels.shape[1]), device=angles.device)
        neg_labels = torch.zeros((angles.shape[0], angles.shape[1], labels.shape[1]), device=angles.device)

        for d in range(angles.shape[1]):
            for j in range(0, angles.shape[0]):
                mask = _angular_greater(angles[:, d], angles[j, d])
                pos_labels_entry = mask.float() * labels
                neg_labels_entry = ~mask * labels
                pos_labels[j, d, :] = pos_labels_entry.sum(dim=0)
                neg_labels[j, d, :] = neg_labels_entry.sum(dim=0)

        n_pos = pos_labels.sum(dim=-1) + eps
        n_neg = neg_labels.sum(dim=-1) + eps
        n_total = n_pos + n_neg

        pos_probs = pos_labels / n_pos.unsqueeze(-1)
        neg_probs = neg_labels / n_neg.unsqueeze(-1)
        total_probs = (pos_labels + neg_labels) / n_total.unsqueeze(-1)

        gini_pos = 1 - (pos_probs**2).sum(dim=-1)
        gini_neg = 1 - (neg_probs**2).sum(dim=-1)
        gini_total = 1 - (total_probs**2).sum(dim=-1)

    elif criterion == "mse":
        n_rows, n_dims = angles.shape
        n_pos = torch.zeros((n_rows, n_dims), device=angles.device)
        n_neg = torch.zeros((n_rows, n_dims), device=angles.device)
        ss_pos = torch.zeros((n_rows, n_dims), device=angles.device)
        ss_neg = torch.zeros((n_rows, n_dims), device=angles.device)

        for d in range(n_dims):
            for j in range(n_rows):
                mask = _angular_greater(angles[:, d], angles[j, d]).flatten()
                pos, neg = labels[mask], labels[~mask]
                n_pos[j, d], n_neg[j, d] = pos.shape[0], neg.shape[0]
                if pos.shape[0] > 0:
                    ss_pos[j, d] = ((pos - pos.mean()) ** 2).sum()
                if neg.shape[0] > 0:
                    ss_neg[j, d] = ((neg - neg.mean()) ** 2).sum()

        n_pos = n_pos + eps
        n_neg = n_neg + eps
        n_total = n_pos + n_neg

        gini_pos = ss_pos / n_pos
        gini_neg = ss_neg / n_neg
        gini_total = ((labels - labels.mean()) ** 2).mean()
    else:
        raise ValueError(f"Invalid criterion: {criterion}")

    ig = gini_total - (gini_pos * n_pos + gini_neg * n_neg) / n_total
    assert not ig.isnan().any()
    invalid_mask = torch.logical_or(n_pos < min_values_leaf, n_neg < min_values_leaf)
    ig[invalid_mask] = 0.0
    return ig


def _random_classification(n_rows, n_dims, n_classes, seed, duplicates=False):
    torch.manual_seed(seed)
    angles = (torch.rand(n_rows, n_dims) * 2 - 1) * torch.pi
    if duplicates and n_rows > 4:
        angles[1] = angles[0]
        angles[2] = angles[0]
    y = torch.randint(0, n_classes, (n_rows,))
    labels = torch.nn.functional.one_hot(y, n_classes).float()
    return angles, labels


SHAPES = [
    (2, 1, 2),
    (5, 2, 2),
    (10, 3, 3),
    (23, 4, 2),
    (37, 5, 4),
    (64, 3, 5),
    (80, 6, 3),
    (128, 2, 2),
]
SEEDS = [0, 1, 2, 3, 7]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_gini_bit_exact(shape, seed):
    """Vectorized GINI ig is bit-identical to the reference double loop."""
    n_rows, n_dims, n_classes = shape
    angles, labels = _random_classification(n_rows, n_dims, n_classes, seed)
    ref = _reference_info_gains(angles, labels, criterion="gini")
    got = _get_info_gains_nobatch(angles=angles, labels=labels, criterion="gini")
    assert torch.equal(ref, got), f"gini ig differs for shape={shape} seed={seed}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_gini_bit_exact_with_duplicates(shape, seed):
    """Bit-exact GINI even when angles contain exact duplicate values (tie handling)."""
    n_rows, n_dims, n_classes = shape
    angles, labels = _random_classification(n_rows, n_dims, n_classes, seed, duplicates=True)
    ref = _reference_info_gains(angles, labels, criterion="gini")
    got = _get_info_gains_nobatch(angles=angles, labels=labels, criterion="gini")
    assert torch.equal(ref, got), f"gini ig (dup) differs for shape={shape} seed={seed}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", [(s[0], s[1], 1) for s in SHAPES])
def test_mse_allclose(shape, seed):
    """Vectorized MSE ig matches the reference double loop to ~1e-6 (option (b))."""
    n_rows, n_dims, _ = shape
    torch.manual_seed(seed)
    angles = (torch.rand(n_rows, n_dims) * 2 - 1) * torch.pi
    labels = torch.randn(n_rows)
    ref = _reference_info_gains(angles, labels, criterion="mse")
    got = _get_info_gains_nobatch(angles=angles, labels=labels, criterion="mse")
    assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5), (
        f"mse ig differs beyond tol for shape={shape} seed={seed}: max abs diff {(ref - got).abs().max().item():.3e}"
    )


def test_mse_matches_sklearn_euclidean():
    """On a pure-Euclidean signature the MSE tree degenerates to ordinary sorted
    CART; its R^2 must match sklearn's DecisionTreeRegressor to high precision.

    This pins MSE *correctness* independently of the (option (b)) bit-noisy
    parity vs the legacy matmul MSE.
    """
    from sklearn.tree import DecisionTreeRegressor

    from manify.manifolds import ProductManifold
    from manify.predictors.decision_tree import ProductSpaceDT

    pm = ProductManifold(signature=[(0.0, 4)])
    X, y = pm.gaussian_mixture(num_points=120, num_classes=3, seed=0, task="regression")

    ours = ProductSpaceDT(pm=pm, task="regression", max_depth=4, n_features="d").fit(X, y)
    r2_ours = 1 - ((ours.predict(X) - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    skl = DecisionTreeRegressor(max_depth=4).fit(X.numpy(), y.numpy())
    r2_skl = skl.score(X.numpy(), y.numpy())

    assert abs(r2_ours.item() - r2_skl) < 1e-3, f"R2 mismatch: ours={r2_ours.item():.6f} sklearn={r2_skl:.6f}"


def test_argmax_picks_same_element():
    """The row-major argmax of the vectorized ig matches the reference (downstream selection)."""
    for seed in SEEDS:
        angles, labels = _random_classification(60, 4, 3, seed)
        ref = _reference_info_gains(angles, labels, criterion="gini")
        got = _get_info_gains_nobatch(angles=angles, labels=labels, criterion="gini")
        assert ref.argmax() == got.argmax(), f"argmax differs for seed={seed}"
