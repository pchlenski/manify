"""CART refactor parity tests: the live ProductSpaceDT/RF must match the frozen
legacy oracle (tests/legacy/decision_tree_legacy.py).

Phase A enforces bit-exact parity on ALL signatures. The noncircular tests are
PERMANENT (Euclidean/Hyperbolic-only signatures never hit the spherical
circular-midpoint path); the circular tests are Phase-A-only and are removed in
Phase B when the spherical wraparound behaviour is intentionally fixed. See
docs/design/cart_refactor.md.

Discrete decisions (class predictions) are compared with torch.equal; continuous
outputs (class probabilities, regression values) use a tight allclose to tolerate
ULP-level reordering from vectorization while catching any real change.
"""

import pytest
import torch

from manify.manifolds import ProductManifold
from manify.predictors.decision_tree import ProductSpaceDT, ProductSpaceRF
from tests.legacy.decision_tree_legacy import ProductSpaceDT as LegacyDT
from tests.legacy.decision_tree_legacy import ProductSpaceRF as LegacyRF

NONCIRCULAR_SIGS = [
    [(0.0, 3)],
    [(-1.0, 3)],
    [(-1.0, 2), (0.0, 2)],
    [(-1.0, 2), (-1.0, 2)],
]
CIRCULAR_SIGS = [
    [(1.0, 3)],
    [(1.0, 2), (-1.0, 2), (0.0, 2)],
]
TASKS = ["classification", "regression"]
SEEDS = [0, 1, 2]
NFEATS = ["d", "d_choose_2"]

_ATOL, _RTOL = 1e-6, 1e-5


def _data(sig, seed, task):
    pm = ProductManifold(signature=sig)
    X, y = pm.gaussian_mixture(num_points=80, num_classes=3, seed=seed, task=task)
    return pm, X, y


def _assert_dt_parity(sig, task, seed, nfeat):
    pm, X, y = _data(sig, seed, task)
    kw = dict(task=task, max_depth=4, n_features=nfeat)
    new = ProductSpaceDT(pm=pm, **kw).fit(X, y)
    old = LegacyDT(pm=pm, **kw).fit(X, y)
    if task == "classification":
        assert torch.equal(new.predict(X), old.predict(X)), "class predictions differ from legacy"
        assert torch.allclose(new.predict_proba(X), old.predict_proba(X), atol=_ATOL, rtol=_RTOL)
    else:
        assert torch.allclose(new.predict(X), old.predict(X), atol=_ATOL, rtol=_RTOL)


def _assert_rf_parity(sig, task, seed, nfeat):
    pm, X, y = _data(sig, seed, task)
    kw = dict(task=task, max_depth=4, n_features=nfeat, n_estimators=5, random_state=seed)
    new = ProductSpaceRF(pm=pm, **kw).fit(X, y)
    old = LegacyRF(pm=pm, **kw).fit(X, y)
    if task == "classification":
        assert torch.equal(new.predict(X), old.predict(X)), "RF class predictions differ from legacy"
    else:
        assert torch.allclose(new.predict(X), old.predict(X), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("nfeat", NFEATS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("sig", NONCIRCULAR_SIGS)
def test_dt_parity_noncircular(sig, task, seed, nfeat):
    """PERMANENT: live DT is bit-exact vs legacy on noncircular signatures."""
    _assert_dt_parity(sig, task, seed, nfeat)


@pytest.mark.parametrize("nfeat", NFEATS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("sig", CIRCULAR_SIGS)
def test_dt_parity_circular_phase_a(sig, task, seed, nfeat):
    """PHASE A ONLY (removed in Phase B): bit-exact vs legacy on circular sigs."""
    _assert_dt_parity(sig, task, seed, nfeat)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("sig", NONCIRCULAR_SIGS)
def test_rf_parity_noncircular(sig, task, seed):
    """PERMANENT: live RF is bit-exact vs legacy on noncircular signatures."""
    _assert_rf_parity(sig, task, seed, "d")
