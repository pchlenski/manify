"""CART refactor parity tests: the live ProductSpaceDT/RF must match the frozen
legacy oracle (tests/legacy/decision_tree_legacy.py).

The batched code path was deleted; the live tree is now the single (formerly
"nobatch") path. We therefore re-anchor parity to the legacy oracle as follows:

* Classification parity (PERMANENT, ALL signatures incl. circular/spherical):
  the live tree must be bit-exact vs the legacy tree run in its nobatch mode
  (``LegacyDT(..., batch_size=1)`` / ``LegacyRF(..., batch_size=1)``). The legacy
  nobatch path is the correct reference on spheres (no +/-pi wraparound bug), so
  circular classification parity is permanent, not Phase-A-only.
* Regression parity (NONCIRCULAR only): the live tree vs the legacy *batched*
  path (``LegacyDT()`` default). Legacy batched is correct where there is no
  wraparound, and legacy nobatch MSE has no frozen reference (it raised
  NotImplementedError). Circular regression parity is dropped here and covered
  instead by the oracle-free invariants (see tests/test_cart_invariants.py and
  docs/design/cart_refactor.md).

  Regression parity is asserted on ``n_features="d"`` only. The live MSE
  (variance via gathered ``sum``/``sumsq``) and the legacy batched MSE (matmul
  over the comparisons tensor) are algebraically equal but differ at ULP scale
  (~1e-8 in the information gain). With ``n_features="d_choose_2"`` the many
  near-collinear candidate features create frequent near-ties whose argmax flips
  on that ~1e-8 noise, yielding a different-but-equally-valid tree. That is a
  float tie-break artifact of comparing two distinct implementations, not a
  semantic change; the live MSE itself is verified correct against sklearn on a
  Euclidean signature. So we pin regression parity where it is bit-stable (``d``)
  and rely on sklearn/invariants for ``d_choose_2``.

Discrete decisions (class predictions) are compared with torch.equal; continuous
outputs (class probabilities, regression values) use a tight allclose to tolerate
ULP-level reordering while catching any real change.
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
ALL_SIGS = NONCIRCULAR_SIGS + CIRCULAR_SIGS
SEEDS = [0, 1, 2]
NFEATS = ["d", "d_choose_2"]

_ATOL, _RTOL = 1e-6, 1e-5


def _data(sig, seed, task):
    pm = ProductManifold(signature=sig)
    X, y = pm.gaussian_mixture(num_points=80, num_classes=3, seed=seed, task=task)
    return pm, X, y


def _assert_dt_classification_parity(sig, seed, nfeat):
    """Live DT vs legacy nobatch (batch_size=1), classification, any signature."""
    pm, X, y = _data(sig, seed, "classification")
    kw = dict(task="classification", max_depth=4, n_features=nfeat)
    new = ProductSpaceDT(pm=pm, **kw).fit(X, y)
    old = LegacyDT(pm=pm, batch_size=1, **kw).fit(X, y)
    assert torch.equal(
        new.predict(X), old.predict(X)
    ), "class predictions differ from legacy nobatch"
    assert torch.allclose(
        new.predict_proba(X), old.predict_proba(X), atol=_ATOL, rtol=_RTOL
    )


def _assert_dt_regression_parity(sig, seed):
    """Live DT vs legacy batched (default), regression, noncircular, n_features=d."""
    pm, X, y = _data(sig, seed, "regression")
    kw = dict(task="regression", max_depth=4, n_features="d")
    new = ProductSpaceDT(pm=pm, **kw).fit(X, y)
    old = LegacyDT(pm=pm, **kw).fit(X, y)
    assert torch.allclose(new.predict(X), old.predict(X), atol=_ATOL, rtol=_RTOL)


def _assert_rf_classification_parity(sig, seed):
    """Live RF vs legacy nobatch (batch_size=1), classification, any signature."""
    pm, X, y = _data(sig, seed, "classification")
    kw = dict(
        task="classification",
        max_depth=4,
        n_features="d",
        n_estimators=5,
        random_state=seed,
    )
    new = ProductSpaceRF(pm=pm, **kw).fit(X, y)
    old = LegacyRF(pm=pm, batch_size=1, **kw).fit(X, y)
    assert torch.equal(
        new.predict(X), old.predict(X)
    ), "RF class predictions differ from legacy nobatch"


def _assert_rf_regression_parity(sig, seed):
    """Live RF vs legacy batched (default), regression, noncircular only."""
    pm, X, y = _data(sig, seed, "regression")
    kw = dict(
        task="regression",
        max_depth=4,
        n_features="d",
        n_estimators=5,
        random_state=seed,
    )
    new = ProductSpaceRF(pm=pm, **kw).fit(X, y)
    old = LegacyRF(pm=pm, **kw).fit(X, y)
    assert torch.allclose(new.predict(X), old.predict(X), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("nfeat", NFEATS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("sig", ALL_SIGS)
def test_dt_classification_parity(sig, seed, nfeat):
    """PERMANENT: live DT classification is bit-exact vs legacy nobatch on every signature."""
    _assert_dt_classification_parity(sig, seed, nfeat)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("sig", NONCIRCULAR_SIGS)
def test_dt_regression_parity_noncircular(sig, seed):
    """Live DT regression matches legacy batched on noncircular signatures (n_features=d)."""
    _assert_dt_regression_parity(sig, seed)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("sig", ALL_SIGS)
def test_rf_classification_parity(sig, seed):
    """PERMANENT: live RF classification is bit-exact vs legacy nobatch on every signature."""
    _assert_rf_classification_parity(sig, seed)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("sig", NONCIRCULAR_SIGS)
def test_rf_regression_parity_noncircular(sig, seed):
    """Live RF regression matches legacy batched on noncircular signatures."""
    _assert_rf_regression_parity(sig, seed)
