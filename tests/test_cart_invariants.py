"""Oracle-free behavioural invariants for the product-space decision tree.

These do not depend on the legacy oracle and are the source of truth for the
spherical behaviour after the Phase B wraparound fix.
"""

import torch

from manify.manifolds import ProductManifold
from manify.predictors.decision_tree import ProductSpaceDT


def test_sphere_train_consistency():
    """A full-depth DT must reproduce its own training partition on the sphere.

    If the split threshold stored at a node routes inference differently from the
    training partition (the ±pi wraparound bug), train accuracy drops below 1.0.
    Guards the Phase B fix.
    """
    pm = ProductManifold(signature=[(1.0, 3)])
    X, y = pm.gaussian_mixture(num_points=80, num_classes=3, seed=0)
    dt = ProductSpaceDT(
        pm=pm, task="classification", max_depth=None, min_samples_split=2
    ).fit(X, y)
    acc = (dt.predict(X) == y).float().mean().item()
    assert (
        acc == 1.0
    ), f"train accuracy {acc} < 1.0: split misroutes its own training data"
