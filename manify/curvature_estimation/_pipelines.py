from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

    from jaxtyping import Float

import torch
from sklearn.model_selection import train_test_split

from ..embedders._losses import distortion_loss
from ..embedders.coordinate_learning import CoordinateLearning
from ..manifolds import ProductManifold
from ..predictors._base import BasePredictor
from ..predictors.decision_tree import ProductSpaceDT


def distortion_pipeline(
    pm: ProductManifold,
    dists: Float[torch.Tensor, "n_nodes n_nodes"],
    embedder_init_kwargs: dict[str, Any] | None = None,
    embedder_fit_kwargs: dict[str, Any] | None = None,
) -> float:
    """Distortion‐based pipeline function for greedy signature selection.

    Args:
        pm: Product manifold to use for the pipeline.
        dists: Pairwise distances to approximate.
        embedder_init_kwargs: Additional keyword arguments for initializing the embedder model.
        embedder_fit_kwargs: Additional keyword arguments for fitting the embedder model.

    Returns:
        The distortion loss of the embeddings learned on the given product manifold.
    """
    embedder_init_kwargs = embedder_init_kwargs or {}
    embedder_fit_kwargs = embedder_fit_kwargs or {}

    dists = dists.to(pm.device)
    dists_rescaled = dists / dists.max()

    # Initialize embedder model
    model = CoordinateLearning(pm=pm, device=pm.device, **embedder_init_kwargs)

    # Fit the model
    model.fit(X=None, D=dists_rescaled, **embedder_fit_kwargs)

    # Loss is the distortion loss of the new embeddings
    embeddings = model.embeddings_
    new_dists = pm.pdist(X=embeddings)
    return float(distortion_loss(new_dists, dists_rescaled).item())


def predictor_pipeline(
    pm: ProductManifold,
    dists: Float[torch.Tensor, "n_nodes n_nodes"],
    labels: Float[torch.Tensor, "n_nodes"],
    classifier: type[BasePredictor] = ProductSpaceDT,
    task: Literal["classification", "regression"] = "classification",
    embedder_init_kwargs: dict[str, Any] | None = None,
    embedder_fit_kwargs: dict[str, Any] | None = None,
    model_init_kwargs: dict[str, Any] | None = None,
    model_fit_kwargs: dict[str, Any] | None = None,
) -> float:
    """Classifier‐based pipeline function for greedy signature selection.

    Args:
        pm: Product manifold to use for the pipeline.
        dists: Pairwise distances to approximate.
        labels: Labels for the nodes, used for training the classifier.
        classifier: Classifier to use for evaluating the signature.
        task: Task type, either "classification" or "regression".
        embedder_init_kwargs: Additional keyword arguments for initializing the coordinate learning model.
        embedder_fit_kwargs: Additional keyword arguments for fitting the coordinate learning model.
        model_init_kwargs: Additional keyword arguments for initializing the classifier.
        model_fit_kwargs: Additional keyword arguments for fitting the classifier.

    Returns:
        The loss of the classifier on the test set after embedding the distances using the product manifold.
    """
    embedder_init_kwargs = embedder_init_kwargs or {}
    embedder_fit_kwargs = embedder_fit_kwargs or {}
    model_init_kwargs = model_init_kwargs or {}
    model_fit_kwargs = model_fit_kwargs or {}

    dists = dists.to(pm.device)
    dists_rescaled = dists / dists.max()

    # Embedding steps
    embedder = CoordinateLearning(pm=pm, device=pm.device, **embedder_init_kwargs)
    embedder.fit(X=None, D=dists_rescaled, **embedder_fit_kwargs)
    X = embedder.embeddings_

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, labels)

    # Train classifier
    model_init_kwargs["task"] = task
    model = classifier(pm=pm, **model_init_kwargs)
    model.fit(X=X_train, y=y_train, **model_fit_kwargs)
    loss = model.score(X=X_test, y=y_test)

    # For classification, we want to maximize accuracy; for regression, we minimize MSE.
    return -loss if task == "classification" else loss
