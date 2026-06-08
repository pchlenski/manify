import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from manify.manifolds import ProductManifold
from manify.predictors.decision_tree import ProductSpaceDT, ProductSpaceRF
from manify.predictors.kappa_gcn import KappaGCN, get_A_hat
from manify.predictors.perceptron import ProductSpacePerceptron
from manify.predictors.svm import ProductSpaceSVM
from manify.utils.link_prediction import (
    make_link_prediction_dataset,
    split_link_prediction_dataset,
)


def _test_base_classifier(
    model, X_train, X_test, y_train, y_test, task="classification"
):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert (
        preds.shape[0] == X_test.shape[0]
    ), "Predictions should match the number of test samples"
    assert preds.ndim == 1, "Predictions should be a 1D array"

    if task == "classification":
        probs = model.predict_proba(X_test)
        assert probs.shape == (
            X_test.shape[0],
            2,
        ), "Probabilities should match the number of test samples and classes"
        assert probs.ndim == 2, "Probabilities should be a 2D array"
        assert (
            torch.argmax(probs, dim=1) == preds
        ).all(), "Predictions should match the class with the highest probability"

        accuracy = (preds == y_test).float().mean()
        assert (
            accuracy >= 0.5
        ), f"Model {model.__class__.__name__} did not achieve sufficient accuracy"

        score = model.score(X_test, y_test)
        assert (
            score >= 0.5
        ), f"Model {model.__class__.__name__} did not achieve sufficient score"
        assert (
            score == accuracy
        ), "Score and accuracy should match for classification tasks"
    elif task == "regression":
        pass  # No further tests are really possible for regression


def _test_kappa_gcn_model(
    model, X_train, X_test, y_train, y_test, pm, task="classification"
):
    X_train_kernel = torch.exp(-pm.pdist2(X_train))
    X_test_kernel = torch.exp(-pm.pdist2(X_test))
    A_train = get_A_hat(X_train_kernel)
    A_test = get_A_hat(X_test_kernel)
    model.fit(X_train, y_train, A=A_train, use_tqdm=False, epochs=100)

    preds = model.predict(X_test, A=A_test)
    assert (
        preds.shape[0] == X_test.shape[0]
    ), "Predictions should match the number of test samples"

    if task == "classification":
        assert preds.ndim == 1, "Predictions should be a 1D array"

        probs = model.predict_proba(X_test, A=A_test)
        assert probs.shape == (
            X_test.shape[0],
            2,
        ), "Probabilities should match the number of test samples and classes"
        assert (
            torch.argmax(probs, dim=1) == preds
        ).all(), "Predictions should match the class with the highest probability"
        assert (
            torch.argmax(probs, dim=1).shape[0] == preds.shape[0]
        ), "The number of predicted classes should match the number of predictions"

        accuracy = (preds == y_test).float().mean()
        assert (
            accuracy >= 0.5
        ), f"Model {model.__class__.__name__} did not achieve sufficient accuracy"
    elif task == "regression":
        assert preds.ndim == 1, "Predictions should be a 1D array"
    elif task == "link_prediction":
        assert preds.ndim == 2, "Link prediction predictions should be a 2D array"
        assert preds.shape == (
            X_test.shape[0],
            X_test.shape[0],
        ), "Link prediction predictions should match the number of pairs"
        assert (
            (preds == 0) | (preds == 1)
        ).all(), "Link prediction predictions should be binary (0 or 1)"


def test_all_classifiers():
    print("Testing basic classifier functionality")
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Init models
    for model_class in [
        ProductSpaceDT,
        ProductSpaceRF,
        ProductSpacePerceptron,
        ProductSpaceSVM,
    ]:
        print(f"Testing model: {model_class.__name__}")
        model = model_class(pm=pm)
        _test_base_classifier(model, X_train, X_test, y_train, y_test)

    # Kappa-GCN needs its own thing
    pm_stereo, X_train_stereo, X_test_stereo = pm.stereographic(X_train, X_test)
    kappa_gcn = KappaGCN(pm=pm_stereo, output_dim=2, num_hidden=2)
    _test_kappa_gcn_model(
        kappa_gcn, X_train_stereo, X_test_stereo, y_train, y_test, pm=pm_stereo
    )

    print("All classifiers tested successfully.")


def test_all_regressors():
    print("Testing basic regressor functionality")
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(
        num_points=100, num_classes=2, seed=42, task="regression"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Init models
    for model_class in [
        ProductSpaceDT,
        ProductSpaceRF,
        # ProductSpacePerceptron,
        # ProductSpaceSVM,
    ]:
        model = model_class(pm=pm, task="regression")
        _test_base_classifier(
            model, X_train, X_test, y_train, y_test, task="regression"
        )

    # Kappa-GCN needs its own thing
    pm_stereo, X_train_stereo, X_test_stereo = pm.stereographic(X_train, X_test)
    kappa_gcn = KappaGCN(pm=pm_stereo, output_dim=1, num_hidden=2, task="regression")
    _test_kappa_gcn_model(
        kappa_gcn,
        X_train_stereo,
        X_test_stereo,
        y_train,
        y_test,
        pm=pm_stereo,
        task="regression",
    )

    print("All regressors tested successfully.")


def test_regression_score_is_r2():
    """Regression .score() must return R^2 (higher is better), matching sklearn."""
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(
        num_points=100, num_classes=2, seed=42, task="regression"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = ProductSpaceDT(pm=pm, task="regression")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    score = model.score(X_test, y_test)
    expected = r2_score(y_test.numpy(), preds.numpy())
    assert (
        abs(score - expected) < 1e-5
    ), f"score {score} should equal r2_score {expected}"
    assert score <= 1.0, "R^2 cannot exceed 1.0"


def test_get_a_hat_does_not_mutate_input():
    """get_A_hat must not modify the adjacency/distance matrix passed in."""
    A = torch.tensor([[float("nan"), 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    A_original = A.clone()
    _ = get_A_hat(A)
    assert torch.equal(A.isnan(), A_original.isnan()), "NaNs in input were overwritten"
    finite = ~A_original.isnan()
    assert torch.allclose(A[finite], A_original[finite]), "Input matrix was mutated"


def test_all_link_predictors():
    print("Testing basic link predictor functionality")
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, _ = pm.gaussian_mixture(
        num_points=100, num_classes=1, seed=42, task="classification"
    )

    # Compute adjacency matrix: top 3 neighbors for each node
    with torch.no_grad():
        dists = pm.pdist2(X)
        dists.fill_diagonal_(float("inf"))
        adj = torch.zeros_like(dists)
        topk = torch.topk(-dists, k=3, dim=1)
        for i in range(X.shape[0]):
            adj[i, topk.indices[i]] = 1.0
        adj = ((adj + adj.t()) > 0).float()

    # Create link prediction dataset
    X_lp, y_lp, pm_with_dists = make_link_prediction_dataset(
        X_embed=X, pm=pm, adj=adj, add_dists=True
    )

    # Use split_link_prediction_dataset for train/test split
    X_train_lp, X_test_lp, y_train_lp, y_test_lp, idx_train, idx_test = (
        split_link_prediction_dataset(
            X_lp, y_lp, test_size=0.2, random_state=42, downsample=20
        )
    )

    # Test traditional models - these treat data as a classification task
    # for model_class in [
    #     ProductSpaceDT,
    #     ProductSpaceRF,
    #     ProductSpacePerceptron,
    #     ProductSpaceSVM,
    # ]:
    #     print(f"Testing model: {model_class.__name__}")
    #     model = model_class(pm=pm_with_dists, task="classification")
    #     _test_base_classifier(model, X_train_lp, X_test_lp, y_train_lp, y_test_lp, task="classification")

    # For KappaGCN, work with original node embeddings
    X_train = X[idx_train]
    X_test = X[idx_test]
    adj_train = adj[idx_train][:, idx_train]
    adj_test = adj[idx_test][:, idx_test]

    # Convert to stereographic
    pm_stereo, X_train_stereo, X_test_stereo = pm.stereographic(X_train, X_test)

    # Test KappaGCN
    kappa_gcn = KappaGCN(
        pm=pm_stereo, output_dim=2, num_hidden=2, task="link_prediction"
    )
    _test_kappa_gcn_model(
        kappa_gcn,
        X_train_stereo,
        X_test_stereo,
        adj_train,
        adj_test,
        pm=pm_stereo,
        task="link_prediction",
    )
    print("All link predictors tested successfully.")


def test_random_forest_batch():
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    batch_sizes = [1, 10, None]
    preds_list = []

    for batch_size in batch_sizes:
        rf = ProductSpaceRF(
            pm=pm,
            batch_size=batch_size,
            n_estimators=2,
            random_state=42,
            max_features="none",
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        assert (
            preds.shape[0] == X_test.shape[0]
        ), "Predictions should match the number of test samples"
        assert preds.ndim == 1, "Predictions should be a 1D array"
        assert (
            preds == y_test
        ).float().mean() >= 0.5, "Model did not achieve sufficient accuracy"
        preds_list.append(preds)

    # Check equality for all outputs
    for i in range(len(preds_list)):
        for j in range(i + 1, len(preds_list)):
            assert torch.allclose(
                preds_list[i], preds_list[j]
            ), f"Predictions should be the same for batch sizes {batch_sizes[i]} and {batch_sizes[j]}"


def test_decision_tree_special_dims_ablate_midpoints():
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Special dims
    rf = ProductSpaceRF(pm=pm, use_special_dims=True, n_estimators=2, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert (
        preds.shape[0] == X_test.shape[0]
    ), "Predictions should match the number of test samples"
    assert preds.ndim == 1, "Predictions should be a 1D array"
    assert (
        preds == y_test
    ).float().mean() >= 0.5, "Model did not achieve sufficient accuracy"

    # Ablate midpoints
    rf = ProductSpaceRF(pm=pm, ablate_midpoints=True, n_estimators=2, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert (
        preds.shape[0] == X_test.shape[0]
    ), "Predictions should match the number of test samples"
    assert preds.ndim == 1, "Predictions should be a 1D array"
    assert (
        preds == y_test
    ).float().mean() >= 0.5, "Model did not achieve sufficient accuracy"


def test_random_forest_max_features():
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # max_features = sqrt
    dt = ProductSpaceRF(pm=pm, max_features="sqrt", n_estimators=2)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    assert (
        preds.shape[0] == X_test.shape[0]
    ), "Predictions should match the number of test samples"
    assert preds.ndim == 1, "Predictions should be a 1D array"
    assert (
        preds == y_test
    ).float().mean() >= 0.5, "Model did not achieve sufficient accuracy"

    # max_features = log2
    dt = ProductSpaceRF(pm=pm, max_features="log2", n_estimators=2)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    assert (
        preds.shape[0] == X_test.shape[0]
    ), "Predictions should match the number of test samples"
    assert preds.ndim == 1, "Predictions should be a 1D array"
    assert (
        preds == y_test
    ).float().mean() >= 0.5, "Model did not achieve sufficient accuracy"

    # max_features = int: each tree should subsample exactly that many feature columns
    n_features_int = 3
    rf = ProductSpaceRF(pm=pm, max_features=n_features_int, n_estimators=2)
    rf.fit(X_train, y_train)
    n_cols = rf.trees[0].permutations.shape[0]
    assert (
        n_cols == n_features_int
    ), f"Expected {n_features_int} subsampled features, got {n_cols}"
    assert all(tree.permutations.shape[0] == n_features_int for tree in rf.trees)

    # max_features larger than the available columns should clamp to the number of columns
    rf = ProductSpaceRF(pm=pm, max_features=10_000, n_estimators=2)
    rf.fit(X_train, y_train)
    total_angles = rf.trees[0].permutations.shape[0]
    rf_all = ProductSpaceRF(pm=pm, max_features="none", n_estimators=2)
    rf_all.fit(X_train, y_train)
    assert total_angles == rf_all.trees[0].permutations.shape[0]
