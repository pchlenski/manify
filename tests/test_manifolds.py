import math

import geoopt
import torch

from manify.manifolds import Manifold, ProductManifold


def _shared_tests(M, X1, X2, is_euclidean):
    # Does device switching work?
    M.to("cpu")

    # Verify points are on manifold
    assert M.manifold.check_point(X1), "X1 is not on the manifold"
    assert M.manifold.check_point(X2), "X2 is not on the manifold"

    # Inner products
    ip_11 = M.inner(X1, X1)
    assert ip_11.shape == (10, 10), "Inner product shape mismatch for X1"
    ip_12 = M.inner(X1, X2)
    assert ip_12.shape == (10, 5), "Inner product shape mismatch for X1 and X2"
    if is_euclidean:
        assert torch.allclose(ip_11, X1 @ X1.T, atol=1e-5), "Euclidean inner products do not match for X1"
        assert torch.allclose(ip_12, X1 @ X2.T, atol=1e-5), "Euclidean inner products do not match for X1 and X2"

    # Sampling shapes should support a variety of inputs
    stacked_means = torch.stack([M.mu0] * 5)
    s1 = M.sample(100)
    assert s1.shape == (100, M.ambient_dim), "Sampled points should have the correct shape"
    s2 = M.sample(100, z_mean=M.mu0)
    assert s2.shape == (100, M.ambient_dim), "Sampled points should have the correct shape"
    s3 = M.sample(z_mean=stacked_means)
    assert s3.shape == (5, M.ambient_dim), "Sampled points should have the correct shape"
    s3 = M.sample(100, z_mean=stacked_means)
    assert s3.shape == (500, M.ambient_dim), "Sampled points should have the correct shape"

    # Dists
    dists_11 = M.dist(X1, X1)
    assert dists_11.shape == (10, 10), "Distance shape mismatch for X1"
    dists_12 = M.dist(X1, X2)
    assert dists_12.shape == (10, 5), "Distance shape mismatch for X1 and X2"
    if is_euclidean:
        assert torch.allclose(dists_12, torch.linalg.norm(X1[:, None] - X2[None, :], dim=-1), atol=1e-5), (
            "Euclidean distances do not match for X1 and X2"
        )
        assert torch.allclose(dists_11, torch.linalg.norm(X1[:, None] - X1[None, :], dim=-1), atol=1e-5), (
            f"Euclidean distances do not match for X1 {M.signature}"
        )
    assert (dists_11.triu(1) >= 0).all(), "Distances for X1 should be non-negative"
    assert (dists_12.triu(1) >= 0).all(), "Distances for X2 should be non-negative"
    assert torch.allclose(dists_11.triu(1), M.pdist(X1).triu(1), atol=1e-5), "dist and pdist diverge for X1"

    # Square dists
    sqdists_11 = M.dist2(X1, X1)
    assert sqdists_11.shape == (10, 10), "Squared distance shape mismatch for X1"
    sqdists_12 = M.dist2(X1, X2)
    assert sqdists_12.shape == (10, 5), "Squared distance shape mismatch for X1 and X2"
    if is_euclidean:
        assert torch.allclose(sqdists_12, torch.linalg.norm(X1[:, None] - X2[None, :], dim=-1) ** 2, atol=1e-5), (
            "Euclidean squared distances do not match for X1 and X2"
        )
        assert torch.allclose(sqdists_11, torch.linalg.norm(X1[:, None] - X1[None, :], dim=-1) ** 2, atol=1e-5), (
            "Euclidean squared distances do not match for X1"
        )
    assert (sqdists_11.triu(1) >= 0).all(), "Squared distances for X1 should be non-negative"
    assert (sqdists_12.triu(1) >= 0).all(), "Squared distances for X1 and X2 should be non-negative"
    assert torch.allclose(sqdists_11.triu(1), M.pdist2(X1).triu(1), atol=1e-5), "sqdists_11 and pdist2 diverge for X1"
    # Equivalence between dist2 and dist squared
    assert torch.allclose(sqdists_11, dists_11 ** 2, atol=1e-5), "dist2 does not match dist squared for X1"
    assert torch.allclose(sqdists_12, dists_12 ** 2, atol=1e-5), "dist2 does not match dist squared for X1 and X2"
    assert torch.allclose(M.pdist2(X1), M.pdist(X1) ** 2, atol=1e-5), "pdist2 does not match pdist squared for X1"

    # the other way around: sqrt(dist2) should match dist
    assert torch.allclose(sqdists_11.sqrt(), dists_11, atol=1e-5), "sqrt(dist2) does not match dist for X1"
    assert torch.allclose(sqdists_12.sqrt(), dists_12, atol=1e-5), "sqrt(dist2) does not match dist for X1 and X2"
    assert torch.allclose(M.pdist2(X1).sqrt(), M.pdist(X1), atol=1e-5), "sqrt(pdist2) does not match pdist for X1"

    # Log-likelihood
    lls = M.log_likelihood(X1)
    if is_euclidean:
        # Evaluate as ll of gaussian with mean 0, variance 1:
        assert torch.allclose(
            lls,
            -0.5 * (torch.sum(X1**2, dim=-1) + X1.size(-1) * math.log(2 * math.pi)),
            atol=1e-5,
        ), "Log-likelihood mismatch for Gaussian"
    assert (lls <= 0).all(), "Log-likelihood should be non-positive"

    # Logmap and expmap
    logmap_x1 = M.logmap(X1)
    assert M.manifold.check_vector(logmap_x1), "Logmap point should be in the tangent plane"
    expmap_x1 = M.expmap(logmap_x1)
    assert M.manifold.check_point(expmap_x1), "Expmap point should be on the manifold"

    # Higher-tolerance check for expmap inversion because of numerical issues
    assert torch.allclose(expmap_x1, X1, atol=1e-3), "Expmap does not return the original points"

    # Stereographic conversions
    M_stereo, X1_stereo, X2_stereo = M.stereographic(X1, X2)
    assert M_stereo.is_stereographic
    X_inv_stereo, X1_inv_stereo, X2_inv_stereo = M_stereo.inverse_stereographic(X1_stereo, X2_stereo)
    assert not X_inv_stereo.is_stereographic

    # Assert calling stereographic and inverse_stereographic returns the same points, if the manifold is already
    # in the necessary geometry
    assert M.inverse_stereographic(X1, X2) == (M, X1, X2), (
        "Inverse stereographic does not return the original points for X1"
    )
    assert M_stereo.stereographic(X1_stereo, X2_stereo) == (M_stereo, X1_stereo, X2_stereo), (
        "Inverse stereographic does not return the original points for X2"
    )

    # Higher-tolerance check for stereographic projection inversion
    assert torch.allclose(X1_inv_stereo, X1, atol=1e-3), "Inverse stereographic conversion mismatch for X1"
    assert torch.allclose(X2_inv_stereo, X2, atol=1e-3), "Inverse stereographic conversion mismatch for X2"

    # Apply
    @M.apply
    def apply_function(x):
        return torch.nn.functional.relu(x)

    result = apply_function(X1)
    assert result.shape == X1.shape, "Result shape mismatch for apply_function"
    assert M.manifold.check_point(result)


def test_manifold_methods():
    print("Checking Manifold class...")
    for curv, dim in [(-1.0, 2), (0.0, 2), (1.0, 2), (-1.0, 64), (0.0, 64), (1.0, 64)]:
        print(f"  Signature: [({curv}, {dim})]")
        M = Manifold(curvature=curv, dim=dim)

        # get some vectors via gaussian mixture
        cov = torch.eye(M.dim) / M.dim / 100
        means = torch.vstack([M.mu0] * 10)
        covs = torch.stack([cov] * 10)
        torch.random.manual_seed(42)
        X1 = M.sample(z_mean=means, sigma=covs)
        X2 = M.sample(z_mean=means[:5], sigma=covs[:5])

        # Do attributes work correctly?
        if curv < 0:
            assert M.type == "H" and isinstance(M.manifold.base, geoopt.Lorentz)
        elif curv == 0:
            assert M.type == "E" and isinstance(M.manifold.base, geoopt.Euclidean)
        else:
            assert M.type == "S" and isinstance(M.manifold.base, geoopt.Sphere)

        _shared_tests(M, X1, X2, is_euclidean=curv == 0)


def test_product_manifold_methods():
    print("Checking ProductManifold class...")
    for signature in [
        [(-1.0, 8)],
        [(0.0, 8)],
        [(1.0, 8)],
        [(-1.0, 8), (1.0, 8)],
        [(-1.0, 8), (0.0, 8), (1.0, 8)],
        [(0.0, 8), (0.0, 8)],
    ]:
        print(f"  Signature: [({signature})]")
        pm = ProductManifold(signature=signature)

        # get some vectors via gaussian mixture
        covs = [torch.stack([torch.eye(M.dim) / M.dim / 100] * 10) for M in pm.P]
        means = torch.vstack([pm.mu0] * 10)
        torch.random.manual_seed(42)
        X1 = pm.sample(z_mean=means, sigma_factorized=covs)
        X2 = pm.sample(z_mean=means[:5], sigma_factorized=[cov[:5] for cov in covs])
        X3 = pm.sample()

        # Do attributes work correctly?
        for M in pm.P:
            curv = M.curvature
            if curv < 0:
                assert M.type == "H" and isinstance(M.manifold.base, geoopt.Lorentz)
            elif curv == 0:
                assert M.type == "E" and isinstance(M.manifold.base, geoopt.Euclidean)
            else:
                assert M.type == "S" and isinstance(M.manifold.base, geoopt.Sphere)

        _shared_tests(pm, X1, X2, is_euclidean=all(M.curvature == 0 for M in pm.P))
        # dist2 and pdist2 are the sum of component dist2 and pdist2
        if len(pm.P) > 1:
            pdist2_total = pm.pdist2(X1)
            dist2_total = pm.dist2(X1, X2)
            
            pdist2_sum = sum(M.pdist2(X1[:, pm.man2dim[M]]) for M in pm.P)
            dist2_sum = sum(M.dist2(X1[:, pm.man2dim[M]], X2[:, pm.man2dim[M]]) for M in pm.P)
            
            assert torch.allclose(pdist2_total, pdist2_sum, atol=1e-5), "pdist2 does not match sum of component pdist2"
            assert torch.allclose(dist2_total, dist2_sum, atol=1e-5), "dist2 does not match sum of component dist2"
        # Also test gaussian mixture
        X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42, adjust_for_dims=True)

        # Test that dist_component_by_manifold contributions sum to 1
        from manify.embedders._losses import dist_component_by_manifold # type: ignore
        if len(pm.P) > 1:  # Requires multiple components
            contributions = dist_component_by_manifold(pm, X1)
            assert torch.isclose(torch.tensor(sum(contributions)), torch.tensor(1.0), atol=1e-5), "Contributions do not sum to 1"
