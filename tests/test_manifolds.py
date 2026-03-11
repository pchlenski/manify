import geoopt
import torch

from manify.embedders._losses import dist_component_by_manifold  # type: ignore
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
    assert torch.allclose(sqdists_11, dists_11**2, atol=1e-5), "dist2 does not match dist squared for X1"
    assert torch.allclose(sqdists_12, dists_12**2, atol=1e-5), "dist2 does not match dist squared for X1 and X2"
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
            -0.5 * (torch.sum(X1**2, dim=-1) + X1.size(-1) * torch.log(torch.tensor(2 * torch.pi))),
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

    # Test log-likelihood differences and KL divergence
    _test_log_likelihood_properties(M, X1)


def _test_log_likelihood_properties(M, X1):
    """Test log-likelihood differences and KL divergence."""
    torch.manual_seed(42)

    mu = M.sample(z_mean=M.mu0)

    if hasattr(M, "P"):  # ProductManifold
        sigma_factorized = []
        for component_M in M.P:
            Sigma = torch.diag(torch.randn(component_M.dim)) ** 2
            sigma_factorized.append(Sigma)

        log_probs_p = M.log_likelihood(z=X1)
        log_probs_q = M.log_likelihood(z=X1, mu=mu, sigma_factorized=sigma_factorized)
    else:  # Single Manifold
        Sigma = torch.diag(torch.randn(M.dim)) ** 2

        log_probs_p = M.log_likelihood(z=X1)
        log_probs_q = M.log_likelihood(z=X1, mu=mu, sigma=Sigma)

    log_likelihood_diff = log_probs_q - log_probs_p

    # Differences should be finite and vary across samples
    assert torch.isfinite(log_likelihood_diff).all(), "Log-likelihood differences should be finite"
    assert log_likelihood_diff.std() > 0, "Log-likelihood differences should have variance"

    # For Euclidean manifolds, compare against analytic Gaussian KL in a simple 1D case
    if not hasattr(M, "P") and getattr(M, "type", None) == "E" and M.dim == 1:
        # Construct a simple 1D Gaussian example where analytic KL is known
        base = torch.distributions.Normal(loc=0.0, scale=1.0)
        shifted = torch.distributions.Normal(loc=1.0, scale=2.0)

        samples = base.sample((X1.shape[0],))
        kl_samples = shifted.log_prob(samples) - base.log_prob(samples)

        analytic_kl = torch.log(torch.tensor(2.0)) + (1 + 1**2) / (2 * 2**2) - 0.5
        mc_kl = kl_samples.mean()

        assert torch.isfinite(mc_kl), "Monte Carlo KL estimate should be finite"
        assert torch.allclose(mc_kl, analytic_kl, atol=5e-2), "Monte Carlo KL should match analytic Gaussian KL"

    kl_divergence_approx = log_likelihood_diff.mean()
    assert torch.isfinite(kl_divergence_approx), "KL divergence should be finite"


def _product_manifold_tests(pm, X1, X2):
    """Test ProductManifold-specific functionality."""
    if len(pm.P) > 1:
        pdist2_total = pm.pdist2(X1)
        dist2_total = pm.dist2(X1, X2)

        X1_factorized = pm.factorize(X1)
        X2_factorized = pm.factorize(X2)

        pdist2_sum = sum(M.pdist2(x) for M, x in zip(pm.P, X1_factorized, strict=True))
        dist2_sum = sum(M.dist2(x1, x2) for M, x1, x2 in zip(pm.P, X1_factorized, X2_factorized, strict=True))

        assert torch.allclose(pdist2_total, pdist2_sum, atol=1e-5), "pdist2 does not match sum of component pdist2"
        assert torch.allclose(dist2_total, dist2_sum, atol=1e-5), "dist2 does not match sum of component dist2"

    if len(pm.P) > 1:
        contributions = dist_component_by_manifold(pm, X1)
        assert torch.isclose(torch.tensor(sum(contributions)), torch.tensor(1.0), atol=1e-5), (
            "Contributions do not sum to 1"
        )


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
        _product_manifold_tests(pm, X1, X2)

        # Also test gaussian mixture
        X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42, adjust_for_dims=True)


def test_sampling_distances_to_origin():
    """Test that distances to origin follow expected statistical properties."""
    print("Testing distances to origin for wrapped normal distributions...")

    N_SAMPLES = 1000
    curvatures = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    for K in curvatures:
        print(f"  Testing curvature K = {K}")
        torch.manual_seed(42)
        m = Manifold(K, 4)

        # Sample mu at a reasonable distance from origin
        mu = m.sample(z_mean=m.mu0)
        mu_distance_to_origin = m.dist(mu.unsqueeze(0), m.mu0.unsqueeze(0)).item()

        # Create Sigma with controlled scale
        Sigma = torch.eye(m.dim) * 0.1  # Small, controlled variance

        samples = m.sample(n_samples=N_SAMPLES, z_mean=mu, sigma=Sigma)
        distances_to_origin = m.dist(samples, m.mu0.unsqueeze(0)).squeeze()

        # Statistical expectations
        mean_distance = distances_to_origin.mean().item()
        std_distance = distances_to_origin.std().item()

        # Mean distance should be close to mu's distance to origin
        distance_tolerance = 0.5  # Allow some tolerance for sampling variance
        assert abs(mean_distance - mu_distance_to_origin) < distance_tolerance, (
            f"Mean distance {mean_distance:.3f} should be close to mu distance {mu_distance_to_origin:.3f} for K={K}"
        )

        # Standard deviation should be related to Sigma scale
        expected_std = torch.sqrt(torch.trace(Sigma)).item() * 0.5  # Rough approximation
        std_tolerance = 0.3
        assert abs(std_distance - expected_std) < std_tolerance, (
            f"Distance std {std_distance:.3f} should be related to Sigma scale {expected_std:.3f} for K={K}"
        )

        # Basic sanity checks
        assert distances_to_origin.min() > 0, f"Distances to origin should be positive for K={K}"
        assert distances_to_origin.std() > 1e-6, f"Distances to origin should have variance for K={K}"

        # For spherical manifolds, check bounds
        if K > 0:
            max_dist = torch.pi / torch.sqrt(torch.tensor(K))
            assert distances_to_origin.max() < max_dist, f"Distances should be bounded for spherical K={K}"


def test_sampling_consistency():
    """Test that sampling produces equivalent results with different input formats when seeded."""
    print("Testing sampling consistency...")

    curvatures = [-1.0, 0.0, 1.0]

    for K in curvatures:
        print(f"  Testing curvature K = {K}")
        m = Manifold(K, 4)

        # Test 1: n_samples vs stacked z_mean (no sigma)
        torch.manual_seed(42)
        mu = m.sample(z_mean=m.mu0)

        torch.manual_seed(42)
        samples1 = m.sample(n_samples=100, z_mean=mu)

        torch.manual_seed(42)
        stacked_mu = torch.stack([mu] * 100)
        samples2 = m.sample(z_mean=stacked_mu)

        # Should produce identical samples when seeded equivalently
        assert torch.allclose(samples1, samples2, atol=1e-6), f"Samples should be identical for K={K}"
        assert samples1.shape == (100, m.ambient_dim), f"Sample shape mismatch for K={K}"

        # Test 2: n_samples vs stacked z_mean (with sigma)
        torch.manual_seed(42)
        Sigma = torch.diag(torch.randn(m.dim)) ** 2

        torch.manual_seed(42)
        samples1 = m.sample(n_samples=100, z_mean=mu, sigma=Sigma)

        torch.manual_seed(42)
        stacked_Sigma = torch.stack([Sigma] * 100)
        samples2 = m.sample(z_mean=stacked_mu, sigma=stacked_Sigma)

        # Should produce identical samples when seeded equivalently
        assert torch.allclose(samples1, samples2, atol=1e-6), f"Samples with sigma should be identical for K={K}"
        assert samples1.shape == (100, m.ambient_dim), f"Sample shape mismatch for K={K}"

        # Test 3: Verify the sampling formats from _shared_tests work as expected
        torch.manual_seed(42)
        stacked_means = torch.stack([m.mu0] * 5)

        torch.manual_seed(42)
        s1 = m.sample(100)
        assert s1.shape == (100, m.ambient_dim), f"Sample shape mismatch for s1, K={K}"

        torch.manual_seed(42)
        s2 = m.sample(100, z_mean=m.mu0)
        assert s2.shape == (100, m.ambient_dim), f"Sample shape mismatch for s2, K={K}"

        torch.manual_seed(42)
        s3 = m.sample(z_mean=stacked_means)
        assert s3.shape == (5, m.ambient_dim), f"Sample shape mismatch for s3, K={K}"

        torch.manual_seed(42)
        s4 = m.sample(100, z_mean=stacked_means)
        assert s4.shape == (500, m.ambient_dim), f"Sample shape mismatch for s4, K={K}"


def test_sampling_edge_cases():
    """Test sampling with moderate edge cases within supported ranges."""
    print("Testing sampling edge cases...")

    moderate_curvatures = [-3.0, -2.0, 2.0, 3.0]

    for K in moderate_curvatures:
        print(f"  Testing moderate curvature K = {K}")
        torch.manual_seed(42)
        m = Manifold(K, 4)

        samples = m.sample(100)
        assert samples.shape == (100, m.ambient_dim), f"Sampling failed for moderate curvature K={K}"
        assert m.manifold.check_point(samples), f"Sampled points not on manifold for K={K}"

    torch.manual_seed(42)
    m = Manifold(0.0, 4)  # Use Euclidean for simplicity
    mu = m.sample(z_mean=m.mu0)

    small_Sigma = torch.eye(m.dim) * 1e-3
    samples_small = m.sample(n_samples=100, z_mean=mu, sigma=small_Sigma)
    assert samples_small.shape == (100, m.ambient_dim), "Sampling failed for small covariance"

    large_Sigma = torch.eye(m.dim) * 10.0
    samples_large = m.sample(n_samples=100, z_mean=mu, sigma=large_Sigma)
    assert samples_large.shape == (100, m.ambient_dim), "Sampling failed for large covariance"
