import geoopt
import pytest
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


def test_stereographic_conversion_isometry():
    """Stereographic conversion must preserve distances at every curvature, not just |K| == 1.

    The forward projection used to fold the curvature scale into the denominator, which only matches
    the correct formula when |K| == 1. For other curvatures the conversion silently distorted
    distances. This escaped the existing checks in `_shared_tests` because those only run at
    curvatures -1, 0, and 1. Here we verify, across a range of curvatures, that:
      * converting to stereographic coordinates preserves pairwise and to-origin distances,
      * converting back preserves them too, and
      * the round trip returns the original points.
    """
    print("Testing stereographic conversion isometry across curvatures...")

    curvatures = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    for K in curvatures:
        print(f"  Testing curvature K = {K}")
        M = Manifold(K, 4, stereographic=False)
        torch.manual_seed(0)
        # Keep the covariance modest so points stay well away from the boundary, where float32 distance
        # computations get noisy. The curvature-scaling bug is multiplicative, so it is glaringly visible
        # even at these moderate distances.
        means = torch.vstack([M.mu0] * 10)
        covs = torch.stack([torch.eye(M.dim) * 0.1] * 10)
        X = M.sample(z_mean=means, sigma=covs)

        M_stereo, X_stereo = M.stereographic(X)
        assert M_stereo.is_stereographic, f"Converted manifold should be stereographic for K={K}"
        assert M_stereo.manifold.check_point(X_stereo), f"Converted points not on stereographic manifold for K={K}"

        # Forward isometry: distances on the original manifold match distances on the stereographic one.
        # The buggy formula distorted these by ~0.2-0.25 here for |K| != 1, far above this tolerance.
        d_orig = M.dist(X, X)
        d_stereo = M_stereo.dist(X_stereo, X_stereo)
        assert torch.allclose(d_orig, d_stereo, atol=5e-2), (
            f"Stereographic conversion is not isometric for K={K} "
            f"(max diff {(d_orig - d_stereo).abs().max().item():.4f})"
        )

        # Distances to the origin should be preserved as well
        d_orig_0 = M.dist(X, M.mu0).flatten()
        d_stereo_0 = M_stereo.dist(X_stereo, M_stereo.mu0).flatten()
        assert torch.allclose(d_orig_0, d_stereo_0, atol=5e-2), (
            f"Distances to origin not preserved under stereographic conversion for K={K}"
        )

        # Round trip back to the original coordinates
        M_back, X_back = M_stereo.inverse_stereographic(X_stereo)
        assert not M_back.is_stereographic, f"Inverse-converted manifold should not be stereographic for K={K}"
        assert torch.allclose(X_back, X, atol=1e-3), f"Stereographic round trip does not recover points for K={K}"


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


def test_stereographic_sampling():
    """Regression test for sampling on stereographic manifolds (see issue #37).

    Sampling on stereographic manifolds used to crash because the intrinsic tangent vector was
    embedded with an extra ambient coordinate (correct for the Lorentz/sphere models, but wrong
    for stereographic coordinates where ambient_dim == dim). This verifies that sampling now (a)
    runs and lands on the manifold, and (b) produces the same wrapped-normal distribution as the
    equivalent non-stereographic manifold (distances to the origin are an intrinsic quantity).
    """
    print("Testing sampling on stereographic manifolds...")

    curvatures = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    # Single stereographic manifolds: no crash, points on manifold, distribution matches reference
    quantiles = torch.tensor([0.25, 0.5, 0.75, 0.9])
    for K in curvatures:
        print(f"  Testing curvature K = {K}")
        m_stereo = Manifold(K, 4, stereographic=True)
        assert m_stereo.is_stereographic

        torch.manual_seed(42)
        samples = m_stereo.sample(100)
        assert samples.shape == (100, m_stereo.ambient_dim), f"Sample shape mismatch for stereographic K={K}"
        assert m_stereo.manifold.check_point(samples), f"Stereographic samples not on manifold for K={K}"

        # Wrapped-normal distance-to-origin should match the non-stereographic manifold of the same curvature
        m_ref = Manifold(K, 4, stereographic=False)
        torch.manual_seed(0)
        d_ref = m_ref.dist(m_ref.sample(20000), m_ref.mu0).flatten()
        torch.manual_seed(0)
        d_stereo = m_stereo.dist(m_stereo.sample(20000), m_stereo.mu0).flatten()
        q_ref = torch.quantile(d_ref, quantiles.to(d_ref.dtype))
        q_stereo = torch.quantile(d_stereo, quantiles.to(d_stereo.dtype))
        max_rel_diff = ((q_ref - q_stereo).abs() / q_ref.clamp_min(1e-6)).max().item()
        assert max_rel_diff < 0.05, (
            f"Stereographic wrapped-normal does not match reference for K={K} (max rel diff {max_rel_diff:.4f})"
        )

    # Stereographic product manifold: sampling and gaussian_mixture should both work
    pm_stereo = ProductManifold([(-1.0, 4), (0.0, 4), (1.0, 4)], stereographic=True)
    torch.manual_seed(42)
    samples = pm_stereo.sample(50)
    assert samples.shape == (50, pm_stereo.ambient_dim), "Sample shape mismatch for stereographic product manifold"
    assert pm_stereo.manifold.check_point(samples), "Stereographic product samples not on manifold"

    X, y = pm_stereo.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    assert X.shape == (100, pm_stereo.ambient_dim), "gaussian_mixture shape mismatch on stereographic product manifold"
    assert pm_stereo.manifold.check_point(X), "gaussian_mixture samples not on stereographic product manifold"


def test_default_dtype_is_float64():
    """Manifold math needs the extra range/precision, so both classes default to float64."""
    assert Manifold(curvature=-1.0, dim=4).dtype == torch.float64
    assert ProductManifold([(-1.0, 4), (1.0, 4)]).dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_manifold_dtype_propagation(dtype):
    """The requested dtype should flow through the origin, sampling, and distance computations."""
    print(f"Checking dtype propagation for {dtype}...")
    for curv in [-1.0, 0.0, 1.0]:
        M = Manifold(curvature=curv, dim=16, dtype=dtype)
        assert M.dtype == dtype
        assert M.mu0.dtype == dtype

        X = M.sample(20, sigma=torch.eye(16, dtype=dtype))
        assert X.dtype == dtype, f"sample dtype mismatch for K={curv}"

        D = M.pdist(X)
        assert D.dtype == dtype, f"pdist dtype mismatch for K={curv}"
        assert torch.isfinite(D).all(), f"distances should be finite for K={curv}"

        ll = M.log_likelihood(X)
        assert ll.dtype == dtype, f"log_likelihood dtype mismatch for K={curv}"


def test_float64_avoids_high_curvature_overflow():
    """Regression: the wrapped-normal sampler must stay finite at large |K| * sigma^2 * dim.

    The ambient hyperboloid/sphere coordinates are cosh/sin of the tangent norm, which scales like
    sqrt(|K| * sigma^2 * dim). Around sqrt(kappa) ~ 44 those coordinates exceed the float32 range and the
    sampled points (and hence every distance) become non-finite. float64 pushes that boundary out to
    sqrt(kappa) ~ 354, covering any realistic dimension.
    """
    torch.manual_seed(0)
    K, dim = -1.0, 4096  # kappa = |K| * 1 * dim = 4096, well past the float32 overflow threshold

    M64 = Manifold(curvature=K, dim=dim, dtype=torch.float64)
    X64 = M64.sample(n_samples=32, sigma=torch.eye(dim, dtype=torch.float64))
    assert torch.isfinite(X64).all(), "float64 sample should be finite at high curvature x dimension"
    D64 = M64.pdist(X64)
    assert torch.isfinite(D64).all(), "float64 pdist should be finite at high curvature x dimension"
    assert (D64.triu(1) >= 0).all(), "distances should be non-negative"

    # Document the motivating failure: the same configuration overflows in float32 (hence the float64 default).
    M32 = Manifold(curvature=K, dim=dim, dtype=torch.float32)
    X32 = M32.sample(n_samples=32, sigma=torch.eye(dim))
    assert not torch.isfinite(X32).all(), "float32 is expected to overflow here, which is why dtype is configurable"
