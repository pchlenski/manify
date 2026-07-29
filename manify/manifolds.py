"""Tools for generating Riemannian manifolds and product manifolds.

The module consists of two classes: `Manifold` and `ProductManifold`. The `Manifold` class represents hyperbolic,
Euclidean, or spherical manifolds of constant Gaussian curvature. The `ProductManifold` class supports Cartesian
products of multiple manifolds, combining their geometric properties to create mixed-curvature. Both classes
include methods for different key geometric operations, and are built on top of their corresponding `geoopt` classes
(`Lorentz`, `Euclidean`, `Sphere`, `Scaled` and `ProductManifold`)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import geoopt
import torch

if TYPE_CHECKING:
    from beartype.typing import Callable, Literal
    from jaxtyping import Float, Real

warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributions")  # Singular samples from Wishart


class Manifold:
    """Constant-curvature Riemannian manifold class.

    This class provides tools for creating and manipulating Riemannian manifolds with constant curvature (hyperbolic,
    Euclidean, or spherical).

    Args:
        curvature: The curvature of the manifold.
        dim: The dimension of the manifold.
        device: The device on which the manifold is stored.
        stereographic: Whether to use stereographic coordinates.
        dtype: The floating-point dtype for the origin, sampling, and geometric operations. Defaults to
            ``torch.float32``. Use ``torch.float64`` for large ``|curvature| * sigma^2 * dim``, where the ambient
            hyperboloid/sphere coordinates (``cosh``/``sin`` of the tangent norm) otherwise overflow float32.

    Attributes:
        curvature: The curvature of the manifold. Negative for hyperbolic, zero for Euclidean, and positive for
            spherical manifolds.
        dim: The dimension of the manifold.
        device: The device on which the manifold is stored.
        is_stereographic: Whether stereographic coordinates are used for the manifold.
        dtype: The floating-point dtype used for manifold tensors and operations.
        scale: The scale factor derived from the curvature.
        type: A string identifier for the manifold type ('H' for hyperbolic, 'E' for Euclidean, 'S' for spherical,
            'P' for poincaré ball, 'D' for stereographic sphere).
        ambient_dim: The dimension of the ambient space.
        manifold: The underlying geoopt manifold object.
        mu0: The origin point on the manifold.
        name: A string identifier for the manifold.
    """

    def __init__(
        self,
        curvature: float,
        dim: int,
        device: str = "cpu",
        stereographic: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        # Device management
        self.device = device

        # Basic properties
        self.curvature = curvature
        self.dim = dim
        self.scale = abs(curvature) ** -0.5 if curvature != 0 else 1
        self.is_stereographic = stereographic
        self.dtype = dtype

        # A couple of manifold-specific quirks we need to deal with here
        if stereographic:
            self.manifold = geoopt.Stereographic(k=curvature, learnable=True).to(self.device)
            if curvature < 0:
                self.type = "P"
            elif curvature == 0:
                self.type = "E"
            else:  # curvature > 0
                self.type = "D"
            self.ambient_dim = dim
            self.mu0 = torch.zeros(self.dim).to(self.device).reshape(1, -1)
        else:
            if curvature < 0:
                self.type = "H"
                man = geoopt.Lorentz(k=1.0)
                # Use 'k=1.0' because the scale will take care of the curvature
                # For more information, see the bottom of page 5 of Gu et al. (2019):
                # https://openreview.net/pdf?id=HJxeWnCcF7
            elif curvature == 0:
                self.type = "E"
                man = geoopt.Euclidean(ndim=1)
                # Use 'ndim=1' because dim means the *shape* of the Euclidean space, not the dimensionality...
            else:
                self.type = "S"
                man = geoopt.Sphere()
            self.manifold = geoopt.Scaled(man, self.scale, learnable=True).to(self.device)

            self.ambient_dim = dim if curvature == 0 else dim + 1
            if curvature == 0:
                self.mu0 = torch.zeros(self.dim).to(self.device).reshape(1, -1)
            else:
                self.mu0 = torch.Tensor([1.0] + [0.0] * dim).to(self.device).reshape(1, -1)

        self.name = f"{self.type}_{abs(self.curvature):.1f}^{dim}"

        # Cast the geoopt manifold (incl. its learnable scale) and the origin to the requested dtype. float64
        # is needed at large |K|*sigma^2*dim, where the ambient cosh/sin coordinates overflow float32.
        self.manifold = self.manifold.to(self.dtype)
        self.mu0 = self.mu0.to(self.dtype)

        # Couple of assertions to check
        assert self.manifold.check_point(self.mu0)

    def to(self, device: str) -> Manifold:
        """Move the Manifold object to a specified device.

        Args:
            device: The device to which the manifold will be moved.

        Returns:
            manifold: The updated manifold object on the specified device.
        """
        self.device = device
        self.manifold = self.manifold.to(device)
        self.mu0 = self.mu0.to(device)
        return self

    def inner(
        self, X: Float[torch.Tensor, "n_points1 n_dim"], Y: Float[torch.Tensor, "n_points2 n_dim"]
    ) -> Float[torch.Tensor, "n_points1 n_points2"]:
        """Compute the inner product between two points on the manifold.

        This ensures the correct inner product is computed for all manifold types
        (flipping the sign of dim 0 for hyperbolic manifolds).

        Args:
            X: Tensor of points in the manifold.
            Y: Tensor of points in the manifold.

        Returns:
            inner_products: Tensor of inner products between points.
        """
        # "Not inherited because of weird broadcasting stuff, plus need for scale.
        # This ensures we compute the right inner product for all manifolds (flip sign of dim 0 for hyperbolic)
        X_fixed = torch.cat([-X[:, 0:1], X[:, 1:]], dim=1) if self.type == "H" else X

        # This prevents dividing by zero in the Euclidean case
        scaler = 1 / abs(self.curvature) if self.type != "E" else 1
        return X_fixed @ Y.T * scaler

    def dist(
        self, X: Float[torch.Tensor, "n_points1 n_dim"], Y: Float[torch.Tensor, "n_points2 n_dim"]
    ) -> Float[torch.Tensor, "n_points1 n_points2"]:
        """Compute the distance between two sets of points on the manifold.

        Args:
            X: Tensor of points in the manifold.
            Y: Tensor of points in the manifold.

        Returns:
            distances: Tensor of distances between points.
        """
        return self.manifold.dist(X[:, None], Y[None, :])

    def dist2(
        self, X: Float[torch.Tensor, "n_points1 n_dim"], Y: Float[torch.Tensor, "n_points2 n_dim"]
    ) -> Float[torch.Tensor, "n_points1 n_points2"]:
        """Compute the squared distance between two sets of points on the manifold.

        Args:
            X: Tensor of points in the manifold.
            Y: Tensor of points in the manifold.

        Returns:
            squared_distances: Tensor of squared distances between points.
        """
        return self.manifold.dist2(X[:, None], Y[None, :])

    def pdist(self, X: Float[torch.Tensor, "n_points n_dim"]) -> Float[torch.Tensor, "n_points n_points"]:
        """Compute pairwise distances between points on the manifold.

        Args:
            X: Tensor of points in the manifold.

        Returns:
            pairwise_distances: Tensor of pairwise distances.
        """
        dists = self.dist(X, X)

        # Fill diagonal with zeros
        dists.fill_diagonal_(0.0)

        return dists

    def pdist2(self, X: Float[torch.Tensor, "n_points n_dim"]) -> Float[torch.Tensor, "n_points n_points"]:
        """Compute pairwise squared distances between points on the manifold.

        Args:
            X: Tensor of points in the manifold.

        Returns:
            pairwise_squared_distances: Tensor of pairwise squared distances.
        """
        dists2 = self.dist2(X, X)

        dists2.fill_diagonal_(0.0)

        return dists2

    def _to_tangent_plane_mu0(
        self, x: Float[torch.Tensor, "n_points n_dim"]
    ) -> Float[torch.Tensor, "n_points n_ambient_dim"]:
        """Map points to the tangent plane at the origin of the manifold.

        Args:
            x: Tensor of points to map to the tangent plane.

        Returns:
            tangent_points: Tensor of points in the tangent plane at the origin.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device).reshape(-1, self.dim)
        if self.is_stereographic:
            # Stereographic models keep the intrinsic dimension (ambient_dim == dim), so there is no
            # extra coordinate to prepend. The conformal factor at the origin is
            # lambda_0 = 2 / (1 + curvature * ||0||^2) = 2 (for every curvature, including the flat
            # case), so we divide the orthonormal tangent vector by lambda_0 to express it in
            # stereographic coordinates and recover a correctly scaled wrapped-normal sample
            # (matching the equivalent non-stereographic manifold).
            return x / 2.0
        if self.type == "E":
            return x
        return torch.cat([torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device), x], dim=1)

    def sample(
        self,
        n_samples: int = 1,
        z_mean: Float[torch.Tensor, "n_points n_ambient_dim"] | Float[torch.Tensor, "n_ambient_dim"] | None = None,
        sigma: Float[torch.Tensor, "n_points n_dim n_dim"] | None = None,
        return_tangent: bool = False,
    ) -> (
        tuple[Float[torch.Tensor, "n_points n_ambient_dim"], Float[torch.Tensor, "n_points n_dim"]]
        | Float[torch.Tensor, "n_points n_ambient_dim"]
    ):
        """Sample points from the variational distribution on the manifold.

        Args:
            n_samples: Number of points to sample.
            z_mean: Tensor representing the mean of the sample distribution.
            sigma: Optional tensor representing the covariance matrix. If None, defaults to an identity matrix.
            return_tangent: Whether to return the tangent vectors along with the sampled points.

        Returns:
            x: Tensor of sampled points on the manifold
            v: Tensor of tangent vectors (if `return_tangent` is True).
        """
        z_mean = self.mu0 if z_mean is None else z_mean
        z_mean = torch.as_tensor(z_mean, dtype=self.dtype, device=self.device).reshape(-1, self.ambient_dim)
        n = z_mean.shape[0]

        sigma = torch.stack([torch.eye(self.dim, dtype=self.dtype)] * n).to(self.device) if sigma is None else sigma
        sigma = torch.as_tensor(sigma, dtype=self.dtype, device=self.device).reshape(-1, self.dim, self.dim)
        assert sigma.shape == (
            n,
            self.dim,
            self.dim,
        ), f"Expected sigma shape {(n, self.dim, self.dim)}, got {sigma.shape}"
        assert torch.allclose(sigma, sigma.transpose(-1, -2)), "Covariance matrix must be symmetric"
        assert z_mean.shape[-1] == self.ambient_dim, f"Expected z_mean shape {self.ambient_dim}, got {z_mean.shape[-1]}"

        # Adjust for n_points:
        z_mean = torch.repeat_interleave(z_mean, n_samples, dim=0)
        sigma = torch.repeat_interleave(sigma, n_samples, dim=0)

        # Sample initial vector from N(0, sigma)
        N = torch.distributions.MultivariateNormal(
            loc=torch.zeros((n * n_samples, self.dim), dtype=self.dtype, device=self.device), covariance_matrix=sigma
        )
        v = N.sample()

        # Don't need to adjust normal vectors for the Scaled manifold class in geoopt - very cool!

        # Enter tangent plane
        v_tangent = self._to_tangent_plane_mu0(v)

        # Move to z_mean via parallel transport
        z = self.manifold.transp(x=self.mu0, y=z_mean, v=v_tangent)

        # If we're sampling at the origin, z and v should be the same
        mask = torch.all(z == self.mu0, dim=1)
        assert torch.allclose(v_tangent[mask], z[mask]), (
            "Tangent vectors at the origin should be equal to the sampled points at the origin"
        )

        # Exp map onto the manifold
        x = self.manifold.expmap(x=z_mean, u=z)

        return (x, v) if return_tangent else x

    def log_likelihood(
        self,
        z: Float[torch.Tensor, "n_points n_ambient_dim"],
        mu: Float[torch.Tensor, "n_points n_ambient_dim"] | None = None,
        sigma: Float[torch.Tensor, "n_points n_dim n_dim"] | None = None,
    ) -> Float[torch.Tensor, "n_points"]:
        r"""Compute probability density function for $\mathcal{WN}(\mathbf{z}; \mu, \Sigma)$ on the manifold.

        Args:
            z: Tensor of points on the manifold for which to compute the likelihood.
            mu: Tensor representing the mean of the distribution. If None, defaults to the origin `self.mu0`.
            sigma: Tensor representing the covariance matrix. If None, defaults to an identity matrix.

        Returns:
            log_likelihoods: Tensor containing the log-likelihood of the points `z` under the distribution with mean
                `mu` and covariance `sigma`.
        """
        # Default to mu=self.mu0 and sigma=I
        mu = self.mu0 if mu is None else mu
        mu = torch.as_tensor(mu, dtype=self.dtype, device=self.device).reshape(-1, self.ambient_dim)
        n = mu.shape[0]
        sigma = torch.stack([torch.eye(self.dim, dtype=self.dtype)] * n).to(self.device) if sigma is None else sigma
        sigma = torch.as_tensor(sigma, dtype=self.dtype, device=self.device).reshape(-1, self.dim, self.dim)

        # Euclidean case is regular old Gaussian log-likelihood
        if self.type == "E":
            return torch.distributions.MultivariateNormal(mu, sigma).log_prob(z)

        u = self.manifold.logmap(x=mu, y=z)  # Map z to tangent space at mu
        v = self.manifold.transp(x=mu, y=self.mu0, v=u)  # Parallel transport to origin
        # assert torch.allclose(v[:, 0], torch.Tensor([0.])) # For tangent vectors at origin this should be true
        # OK, so this assertion doesn't actually pass, but it's spiritually true
        if torch.isnan(v).any():
            print("NANs in parallel transport")
            v = torch.nan_to_num(v, nan=0.0)
        N = torch.distributions.MultivariateNormal(torch.zeros(self.dim, dtype=self.dtype, device=self.device), sigma)
        ll = N.log_prob(v[:, 1:])

        # For convenience
        R = self.scale
        n = self.dim

        # Final formula (epsilon to avoid log(0))
        if self.type == "S":
            sin_M = torch.sin
            u_norm = self.manifold.norm(x=mu, u=u)

        else:
            sin_M = torch.sinh
            u_norm = self.manifold.base.norm(u=u)  # Horrible workaround needed for geoopt bug # type: ignore

        return ll - (n - 1) * torch.log(R * torch.abs(sin_M(u_norm / R) / u_norm) + 1e-8)

    def logmap(
        self,
        x: Float[torch.Tensor, "n_points n_dim"],
        base: Float[torch.Tensor, "n_points n_dim"] | Float[torch.Tensor, "1 n_dim"] | None = None,
    ) -> Float[torch.Tensor, "n_points n_dim"]:
        """Compute the logarithmic map of points on the manifold at a base point.

        Args:
            x: Tensor representing points on the manifold.
            base: Tensor representing the base point for the map. If None, defaults to the origin `self.mu0`.

        Returns:
            logmap_result: Tensor representing the result of the logarithmic map from `base` to `x` on the manifold.
        """
        base = self.mu0 if base is None else base
        return self.manifold.logmap(x=base, y=x)

    def expmap(
        self,
        u: Float[torch.Tensor, "n_points n_dim"],
        base: Float[torch.Tensor, "n_points n_dim"] | Float[torch.Tensor, "1 n_dim"] | None = None,
    ) -> Float[torch.Tensor, "n_points n_dim"]:
        r"""Compute the exponential map of a tangent vector $\mathbf{u}$ at base point.

        Args:
            u: Tensor representing the tangent vector at the base point to map.
            base: Tensor representing the base point for the exponential map. If None, defaults to the origin
                `self.mu0`.

        Returns:
            expmap_result: Tensor representing the result of the exponential map applied to `u` at the base point.
        """
        base = self.mu0 if base is None else base
        return self.manifold.expmap(x=base, u=u)

    def stereographic(self, *points: Float[torch.Tensor, "n_points n_dim"]) -> tuple[Manifold, ...]:
        r"""Convert the manifold to its stereographic equivalent. If points are given, convert them as well.

        The points lie on the unit hyperboloid/sphere (geoopt uses unit curvature and carries the
        curvature in the scale), so the unit-curvature projection is rescaled by $R = |K|^{-1/2}$
        (for $i \geq 1$):
        \begin{equation}
            \rho_K(x_i) = |K|^{-1/2} \cdot \frac{x_i}{1 + x_0}
        \end{equation}

        For more information, see https://arxiv.org/pdf/1911.08411

        Args:
            *points: Variable number of tensors representing points on the manifold to convert to stereographic coords.

        Returns:
            stereo_manifold: The manifold in stereographic coordinates.
            stereo_points: The provided points converted to stereographic coordinates (if any).
        """
        if self.is_stereographic:
            print("Manifold is already in stereographic coordinates.")
            return self, *points

        # Convert manifold
        stereo_manifold = Manifold(self.curvature, self.dim, device=self.device, stereographic=True)

        # Euclidean edge case
        if self.type == "E":
            return stereo_manifold, *points

        # Convert points. The points live on the *unit* hyperboloid/sphere (geoopt uses k=1 and lets
        # the scale carry the curvature), so we apply the unit-curvature projection x_i / (1 + x_0) and
        # then rescale by R = |K|^{-1/2} to land on the curvature-K stereographic model.
        scale = abs(self.curvature) ** -0.5
        num = [scale * X[:, 1:] for X in points]
        denom = [1 + X[:, 0:1] for X in points]
        for X in denom:
            X[X.abs() < 1e-6] = 1e-6  # Avoid division by zero
        stereo_points = [n / d for n, d in zip(num, denom, strict=False)]
        assert all(stereo_manifold.manifold.check_point(X) for X in stereo_points), (
            "Generated points do not lie on the target manifold"
        )

        return stereo_manifold, *stereo_points

    def inverse_stereographic(self, *points: Float[torch.Tensor, "n_points n_dim_stereo"]) -> tuple[Manifold, ...]:
        r"""Convert the manifold from its stereographic coordinates back to the original coordinates.

        If points are given, convert them as well.

        Formula for inverse stereographic projection:
        \begin{align}
            x_0 &= \frac{1 + sign(K) \cdot \|y\|^2}{1 - sign(K) \cdot \|y\|^2} \\\\
            x_i &= \frac{2 \cdot y_i}{1 - sign(K) \cdot \|y\|^2}
        \end{align}

        Args:
            *points: Variable number of tensors representing points in stereographic coords to convert back to original
                coords.

        Returns:
            inv_stereo_manifold: The manifold in original coords.
            inv_stereo_points: The provided points converted from stereographic to original coords (if any).
        """
        if not self.is_stereographic:
            print("Manifold is already in original coordinates.")
            return self, *points

        # Convert manifold
        orig_manifold = Manifold(self.curvature, self.dim, device=self.device, stereographic=False)

        # Euclidean edge case
        if self.type == "E":
            return orig_manifold, *points

        # Inverse projection for points
        out = []
        for X in points:
            # Calculate squared norm
            # let σ = sign(K)  and  λ = sqrt(|K|)
            sign = torch.sign(torch.tensor(self.curvature, device=self.device))
            lam = abs(self.curvature) ** 0.5

            # compute the ‖·‖² in the *scaled* ball
            norm2 = torch.sum((lam * X) ** 2, dim=1)

            # inverse‐stereographic denom must be (1 + σ⋅‖y‖²), *not* (1 – σ⋅‖y‖²)
            denom = 1.0 + sign * norm2
            # clamp to avoid blow‐up at the boundary
            denom = torch.clamp_min(denom.abs(), 1e-6) * denom.sign()

            # then
            X0 = (1.0 - sign * norm2) / denom
            Xi = 2.0 * lam * X / denom.unsqueeze(1)

            # Combine into full coordinates
            inv_points = torch.cat([X0.unsqueeze(1), Xi], dim=1)

            # Let the manifold class validate the points
            if not orig_manifold.manifold.check_point(inv_points):
                raise ValueError("Generated points do not lie on the target manifold")

            out.append(inv_points)

        return orig_manifold, *out

    def apply(self, f: Callable) -> Callable:
        """Create a decorator for logmap -> function -> expmap. If a base point is not provided, use the origin.

        Args:
            f: Function to apply in the tangent space.

        Returns:
            wrapper: Callable representing the composed map that:

                1. Maps points to the tangent space using logmap
                2. Applies the function f
                3. Maps the result back to the manifold using expmap
        """

        def wrapper(x: Float[torch.Tensor, "n_points n_dim"]) -> Float[torch.Tensor, "n_points n_dim"]:
            return self.expmap(
                f(self.logmap(x, base=self.mu0)),
                base=self.mu0,
            )

        return wrapper


class ProductManifold(Manifold):
    """Tools for constructing product manifolds with multiple factors.

    A product manifold combines multiple manifolds with different curvatures and dimensions into a single product space.

    Args:
        signature: List of (curvature, dimension) tuples for each factor manifold.
        device: The device on which the manifold is stored.
        stereographic: Whether to use stereographic coordinates.

    Attributes:
        signature: List of tuples defining the curvature and dimension of each factor manifold.
        device: The device on which the manifold is stored.
        is_stereographic: Whether stereographic coordinates are used.
        type: String identifier for product manifold (always 'P').
        curvatures: List of curvature values for each factor manifold.
        dims: List of dimensions for each factor manifold.
        n_manifolds: Number of factor manifolds.
        P: List of individual Manifold objects that make up this product manifold.
        manifold: The underlying geoopt ProductManifold object.
        name: String identifier for the product manifold.
        mu0: The origin point on the product manifold.
        ambient_dim: Total ambient dimension of the product manifold.
        dim: Total intrinsic dimension of the product manifold.
        dim2man: Dictionary mapping dimensions to manifold indices.
        man2dim: Dictionary mapping manifold indices to their dimensions.
        man2intrinsic: Dictionary mapping manifold indices to their intrinsic dimensions.
        intrinsic2man: Dictionary mapping intrinsic dimensions to manifold indices.
        projection_matrix: Matrix for projecting from intrinsic to ambient dimensions.
    """

    def __init__(self, signature: list[tuple[float, int]], device: str = "cpu", stereographic: bool = False):
        # Device management
        self.device = device

        # Basic properties
        self.type = "P"
        self.signature = signature
        self.curvatures = [curvature for curvature, _ in signature]
        self.dims = [dim for _, dim in signature]
        self.n_manifolds = len(signature)
        self.is_stereographic = stereographic

        # Actually initialize the geoopt manifolds; other derived properties
        self.P = [Manifold(curvature, dim, device=device, stereographic=stereographic) for curvature, dim in signature]
        manifold_class = geoopt.StereographicProductManifold if stereographic else geoopt.ProductManifold
        self.manifold = manifold_class(*[(M.manifold, M.ambient_dim) for M in self.P]).to(device)
        self.name = " x ".join([M.name for M in self.P])

        # Origin
        self.mu0 = torch.cat([M.mu0 for M in self.P], axis=1).to(self.device)

        # Manifold <-> Dimension mapping
        self.ambient_dim, self.n_manifolds, self.dim = 0, 0, 0
        self.dim2man, self.man2dim, self.man2intrinsic, self.intrinsic2man = {}, {}, {}, {}

        for M in self.P:
            for d in range(self.ambient_dim, self.ambient_dim + M.ambient_dim):
                self.dim2man[d] = self.n_manifolds
            for d in range(self.dim, self.dim + M.dim):
                self.intrinsic2man[d] = self.n_manifolds
            self.man2dim[self.n_manifolds] = list(range(self.ambient_dim, self.ambient_dim + M.ambient_dim))
            self.man2intrinsic[self.n_manifolds] = list(range(self.dim, self.dim + M.dim))

            self.ambient_dim += M.ambient_dim
            self.n_manifolds += 1
            self.dim += M.dim

        # Lift matrix - useful for tensor stuff
        # The idea here is to right-multiply by this to lift a vector in R^dim to a vector in R^ambient_dim
        # such that there are zeros in all the right places, i.e. to make it a tangent vector at the origin of P
        self.projection_matrix = torch.zeros(self.dim, self.ambient_dim, device=self.device)
        for i in range(len(self.P)):
            intrinsic_dims = self.man2intrinsic[i]
            ambient_dims = self.man2dim[i]
            for j, k in zip(intrinsic_dims, ambient_dims[-len(intrinsic_dims) :], strict=False):
                self.projection_matrix[j, k] = 1.0

    def parameters(self) -> list[torch.nn.parameter.Parameter]:
        """Get scale parameters for all component manifolds.

        Returns:
            scales: List of scale parameters for each component manifold.
        """
        return [x._log_scale for x in self.manifold.manifolds]

    def to(self, device: str) -> ProductManifold:
        """Move all components to a new device.

        Args:
            device: The device to which to move all components.

        Returns:
            manifold: The updated ProductManifold object on the specified device.
        """
        self.device = device
        self.P = [M.to(device) for M in self.P]
        self.manifold = self.manifold.to(device)
        self.mu0 = self.mu0.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        return self

    def inner(
        self, X: Float[torch.Tensor, "n_points1 n_dim"], Y: Float[torch.Tensor, "n_points2 n_dim"]
    ) -> Float[torch.Tensor, "n_points1 n_points2"]:
        """Compute the inner product between points on the product manifold.

        The inner product is the sum of inner products in each component manifold.

        Args:
            X: Tensor of points in the product manifold.
            Y: Tensor of points in the product manifold.

        Returns:
            inner_products: Tensor of inner products between points.
        """
        ips = [M.inner(x, y) for x, y, M in zip(self.factorize(X), self.factorize(Y), self.P, strict=False)]
        return torch.stack(ips, dim=0).sum(dim=0)

    def factorize(
        self, X: Float[torch.Tensor, "n_points n_dim"], intrinsic: bool = False
    ) -> list[Float[torch.Tensor, "n_points n_dim_manifold"]]:
        """Factorize the embeddings into the individual manifolds.

        Args:
            X: Tensor representing the embeddings to be factorized.
            intrinsic: bool for whether to use intrinsic dimensions of the manifolds.

        Returns:
            X_factorized: A list of tensors representing the factorized embeddings in each manifold.
        """
        dims_dict = self.man2intrinsic if intrinsic else self.man2dim
        return [X[..., dims_dict[i]] for i in range(len(self.P))]

    def sample(
        self,
        n_samples: int = 1,
        z_mean: Float[torch.Tensor, "n_points n_ambient_dim"] | None = None,
        sigma_factorized: list[Float[torch.Tensor, "n_points ..."]] | None = None,
        return_tangent: bool = False,
    ) -> (
        tuple[Float[torch.Tensor, "n_points n_ambient_dim"], Float[torch.Tensor, "n_points total_intrinsic_dim"]]
        | Float[torch.Tensor, "n_points n_ambient_dim"]
    ):
        """Sample from the variational distribution.

        Args:
            n_samples: Number of points to sample.
            z_mean: Tensor representing the mean of the sample distribution. If None, defaults to the origin `self.mu0`.
            sigma_factorized: List of tensors representing factorized covariance matrices for each manifold. If None,
                defaults to a list of identity matrices for each manifold.
            return_tangent: Whether to return the tangent vectors along with the sampled points.

        Returns:
            x: Tensor of sampled points on the manifold
            v: Tensor of tangent vectors (if `return_tangent` is True).
        """
        z_mean = self.mu0 if z_mean is None else z_mean
        z_mean = torch.Tensor(z_mean).reshape(-1, self.ambient_dim).to(self.device)
        n = z_mean.shape[0]

        sigma_factorized = (
            [torch.stack([torch.eye(M.dim)] * n) for M in self.P] if sigma_factorized is None else sigma_factorized
        )
        sigma_factorized = [
            torch.Tensor(sigma).reshape(-1, M.dim, M.dim).to(self.device)
            for M, sigma in zip(self.P, sigma_factorized, strict=False)
        ]

        # Adjust for n_points:
        z_mean = torch.repeat_interleave(z_mean, n_samples, dim=0)
        sigma_factorized = [torch.repeat_interleave(sigma, n_samples, dim=0) for sigma in sigma_factorized]

        assert all(
            sigma.shape == (n * n_samples, M.dim, M.dim) for M, sigma in zip(self.P, sigma_factorized, strict=False)
        ), "Sigma matrices must match the dimensions of the manifolds."
        assert z_mean.shape == (n * n_samples, self.ambient_dim), (
            "z_mean must have the same ambient dimension as the product manifold."
        )

        # Sample initial vector from N(0, sigma)
        samples = [
            M.sample(1, z_M, sigma_M, return_tangent=True)
            for M, z_M, sigma_M in zip(self.P, self.factorize(z_mean), sigma_factorized, strict=False)
        ]

        x = torch.cat([s[0] for s in samples], dim=1)
        v = torch.cat([s[1] for s in samples], dim=1)

        # Different samples and tangent vectors
        return (x, v) if return_tangent else x

    def log_likelihood(
        self,
        z: Float[torch.Tensor, "batch_size n_dim"],
        mu: Float[torch.Tensor, "batch_size n_dim"] | None = None,
        sigma_factorized: list[Float[torch.Tensor, "batch_size ..."]] | None = None,
    ) -> Float[torch.Tensor, "batch_size"]:
        r"""Compute probability density function for $\mathcal{WN}(\mathbf{z} ; \mu, \Sigma)$ on the product manifold.

        Args:
            z: Tensor representing the points for which the log-likelihood is computed.
            mu: Tensor representing the mean of the distribution. If None, defaults to the origin `self.mu0`.
            sigma_factorized: List of tensors representing factorized covariance matrices for each manifold. If None,
                defaults to a list of identity matrices for each manifold.

        Returns:
            log_likelihoods: Tensor containing the log-likelihood of the points `z` under the distribution with mean
                `mu` and covariance `sigma`.
        """
        n = z.shape[0]
        mu = torch.vstack([self.mu0] * n).to(self.device) if mu is None else mu

        sigma_factorized = (
            [torch.stack([torch.eye(M.dim)] * n) for M in self.P] if sigma_factorized is None else sigma_factorized
        )
        # Note that this factorization assumes block-diagonal covariance matrices

        mu_factorized = self.factorize(mu)
        z_factorized = self.factorize(z)
        component_lls = [
            M.log_likelihood(z_M, mu_M, sigma_M).unsqueeze(dim=1)
            for M, z_M, mu_M, sigma_M in zip(self.P, z_factorized, mu_factorized, sigma_factorized, strict=False)
        ]
        return torch.cat(component_lls, axis=1).sum(axis=1)

    def stereographic(self, *points: Float[torch.Tensor, "n_points n_dim"]) -> tuple[ProductManifold, ...]:
        r"""Convert the manifold to its stereographic equivalent. If points are given, convert them as well.

        The points lie on the unit hyperboloid/sphere (geoopt uses unit curvature and carries the
        curvature in the scale), so the unit-curvature projection is rescaled by $R = |K|^{-1/2}$
        (for $i \geq 1$):
        \begin{equation}
            \rho_K(x_i) = |K|^{-1/2} \cdot \frac{x_i}{1 + x_0}
        \end{equation}

        For more information, see https://arxiv.org/pdf/1911.08411

        Args:
            *points: Variable number of tensors representing points on the manifold to convert to stereographic coords.

        Returns:
            stereo_manifold: The manifold in stereographic coords.
            stereo_points: The provided points converted to stereographic coords (if any).
        """
        if self.is_stereographic:
            print("Manifold is already in stereographic coords.")
            return self, *points

        # Convert manifold
        stereo_manifold = ProductManifold(self.signature, device=self.device, stereographic=True)

        # Convert points
        stereo_points = [
            torch.hstack([M.stereographic(x)[1] for x, M in zip(self.factorize(X), self.P, strict=False)])
            for X in points
        ]
        assert all(stereo_manifold.manifold.check_point(X) for X in stereo_points), (
            "Generated points do not lie on the target manifold"
        )

        return stereo_manifold, *stereo_points

    def inverse_stereographic(
        self, *points: Float[torch.Tensor, "n_points n_dim_stereo"]
    ) -> tuple[ProductManifold, ...]:
        r"""Convert the manifold from its stereographic coordinates back to the original coordinates.

        If points are given, convert them as well.

        Formula for inverse stereographic projection:
        \begin{align}
            x_0 &= \frac{1 + sign(K) \cdot \|y\|^2}{1 - sign(K) \cdot \|y\|^2} \\\\
            x_i &= \frac{2 \cdot y_i}{1 - sign(K) \cdot \|y\|^2}
        \end{align}

        Args:
            *points: Variable number of tensors representing points in stereographic coords to convert back to original
                coords.

        Returns:
            inv_stereo_manifold: The manifold in original coords.
            inv_stereo_points: The provided points converted from stereographic to original coords (if any).
        """
        if not self.is_stereographic:
            print("Manifold is already in original coordinates.")
            return self, *points

        # Convert manifold
        orig_manifold = ProductManifold(self.signature, device=self.device, stereographic=False)

        orig_points = [
            torch.hstack([M.inverse_stereographic(x)[1] for x, M in zip(self.factorize(X), self.P, strict=False)])
            for X in points
        ]
        assert all(orig_manifold.manifold.check_point(X) for X in orig_points), (
            "Generated points do not lie on the target manifold"
        )

        return orig_manifold, *orig_points

    @torch.no_grad()  # type: ignore
    def gaussian_mixture(
        self,
        num_points: int = 1_000,
        num_classes: int = 2,
        num_clusters: int | None = None,
        seed: int | None = None,
        cov_scale_means: float = 1.0,
        cov_scale_points: float = 1.0,
        regression_noise_std: float = 0.1,
        task: Literal["classification", "regression"] = "classification",
        adjust_for_dims: bool = False,
    ) -> tuple[Float[torch.Tensor, "n_points n_ambient_dim"], Real[torch.Tensor, "n_points"]]:
        """Generate a set of labeled samples from a Gaussian mixture model.

        Args:
            num_points: The number of points to generate.
            num_classes: The number of classes to generate.
            num_clusters: The number of clusters to generate. If None, defaults to num_classes.
            seed: An optional seed for the random number generator. If None, no random seed is set.
            cov_scale_means: The scale of the covariance matrix for the means.
            cov_scale_points: The scale of the covariance matrix for the points.
            regression_noise_std: The standard deviation of the noise for regression labels.
            task: The type of labels to generate. Either "classification" or "regression".
            adjust_for_dims: Whether to adjust the covariance matrices for the number of dimensions in each manifold.

        Returns:
            samples: A tensor of generated samples.
            class_assignments: A tensor of class assignments for the samples.
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)

        # Deal with clusters
        num_clusters = num_clusters or num_classes
        assert num_clusters >= num_classes, "Number of clusters must be at least as large as number of classes."

        # Adjust covariance matrices for number of dimensions
        if adjust_for_dims:
            cov_scale_points /= self.dim
            cov_scale_means /= self.dim

        # Generate cluster means
        cluster_means = self.sample(num_clusters, sigma_factorized=[torch.eye(M.dim) * cov_scale_means for M in self.P])
        assert cluster_means.shape == (num_clusters, self.ambient_dim), "Cluster means shape mismatch."  # type: ignore

        # Generate class assignments
        cluster_probs = torch.rand(num_clusters)
        cluster_probs /= cluster_probs.sum()

        # Draw cluster assignments: ensure at least 2 points per cluster. This is to ensure splits can always happen.
        cluster_assignments = torch.multinomial(input=cluster_probs, num_samples=num_points, replacement=True)
        while (cluster_assignments.bincount() < 2).any():
            cluster_assignments = torch.multinomial(input=cluster_probs, num_samples=num_points, replacement=True)
        assert cluster_assignments.shape == (num_points,), "Cluster assignments shape mismatch."

        # Generate covariance matrices for each class - Wishart distribution
        cov_matrices = [
            torch.distributions.Wishart(df=M.dim + 1, covariance_matrix=torch.eye(M.dim) * cov_scale_points).sample(
                sample_shape=(num_clusters,)
            )
            + torch.eye(M.dim) * 1e-5  # jitter to avoid singularity
            for M in self.P
        ]

        # Generate random samples for each cluster
        sample_means = torch.stack([cluster_means[c] for c in cluster_assignments])
        assert sample_means.shape == (num_points, self.ambient_dim), "Sample means shape mismatch."
        sample_covs = [torch.stack([cov_matrix[c] for c in cluster_assignments]) for cov_matrix in cov_matrices]
        samples, tangent_vals = self.sample(z_mean=sample_means, sigma_factorized=sample_covs, return_tangent=True)
        assert samples.shape == (num_points, self.ambient_dim), "Sample shape mismatch."

        # Map clusters to classes
        cluster_to_class = torch.cat(
            [
                torch.arange(num_classes, device=self.device),
                torch.randint(0, num_classes, (num_clusters - num_classes,), device=self.device),
            ]
        )
        assert cluster_to_class.shape == (num_clusters,), "Cluster to class mapping shape mismatch."
        assert torch.unique(cluster_to_class).shape == (num_classes,), (
            "Cluster to class mapping must cover all classes."
        )

        # Generate outputs
        if task == "classification":
            labels = cluster_to_class[cluster_assignments]
        elif task == "regression":
            slopes = (0.5 - torch.randn(num_clusters, self.dim, device=self.device)) * 2
            intercepts = (0.5 - torch.randn(num_clusters, device=self.device)) * 20
            labels = (
                torch.einsum("ij,ij->i", slopes[cluster_assignments], tangent_vals) + intercepts[cluster_assignments]
            )

            # Noise component
            N = torch.distributions.Normal(0, regression_noise_std)
            v = N.sample((num_points,)).to(self.device)
            labels += v

            # Normalize regression labels to range [0, 1] so that RMSE can be more easily interpreted
            labels = (labels - labels.min()) / (labels.max() - labels.min())

        return samples, labels
