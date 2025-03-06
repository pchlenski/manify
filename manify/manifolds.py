"""
Tools for generating Riemannian manifolds and product manifolds.

The module consists of two classes: Manifold and ProductManifold .The Manifold class
represents hyperbolic, Euclidean, or spherical manifolds based on curvature.
The ProductManifold class supports products of multiple manifolds,
combining their geometric properties to create mixed-curvature. Both classes
includes functions for different key geometric operations.
"""

from typing import List, Optional, Tuple, Callable, Literal, Union
from jaxtyping import Float

import torch
import geoopt


class Manifold:
    """
    Tools for generating Riemannian manifolds.

    Parameters
    ----------
    curvature: (float) The curvature of the manifold. Negative for hyperbolic,
    zero for Euclidean, and positive for spherical manifolds.
    dim: (int) The dimension of the manifold.
    device: (str) The device on which the manifold is stored (default: "cpu").
    stereographic: (bool) Whether to use stereographic coordinates for the manifold.
    """

    def __init__(self, curvature: float, dim: int, device: str = "cpu", stereographic: bool = False):
        # Device management
        self.device = device

        # Basic properties
        self.curvature = curvature
        self.dim = dim
        self.scale = abs(curvature) ** -0.5 if curvature != 0 else 1
        self.is_stereographic = stereographic

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

        # Couple of assertions to check
        assert self.manifold.check_point(self.mu0)

    def to(self, device: str) -> "Manifold":
        """
        Move objects to a specified device

        Args:
            device: (str) The device to which the manifold will be moved.

        Returns:
            self: The updated manifold object.
        """
        self.device = device
        self.manifold = self.manifold.to(device)
        self.mu0 = self.mu0.to(device)
        return self

    def inner(
        self, X: Float[torch.Tensor, "n_points1 n_dim"], Y: Float[torch.Tensor, "n_points2 n_dim"]
    ) -> Float[torch.Tensor, "n_points1 n_points2"]:
        """
        Compute the inner product of manifolds.

        Args:
            X: (n_points1, n_dim) Tensor of points in the manifold.
            Y: (n_points2, n_dim) Tensor of points in the manifold.

        Returns:
            inner_product: (n_points1, n_points2) Tensor of inner products between points.
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
        """
        Inherit distance function from the geoopt manifold.

        Args:
            X: (n_points1, n_dim) Tensor of points in the manifold.
            Y: (n_points2, n_dim) Tensor of points in the manifold.

        Returns:
            distance: (n_points1, n_points2) Tensor of distances between points.
        """
        return self.manifold.dist(X[:, None], Y[None, :])

    def dist2(
        self, X: Float[torch.Tensor, "n_points1 n_dim"], Y: Float[torch.Tensor, "n_points2 n_dim"]
    ) -> Float[torch.Tensor, "n_points1 n_points2"]:
        """
        Inherit squared distance function from the geoopt manifold.

        Args:
            X: (n_points1, n_dim) Tensor of points in the manifold.
            Y: (n_points2, n_dim) Tensor of points in the manifold.

        Returns:
            distance: (n_points1, n_points2) Tensor of squared distances between points.
        """
        return self.manifold.dist2(X[:, None], Y[None, :])

    def pdist(self, X: Float[torch.Tensor, "n_points n_dim"]) -> Float[torch.Tensor, "n_points n_points"]:
        """
        Compute pairwise distances between embeddings.

        Args:
            X: (n_points, n_dim) Tensor of points in the manifold.

        Returns:
            dists: (n_points, n_points) Tensor of pairwise distances.
        """
        dists = self.dist(X, X)

        # Fill diagonal with zeros
        dists.fill_diagonal_(0.0)

        return dists

    def pdist2(self, X: Float[torch.Tensor, "n_points n_dim"]) -> Float[torch.Tensor, "n_points n_points"]:
        """
        Compute pairwise squared distances between embeddings.

        Args:
            X: (n_points, n_dim) Tensor of points in the manifold.

        Returns:
            dists2: (n_points, n_points) Tensor of squared distances.
        """
        dists2 = self.dist2(X, X)

        dists2.fill_diagonal_(0.0)

        return dists2

    def _to_tangent_plane_mu0(
        self, x: Float[torch.Tensor, "n_points n_dim"]
    ) -> Float[torch.Tensor, "n_points n_ambient_dim"]:
        """Map points to the tangent plane at the origin of the manifold."""
        x = torch.Tensor(x).reshape(-1, self.dim)
        if self.type == "E":
            return x
        else:
            return torch.cat([torch.zeros((x.shape[0], 1), device=self.device), x], dim=1)

    def sample(
        self,
        z_mean: Optional[Float[torch.Tensor, "n_points n_ambient_dim"]] = None,
        sigma: Optional[Float[torch.Tensor, "n_points n_dim n_dim"]] = None,
    ) -> Union[
        Float[torch.Tensor, "n_points n_ambient_dim"],
        Tuple[Float[torch.Tensor, "n_points n_ambient_dim"], Float[torch.Tensor, "n_points n_dim"]],
    ]:
        """
        Sample from the variational distribution.

        Args:
            z_mean: (n_points, n_dim) Tensor representing the mean of the sample distribution. Defaults to `self.mu0`.
            sigma_factorized: List of tensors representing factorized covariance matrices for each manifold.

        Returns:
            Tensor or tuple of tensor(n_points, n_ambient_dim) representing the sampled points on the manifold.
            If `return_tangent` is True, also returns the tangent vectors with shape `(n_points, n_dim)`.
        """
        if z_mean is None:
            z_mean = self.mu0
        z_mean = torch.Tensor(z_mean).reshape(-1, self.ambient_dim).to(self.device)
        n = z_mean.shape[0]
        if sigma is None:
            sigma = torch.stack([torch.eye(self.dim)] * n).to(self.device)
        else:
            sigma = torch.Tensor(sigma).reshape(-1, self.dim, self.dim).to(self.device)
        assert sigma.shape == (n, self.dim, self.dim)
        assert torch.all(sigma == sigma.transpose(-1, -2))
        assert z_mean.shape[-1] == self.ambient_dim

        # Sample initial vector from N(0, sigma)
        N = torch.distributions.MultivariateNormal(
            loc=torch.zeros((n, self.dim), device=self.device), covariance_matrix=sigma
        )
        v = N.sample(sample_shape=(1,))  # type: ignore

        # Don't need to adjust normal vectors for the Scaled manifold class in geoopt - very cool!

        # Enter tangent plane
        v_tangent = self._to_tangent_plane_mu0(v)

        # Move to z_mean via parallel transport
        z = self.manifold.transp(x=self.mu0, y=z_mean, v=v_tangent)

        # If we're sampling at the origin, z and v should be the same
        mask = torch.all(z == self.mu0, dim=1)
        assert torch.allclose(v_tangent[mask], z[mask])

        # Exp map onto the manifold
        x = self.manifold.expmap(x=z_mean, u=z)

        # Different samples and tangent vectors
        return x, v

    def log_likelihood(
        self,
        z: Float[torch.Tensor, "n_points n_ambient_dim"],
        mu: Optional[Float[torch.Tensor, "n_points n_ambient_dim"]] = None,
        sigma: Optional[Float[torch.Tensor, "n_points n_dim n_dim"]] = None,
    ) -> Float[torch.Tensor, "n_points"]:
        """
        Probability density function for WN(z ; mu, Sigma) in manifold

        Args:
            z: (n_points", "n_ambient_dim) Tensor of points on the manifold for which the likelihood is computed.
            mu: (n_points", "n_ambient_dim) Tensor representing the mean of the distribution. Defaults to `self.mu0`.
            sigma: (n_points", "n_dim", "n_dim) Tensor representing the covariance matrix. Defaults to identity matrix.

        Returns:
            (n_points) Tensor containing the log-likelihood of the points `z` under the distribution
            with mean `mu` and covariance `sigma.`

        """

        # Default to mu=self.mu0 and sigma=I
        if mu is None:
            mu = self.mu0
        mu = torch.Tensor(mu).reshape(-1, self.ambient_dim).to(self.device)
        n = mu.shape[0]
        if sigma is None:
            sigma = torch.stack([torch.eye(self.dim)] * n).to(self.device)
        else:
            sigma = torch.Tensor(sigma).reshape(-1, self.dim, self.dim).to(self.device)

        # Euclidean case is regular old Gaussian log-likelihood
        if self.type == "E":
            return torch.distributions.MultivariateNormal(mu, sigma).log_prob(z)

        else:
            u = self.manifold.logmap(x=mu, y=z)  # Map z to tangent space at mu
            v = self.manifold.transp(x=mu, y=self.mu0, v=u)  # Parallel transport to origin
            # assert torch.allclose(v[:, 0], torch.Tensor([0.])) # For tangent vectors at origin this should be true
            # OK, so this assertion doesn't actually pass, but it's spiritually true
            if torch.isnan(v).any():
                print("NANs in parallel transport")
                v = torch.nan_to_num(v, nan=0.0)
            N = torch.distributions.MultivariateNormal(torch.zeros(self.dim, device=self.device), sigma)
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
        self, x: Float[torch.Tensor, "n_points n_dim"], base: Optional[Float[torch.Tensor, "n_points n_dim"]] = None
    ) -> Float[torch.Tensor, "n_points n_dim"]:
        """
        Logarithmic map of point on manifold x at base point.

        Args:
            x: Tensor representing the point on the manifold for the logarithmic map.
            base: Tensor representing the base point for the map. Defaults to `self.mu0` if not provided.

        Returns:
            Tensor representing the result of the logarithmic map from `base` to `x` on the manifold.

        """
        if base is None:
            base = self.mu0
        return self.manifold.logmap(x=base, y=x)

    def expmap(
        self, u: Float[torch.Tensor, "n_points n_dim"], base: Optional[Float[torch.Tensor, "n_points n_dim"]] = None
    ) -> Float[torch.Tensor, "n_points n_dim"]:
        """
        Exponential map of tangent vector u at base point.

        Args:
            u: Tensor representing the tangent vector at the base point to map.
            base: Tensor representing the base point for the exponential map. Defaults to `self.mu0` if not provided.

        Returns:
            Tensor representing the result of the exponential map applied to `u` at the base point.
        """
        if base is None:
            base = self.mu0
        return self.manifold.expmap(x=base, u=u)

    def stereographic(self, *points: Float[torch.Tensor, "n_points n_dim"]) -> Tuple["Manifold", ...]:
        """
        Convert the manifold to its stereographic equivalent. If points are given, convert them as well.

        Formula for stereographic projection:
        rho_K(x) = x[1:] / (1 + sqrt(|K|) * x[0])

        Source:
        https://arxiv.org/pdf/1911.08411
        """

        if self.is_stereographic:
            print("Manifold is already in stereographic coordinates.")
            return self, *points  # type: ignore

        # Convert manifold
        stereo_manifold = Manifold(self.curvature, self.dim, device=self.device, stereographic=True)

        # Euclidean edge case
        if self.type == "E":
            return stereo_manifold, *points  # type: ignore

        # Convert points
        num = [X[:, 1:] for X in points]
        denom = [1 + abs(self.curvature) ** 0.5 * X[:, 0:1] for X in points]
        for X in denom:
            X[X.abs() < 1e-6] = 1e-6  # Avoid division by zero
        stereo_points = [n / d for n, d in zip(num, denom)]
        assert all([stereo_manifold.manifold.check_point(X) for X in stereo_points])

        return stereo_manifold, *stereo_points  # type: ignore

    def inverse_stereographic(self, *points: Float[torch.Tensor, "n_points n_dim_stereo"]) -> Tuple["Manifold", ...]:
        """
        Convert the manifold from its stereographic coordinates back to the original coordinates.
        If points are given, convert them as well.

        Formula for inverse stereographic projection:
        X0 = (1 + sign(K) * ||y||**2) / (1 - sign(K) * ||y||**2)
        Xi = 2 * yi / (1 - sign(K) * ||y||**2)

        Source:
        https://arxiv.org/pdf/1911.08411
        """
        if not self.is_stereographic:
            print("Manifold is already in original coordinates.")
            return self, *points  # type: ignore

        # Convert manifold
        orig_manifold = Manifold(self.curvature, self.dim, device=self.device, stereographic=False)

        # Euclidean edge case
        if self.type == "E":
            return orig_manifold, *points  # type: ignore

        # Inverse projection for points
        norm_squared = [(Y**2).sum(dim=1, keepdim=True) for Y in points]
        sign = torch.sign(self.curvature)  # type: ignore

        X0 = (1 + sign * norm_squared) / (1 - sign * norm_squared)
        Xi = 2 * points / (1 - sign * norm_squared)

        inv_points = [torch.cat([x0, xi], dim=1) for x0, xi in zip(X0, Xi)]
        assert all([orig_manifold.manifold.check_point(X) for X in inv_points])

        return orig_manifold, *inv_points  # type: ignore

    def apply(self, f: Callable) -> Callable:
        """
        Decorator for logmap -> function -> expmap. If a base point is not provided, use the origin.

        Args:
            f: Function.

        Returns:
            Callable representing the composed map.
        """

        def wrapper(x: Float[torch.Tensor, "n_points n_dim"]) -> Float[torch.Tensor, "n_points n_dim"]:
            return self.expmap(
                f(self.logmap(x, base=self.mu0)),
                base=self.mu0,
            )

        return wrapper


class ProductManifold(Manifold):
    """
    Tools for generating Riemannian manifolds.

    Parameters
    ----------
    Signature: (Tuple[float, int]) A list for the signature for the product manifold
    device: (str) The device on which the manifold is stored (default: "cpu").
    stereographic: (bool) Whether to use stereographic coordinates for the manifold.
    """

    def __init__(self, signature: List[Tuple[float, int]], device: str = "cpu", stereographic: bool = False):
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
        self.manifold = manifold_class(*[(M.manifold, M.ambient_dim) for M in self.P]).to(device)  # type: ignore
        self.name = " x ".join([M.name for M in self.P])

        # Origin
        self.mu0 = torch.cat([M.mu0 for M in self.P], axis=1).to(self.device)  # type: ignore

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
            for j, k in zip(intrinsic_dims, ambient_dims[-len(intrinsic_dims) :]):
                self.projection_matrix[j, k] = 1.0

    def params(self):
        """Returns scales for all component manifolds"""
        return [x.scale() for x in self.manifold.manifolds]

    def to(self, device: str):
        """Move all components to a new device"""
        self.device = device
        self.P = [M.to(device) for M in self.P]
        self.manifold = self.manifold.to(device)
        self.mu0 = self.mu0.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        return self

    def factorize(
        self, X: Float[torch.Tensor, "n_points n_dim"], intrinsic: bool = False
    ) -> List[Float[torch.Tensor, "n_points n_dim_manifold"]]:
        """
        Factorize the embeddings into the individual manifolds.

        Args:
            X: (n_points", "n_dim) tensor representing the embeddings to be factorized.
            intrinsic: bool for whether to use intrinsic dimensions of the manifolds. Defaults to False.

        Returns:
            (List[Tensor]) list of tensors representing the factorized embeddings in each manifold.
        """
        dims_dict = self.man2intrinsic if intrinsic else self.man2dim
        return [X[..., dims_dict[i]] for i in range(len(self.P))]

    def sample(
        self,
        z_mean: Optional[Float[torch.Tensor, "n_points n_dim"]] = None,
        sigma_factorized: Optional[List[Float[torch.Tensor, "n_points n_dim_manifold n_dim_manifold"]]] = None,
    ) -> Union[
        Float[torch.Tensor, "n_points n_ambient_dim"],
        Tuple[Float[torch.Tensor, "n_points n_ambient_dim"], Float[torch.Tensor, "n_points n_dim"]],
    ]:
        """
        Sample from the variational distribution.

        Args:
            z_mean: (n_points, n_dim) Tensor representing the mean of the sample distribution. Defaults to `self.mu0`.
            sigma_factorized: List of tensors representing factorized covariance matrices for each manifold.

        Returns:
            Tensor or tuple of tensor of shape `(n_points, n_ambient_dim)` representing sampled points on the manifold.
            If `return_tangent` is True, also returns the tangent vectors with shape `(n_points, n_dim)`.
        """
        if z_mean is None:
            z_mean = self.mu0
        z_mean = torch.Tensor(z_mean).reshape(-1, self.ambient_dim).to(self.device)
        n = z_mean.shape[0]

        if sigma_factorized is None:
            sigma_factorized = [torch.stack([torch.eye(M.dim)] * n) for M in self.P]
        else:
            sigma_factorized = [
                torch.Tensor(sigma).reshape(-1, M.dim, M.dim).to(self.device)
                for M, sigma in zip(self.P, sigma_factorized)
            ]

        assert sum([sigma.shape == (n, M.dim, M.dim) for M, sigma in zip(self.P, sigma_factorized)]) == len(self.P)
        assert z_mean.shape[-1] == self.ambient_dim

        # Sample initial vector from N(0, sigma)
        samples = [M.sample(z_M, sigma_M) for M, z_M, sigma_M in zip(self.P, self.factorize(z_mean), sigma_factorized)]

        x = torch.cat([s[0] for s in samples], dim=1)
        v = torch.cat([s[1] for s in samples], dim=1)

        # Different samples and tangent vectors
        return x, v

    def log_likelihood(
        self,
        z: Float[torch.Tensor, "batch_size n_dim"],
        mu: Optional[Float[torch.Tensor, "n_dim"]] = None,
        sigma_factorized: Optional[List[Float[torch.Tensor, "n_points n_dim_manifold n_dim_manifold"]]] = None,
    ) -> Float[torch.Tensor, "batch_size"]:
        """
        Probability density function for WN(z ; mu, Sigma) in manifold

        Args:
            z: (batch_size, n_dim) Tensor representing the points for which the log-likelihood is computed.
            mu: (n_dim,) Tensor representing the mean of the distribution.
            sigma_factorized: List of tensors representing factorized covariance matrices for each manifold.

        Returns:
            (batch_size,) Tensor of shape containing the log-likelihood
            of each point in `z`  with mean `mu` and covariance `sigma`.
        """
        n = z.shape[0]
        if mu is None:
            mu = torch.stack([self.mu0] * n).to(self.device)

        if sigma_factorized is None:
            sigma_factorized = [torch.stack([torch.eye(M.dim)] * n) for M in self.P]
            # Note that this factorization assumes block-diagonal covariance matrices

        mu_factorized = self.factorize(mu)
        z_factorized = self.factorize(z)
        component_lls = [
            M.log_likelihood(z_M, mu_M, sigma_M).unsqueeze(dim=1)
            for M, z_M, mu_M, sigma_M in zip(self.P, z_factorized, mu_factorized, sigma_factorized)
        ]
        return torch.cat(component_lls, axis=1).sum(axis=1)  # type: ignore

    def stereographic(self, *points: Float[torch.Tensor, "n_points n_dim"]) -> Tuple[Manifold, ...]:
        if self.is_stereographic:
            print("Manifold is already in stereographic coordinates.")
            return self, *points  # type: ignore

        # Convert manifold
        stereo_manifold = ProductManifold(self.signature, device=self.device, stereographic=True)

        # Convert points
        stereo_points = [
            torch.hstack([M.stereographic(x)[1] for x, M in zip(self.factorize(X), self.P)])  # type: ignore
            for X in points
        ]
        assert all([stereo_manifold.manifold.check_point(X) for X in stereo_points])

        return stereo_manifold, *stereo_points  # type: ignore

    @torch.no_grad()
    def gaussian_mixture(
        self,
        num_points: int = 1_000,
        num_classes: int = 2,
        num_clusters: Optional[int] = None,
        seed: Optional[int] = None,
        cov_scale_means: float = 1.0,
        cov_scale_points: float = 1.0,
        regression_noise_std: float = 0.1,
        task: Literal["classification", "regression"] = "classification",
    ) -> Tuple[Float[torch.Tensor, "n_points n_ambient_dim"], Float[torch.Tensor, "n_points"]]:
        """
        Generate a set of labeled samples from a Gaussian mixture model.

        Args:
            num_points: The number of points to generate.
            num_classes: The number of classes to generate.
            num_clusters: The number of clusters to generate. If None, defaults to num_classes.
            seed: An optional seed for the random number generator.
            cov_scale_means: The scale of the covariance matrix for the means.
            cov_scale_points: The scale of the covariance matrix for the points.
            regression_noise_std: The standard deviation of the noise for regression labels.
            task: The type of labels to generate. Either "classification" or "regression".

        Returns:
            samples: A tensor of shape (num_points, ambient_dim) containing the generated samples.
            class_assignments: A tensor of shape (num_points,) containing the class assignments of the samples.
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)

        # Deal with clusters
        if num_clusters is None:
            num_clusters = num_classes
        else:
            assert num_clusters >= num_classes

        # Generate cluster means
        cluster_means, _ = self.sample(
            z_mean=torch.stack([self.mu0] * num_clusters),
            sigma_factorized=[torch.stack([torch.eye(M.dim)] * num_clusters) * cov_scale_means for M in self.P],
        )
        assert cluster_means.shape == (num_clusters, self.ambient_dim)  # type: ignore

        # Generate class assignments
        cluster_probs = torch.rand(num_clusters)
        cluster_probs /= cluster_probs.sum()
        cluster_assignments = torch.multinomial(input=cluster_probs, num_samples=num_points, replacement=True)
        assert cluster_assignments.shape == (num_points,)

        # Generate covariance matrices for each class - Wishart distribution
        cov_matrices = [
            torch.distributions.Wishart(
                df=M.dim + 1, covariance_matrix=torch.eye(M.dim) * cov_scale_points  # type: ignore
            ).sample(
                sample_shape=(num_clusters,)  # type: ignore
            )
            + torch.eye(M.dim) * 1e-5  # jitter to avoid singularity
            for M in self.P
        ]

        # Generate random samples for each cluster
        sample_means = torch.stack([cluster_means[c] for c in cluster_assignments])
        assert sample_means.shape == (num_points, self.ambient_dim)
        sample_covs = [torch.stack([cov_matrix[c] for c in cluster_assignments]) for cov_matrix in cov_matrices]
        samples, tangent_vals = self.sample(z_mean=sample_means, sigma_factorized=sample_covs)
        assert samples.shape == (num_points, self.ambient_dim)

        # Map clusters to classes
        cluster_to_class = torch.cat(
            [
                torch.arange(num_classes, device=self.device),
                torch.randint(0, num_classes, (num_clusters - num_classes,), device=self.device),
            ]
        )
        assert cluster_to_class.shape == (num_clusters,)
        assert torch.unique(cluster_to_class).shape == (num_classes,)

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
            v = N.sample((num_points,)).to(self.device)  # type: ignore
            labels += v

            # Normalize regression labels to range [0, 1] so that RMSE can be more easily interpreted
            labels = (labels - labels.min()) / (labels.max() - labels.min())

        return samples, labels
