"""Variational autoencoder implementation for product manifold spaces.

This module provides a variational autoencoder (VAE) implementation specifically designed for learning representations
in mixed-curvature product spaces. The implementation handles the complexities of sampling, KL divergence calculation,
and reparameterization in curved spaces, supporting combinations of hyperbolic, Euclidean, and spherical geometries
within a single latent space.

For more information, see Skopek et al (2020): Mixed Curvature Variational Autoencoders
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from jaxtyping import Float

from ..manifolds import ProductManifold
from ._base import BaseEmbedder

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ProductSpaceVAE(BaseEmbedder, torch.nn.Module):
    r"""Product Space Variational Autoencoder.

    The probabilistic model is defined as:

    - Prior: $p(z) = \mathcal{WN}(z; \mu_0, I)$ (wrapped normal distribution centered at manifold origin)
    - Likelihood: $p_\theta(x|z) = \mathcal{N}(x; f_\theta(z), \sigma^2 I)$ or other reconstruction distribution
    - Posterior approximation: $q_\phi(z|x) = \mathcal{WN}(z; \mu_\phi(x), \Sigma_\phi(x))$

    where $\mathcal{WN}$ is a wrapped normal distribution on the manifold.

    The model is trained by maximizing the evidence lower bound (ELBO):

    $\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) || p(z))$

    Attributes:
        pm: Product manifold defining the structure of the latent space.
        random_state: Random state for reproducibility.
        encoder: Neural network that outputs mean and log-variance parameters.
        decoder: Neural network that reconstructs inputs from latent embeddings.
        beta: Weight for the KL divergence term in the ELBO.
        device: Device for tensor computations.
        n_samples: Number of samples for Monte Carlo estimation of KL divergence.
        reconstruction_loss: Type of reconstruction loss to use.
        loss_history_: Dictionary to store the history of loss values during training.
        is_fitted_: Boolean flag indicating whether the model has been fitted.


    Args:
        pm: Product manifold defining the structure of the latent space.
        encoder: Neural network module that produces mean (first half of output) and log-variance (second half of
            output) of the posterior distribution. The output dimension should match twice the intrinsic dimension of
            the product manifold.
        decoder: Neural network module that maps latent representations back to the input space.
        random_state: Optional random state for reproducibility.
        device: Optional device for tensor computations.
        beta: Weight of the KL divergence term in the ELBO loss. Values < 1 give a $\beta$-VAE with a looser constraint
            on the latent space.
        reconstruction_loss: Type of reconstruction loss to use. Currently only "mse" (mean squared error) is supported.
        n_samples: Number of Monte Carlo samples to use when estimating the KL divergence.
    """

    def __init__(
        self,
        pm: ProductManifold,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        random_state: int | None = None,
        device: str = "cpu",
        beta: float = 1.0,
        reconstruction_loss: torch.nn.modules.loss._Loss | None = None,
        n_samples: int = 16,
    ):
        # Init both base classes
        torch.nn.Module.__init__(self)
        BaseEmbedder.__init__(self, pm=pm, random_state=random_state, device=device)

        # Now we assign
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.beta = beta
        self.n_samples = n_samples
        self.reconstruction_loss = (
            reconstruction_loss if reconstruction_loss is not None else torch.nn.MSELoss(reduction="none")
        )
        self.model_ = None
        self.loss_history_ = {}
        self.is_fitted_ = False

        # Ensure encoder last dimension is 2 * pm.intrinsic_dim:
        assert encoder[-1].out_features == 2 * pm.dim, "Encoder output must match 2 * intrinsic dimension of manifold."

        # Ensure decoder input dimension is pm.intrinsic_dim
        assert decoder[0].in_features == pm.ambient_dim, "Decoder input must match ambient dimension of manifold."

    def encode(
        self, x: Float[torch.Tensor, "batch_size n_features"]
    ) -> tuple[
        Float[torch.Tensor, "batch_size n_latent"],
        Float[torch.Tensor, "batch_size n_latent"],
    ]:
        r"""Encodes input data to obtain latent means and log-variances in the manifold.

        This method processes input data through the encoder network to obtain parameters of the approximate posterior
        distribution $q(z|x)$ in the product manifold space. For non-Euclidean components, the method:

        1. Gets tangent space vectors and log-variances from the encoder,
        2. Projects tangent vectors to the ambient space by adding zeros in the right places, and
        3. Maps the ambient space vectors to the manifold using the exponential map

        Args:
            x: Input data tensor.

        Returns:
            z_mean_tangent: Mean of the posterior distribution in the tangent plane at the origin.
            z_logvar: Log-variance of the posterior distribution, used for constructing the covariance matrices.
        """
        z = self.encoder(x)
        z_mean_tangent, z_logvar = z[..., : self.pm.dim], z[..., self.pm.dim :]
        # z_mean_ambient = z_mean_tangent @ self.pm.projection_matrix  # Adds zeros in the right places
        # z_mean = self.pm.expmap(u=z_mean_ambient, base=None)
        return z_mean_tangent, z_logvar

    def decode(self, z: Float[torch.Tensor, "batch_size n_ambient"]) -> Float[torch.Tensor, "batch_size n_features"]:
        """Decodes latent points from the manifold space back to the input space.

        Takes points from the product manifold latent space and passes them through
        the decoder network to reconstruct the original input data.

        Args:
            z: Latent points in the product manifold

        Returns:
            reconstructed: Tensor containing the reconstructed input data,
                with shape (batch_size, n_features).
        """
        return self.decoder(z)

    def forward(
        self, x: Float[torch.Tensor, "batch_size n_features"]
    ) -> tuple[
        Float[torch.Tensor, "batch_size n_features"],
        Float[torch.Tensor, "batch_size n_ambient"],
        list[Float[torch.Tensor, "batch_size n_latent n_latent"]],
    ]:
        r"""Performs the forward pass of the VAE in product manifold space.

        This method implements the complete VAE forward pass, with manifold projection:

        1. Encode the input to get posterior parameters (`z_means`, `z_logvars`)
        2. Project means onto the manifold using exponential map
        3. Factorize the log-variances for each manifold component and convert to covariance matrices
        4. Sample points from the posterior distributions in the product manifold
        5. Decode the sampled points to get reconstructions

        Args:
            x: Input data tensor.

        Returns:
            x_reconstructed: Reconstructed data tensor with the same shape as the input.
            z_means: Means of the posterior distributions in the manifold space.
            sigmas: List of covariance matrices for each manifold component.
        """
        z_mean_tangent, z_logvars = self.encode(x)

        # Need to convert from implicit parameterization to extrinsic coordinates
        z_mean_ambient = z_mean_tangent @ self.pm.projection_matrix  # Adds zeros in the right places
        z_means = self.pm.expmap(u=z_mean_ambient, base=None)

        # Factorize log-variances; convert to covariances
        sigma_factorized = self.pm.factorize(z_logvars, intrinsic=True)
        sigmas = [torch.diag_embed(torch.exp(z_logvar) + 1e-8) for z_logvar in sigma_factorized]

        # Sample and decode
        z = self.pm.sample(z_mean=z_means, sigma_factorized=sigmas)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_means, sigmas

    def kl_divergence(
        self,
        z_mean: Float[torch.Tensor, "batch_size n_latent"],
        sigma_factorized: list[Float[torch.Tensor, "batch_size manifold_dim manifold_dim"]],
    ) -> Float[torch.Tensor, "batch_size"]:
        r"""Computes the KL divergence between posterior and prior distributions in the manifold.

        For distributions in Riemannian manifolds, computing the KL divergence analytically
        is often intractable. This method uses Monte Carlo sampling to approximate the KL divergence:

        $$D_{KL}(q(z|x) || p(z)) \approx \frac{1}{N} \sum_{i=1}^{N} [\log q(z_i|x) - \log p(z_i)]$$

        where $z_i$ are samples from $q(z|x)$.

        This implementation follows the approach described in:
        http://joschu.net/blog/kl-approx.html

        Args:
            z_mean: Means of the posterior distributions in the manifold.
            sigma_factorized: List of covariance matrices for each manifold component.

        Returns:
            kl_divergence: KL divergence values for each data point in the batch.
        """
        # Get KL divergence as the average of log q(z|x) - log p(z)
        means = torch.repeat_interleave(z_mean, self.n_samples, dim=0)
        sigmas_factorized_interleaved = [
            torch.repeat_interleave(sigma, self.n_samples, dim=0) for sigma in sigma_factorized
        ]
        # We want to use n_samples = 1 here, since we'll need to pass the interleaved means/sigmas to the log-likelihood
        z_samples = self.pm.sample(z_mean=means, sigma_factorized=sigmas_factorized_interleaved)
        log_qz = self.pm.log_likelihood(z_samples, means, sigmas_factorized_interleaved)
        log_pz = self.pm.log_likelihood(z_samples)
        return (log_qz - log_pz).view(-1, self.n_samples).mean(dim=1)

    def elbo(
        self, x: Float[torch.Tensor, "batch_size n_features"]
    ) -> tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        r"""Computes the Evidence Lower Bound (ELBO) for the VAE objective.

        The ELBO is the standard objective function for variational autoencoders, consisting of a reconstruction term
        (log-likelihood) and a regularization term (KL divergence):

        $$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) || p(z)),$$

        where:

        - $\theta$ are the decoder parameters
        - $\phi$ are the encoder parameters
        - $\beta$ is a weight for the KL term (setting $\beta < 1$ creates a $\beta$-VAE)

        Args:
            x: Input data tensor.

        Returns:
            elbo: Mean ELBO value across the batch (higher is better).
            log_likelihood: Mean reconstruction log-likelihood across the batch.
            kl_divergence: Mean KL divergence across the batch.
        """
        x_reconstructed, z_means, sigma_factorized = self(x)
        kld = self.kl_divergence(z_means, sigma_factorized)
        ll = -self.reconstruction_loss(x_reconstructed.view(x.shape[0], -1), x.view(x.shape[0], -1)).sum(dim=1)
        return (ll - self.beta * kld).mean(), ll.mean(), kld.mean()

    def _grads_ok(self) -> bool:
        """Checks if all gradients are valid (no NaN or Inf values).

        This is a helper method used during training to ensure numerical stability. It checks each parameter's gradient
        for NaN or Inf values and reports any issues.

        Returns:
            valid: True if all gradients are valid, False otherwise.
        """
        out = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    out = False
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient in {name}")
                    out = False
        return out

    def fit(  # type: ignore[override]
        self,
        X: Float[torch.Tensor, "n_points n_features"],
        D: None = None,
        lr: float = 1e-3,
        burn_in_lr: float = 1e-4,
        curvature_lr: float = 0.0,  # Off by default
        burn_in_iterations: int = 1,
        training_iterations: int = 9,
        loss_window_size: int = 100,
        logging_interval: int = 10,
        batch_size: int = 32,
        clip_grad: bool = True,
    ) -> "ProductSpaceVAE":
        """Trains the VAE model on the provided data.

        The training process consists of two phases:

        1. Burn-in phase: Initial training with a lower learning rate for stability
        2. Main training phase: Training with the full learning rate and optional curvature optimization

        Training uses Adam optimizer with gradient clipping to prevent exploding gradients. During training, the model
        maximizes the Evidence Lower Bound (ELBO).

        Args:
            X: Training data tensor.
            D: Ignored.
            lr: Learning rate for the main training phase.
            burn_in_lr: Learning rate for the burn-in phase.
            curvature_lr: Learning rate for optimizing manifold scale factors. Off (no learning) by default.
            burn_in_iterations: Number of iterations for the burn-in phase.
            training_iterations: Number of iterations for the main training phase.
            loss_window_size: Window size for computing moving average loss.
            logging_interval: Interval for logging training progress.
            batch_size: Batch size for training.
            clip_grad: Whether to apply gradient clipping.

        Returns:
            losses: List of loss values recorded during training.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        my_tqdm = tqdm(total=(burn_in_iterations + training_iterations) * len(X))
        opt = torch.optim.Adam(
            [
                {
                    "params": [p for p in self.parameters() if p not in set(self.pm.parameters())],
                    "lr": burn_in_lr,
                },
                {"params": self.pm.parameters(), "lr": 0},
            ]
        )
        losses: dict[str, list[float]] = {"elbo": [], "ll": [], "kl": []}
        for epoch in range(burn_in_iterations + training_iterations):
            if epoch == burn_in_iterations:
                opt.param_groups[0]["lr"] = lr
                opt.param_groups[1]["lr"] = curvature_lr

            for i in range(0, len(X), batch_size):
                opt.zero_grad()
                X_batch = X[i : i + batch_size]
                elbo, ll, kl = self.elbo(X_batch)
                L = -elbo
                L.backward()

                # Add to losses
                losses["elbo"].append(elbo.item())
                losses["ll"].append(ll.item())
                losses["kl"].append(kl.item())

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.pm.parameters(), max_norm=1.0)
                if torch.isnan(L) or torch.isinf(L):
                    print(f"Invalid loss detected at epoch {epoch}, batch {i}")
                    continue
                elif self._grads_ok():
                    opt.step()

                # TQDM management
                my_tqdm.update(batch_size)
                my_tqdm.set_description(f"L: {L.item():.3e}, ll: {ll.item():.3e}, kl: {kl.item():.3e}")

                # Logging
                if i % logging_interval == 0:
                    d = {f"r{i}": f"{logscale.item():.3f}" for i, logscale in enumerate(self.pm.parameters())}
                    # d["D_avg"] = f"{d_avg(D_tt, D[train][:, train], pairwise=True):.4f}"
                    d["L_avg"] = f"{np.mean(losses['elbo'][-loss_window_size:]):.3e}"
                    d["ll_avg"] = f"{np.mean(losses['ll'][-loss_window_size:]):.3e}"
                    d["kl_avg"] = f"{np.mean(losses['kl'][-loss_window_size:]):.3e}"
                    my_tqdm.set_postfix(d)

        # Final maintenance: update attributes
        self.loss_history_ = losses
        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: Float[torch.Tensor, "n_points n_features"],
        D: None = None,
        batch_size: int = 32,
        expmap: bool = True,
    ) -> Float[torch.Tensor, "n_points embedding_dim"]:
        """Transform data using the trained VAE. Outputs means of the variational distribution.

        Args:
            X: Features to embed with VAE.
            D: Ignored.
            batch_size: Number of samples per batch.
            expmap: Whether to use exponential map for embedding.

        Returns:
            embeddings: Learned embeddings.
        """
        # Set random state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Save the test embeddings
        embeddings_list = []
        for i in range(0, len(X), batch_size):
            x_batch = X[i : i + batch_size]
            z_mean_tangent, _ = self.encode(x_batch)
            if expmap:
                z_mean_ambient = z_mean_tangent @ self.pm.projection_matrix  # Adds zeros in the right places
                z_mean = self.pm.expmap(u=z_mean_ambient, base=None)
            else:
                z_mean = z_mean_tangent
            embeddings_list.append(z_mean.detach().cpu())

        embeddings = torch.cat(embeddings_list, dim=0)

        return embeddings
