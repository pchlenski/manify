"""Siamese network implementation for manifold embedding.

This module provides a Siamese network architecture that can be used for embedding data into product manifolds. Siamese
networks are particularly useful for metric learning tasks, where the goal is to learn a distance-preserving embedding,
while also encoding a set of features.

The SiameseNetwork class supports both encoding (embedding) data into a manifold space and optionally decoding
(reconstructing) from the embedding space back to the original data space.
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
from ._losses import distortion_loss

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class SiameseNetwork(BaseEmbedder, torch.nn.Module):
    """Siamese network for embedding data into a product manifold space.

    A Siamese network consists of an encoder network that maps input data to a latent representation in a product
    manifold, and optionally a decoder network that maps the latent representation back to the original feature space.

    Attributes:
        pm: Product manifold defining the structure of the latent space.
        random_state: Random state for reproducibility.
        encoder: Neural network that maps inputs to latent embeddings.
        decoder: Neural network that reconstructs inputs from latent embeddings.
        beta: Weight for the distortion term in the loss function.
        device: Device for tensor computations.
        reconstruction_loss: Type of reconstruction loss to use.


    Args:
        pm: Product manifold defining the structure of the latent space.
        encoder: Neural network module that maps inputs to the manifold's intrinsic dimension.
            The output dimension should match the intrinsic dimension of the product manifold.
        decoder: Neural network module that maps latent representations back to the input space.
        random_state: Optional random state for reproducibility.
        device: Optional device for tensor computations.
        beta: Weight of the distortion term in the loss function.
        reconstruction_loss: Type of reconstruction loss to use.
    """

    def __init__(
        self,
        pm: ProductManifold,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module | None = None,
        reconstruction_loss: str = "mse",
        beta: float = 1.0,
        random_state: int | None = None,
        device: str = "cpu",
    ):
        # Init both base classes
        torch.nn.Module.__init__(self)
        BaseEmbedder.__init__(self, pm=pm, random_state=random_state, device=device)

        # Now we assign
        self.pm = pm
        self.encoder = encoder
        self.beta = beta

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = torch.nn.Identity()
            self.decoder.requires_grad_(False)
            self.decoder.to(pm.device)

        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    def encode(
        self, x: Float[torch.Tensor, "batch_size n_features"]
    ) -> Float[torch.Tensor, "batch_size n_latent"]:
        """Encodes input data into the manifold embedding space.

        Takes a batch of input data and passes it through the encoder network to obtain embeddings in the manifold.

        Args:
            x: Input data tensor..

        Returns:
            embeddings: Tensor containing the embeddings in the manifold space.
        """
        return self.encoder(x)

    def decode(
        self, z: Float[torch.Tensor, "batch_size n_latent"]
    ) -> Float[torch.Tensor, "batch_size n_features"]:
        """Decodes manifold embeddings back to the original input space.

        Takes a batch of embeddings from the manifold space and passes them through
        the decoder network to reconstruct the original input data.

        Args:
            z: Embedding tensor from the manifold space.

        Returns:
            reconstructed: Tensor containing the reconstructed input data.
        """
        return self.decoder(z)

    def forward(
        self,
        x1: Float[torch.Tensor, "batch_size n_features"],
        x2: Float[torch.Tensor, "batch_size n_features"],
    ) -> tuple[
        Float[torch.Tensor, "batch_size n_latent"],
        Float[torch.Tensor, "batch_size n_latent"],
        Float[torch.Tensor, "batch_size"],
        Float[torch.Tensor, "batch_size n_features"],
        Float[torch.Tensor, "batch_size n_features"],
    ]:
        """Given two points, return their encodings, reconstructions, and embedding distance.

        Args:
            x1: First input tensor.
            x2: Second input tensor.

        Returns:
            z1: Encoded representation of the first input.
            z2: Encoded representation of the second input.
            D_hat: Estimated distance between the two embeddings.
            reconstructed1: Reconstructed input from the first embedding.
            reconstructed2: Reconstructed input from the second embedding.
        """
        z1 = self.pm.expmap(self.encode(x1) @ self.pm.projection_matrix)
        z2 = self.pm.expmap(self.encode(x2) @ self.pm.projection_matrix)
        D_hat = self.pm.manifold.dist(
            z1, z2
        )  # use manifold dist to get (batch_size, ) vector of dists
        reconstructed1 = self.decode(z1)
        reconstructed2 = self.decode(z2)
        return z1, z2, D_hat, reconstructed1, reconstructed2

    def fit(  # type: ignore[override]
        self,
        X: Float[torch.Tensor, "n_points n_features"],
        D: Float[torch.Tensor, "n_points n_points"],
        lr: float = 1e-3,
        burn_in_lr: float = 1e-4,
        curvature_lr: float = 0.0,  # Off by default
        burn_in_iterations: int = 1,
        training_iterations: int = 9,
        loss_window_size: int = 100,
        logging_interval: int = 10,
        batch_size: int = 32,
        clip_grad: bool = True,
    ) -> "SiameseNetwork":
        """Fit the SiameseNetwork embedder.

        Args:
            X: Input data features to encode.
            D: Pairwise distances to emulate.
            lr: Learning rate for the optimizer.
            burn_in_lr: Learning rate during burn-in phase.
            curvature_lr: Learning rate for curvature updates.
            burn_in_iterations: Number of iterations for burn-in phase.
            training_iterations: Number of iterations for training phase.
            loss_window_size: Size of the window for loss averaging.
            logging_interval: Interval for logging progress.
            batch_size: Number of samples per batch.
            clip_grad: Whether to clip gradients.

        Returns:
            self: Fitted SiameseNetwork instance.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        n_samples = len(X)

        # Generate all upper triangular pairs using torch
        indices = torch.triu_indices(n_samples, n_samples, offset=1)
        pairs = torch.hstack([indices]).T  # (n_pairs, 2)

        # Number of pairs and batches
        n_pairs = len(pairs)
        n_batches_per_epoch = (
            n_pairs + batch_size - 1
        ) // batch_size  # Ceiling division
        total_iterations = (
            burn_in_iterations + training_iterations
        ) * n_batches_per_epoch

        my_tqdm = tqdm(total=total_iterations)

        opt = torch.optim.Adam(
            [
                {
                    "params": [
                        p
                        for p in self.parameters()
                        if p not in set(self.pm.parameters())
                    ],
                    "lr": burn_in_lr,
                },
                {"params": self.pm.parameters(), "lr": 0},
            ]
        )
        losses: dict[str, list[float]] = {
            "total": [],
            "reconstruction": [],
            "distortion": [],
        }

        for epoch in range(burn_in_iterations + training_iterations):
            if epoch == burn_in_iterations:
                opt.param_groups[0]["lr"] = lr
                opt.param_groups[1]["lr"] = curvature_lr

            # Shuffle all pairs
            shuffle_idx = torch.randperm(n_pairs)
            shuffled_pairs = pairs[shuffle_idx]

            for batch_start in range(0, n_pairs, batch_size):
                batch_end = min(batch_start + batch_size, n_pairs)
                batch_pairs = shuffled_pairs[batch_start:batch_end]

                # Extract indices for this batch
                batch_indices1 = batch_pairs[:, 0]
                batch_indices2 = batch_pairs[:, 1]

                # Get data for these indices
                X1 = X[batch_indices1]
                X2 = X[batch_indices2]

                # Extract the corresponding distances from D using advanced indexing
                D_batch = D[batch_indices1, batch_indices2]

                # Forward pass
                opt.zero_grad()
                _, _, D_hat, Y1, Y2 = self(X1, X2)
                mse1 = torch.nn.functional.mse_loss(Y1, X1)
                mse2 = torch.nn.functional.mse_loss(Y2, X2)

                # D_hat and D_batch are now 1D tensors of pairwise distances
                distortion = distortion_loss(D_hat, D_batch, pairwise=False)
                L = mse1 + mse2 + self.beta * distortion
                L.backward()

                # Add to losses
                losses["total"].append(L.item())
                losses["reconstruction"].append(mse1.item() + mse2.item())
                losses["distortion"].append(distortion.item())

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.pm.parameters(), max_norm=1.0)

                opt.step()

                # TQDM management
                my_tqdm.update(1)
                my_tqdm.set_description(
                    f"L: {L.item():.3e}, recon: {mse1.item() + mse2.item():.3e}, dist: {distortion.item():.3e}"
                )

                # Logging
                if my_tqdm.n % logging_interval == 0:
                    d = {
                        f"r{i}": f"{logscale.item():.3f}"
                        for i, logscale in enumerate(self.pm.parameters())
                    }
                    d["L_avg"] = f"{np.mean(losses['total'][-loss_window_size:]):.3e}"
                    d["recon_avg"] = (
                        f"{np.mean(losses['reconstruction'][-loss_window_size:]):.3e}"
                    )
                    d["dist_avg"] = (
                        f"{np.mean(losses['distortion'][-loss_window_size:]):.3e}"
                    )
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
    ) -> Float[torch.Tensor, "n_points n_latent"]:
        """Transforms input data into manifold embeddings.

        Args:
            X: Features to embed with SiameseNetwork.
            D: Ignored.
            batch_size: Number of samples per batch.
            expmap: Whether to use exponential map for embedding.

        Returns:
            embeddings: Embeddings produced by forward pass of trained SiameseNetwork model.
        """
        # Set random state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Save the  embeddings
        embeddings_list = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            embeddings = self.encode(batch)
            if expmap:
                embeddings = self.pm.expmap(embeddings @ self.pm.projection_matrix)
            embeddings_list.append(embeddings)
        embeddings = torch.cat(embeddings_list, dim=0)

        return embeddings
