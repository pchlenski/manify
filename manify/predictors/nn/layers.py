"""Neural network layers for product manifolds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geoopt
import torch
from torch import nn

if TYPE_CHECKING:
    from beartype.typing import Callable
    from jaxtyping import Float

from ...manifolds import Manifold, ProductManifold


class KappaGCNLayer(torch.nn.Module):
    """Implementation for the Kappa GCN layer.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        manifold: Manifold object for the Kappa GCN
        nonlinearity: Function for nonlinear activation.

    Attributes:
        W: Weight matrix parameter.
        sigma: Nonlinear activation function applied via the manifold.
        manifold: The manifold object for geometric operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        nonlinearity: Callable | None = torch.relu,
    ):
        super().__init__()

        # Parameters are Euclidean, straightforwardly
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.01)

        # Nonlinearity must be applied via the manifold
        self.sigma = manifold.apply(nonlinearity) if nonlinearity else lambda x: x

        # Also store manifold
        self.manifold = manifold

    def _left_multiply(
        self,
        A: Float[torch.Tensor, "n_nodes n_nodes"],
        X: Float[torch.Tensor, "n_nodes dim"],
        M: Manifold,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        r"""$\kappa$-left matrix multiply two matrices $\mathbf{A}$ and $\mathbf{X}$.

        $$\mathbf{A} \boxtimes_\kappa \mathbf{X}$$

        Args:
            A: Adjacency matrix of the graph
            X: Embedding matrix of the graph.
            M: Manifold object for the Kappa GCN - need to specify in case we're going by component

        Returns:
            out: result of the Kappa left matrix multiplication.
        """
        # Vectorized version:
        return M.manifold.weighted_midpoint(
            xs=X.unsqueeze(0),  # (1, N, D)
            weights=A,  # (N, N)
            reducedim=[1],  # Sum over the N points dimension (dim 1)
            dim=-1,  # Compute conformal factors along the points dimension
            keepdim=False,  # Squeeze the batch dimension out
            lincomb=True,  # Scale by sum of weights (A.sum(dim=1))
            posweight=False,
        )

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass for the Kappa GCN layer.

        Args:
            X: Embedding matrix
            A_hat: Normalized adjacency matrix

        Returns:
            AXW: Transformed node features after message passing and nonlinear activation.
        """
        # 1. right-multiply X by W - mobius_matvec broadcasts correctly (verified)
        XW = self.manifold.manifold.mobius_matvec(m=self.W, x=X)

        # 2. left-multiply (X @ W) by A_hat - we need our own implementation for this
        if A_hat is None:
            AXW = XW
        elif isinstance(self.manifold, ProductManifold):
            XWs = self.manifold.factorize(XW)
            AXW = torch.hstack(
                [
                    self._left_multiply(A_hat, XW, M)
                    for XW, M in zip(XWs, self.manifold.P, strict=False)
                ]
            )
        else:
            AXW = self._left_multiply(A_hat, XW, self.manifold)

        # 3. Apply nonlinearity - note that sigma is wrapped with our manifold.apply decorator
        AXW = self.sigma(AXW)

        return AXW


class KappaSequential(nn.Module):
    """Sequential container for κ-layers that properly handles adjacency matrices.

    Similar to nn.Sequential but passes the adjacency matrix through each layer.
    All layers should accept (X, A_hat) and return X.

    Args:
        *layers: Variable number of layers to be added to the sequence.
        Each layer should be a subclass of nn.Module that implements a forward method accepting (X, A_hat).
    """

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes out_dim"]:
        """Forward pass through all layers.

        Args:
            X: Input features
            A_hat: Adjacency matrix passed to each layer

        Returns:
            Output after passing through all layers
        """
        for layer in self.layers:
            X = layer(X, A_hat)
        return X

    def append(self, layer: nn.Module) -> None:
        """Add a layer to the end of the sequence."""
        self.layers.append(layer)

    def __len__(self) -> int:
        """Return the number of layers in the sequence.

        Returns:
            int: Number of layers in the KappaSequential.
        """
        return len(self.layers)

    def __getitem__(self, idx: int) -> nn.Module:
        """Get a layer by index.

        Args:
            idx: Index of the layer to retrieve

        Returns:
            nn.Module: The layer at the specified index.
        """
        return self.layers[idx]


class StereographicLogits(nn.Module):
    """Stereographic logits layer for classification and regression on product manifolds.

    Computes signed distances from hyperplanes in the product manifold space.
    Can optionally apply softmax for classification tasks.

    Args:
        out_features: Number of output classes (dimensionality of output space)
        manifold: Manifold or ProductManifold object defining the geometry
        apply_softmax: Whether to apply softmax to the output logits (default: False)
    """

    def __init__(
        self,
        out_features: int,
        manifold: Manifold | ProductManifold,
        apply_softmax: bool = False,
    ):
        super().__init__()

        self.out_features = out_features
        self.manifold = manifold
        self.apply_softmax = apply_softmax

        # Weight matrix (Euclidean parameters)
        self.W = nn.Parameter(torch.randn(manifold.dim, out_features) * 0.01)

        # Bias points on the manifold
        self.p_ks = geoopt.ManifoldParameter(
            torch.zeros(out_features, manifold.dim), manifold=manifold.manifold
        )

    def _get_logits_single_manifold(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        W: Float[torch.Tensor, "dim n_classes"],
        b: Float[torch.Tensor, "n_classes dim"],
        M: Manifold,
        return_inner_products: bool = False,
    ) -> (
        tuple[
            Float[torch.Tensor, "n_nodes n_classes"],
            Float[torch.Tensor, "n_nodes n_classes"],
        ]
        | Float[torch.Tensor, "n_nodes n_classes"]
    ):
        """Compute logits for a single manifold."""
        kappa = torch.tensor(M.curvature, dtype=X.dtype, device=X.device)

        # Change shapes
        b = b[None, :]  # (1, k)
        X = X[:, None]  # (n, 1, d)

        # 1. Get z_k = -p_k ⊕_κ x (vectorized)
        z_ks = M.manifold.mobius_add(-b, X)  # (n, k, d)

        # 2. Get norms for relevant terms
        z_k_norms = torch.norm(z_ks, dim=-1).clamp_min(1e-10)  # (n, k)
        a_k_norms = torch.norm(W, dim=0).clamp_min(1e-10)  # (k,)

        # 3. Get the distance to the hyperplane
        za = torch.einsum("nkd,dk->nk", z_ks, W)  # (n, k)

        # 4. Get the logits
        if abs(kappa) < 1e-4:
            # Euclidean case: it's just a dot product
            logits = 4 * za
        else:
            # Non-Euclidean case: need to do the arsinh
            dist = 2 * za / ((1 + kappa * z_k_norms**2) * a_k_norms)
            # arsin_k takes the curvature kappa directly: it scales the argument by
            # sqrt(|kappa|) internally and multiplies the result by 1/sqrt(|kappa|).
            dist = geoopt.manifolds.stereographic.math.arsin_k(dist, kappa)

            # Get the coefficients
            lambda_pks = M.manifold.lambda_x(b)  # (k,)
            coeffs = lambda_pks * a_k_norms
            logits = coeffs * dist

        if return_inner_products:
            return logits, za
        else:
            return logits

    def _get_logits_product_manifold(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        W: Float[torch.Tensor, "dims n_classes"],
        b: Float[torch.Tensor, "n_classes dim"],
        M: ProductManifold,
    ) -> Float[torch.Tensor, "n_nodes n_classes"]:
        """Helper function for get_logits."""
        # For convenience, get curvature and manifold
        # kappas = [man.curvature for manifold in M.P]
        Xs = M.factorize(X)
        bs = M.factorize(b)
        Ws = [w.T for w in M.factorize(W.T)]
        res = [
            self._get_logits_single_manifold(
                X_man, W_man, b_man, man, return_inner_products=True
            )
            for X_man, W_man, b_man, man in zip(Xs, Ws, bs, M.P, strict=False)
        ]

        # Each result is (n, k) logits and (n, k) inner products
        logits, inner_products = zip(*res, strict=False)

        # Final logits: l2 norm of logits * sign of inner product
        stacked_logits = torch.stack(logits, dim=2)  # (n, k, m)
        stack_products = torch.stack(inner_products, dim=2)  # (n, k, m)

        # Reduce
        logits = torch.norm(stacked_logits, dim=2)  # (n, k)
        signs = torch.sign(stack_products.sum(dim=2))  # (n, k)

        return logits * signs

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
        aggregate_logits: bool = False,
    ) -> Float[torch.Tensor, "n_nodes n_classes"]:
        """Forward pass through stereographic logits.

        Args:
            X: Input features
            A_hat: Optional adjacency matrix for logit aggregation
            aggregate_logits: Whether to aggregate logits using adjacency matrix

        Returns:
            Logits (or probabilities if apply_softmax=True)
        """
        # Compute logits based on manifold type
        if isinstance(self.manifold, ProductManifold):
            logits = self._get_logits_product_manifold(
                X, self.W, self.p_ks, self.manifold
            )
        else:
            logits = self._get_logits_single_manifold(
                X, self.W, self.p_ks, self.manifold, return_inner_products=False
            )

        # Optional aggregation via adjacency matrix
        if A_hat is not None and aggregate_logits:
            logits = A_hat @ logits

        # Optional softmax for classification
        if self.apply_softmax:
            logits = torch.softmax(logits, dim=-1)

        return logits


class FermiDiracDecoder(nn.Module):
    """Fermi-Dirac decoder for link prediction tasks.

    Computes pairwise distances and applies Fermi-Dirac transformation
    to predict edge probabilities.

    Args:
        manifold: Manifold or ProductManifold object defining the geometry
        learnable_params: If True, temperature and bias are learnable parameters. If False, they are fixed to 1.0 and
            0.0, respectively.
    """

    def __init__(
        self, manifold: Manifold | ProductManifold, learnable_params: bool = True
    ):
        super().__init__()

        self.manifold = manifold

        if learnable_params:
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("temperature", torch.tensor(1.0))
            self.register_buffer("bias", torch.tensor(0.0))

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes n_nodes"]:
        """Forward pass through Fermi-Dirac decoder.

        Args:
            X: Node embeddings
            A_hat: Ignored (for compatibility)
            return_pairwise: If True, return full pairwise matrix. If False, return flattened upper triangle.

        Returns:
            Edge probabilities (logits, apply sigmoid if needed)
        """
        # Compute pairwise distances
        pairwise_dist = self.manifold.pdist2(X)

        # Apply Fermi-Dirac transformation
        logits = -(pairwise_dist - self.bias) / self.temperature

        return logits


class StereographicLayerNorm(nn.Module):
    """Stereographic Layer Normalization.

    Args:
        manifold: Manifold or ProductManifold object defining the geometry.
        embedding_dim: Embedding dimension of the input points.
        curvatures: Tensor of shape [num_heads, 1, 1] representing the curvature
                    value used per head in geometric computations.

    Attributes:
        manifold: The manifold object for geometric operations.
        stereographic_norm: Stereographic layernorm used in the tangent space.
        curvatures: Tensor of shape [num_heads, 1, 1] representing the curvature
                    value used per head in geometric computations.
    """

    def __init__(
        self,
        manifold: Manifold | ProductManifold,
        embedding_dim: int,
        curvatures: Float[torch.Tensor, "num_heads 1 1"],
    ):
        super().__init__()

        self.manifold = manifold
        self.stereographic_norm = self.manifold.apply(nn.LayerNorm(embedding_dim))
        self.curvatures = curvatures

    def forward(
        self, X: Float[torch.Tensor, "n_nodes dim"]
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Apply layer normalization on the stereographic manifold."""
        norm_X = self.stereographic_norm(X)
        output = geoopt.manifolds.stereographic.math.project(norm_X, self.curvatures)
        return output


class GeometricLinearizedAttention(nn.Module):
    """Geometric Linearized Attention.

    Args:
        curvatures: Tensor of shape [num_heads, 1, 1] representing the curvature
                    value used per head in geometric computations.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        epsilon: Small epsilon for masking inverse denominator (constant).
        clamp_epsilon: Minimum clamp value for numerical stability in gamma denominator (constant).
    """

    def __init__(self, curvatures: float | list[float], num_heads: int, head_dim: int):
        super().__init__()

        self.num_heads = num_heads
        self.curvatures = curvatures

        self.head_dim = head_dim
        self._epsilon = 1e-5
        self._clamp_epsilon = 1e-10

    def forward(
        self,
        Q: Float[torch.Tensor, "batch_size num_heads n_nodes head_dim"],
        K: Float[torch.Tensor, "batch_size num_heads n_nodes head_dim"],
        V: Float[torch.Tensor, "batch_size num_heads n_nodes head_dim"],
        mask: Float[torch.Tensor, "1 1 n_nodes n_nodes"],
    ) -> Float[torch.Tensor, "batch_size n_nodes dim"]:
        """Forward pass for the geometric linearized attention layer.

        Args:
            Q: Query tensor.
            K: Key tensor.
            V: Value tensor.
            mask: Mask tensor for attention.

        Returns:
            Output tensor after applying attention.
        """
        v1 = geoopt.manifolds.stereographic.math.parallel_transport0back(
            V, Q, k=self.curvatures
        )
        v2 = geoopt.manifolds.stereographic.math.parallel_transport0back(
            V, K, k=self.curvatures
        )

        gamma = geoopt.manifolds.stereographic.math.lambda_x(
            x=V, k=self.curvatures, keepdim=True, dim=-1
        )
        denominator = geoopt.utils.clamp_abs((gamma - 1), self._clamp_epsilon)

        x = ((gamma / denominator) * V) * mask[None, :, None]

        v1 = nn.functional.elu(v1) + 1
        v2 = (denominator * (nn.functional.elu(v2) + 1)) * mask[None, :, None]

        # Linearized approximation
        v2_cumsum = v2.sum(dim=-2)  # [B, H, D]
        D = torch.einsum(
            "...nd,...d->...n", v1, v2_cumsum.type_as(v1)
        )  # normalization terms
        D_inv = 1.0 / D.masked_fill_(D == 0, self._epsilon)
        context = torch.einsum("...nd,...ne->...de", v2, x)
        X = torch.einsum("...de,...nd,...n->...ne", context, v1, D_inv)

        X = geoopt.manifolds.stereographic.math.project(X, k=self.curvatures)
        X = geoopt.manifolds.stereographic.math.mobius_scalar_mul(
            torch.tensor(0.5, dtype=X.dtype, device=X.device),
            X,
            k=self.curvatures,
            dim=-1,
        )
        X = geoopt.manifolds.stereographic.math.project(X, k=self.curvatures)

        return X


class StereographicAttention(nn.Module):
    """Stereographic Attention Layer.

    Args:
        manifold: Manifold or ProductManifold object defining the geometry.
        num_heads: Number of attention heads.
        dim: Embedding dimension of the input points.
        head_dim: Dimension of each attention head.

    Attributes:
        manifold: The manifold object for geometric operations.
        curvatures: Tensor of shape [num_heads, 1, 1] representing the curvature
                    value used per head in geometric computations.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        W_q: Linear layer projecting inputs to query vectors.
        W_k: Linear layer projecting inputs to key vectors.
        W_v: Manifold-aware linear layer projecting to value vectors.
        attn: Stereographic multi-head attention module.
        ff: Manifold-aware linear layer for the feedforward output.
    """

    def __init__(
        self,
        manifold: Manifold | ProductManifold,
        num_heads: int,
        dim: int,
        head_dim: int,
    ):
        super().__init__()

        self.manifold = manifold
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.curvatures = _reshape_curvatures(
            _get_curvatures(self.manifold), self.num_heads
        )

        self.W_q = nn.Linear(
            in_features=dim, out_features=self.num_heads * self.head_dim
        )
        self.W_k = nn.Linear(
            in_features=dim, out_features=self.num_heads * self.head_dim
        )
        self.W_v = KappaGCNLayer(
            in_features=dim,
            out_features=self.num_heads * self.head_dim,
            manifold=self.manifold,
        )

        self.attn = GeometricLinearizedAttention(
            curvatures=self.curvatures, num_heads=self.num_heads, head_dim=self.head_dim
        )
        self.ff = KappaGCNLayer(
            in_features=self.num_heads * self.head_dim,
            out_features=dim,
            manifold=self.manifold,
        )

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass for the stereographic attention layer."""
        Q = self._split_heads(self.W_q(X))  # [B, H, N, D]
        K = self._split_heads(self.W_k(X))
        V = self._split_heads(self.W_v(X=X))

        attn_out = self.attn(Q, K, V, mask.unsqueeze(0).unsqueeze(0))  # type: ignore
        attn_out = self._combine_heads(attn_out)

        out = self.ff(X=attn_out)

        return out

    def _combine_heads(
        self, X: Float[torch.Tensor, "n_nodes num_heads head_dim"]
    ) -> Float[torch.Tensor, "n_nodes num_heads * head_dim"]:
        """Combines multi-head tensor by merging head and feature dimensions.

        Args:
            X: Input tensor with shape.

        Returns:
            X: Reshaped tensor with shape (n_nodes, num_heads * head_dim).
        """
        X = X.transpose(0, 1)
        X = X.reshape(X.size(0), self.num_heads * self.head_dim)
        return X

    def _split_heads(
        self, X: Float[torch.Tensor, "n_nodes num_heads * head_dim"]
    ) -> Float[torch.Tensor, "num_heads n_nodes head_dim"]:
        """Splits the last dimension of the input into (num_heads, head_dim) and transposes to prepare for attention.

        Args:
            X: Input tensor with shape (n_nodes, num_heads * head_dim).

        Returns:
            X: Reshaped tensor with shape (num_heads, n_nodes, head_dim).
        """
        X = X.reshape(X.size(0), self.num_heads, self.head_dim)
        X = X.transpose(0, 1)
        return X


class StereographicTransformer(nn.Module):
    """Stereographic Transformer Block.

    Args:
        manifold: Manifold or ProductManifold object defining the geometry.
        num_heads: Number of attention heads.
        dim: Dimensionality of the input features.
        head_dim: Dimensionality of each attention head.
        use_layer_norm: Whether to apply layer normalization in tangent space.

    Attributes:
        manifold: The manifold object for geometric operations.
        curvatures: Manifold curvatures reshaped to [num_heads, 1, 1] for broadcasting.
        mha: Multi-head stereographic attention module.
        norm1: First normalization layer (can be Identity or StereographicLayerNorm).
        norm2: Second normalization layer.
        mlpblock: Feedforward network in stereographic space.
        stereographic_activation: Activation wrapped to operate in tangent space.
    """

    def __init__(
        self,
        manifold: Manifold | ProductManifold,
        num_heads: int,
        dim: int,
        head_dim: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        # Check that manifold is stereographic
        if not manifold.is_stereographic:
            raise ValueError(
                "Manifold must be stereographic for StereographicLayerNorm to work. Please use manifold.stereographic() to convert."
            )

        self.manifold = manifold
        self.curvatures = _reshape_curvatures(_get_curvatures(self.manifold), num_heads)
        self.stereographic_activation = self.manifold.apply(nn.ReLU())
        self.mha = StereographicAttention(
            manifold=self.manifold, num_heads=num_heads, dim=dim, head_dim=head_dim
        )

        if use_layer_norm:
            self.norm1 = StereographicLayerNorm(
                manifold=self.manifold, embedding_dim=dim, curvatures=self.curvatures
            )
            self.norm2 = StereographicLayerNorm(
                manifold=self.manifold, embedding_dim=dim, curvatures=self.curvatures
            )
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.mlpblock = nn.Sequential(
            KappaGCNLayer(in_features=dim, out_features=dim, manifold=self.manifold),
            self.stereographic_activation,
            KappaGCNLayer(in_features=dim, out_features=dim, manifold=self.manifold),
        )

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass through the stereographic transformer block."""
        X = geoopt.manifolds.stereographic.math.mobius_add(
            self.mha(self.norm1(X), mask), X, self.curvatures
        )
        X = geoopt.manifolds.stereographic.math.project(X, self.curvatures)
        X = geoopt.manifolds.stereographic.math.mobius_add(
            self.mlpblock(self.norm2(X)), X, self.curvatures
        )
        X = geoopt.manifolds.stereographic.math.project(X, self.curvatures)

        return X


def _reshape_curvatures(
    curvatures: float | list[float], num_heads: int
) -> Float[torch.Tensor, "num_heads 1 1"]:
    """Helper function to reshape curvature(s) for use in multi-head stereographic attention."""
    if isinstance(curvatures, float):
        output_curvatures = torch.tensor([curvatures] * num_heads, dtype=torch.float)
    else:
        output_curvatures = torch.tensor(curvatures, dtype=torch.float)
    return output_curvatures[:, None, None]


def _get_curvatures(manifold: Manifold | ProductManifold) -> float | list[float]:
    """Helper function to retrieve curvature(s) from a Manifold or ProductManifold."""
    if isinstance(manifold, ProductManifold):
        return manifold.curvatures
    elif isinstance(manifold, Manifold):
        return manifold.curvature
    else:
        raise TypeError("Expected a Manifold or ProductManifold class.")
