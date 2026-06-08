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
            AXW = torch.hstack([self._left_multiply(A_hat, XW, M) for XW, M in zip(XWs, self.manifold.P, strict=False)])
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
        self.p_ks = geoopt.ManifoldParameter(torch.zeros(out_features, manifold.dim), manifold=manifold.manifold)

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
            self._get_logits_single_manifold(X_man, W_man, b_man, man, return_inner_products=True)
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
            logits = self._get_logits_product_manifold(X, self.W, self.p_ks, self.manifold)
        else:
            logits = self._get_logits_single_manifold(X, self.W, self.p_ks, self.manifold, return_inner_products=False)

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

    def __init__(self, manifold: Manifold | ProductManifold, learnable_params: bool = True):
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

        Returns:
            Edge probabilities (logits, apply sigmoid if needed)
        """
        # Compute pairwise distances
        pairwise_dist = self.manifold.pdist2(X)

        # Apply Fermi-Dirac transformation
        logits = -(pairwise_dist - self.bias) / self.temperature

        return logits


def _tangent_module(manifold: Manifold | ProductManifold, module: nn.Module) -> nn.Module:
    """Wrap a Euclidean ``nn.Module`` so it operates in the tangent space at the origin.

    The returned module maps inputs to the tangent space at ``mu0`` via ``logmap0``, applies the
    wrapped Euclidean module, and maps the result back to the manifold via ``expmap0``. This is the
    module-level analogue of ``manifold.apply`` (which only wraps plain callables) and keeps the
    wrapped parameters registered for ``nn.Module`` bookkeeping. For a curvature-zero (Euclidean)
    stereographic manifold ``logmap0``/``expmap0`` are the identity, so the wrapper reduces exactly to
    the underlying Euclidean module.
    """
    return _TangentModule(manifold, module)


class _TangentModule(nn.Module):
    """Module wrapper implementing ``expmap0(module(logmap0(x)))`` (see :func:`_tangent_module`)."""

    def __init__(self, manifold: Manifold | ProductManifold, module: nn.Module):
        super().__init__()
        self.manifold = manifold
        self.module = module

    def forward(self, X: Float[torch.Tensor, "n_nodes dim"]) -> Float[torch.Tensor, "n_nodes dim"]:
        """Apply the wrapped Euclidean module in the tangent space at the origin."""
        H = self.manifold.manifold.logmap0(X)
        H = self.module(H)
        return self.manifold.manifold.expmap0(H)


class StereographicLayerNorm(nn.Module):
    """Stereographic Layer Normalization.

    Layer normalization is undefined directly on a curved manifold, so we apply an ordinary Euclidean
    ``nn.LayerNorm`` in the tangent space at the origin (``logmap0`` -> ``LayerNorm`` -> ``expmap0``).
    For a stereographic ``ProductManifold`` the tangent space at the origin is Euclidean of dimension
    ``manifold.dim`` and ``logmap0``/``expmap0`` handle the per-component curvatures, so no explicit
    curvature broadcasting is required. The output is re-projected onto the manifold for numerical
    safety. In the curvature-zero limit this reduces to a plain ``LayerNorm``.

    Args:
        manifold: Manifold or ProductManifold object defining the geometry. Must be stereographic.
        embedding_dim: Embedding dimension of the input points (``manifold.dim``).

    Attributes:
        manifold: The manifold object for geometric operations.
        norm: Tangent-space layer-norm wrapper.
    """

    def __init__(self, manifold: Manifold | ProductManifold, embedding_dim: int):
        super().__init__()

        self.manifold = manifold
        self.norm = _tangent_module(manifold, nn.LayerNorm(embedding_dim))

    def forward(self, X: Float[torch.Tensor, "n_nodes dim"]) -> Float[torch.Tensor, "n_nodes dim"]:
        """Apply layer normalization on the stereographic manifold."""
        return self.manifold.manifold.projx(self.norm(X))


class GeometricLinearizedAttention(nn.Module):
    r"""Linear (kernelized) multi-head attention in the tangent space of a stereographic manifold.

    The semantics deliberately follow the rest of ``manify``: a *single graph* of ``n_nodes`` tokens
    with shape ``[n_nodes, dim]`` (no batch dimension), exactly like :class:`KappaGCNLayer`. The
    ``mask`` plays the role of the (normalized) adjacency matrix ``A_hat`` -- an all-ones mask gives
    full self-attention, a sparse mask restricts which tokens may attend to which.

    Attention itself is computed in the tangent space at the origin, where the geometry is Euclidean
    and standard linear attention with the ``elu(x) + 1`` feature map is well defined. The caller is
    responsible for mapping points on/off the manifold (this module receives and returns *tangent*
    vectors). Working in the tangent space sidesteps the ill-posed problem of slicing a product
    manifold's coordinates across heads and yields an exact Euclidean limit as curvature -> 0.

    Linear attention computes, per query ``i``:

    $$ \mathrm{out}_i = \frac{\sum_j m_{ij}\, \phi(q_i)^\top \phi(k_j)\, v_j}
                              {\sum_j m_{ij}\, \phi(q_i)^\top \phi(k_j)}, \qquad \phi(x) = elu(x) + 1 $$

    which is evaluated in the kernel-factorized (linear-time) form.

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self._epsilon = 1e-6

    def forward(
        self,
        Q: Float[torch.Tensor, "num_heads n_nodes head_dim"],
        K: Float[torch.Tensor, "num_heads n_nodes head_dim"],
        V: Float[torch.Tensor, "num_heads n_nodes head_dim"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "num_heads n_nodes head_dim"]:
        """Forward pass for tangent-space linear attention.

        Args:
            Q: Query tensor, shape ``[num_heads, n_nodes, head_dim]``.
            K: Key tensor, shape ``[num_heads, n_nodes, head_dim]``.
            V: Value tensor, shape ``[num_heads, n_nodes, head_dim]``.
            mask: Optional adjacency/attention mask, shape ``[n_nodes, n_nodes]``. Entry ``(i, j)``
                weights how much query ``i`` attends to key/value ``j``. ``None`` means full attention.

        Returns:
            Output tensor, shape ``[num_heads, n_nodes, head_dim]``.
        """
        # Feature map phi(x) = elu(x) + 1 > 0, so attention weights are non-negative.
        Qf = nn.functional.elu(Q) + 1.0  # [H, N, d]
        Kf = nn.functional.elu(K) + 1.0  # [H, N, d]

        if mask is None:
            # Linear-time form: aggregate over keys first, O(N d^2) instead of O(N^2 d).
            kv = torch.einsum("hnd,hne->hde", Kf, V)  # [H, d, d]
            numerator = torch.einsum("hnd,hde->hne", Qf, kv)  # [H, N, d]
            k_sum = Kf.sum(dim=1)  # [H, d]
            denominator = torch.einsum("hnd,hd->hn", Qf, k_sum)  # [H, N]
        else:
            # Masked form: explicit (masked) attention scores. O(N^2 d) but supports adjacency.
            scores = torch.einsum("hnd,hmd->hnm", Qf, Kf)  # [H, N, N]
            scores = scores * mask[None]  # broadcast mask over heads
            numerator = torch.einsum("hnm,hme->hne", scores, V)  # [H, N, d]
            denominator = scores.sum(dim=-1)  # [H, N]

        denominator = denominator.clamp_min(self._epsilon).unsqueeze(-1)  # [H, N, 1]
        return numerator / denominator


class StereographicAttention(nn.Module):
    """Stereographic multi-head attention layer for a single graph of ``[n_nodes, dim]`` tokens.

    Inputs and outputs are points on a stereographic (product) manifold. Queries, keys and values are
    produced by Mobius matrix-vector products (queries/keys) and a :class:`KappaGCNLayer` (values),
    then attention is performed in the tangent space at the origin by
    :class:`GeometricLinearizedAttention`, and the result is mapped back onto the manifold by a
    :class:`KappaGCNLayer` output projection. The ``mask`` is the adjacency matrix ``A_hat`` (or
    ``None`` for full attention).

    Args:
        manifold: Stereographic Manifold or ProductManifold defining the geometry.
        num_heads: Number of attention heads.
        dim: Embedding dimension of the input/output points (``manifold.dim``).
        head_dim: Dimension of each attention head.

    Attributes:
        manifold: The manifold object for geometric operations.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        W_q: Euclidean (tangent-space) linear projection to query vectors.
        W_k: Euclidean (tangent-space) linear projection to key vectors.
        W_v: Euclidean (tangent-space) linear projection to value vectors.
        attn: Tangent-space linear attention module.
        W_o: Euclidean (tangent-space) linear output projection.

    Note:
        Queries/keys/values are computed by ordinary Euclidean linear maps *in the tangent space at
        the origin* rather than by Mobius matrix-vector products. This is deliberate: on a product
        stereographic manifold, ``mobius_matvec`` is applied per component and therefore cannot change
        the per-component dimensionality, so it could not realize an arbitrary ``num_heads * head_dim``
        projection. Operating in the tangent space (where the geometry is Euclidean) removes that
        restriction while remaining curvature-correct via ``logmap0``/``expmap0``.
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
        inner = num_heads * head_dim

        # Linear maps live in the tangent space at the origin (Euclidean), so dimension changes are fine.
        self.W_q = nn.Linear(dim, inner)
        self.W_k = nn.Linear(dim, inner)
        self.W_v = nn.Linear(dim, inner)
        self.W_o = nn.Linear(inner, dim)

        self.attn = GeometricLinearizedAttention(num_heads=num_heads, head_dim=head_dim)

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass for the stereographic attention layer."""
        # Move to the tangent space at the origin; attention is Euclidean there.
        H = self.manifold.manifold.logmap0(X)  # [N, dim]

        Q = self._split_heads(self.W_q(H))  # [H, N, head_dim]
        K = self._split_heads(self.W_k(H))
        V = self._split_heads(self.W_v(H))

        attn_out = self.attn(Q, K, V, mask)  # [H, N, head_dim]
        attn_out = self._combine_heads(attn_out)  # [N, inner]
        attn_out = self.W_o(attn_out)  # [N, dim]

        # Back onto the manifold.
        return self.manifold.manifold.expmap0(attn_out)

    def _combine_heads(
        self, X: Float[torch.Tensor, "num_heads n_nodes head_dim"]
    ) -> Float[torch.Tensor, "n_nodes num_heads*head_dim"]:
        """Merge the head and feature dimensions: ``[H, N, d] -> [N, H * d]``."""
        X = X.transpose(0, 1)  # [N, H, d]
        return X.reshape(X.size(0), self.num_heads * self.head_dim)

    def _split_heads(
        self, X: Float[torch.Tensor, "n_nodes num_heads*head_dim"]
    ) -> Float[torch.Tensor, "num_heads n_nodes head_dim"]:
        """Split the feature dimension into heads: ``[N, H * d] -> [H, N, d]``."""
        X = X.reshape(X.size(0), self.num_heads, self.head_dim)
        return X.transpose(0, 1)


class StereographicTransformer(nn.Module):
    """Stereographic Transformer block operating on a single graph of ``[n_nodes, dim]`` tokens.

    A pre-norm transformer block adapted to a stereographic (product) manifold: each sublayer maps
    points to the tangent space at the origin where the computation is Euclidean, and back onto the
    manifold, with Mobius-addition residual connections. Tokens are graph nodes; the ``mask`` is the
    adjacency matrix ``A_hat`` (``None`` for full attention). In the curvature-zero limit the block
    reduces to a standard Euclidean linear-attention transformer block.

    Args:
        manifold: Stereographic Manifold or ProductManifold defining the geometry.
        num_heads: Number of attention heads.
        dim: Dimensionality of the input features (``manifold.dim``).
        head_dim: Dimensionality of each attention head.
        use_layer_norm: Whether to apply (tangent-space) layer normalization.

    Attributes:
        manifold: The manifold object for geometric operations.
        mha: Multi-head stereographic attention module.
        norm1: First normalization layer (Identity or StereographicLayerNorm).
        norm2: Second normalization layer (Identity or StereographicLayerNorm).
        mlpblock: Feedforward network operating on the manifold.
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

        if not manifold.is_stereographic:
            raise ValueError(
                "Manifold must be stereographic for StereographicTransformer to work. "
                "Please use manifold.stereographic() to convert."
            )

        self.manifold = manifold
        self.stereographic_activation = manifold.apply(nn.ReLU())
        self.mha = StereographicAttention(manifold=manifold, num_heads=num_heads, dim=dim, head_dim=head_dim)

        if use_layer_norm:
            self.norm1: nn.Module = StereographicLayerNorm(manifold=manifold, embedding_dim=dim)
            self.norm2: nn.Module = StereographicLayerNorm(manifold=manifold, embedding_dim=dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.ff1 = KappaGCNLayer(
            in_features=dim,
            out_features=dim,
            manifold=manifold,
            nonlinearity=torch.relu,
        )
        self.ff2 = KappaGCNLayer(in_features=dim, out_features=dim, manifold=manifold, nonlinearity=None)

    def _mlpblock(self, X: Float[torch.Tensor, "n_nodes dim"]) -> Float[torch.Tensor, "n_nodes dim"]:
        """Two-layer manifold feedforward network."""
        return self.ff2(self.ff1(X))

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass through the stereographic transformer block.

        Args:
            X: Node features as points on the manifold, shape ``[n_nodes, dim]``.
            mask: Optional adjacency matrix ``A_hat``; ``None`` means full attention.

        Returns:
            Updated node features as points on the manifold, shape ``[n_nodes, dim]``.
        """
        man = self.manifold.manifold

        # Pre-norm attention sublayer with Mobius-addition residual.
        attn = self.mha(self.norm1(X), mask)
        X = man.projx(man.mobius_add(attn, X))

        # Pre-norm feedforward sublayer with Mobius-addition residual.
        ff = self._mlpblock(self.norm2(X))
        X = man.projx(man.mobius_add(ff, X))

        return X
