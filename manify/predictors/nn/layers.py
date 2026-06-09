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
        self, in_features: int, out_features: int, manifold: Manifold, nonlinearity: Callable | None = torch.relu
    ):
        super().__init__()

        # Parameters are Euclidean, straightforwardly
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.01)

        # Nonlinearity must be applied via the manifold
        self.sigma = manifold.apply(nonlinearity) if nonlinearity else lambda x: x

        # Also store manifold
        self.manifold = manifold

    def _left_multiply(
        self, A: Float[torch.Tensor, "n_nodes n_nodes"], X: Float[torch.Tensor, "n_nodes dim"], M: Manifold
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
        self, X: Float[torch.Tensor, "n_nodes dim"], A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None
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
        self, X: Float[torch.Tensor, "n_nodes dim"], A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None
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

    def __init__(self, out_features: int, manifold: Manifold | ProductManifold, apply_softmax: bool = False):
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
        tuple[Float[torch.Tensor, "n_nodes n_classes"], Float[torch.Tensor, "n_nodes n_classes"]]
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
        self, X: Float[torch.Tensor, "n_nodes dim"], A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None
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
    r"""Faithful gyrovector linear attention (FPS-T, arXiv:2309.04082, Eqs 6, 7, 11).

    This is the kernelized mixed-curvature attention from "Curve Your Attention: Mixed-Curvature
    Transformers for Graph Representation Learning". It operates per head on its own
    :math:`\kappa_h`-stereographic space. Following the rest of ``manify`` it works on a *single
    graph* (no batch dimension): inputs are ``[num_heads, n_nodes, head_dim]`` and the ``mask`` is the
    ``[n_nodes, n_nodes]`` adjacency matrix (``None`` means full attention).

    Inputs:
        * ``V`` -- value *points* on the per-head :math:`\kappa_h`-stereographic manifold.
        * ``Q``, ``K`` -- query/key *tangent vectors at the corresponding* ``V_i`` (Eq 5).

    Scores (Eq 6): parallel-transport ``Q_i`` and ``K_j`` to the origin (``parallel_transport0back``),
    apply the feature map :math:`\phi(x)=\mathrm{ELU}(x)+1`, and take a Euclidean inner product there:
    :math:`\alpha_{ij}\approx\phi(\tilde Q_i)^\top\phi(\tilde K_j)`.

    Aggregation (Eq 7) is the **Einstein midpoint**, kernelized (Eq 11):

    $$ \mathrm{Aggregate}_\kappa(V,\alpha)_i = \tfrac{1}{2}\otimes_\kappa
        \sum_j \frac{\alpha_{ij}\,\lambda^\kappa_{V_j}}
                    {\sum_k \alpha_{ik}(\lambda^\kappa_{V_k}-1)}\, V_j, $$

    where :math:`\lambda^\kappa` is the conformal factor and :math:`\tfrac12\otimes_\kappa` is Mobius
    scalar multiplication. Writing :math:`\tilde V_i=\frac{\lambda^\kappa_{V_i}}{\lambda^\kappa_{V_i}-1}V_i`
    and :math:`\phi'(\tilde K)_i=\phi(\tilde K_i)(\lambda^\kappa_{V_i}-1)`, the per-query output is
    :math:`\tfrac12\otimes_\kappa\big[\phi(\tilde Q)\,(\phi'(\tilde K)^\top \tilde V)\big]_i`, which is
    :math:`O(N+M)` in the full-attention case. As :math:`\kappa\to0` the Einstein midpoint reduces to
    the ordinary (Euclidean) weighted mean.

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
        self._clamp_epsilon = 1e-10

    def forward(
        self,
        Q: Float[torch.Tensor, "num_heads n_nodes head_dim"],
        K: Float[torch.Tensor, "num_heads n_nodes head_dim"],
        V: Float[torch.Tensor, "num_heads n_nodes head_dim"],
        curvatures: Float[torch.Tensor, "num_heads 1 1"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "num_heads n_nodes head_dim"]:
        """Forward pass for faithful gyrovector linear attention.

        Args:
            Q: Query tangent vectors at ``V``, shape ``[num_heads, n_nodes, head_dim]``.
            K: Key tangent vectors at ``V``, shape ``[num_heads, n_nodes, head_dim]``.
            V: Value points on the per-head manifold, shape ``[num_heads, n_nodes, head_dim]``.
            curvatures: Per-head curvatures, shape ``[num_heads, 1, 1]``.
            mask: Optional adjacency/attention mask, shape ``[n_nodes, n_nodes]``. Entry ``(i, j)``
                weights how much query ``i`` attends to key/value ``j``. ``None`` means full attention.

        Returns:
            Aggregated value points on the per-head manifold, shape ``[num_heads, n_nodes, head_dim]``.
        """
        k = curvatures
        math = geoopt.manifolds.stereographic.math

        # Eq 6: parallel-transport Q, K (tangent at V) to the origin, then feature map phi = elu + 1.
        q_tilde = math.parallel_transport0back(V, Q, k=k)  # [H, N, d]
        k_tilde = math.parallel_transport0back(V, K, k=k)  # [H, N, d]
        phi_q = nn.functional.elu(q_tilde) + 1.0  # phi(Q~)
        phi_k = nn.functional.elu(k_tilde) + 1.0  # phi(K~)

        # Conformal factor lambda^kappa at each value point (Eq 7).
        lam = math.lambda_x(x=V, k=k, keepdim=True, dim=-1)  # [H, N, 1]
        denom = geoopt.utils.clamp_abs(lam - 1, self._clamp_epsilon)  # (lambda - 1), [H, N, 1]

        v_tilde = (lam / denom) * V  # V~_i = lambda_i / (lambda_i - 1) * V_i
        phi_k_prime = denom * phi_k  # phi'(K~)_i = phi(K~)_i * (lambda_i - 1)

        if mask is None:
            # Kernelized O(N + M) form (Eq 11). The numerator effectively weights V_j by
            # alpha_ij * lambda_j = (phi(Q~)_i . phi'(K~)_j) * V~_j, and the normalizer by
            # alpha_ij * (lambda_j - 1) = phi(Q~)_i . phi'(K~)_j.
            context = torch.einsum("hnd,hne->hde", phi_k_prime, v_tilde)  # phi'(K)^T V~, [H, d, d]
            numerator = torch.einsum("hnd,hde->hne", phi_q, context)  # [H, N, d]
            normalizer = torch.einsum("hnd,hd->hn", phi_q, phi_k_prime.sum(dim=-2))  # [H, N]
        else:
            # Explicit (masked) attention scores -- supports a sparse adjacency, O(N^2 d).
            # Identical math to the kernelized path: numerator weights V_j by alpha_ij * lambda_j,
            # normalizer by alpha_ij * (lambda_j - 1).
            alpha = torch.einsum("hnd,hmd->hnm", phi_q, phi_k) * mask[None]  # [H, N, N]
            numerator = torch.einsum("hnm,hme->hne", alpha, lam * V)  # [H, N, d]
            normalizer = torch.einsum("hnm,hm->hn", alpha, denom.squeeze(-1))  # [H, N]

        norm_inv = 1.0 / normalizer.masked_fill(normalizer == 0, self._epsilon)  # [H, N]
        midpoint = numerator * norm_inv.unsqueeze(-1)  # weighted gyro-sum, [H, N, d]

        # Einstein midpoint finishes with a Mobius half-scaling; project for numerical safety.
        out = math.project(midpoint, k=k)
        out = math.mobius_scalar_mul(torch.tensor(0.5, dtype=out.dtype, device=out.device), out, k=k, dim=-1)
        out = math.project(out, k=k)
        return out


class StereographicAttention(nn.Module):
    r"""Mixed-curvature multi-head attention for a single graph of ``[n_nodes, dim]`` tokens (FPS-T).

    Faithful implementation of the multi-head attention from "Curve Your Attention" (arXiv:2309.04082).
    Each head ``h`` operates on its OWN :math:`\kappa_h`-stereographic space with an independent
    **learnable** curvature, and the multi-head output is the *product* over heads
    (:math:`\bigotimes_h \mathrm{st}_{\kappa_h}`) -- heads are product-manifold components, so per-head
    outputs are concatenated and never reshaped across heads. The per-head curvatures are decoupled
    from the input manifold.

    Per head (Eq 5): values are points on :math:`\mathrm{st}_{\kappa_h}` and queries/keys live in the
    tangent space at the corresponding value point. We obtain these by mapping the input to the tangent
    space at the origin (``logmap0``), applying an ordinary Euclidean ``nn.Linear`` to the head
    dimension, and exponentiating into :math:`\mathrm{st}_{\kappa_h}` for the values; queries/keys are
    the (Euclidean) projections re-based to the tangent space at each value point via parallel
    transport from the origin. Aggregation is the gyrovector Einstein midpoint
    (:class:`GeometricLinearizedAttention`). The masked output product manifold is mapped back to the
    input manifold by a tangent-space linear projection (``logmap0`` -> ``Linear`` -> ``expmap0``).

    Args:
        manifold: Stereographic Manifold or ProductManifold defining the input/output geometry.
        num_heads: Number of attention heads (product-manifold components).
        dim: Embedding dimension of the input/output points (``manifold.dim``).
        head_dim: Dimension of each attention head.
        init_curvatures: Optional list of initial per-head curvatures (length ``num_heads``). Defaults
            to ``-1`` for the first half of heads and ``+1`` for the rest (mixed curvature).

    Attributes:
        manifold: The (input/output) manifold object.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        curvatures: Learnable per-head curvatures, shape ``[num_heads]``.
        W_q: Euclidean (tangent-space) linear projection to query vectors.
        W_k: Euclidean (tangent-space) linear projection to key vectors.
        W_v: Euclidean (tangent-space) linear projection to value vectors.
        attn: Gyrovector Einstein-midpoint attention module.
        W_o: Euclidean (tangent-space) linear output projection.
    """

    def __init__(
        self,
        manifold: Manifold | ProductManifold,
        num_heads: int,
        dim: int,
        head_dim: int,
        init_curvatures: list[float] | None = None,
    ):
        super().__init__()

        self.manifold = manifold
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        # Learnable per-head curvatures (decoupled from the input manifold). Default: mixed curvature.
        if init_curvatures is None:
            init_curvatures = [-1.0] * (num_heads // 2) + [1.0] * (num_heads - num_heads // 2)
        self.curvatures = nn.Parameter(torch.tensor(init_curvatures, dtype=torch.float))

        # Tangent-space linear projections (paper-faithful; allow free head_dim).
        self.W_q = nn.Linear(dim, inner)
        self.W_k = nn.Linear(dim, inner)
        self.W_v = nn.Linear(dim, inner)
        self.W_o = nn.Linear(inner, dim)

        self.attn = GeometricLinearizedAttention(num_heads=num_heads, head_dim=head_dim)

    def _per_head_curvatures(self) -> Float[torch.Tensor, "num_heads 1 1"]:
        """Reshape the learnable per-head curvatures for broadcasting over ``[H, N, d]``."""
        return self.curvatures.view(self.num_heads, 1, 1)

    def forward(
        self, X: Float[torch.Tensor, "n_nodes dim"], mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass for the mixed-curvature attention layer."""
        math = geoopt.manifolds.stereographic.math
        k = self._per_head_curvatures()  # [H, 1, 1]

        # Tangent space at the origin of the input manifold, then per-head Euclidean projections.
        H = self.manifold.manifold.logmap0(X)  # [N, dim]
        q_tan = self._split_heads(self.W_q(H))  # [H, N, d] (Euclidean, at origin)
        k_tan = self._split_heads(self.W_k(H))
        v_tan = self._split_heads(self.W_v(H))

        # Values are points on the per-head st_{kappa_h}; Q/K are tangent vectors AT each value point.
        V = math.expmap0(v_tan, k=k)  # [H, N, d] points on st_kappa_h
        Q = math.parallel_transport0(V, q_tan, k=k)  # tangent at V_i (Eq 5)
        K = math.parallel_transport0(V, k_tan, k=k)

        attn_out = self.attn(Q, K, V, curvatures=k, mask=mask)  # [H, N, d] points on st_kappa_h

        # Multi-head output is the product over heads: concat per-head tangent coords, project back.
        attn_tan = self._combine_heads(math.logmap0(attn_out, k=k))  # [N, inner] (tangent)
        out_tan = self.W_o(attn_tan)  # [N, dim]
        return self.manifold.manifold.expmap0(out_tan)

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
    r"""Mixed-curvature transformer block on a single graph of ``[n_nodes, dim]`` tokens (FPS-T).

    A pre-norm transformer block adapted to a stereographic (product) manifold per "Curve Your
    Attention" (arXiv:2309.04082): a mixed-curvature multi-head attention sublayer
    (:class:`StereographicAttention`, faithful gyrovector Einstein-midpoint aggregation with learnable
    per-head curvatures) followed by a manifold feedforward sublayer, each wrapped in a Mobius-addition
    residual connection. Tokens are graph nodes; the ``mask`` is the adjacency matrix ``A_hat``
    (``None`` for full attention). As the curvatures vanish the block reduces to a standard Euclidean
    linear-attention transformer block.

    Args:
        manifold: Stereographic Manifold or ProductManifold defining the geometry.
        num_heads: Number of attention heads.
        dim: Dimensionality of the input features (``manifold.dim``).
        head_dim: Dimensionality of each attention head.
        use_layer_norm: Whether to apply (tangent-space) layer normalization.
        init_curvatures: Optional initial per-head curvatures passed to the attention sublayer.

    Attributes:
        manifold: The manifold object for geometric operations.
        mha: Mixed-curvature multi-head attention module.
        norm1: First normalization layer (Identity or StereographicLayerNorm).
        norm2: Second normalization layer (Identity or StereographicLayerNorm).
        ff1: First feedforward KappaGCNLayer (with ReLU nonlinearity).
        ff2: Second feedforward KappaGCNLayer (no nonlinearity).
    """

    def __init__(
        self,
        manifold: Manifold | ProductManifold,
        num_heads: int,
        dim: int,
        head_dim: int,
        use_layer_norm: bool = True,
        init_curvatures: list[float] | None = None,
    ):
        super().__init__()

        if not manifold.is_stereographic:
            raise ValueError(
                "Manifold must be stereographic for StereographicTransformer to work. "
                "Please use manifold.stereographic() to convert."
            )

        self.manifold = manifold
        self.mha = StereographicAttention(
            manifold=manifold, num_heads=num_heads, dim=dim, head_dim=head_dim, init_curvatures=init_curvatures
        )

        if use_layer_norm:
            self.norm1: nn.Module = StereographicLayerNorm(manifold=manifold, embedding_dim=dim)
            self.norm2: nn.Module = StereographicLayerNorm(manifold=manifold, embedding_dim=dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.ff1 = KappaGCNLayer(in_features=dim, out_features=dim, manifold=manifold, nonlinearity=torch.relu)
        self.ff2 = KappaGCNLayer(in_features=dim, out_features=dim, manifold=manifold, nonlinearity=None)

    def _mlpblock(self, X: Float[torch.Tensor, "n_nodes dim"]) -> Float[torch.Tensor, "n_nodes dim"]:
        """Two-layer manifold feedforward network."""
        return self.ff2(self.ff1(X))

    def forward(
        self, X: Float[torch.Tensor, "n_nodes dim"], mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass through the mixed-curvature transformer block.

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
