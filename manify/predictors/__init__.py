r"""Initialize predictors in the product space.

Implemented predictors include:
## Decision Trees and Random Forests: Use geodesic splits in product manifolds via projection angles and manifold-specific geodesic midpoints.
For points in a product manifold $\mathcal{P} = \mathcal{M}_1 \times \mathcal{M}_2 \times \cdots \times \mathcal{M}_k$,
we define splits along two-dimensional subspaces for each component manifold.


### Angular representation of splits
For each component manifold $\mathcal{M}_i$, we project points onto two-dimensional subspaces and represent potential splits using angles. Given a point $\mathbf{x}$ and a basis dimension $d$, the projection angle is computed as:

$$
\theta(\mathbf{x}, d) = \tan^{-1}\left(\frac{x_0}{x_d}\right)
$$

where $x_0$ and $x_d$ are the coordinates in the selected two-dimensional subspace.

The splitting criterion is then:

$$
S(\mathbf{x}, d, \theta) =
\begin{cases}
1 & \text{if } \tan^{-1}\left( \frac{x_0}{x_d} \right) \in [\theta, \theta + \pi) \\\\
0 & \text{otherwise}
\end{cases}
$$
### Geodesic Midpoints for Decision Boundaries

To place decision boundaries optimally between clusters of points, we compute the geodesic midpoint between consecutive angles in the sorted list of projection angles. The midpoint calculation is specific to each manifold type:

$$
\begin{align}
    \theta_u &= \tan^{-1}\left(\frac{u_0}{u_d}\right) \\
    \theta_v &= \tan^{-1}\left(\frac{v_0}{v_d}\right)\\
    m_{\mathbb{E}}(\theta_u, \theta_v) &= \tan^{-1}\left( \frac{2}{u_0 + v_0} \right)\\
    m_{\mathbb{S}}(\theta_u, \theta_v) &= \frac{\theta_u + \theta_v}{2}\\
    m_{\mathbb{H}}(\theta_u, \theta_v) &= \begin{cases}
        \cot^{-1}\left(V - \sqrt{V^2-1}\right) &\text{if } \theta_u + \theta_v < \pi, \\
        \cot^{-1}\left(V + \sqrt{V^2-1}\right) &\text{otherwise.}
    \end{cases}\\
    V &= \frac{\sin(2\theta_u - 2\theta_v)}{2\sin(\theta_u + \theta_v)\sin(\theta_v - \theta_u)}
\end{align}
$$

## $\kappa$-GCNs: Extend standard GCNs to operate on both positive and negative curvature using gyrovector operations. Variants include:

### Graph Convolutional Networks Background

In a typical (Euclidean) graph convolutional network (GCN), each layer takes the form:

$$
\begin{align}
    \mathbf{H}^{(0)} &= \mathbf{X} \\\\
    \mathbf{H}^{(l+1)} &= \sigma\left( \hat{\mathbf{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)} \right)
\end{align}
$$

where $\hat{\mathbf{A}} \in \mathbb{R}^{n \times n}$ is a normalized adjacency matrix with self-connections, $\mathbf{X}^{(l)} \in \mathbb{R}^{n \times d}$ is a matrix of features, $\mathbf{W}^{(l)} \in \mathbb{R}^{d \times e}$ is a weight matrix, $\mathbf{b}^{(l)} \in \mathbb{R}^e$ is a bias term, and $\sigma$ is some nonlinearity (e.g., ReLU).
### Graph Convolution Layers

Bachmann et al. (2020) describe a way to adapt the typical GCN model for use with $\mathbf{X} \in \mathbb{S}^d_\kappa$, using gyrovector operations:

$$
\mathbf{H}^{(l+1)} = \sigma^{\otimes_\kappa} \left( \hat{\mathbf{A}} \boxtimes_\kappa \left( \mathbf{H}^{(l)} \otimes_\kappa \mathbf{W}^{(l)} \right) \right)
$$

$$
\sigma^{\otimes_\kappa}(\cdot) = \exp_{\mathbf{0}} \left( \sigma\left( \log_{\mathbf{0}}(\cdot) \right) \right)
$$
Note that this paper does not include a bias term, although it is reasonable to extend the definition of a GCN layer to include one:

$$
\mathbf{H}^{(l+1)} =
\sigma^{\otimes_\kappa} \left(
    \hat{\mathbf{A}} \boxtimes_\kappa \left(
        \mathbf{H}^{(l)} \otimes_\kappa \mathbf{W}^{(l)}
    \right) \oplus \mathbf{b}
\right)
$$

where $\mathbf{b} \in \mathbb{S}^d_\kappa$ is a bias vector.

Also note that, in order for each $\mathbf{H}^{(i)}$ to remain on the same manifold, $\mathbf{W}^{(i)}$ must be a square matrix. However, this assumption can be relaxed to allow for different dimensionalities and curvatures for each layer.
### Stereographic Logits

For classification, we define a $\kappa$-stereographic equivalent of a logit layer:

$$
\mathbf{H}^{(L)} = \text{softmax} \left(
    \hat{\mathbf{A}} \, \operatorname{logits}_{\mathbb{S}^d_\kappa}
    \left( \mathbf{H}^{(L-1)}, \mathbf{W}^{(L-1)} \right)
\right)
$$

To implement logits in $\mathbb{S}^d_\kappa$, we begin by noting that Euclidean logits can be interpreted as signed distances from a hyperplane. This follows from the linear form $\mathbf{w}_i^\top \mathbf{x} + b_i$ used in traditional classification, where $\mathbf{w}_i$ is a column of the final weight matrix and $b_i$ is its corresponding bias.

The magnitude reflects the point's distance from the decision boundary (the hyperplane $\mathbf{w}_i^\top \mathbf{x} + b_i = 0$), and the sign determines which side of the hyperplane the point lies on. This formulation encodes both the model’s decision and its confidence.

Bachmann et al. (2020) and Ganea et al. (2018) extend this intuition to non-Euclidean spaces by defining logits using appropriate distance functions.

In $\kappa$-GCN, this becomes:

$$
\mathbb{P}(y = k \mid \mathbf{x}) =
\text{Softmax} \left( \operatorname{logits}_\mathcal{M}(\mathbf{x}, k) \right)
$$

$$
\operatorname{logits}_\mathcal{M}(\mathbf{x}, k) =
\frac{ \| \mathbf{a}_k \|_{\mathbf{p}_k} }{ \sqrt{K} } \,
\sin_K^{-1} \left(
    \frac{
        2 \sqrt{|\kappa|} \langle \mathbf{z}_k, \mathbf{a}_k \rangle
    }{
        (1 + \kappa \| \mathbf{z}_k \|^2) \| \mathbf{a}_k \|
    }
\right)
$$

Although it is not explicitly stated in Bachmann et al. (2020), we follow Cho et al. (2023) and later Chlenski et al. (2024) in aggregating logits across product manifolds using the $\ell_2$-norm of component manifold logits, scaled by the sign of the sum of component inner products:

$$
\operatorname{logits}_\mathcal{P}(\mathbf{x}, k) =
\sqrt{
  \sum_{\mathcal{M} \in \mathcal{P}} \left(
    \operatorname{logits}_\mathcal{M}(\mathbf{x}^\mathcal{M}, k)
  \right)^2
}
\cdot
\text{sign} \left(
  \sum_{\mathcal{M} \in \mathcal{P}}
  \langle \mathbf{x}^\mathcal{M}, \mathbf{a}_k^\mathcal{M} \rangle_\mathcal{M}
\right)
$$


Finally, for link prediction, we follow Chami et al. (2019) in adopting the standard approach of applying the Fermi–Dirac decoder [Krioukov et al., 2010; Nickel and Kiela, 2017] to predict edges:

$$
\mathbb{P}\big((i, j) \in \mathcal{E} \,\big|\, \mathbf{x}_i, \mathbf{x}_j\big) =
\left(
  \exp \left(
    \frac{\delta_\mathcal{M}(\mathbf{x}_i, \mathbf{x}_j)^2 - r}{t}
  \right) + 1
\right)^{-1}
$$


- Perceptrons: Use manifold-specific linear combinations of inputs with sine and sinh activation terms.
### Product Space Perceptron

A linear classifier on the product manifold $\mathcal{P}$ is defined as:

$$
\operatorname{LC}(\mathbf{x}, \mathbf{w}) =
\operatorname{sign} \left(
  \langle \mathbf{w}_\mathbb{E}, \mathbf{x}_\mathbb{E} \rangle +
  \alpha_\mathbb{S} \sin^{-1} \left( \langle \mathbf{w}_\mathbb{S}, \mathbf{x}_\mathbb{S} \rangle \right) +
  \alpha_\mathbb{H} \sinh^{-1} \left( \langle \mathbf{w}_\mathbb{H}, \mathbf{x}_\mathbb{H} \rangle_\mathbb{H} \right) + b
\right)
$$

where $\mathbf{x}_\mathcal{M}$ (and similarly $\mathbf{w}_\mathcal{M}$) denotes the restriction of $\mathbf{x} \in \mathcal{P}$ to one of its component manifolds. The coefficients $\alpha_\mathbb{S}$ and $\alpha_\

- SVMs: Extend kernel SVMs with constraints respecting the geometry of each component manifold, optimizing a margin-based objective under convex and relaxed constraints.
### Product Space SVM

The Product Space SVM extends the kernel-based approach described in [Section Perceptron](#product-space-perceptron) by finding a maximum-margin classifier in the product space. The optimization problem is formulated as:

$$
\text{maximize } \varepsilon - \sum_{i=1}^{n} \xi_i
$$

subject to

$$
y_i \sum_{j=1}^{n} \beta_j K(\mathbf{x}_i, \mathbf{x}_j) \geq \varepsilon - \xi_i
\quad \text{for all } i \in \{1, \ldots, n\}
$$

where $\varepsilon > 0$ and $\xi_i \geq 0$.
"""

from manify.predictors.decision_tree import ProductSpaceDT, ProductSpaceRF
from manify.predictors.kappa_gcn import KappaGCN
from manify.predictors.perceptron import ProductSpacePerceptron
from manify.predictors.svm import ProductSpaceSVM
from manify.predictors.transformer import KappaTransformer

__all__ = [
    "ProductSpaceDT",
    "ProductSpaceRF",
    "KappaGCN",
    "KappaTransformer",
    "ProductSpacePerceptron",
    "ProductSpaceSVM",
]
