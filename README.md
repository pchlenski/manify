# `manify`

Manify is a Python library for generating graph/data embeddings and performing machine learning in product spaces with mixed curvature (hyperbolic, Euclidean, and spherical spaces). It provides tools for manifold creation, curvature estimation, embedding generation, and predictive modeling that respects the underlying geometry of complex data.

## Key Features

- Create and manipulate manifolds with different curvatures (hyperbolic, Euclidean, spherical)
- Build product manifolds by combining multiple spaces with different geometric properties
- Learn embeddings of data in these manifolds
- Train machine learning models that respect the geometry of the embedding space
- Generate synthetic data with known geometric properties for benchmarking

## Installation

There are two ways to install `manify`:

1. **From PyPI**:
   ```bash
   pip install manify
   ```

2. **From GitHub**:
   ```bash
   pip install git+https://github.com/pchlenski/manify
   ```

## Quick Example

```python
import torch
from manify.manifolds import ProductManifold
from manify.embedders import coordinate_learning
from manify.predictors.decision_tree import ProductSpaceDT
from manify.utils import dataloaders

# Load graph data
graph_dists, graph_labels, _ = dataloaders.load("polblogs")

# Create product manifold
signature = [(1, 4)]  # Spherical manifold
pm = ProductManifold(signature=signature)

# Learn embeddings
embeddings, _ = coordinate_learning.train_coords(
    pm,
    graph_dists / graph_dists.max(),
    burn_in_iterations=1000,
    training_iterations=9000
)

# Train classifier
tree = ProductSpaceDT(pm=pm, max_depth=3)
tree.fit(embeddings, graph_labels)
```

## Modules

**Manifold Operations**
- `manify.manifolds` - Tools for generating Riemannian manifolds and product manifolds

**Curvature Estimation**
- `manify.curvature_estimation.delta_hyperbolicity` - Compute delta-hyperbolicity of a metric space
- `manify.curvature_estimation.greedy_method` - Greedy selection of signatures
- `manify.curvature_estimation.sectional_curvature` - Sectional curvature estimation using Toponogov's theorem

**Embedders**
- `manify.embedders.coordinate_learning` - Coordinate learning and optimization
- `manify.embedders.losses` - Different measurement metrics
- `manify.embedders.siamese` - Siamese network embedder
- `manify.embedders.vae` - Product space variational autoencoder

**Predictors**
- `manify.predictors.decision_tree` - Decision tree and random forest predictors
- `manify.predictors.kappa_gcn` - Kappa GCN
- `manify.predictors.kernel` - Kernel matrix calculation
- `manify.predictors.midpoint` - Angular midpoints calculation
- `manify.predictors.perceptron` - Product space perceptron
- `manify.predictors.svm` - Product space SVM

**Utilities**
- `manify.utils.benchmarks` - Tools for benchmarking
- `manify.utils.dataloaders` - Loading datasets
- `manify.utils.link_prediction` - Preprocessing graphs with link prediction
- `manify.utils.visualization` - Tools for visualization

## Documentation

For detailed documentation and examples, see `llms.txt` in the repository.

## Research Background

Manify implements geometric machine learning approaches described in academic papers, particularly focusing on handling data with mixed geometric properties. It's especially suited for data that naturally lives in non-Euclidean spaces, such as hierarchical data, networks, and certain types of biological data.