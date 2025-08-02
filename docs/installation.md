# Installing Manify

There are two ways to install `manify`:

1. **From PyPI**:
   ```bash
   pip install manify
   ```

2. **From GitHub** (recommended due to active development of the repo):
   ```bash
   pip install git+https://github.com/pchlenski/manify
   ```

## Quick Example

```python
import manify
from manify.utils.dataloaders import load_hf
from sklearn.model_selection import train_test_split

# Load Polblogs graph from HuggingFace
features, dists, adj, labels = load_hf("polblogs")

# Create an S^4 x H^4 product manifold
pm = manify.ProductManifold(signature=[(1.0, 4), (-1.0, 4)])

# Learn embeddings (Gu et al (2018) method)
embedder = manify.CoordinateLearning(pm=pm)
X_embedded = embedder.fit_transform(X=None, D=dists, burn_in_iterations=200, training_iterations=800)

# Train and evaluate classifier (Chlenski et al (2025) method)
X_train, X_test, y_train, y_test = train_test_split(X_embedded, labels)
model = manify.ProductSpaceDT(pm=pm, max_depth=3, task="classification")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## Tutorial
The official tutorial for Manify can be found in [`tutorial.ipynb`](tutorial.ipynb). This Jupyter
notebook contains a comprehensive overview of all of the library's core features. 
