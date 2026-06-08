"""# Dataloaders Submodule.

The dataloaders module allows users to load datasets from Manify's datasets repo [on Hugging Face](https://huggingface.co/manify).

We provide a summary of the data types available, and their original sources, here.

Earlier versions of Manify included scripts to process raw data, which we have replaced with a single, centralized Hugging Face repo and the function `load_hf`. For transparency, we have preserved the data generation code in [the Dataset-Generation branch of Manify](https://github.com/pchlenski/manify/tree/Dataset-Generation).

| Dataset | Task | Distance Matrix | Features | Labels | Adjacency Matrix | Source/Citation |
|---------|------|----------------|----------|--------|-----------------|-----------------|
| cities | none | ✅ | ❌ | ❌ | ❌ | [Network Repository: Cities](https://networkrepository.com/Cities.php) |
| cs_phds | regression | ✅ | ❌ | ✅ | ✅ | [Network Repository: CS PhDs](https://networkrepository.com/CSphd.php) |
| polblogs | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Polblogs](https://networkrepository.com/polblogs.php) |
| polbooks | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Polbooks](https://networkrepository.com/polbooks.php) |
| cora | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Cora](https://networkrepository.com/cora.php) |
| citeseer | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Citeseer](https://networkrepository.com/citeseer.php) |
| karate_club | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Karate](https://networkrepository.com/karate.php) |
| lesmis | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Lesmis](https://networkrepository.com/lesmis.php) |
| adjnoun | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Adjnoun](https://networkrepository.com/adjnoun.php) |
| football | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Football](https://networkrepository.com/football.php) |
| dolphins | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Dolphins](https://networkrepository.com/dolphins.php) |
| blood_cells | classification | ❌ | ✅ | ✅ | ❌ | See datasets from Zheng et al (2017): Massively parallel digital transcriptional profiling of single cells.<br>- [CD8+ Cytotoxic T-cells](https://www.10xgenomics.com/datasets/cd-8-plus-cytotoxic-t-cells-1-standard-1-1-0)<br>- [CD8+/CD45RA+ Naive Cytotoxic T Cells](https://www.10xgenomics.com/datasets/cd-8-plus-cd-45-r-aplus-naive-cytotoxic-t-cells-1-standard-1-1-0)<br>- [CD56+ Natural Killer Cells](https://www.10xgenomics.com/datasets/cd-56-plus-natural-killer-cells-1-standard-1-1-0)<br>- [CD4+ Helper T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-helper-t-cells-1-standard-1-1-0)<br>- [CD4+/CD45RO+ Memory T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-oplus-memory-t-cells-1-standard-1-1-0)<br>- [CD4+/CD45RA+/CD25- Naive T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-aplus-cd-25-naive-t-cells-1-standard-1-1-0)<br>- [CD4+/CD25+ Regulatory T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-25-plus-regulatory-t-cells-1-standard-1-1-0)<br>- [CD34+ Cells](https://www.10xgenomics.com/datasets/cd-34-plus-cells-1-standard-1-1-0)<br>- [CD19+ B Cells](https://www.10xgenomics.com/datasets/cd-19-plus-b-cells-1-standard-1-1-0)<br>- [CD14+ Monocytes](https://www.10xgenomics.com/datasets/cd-14-plus-monocytes-1-standard-1-1-0) |
| lymphoma | classification | ❌ | ✅ | ✅ | ❌ | See datasets from 10x Genomics:<br>- [Hodgkin's Lymphoma](https://www.10xgenomics.com/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0)<br>- [Healthy Donor PBMCs](https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-targeted-compare-immunology-panel-3-1-standard-4-0-0) |
| cifar_100 | classification | ❌ | ✅ | ✅ | ❌ | [Hugging Face Datasets: CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100) |
| mnist | classification | ❌ | ✅ | ✅ | ❌ | [Hugging Face Datasets: MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| temperature | regression | ❌ | ✅ | ✅ | ❌ | [Citation] |
| landmasses | classification | ❌ | ✅ | ✅ | ❌ | Generated using [basemap.is_land](https://matplotlib.org/basemap/stable/api/basemap_api.html#mpl_toolkits.basemap.Basemap.is_land) |
| neuron_33 | classification | ❌ | ✅ | ✅ | ❌ | [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/623474400) |
| neuron_46 | classification | ❌ | ✅ | ✅ | ❌ | [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/623474400) |
| traffic | regression | ❌ | ✅ | ✅ | ❌ | [Kaggle: Traffic Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset) |
| qiita | none | ✅ | ✅ | ❌ | ❌ | [NeuroSEED Git Repo](https://github.com/gcorso/NeuroSEED) |
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from datasets import load_dataset

if TYPE_CHECKING:
    from jaxtyping import Float, Real


@functools.cache
def _load_hf_cached(
    name: str, namespace: str
) -> tuple[
    Float[torch.Tensor, "n_points ..."] | None,
    Float[torch.Tensor, "n_points n_points"] | None,
    Float[torch.Tensor, "n_points n_points"] | None,
    Real[torch.Tensor, "n_points"] | None,
]:
    """In-process memoized implementation of :func:`load_hf`.

    Results are cached per ``(name, namespace)`` for the lifetime of the process so that
    repeatedly loading the same dataset (e.g. across multiple tests in a single pytest run)
    only resolves and decodes the dataset once. The Hugging Face on-disk cache and the
    ``HF_HUB_OFFLINE`` / ``HF_DATASETS_OFFLINE`` environment variables are still honored by
    :func:`datasets.load_dataset`, so this layer composes with the local cache.

    Args:
        name: The dataset name within the namespace.
        namespace: The Hugging Face namespace/owner of the dataset.

    Returns:
        The same 4-tuple as :func:`load_hf`.
    """
    # 1) fetch the single-row dataset
    ds = load_dataset(f"{namespace}/{name}")
    data = ds.get("train", ds)  # use "train" split if available, else the only split
    row = data[0]

    # 2) helper to turn lists -> torch (or None)
    def to_tensor(key: str, dtype: torch.dtype) -> torch.Tensor | None:
        vals = row.get(key, [])
        if not vals:
            return None
        return torch.tensor(vals, dtype=dtype)

    # 3) reconstruct everything
    dists = to_tensor("distances", torch.float32)
    feats = to_tensor("features", torch.float32)
    adj = to_tensor("adjacency", torch.float32)

    cls_ls = row.get("classification_labels", [])
    reg_ls = row.get("regression_labels", [])
    if cls_ls:
        labels = torch.tensor(cls_ls, dtype=torch.int64)
    elif reg_ls:
        labels = torch.tensor(reg_ls, dtype=torch.float32)
    else:
        labels = None

    return feats, dists, adj, labels


def load_hf(
    name: str, namespace: str = "manify"
) -> tuple[
    Float[torch.Tensor, "n_points ..."] | None,  # features
    Float[torch.Tensor, "n_points n_points"] | None,  # pairwise dists
    Float[torch.Tensor, "n_points n_points"] | None,  # adjacency labels
    Real[torch.Tensor, "n_points"] | None,  # labels
]:
    """Load a dataset from HuggingFace Hub at {namespace}/{name}.

    Results are memoized per ``(name, namespace)`` for the lifetime of the process, so loading
    the same dataset repeatedly (common across a test run) only resolves and decodes it once.
    The Hugging Face on-disk cache and the ``HF_HUB_OFFLINE`` / ``HF_DATASETS_OFFLINE``
    environment variables are honored by the underlying :func:`datasets.load_dataset` call, so
    setting offline mode reuses the local cache without any network access.

    Returns:
        features: The features for each node, if any
        dists: The pairwise distance matrix over all nodes, if any
        adj: The adjacency matrix over all nodes, if any
        labels: The (classification or regression) labels for each node, if any
    """
    return _load_hf_cached(name, namespace)
