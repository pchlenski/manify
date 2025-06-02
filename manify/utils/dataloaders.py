"""Dataloaders for different datasets"""

from __future__ import annotations

import gzip
import pickle
import shlex
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import anndata
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Real
from mpl_toolkits.basemap import Basemap
from scipy.fftpack import fft, fftfreq
from scipy.io import mmread

# Create a data directory constant
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _top_cc_dists(
    G: nx.Graph, to_undirected: bool = True, bypassed: bool = False
) -> Tuple[Real[np.ndarray, "nodes nodes"], List[int]]:
    """Returns the distances between the top connected component of a graph"""
    if to_undirected:
        G = G.to_undirected()
    if bypassed:
        return nx.floyd_warshall_numpy(G), list(range(G.number_of_nodes()))
    top_cc = max(nx.connected_components(G), key=len)
    print(f"Top CC has {len(top_cc)} nodes; original graph has {G.number_of_nodes()} nodes.")
    return nx.floyd_warshall_numpy(G.subgraph(top_cc)), list(top_cc)


def load_cities(
    cities_path: Path = DATA_DIR / "graphs" / "cities" / "cities.txt", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], None, None]:
    dists_flattened = []
    with open(cities_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            dists_flattened += [float(x) for x in line.split()]

    cities_dists = torch.tensor(dists_flattened).reshape(312, 312)

    return cities_dists, None, None


def load_cs_phds(
    cs_phds_path: Path = DATA_DIR / "graphs" / "cs_phds" / "cs_phds.txt", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    G = nx.Graph()

    with open(cs_phds_path, "r") as f:
        lines = f.readlines()

    # Add nodes
    for line in lines[2:1027]:
        num, _, v1, v2, v3 = shlex.split(line)
        num = int(num)  # type: ignore
        v1, v2, v3 = float(v1), float(v2), float(v3)  # type: ignore
        G.add_node(num, attr={"name": line, "val1": v1, "val2": v2, "val3": v3})

    # Add edges
    for line in lines[1028:2071]:
        n1, n2, weight = shlex.split(line)
        n1, n2 = int(n1), int(n2)  # type: ignore
        weight = float(weight)  # type: ignore
        G.add_edge(n1, n2, weight=weight)

    # Add years
    for i, line in enumerate(lines[2075:-1]):
        year = int(line.strip())
        G.nodes[i + 1]["year"] = year  # They're 1-indexed

    phd_dists, idx = _top_cc_dists(G, bypassed=bypassed)
    labels = [G.nodes[i]["year"] for i in idx]
    return (torch.tensor(phd_dists), torch.tensor(labels), torch.tensor(nx.to_numpy_array(G.subgraph(idx))))


def load_facebook() -> Tuple[None, None, None]:
    raise NotImplementedError


def load_power() -> Tuple[None, None, None]:
    raise NotImplementedError


def load_polblogs(
    polblogs_path: Path = DATA_DIR / "graphs" / "polblogs" / "polblogs.mtx",
    polblogs_labels_path: Path = DATA_DIR / "graphs" / "polblogs" / "polblogs_labels.tsv",
    bypassed: bool = False,
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    # Load the graph
    G = nx.from_scipy_sparse_array(mmread(polblogs_path))

    # Load the labels
    polblogs_labels = pd.read_table(polblogs_labels_path, header=None)[0]

    # Filter to match G
    dists, idx = _top_cc_dists(G)
    polblogs_labels = polblogs_labels[idx].tolist()
    return (torch.tensor(dists), torch.tensor(polblogs_labels), torch.tensor(nx.to_numpy_array(G.subgraph(idx))))


def load_polbooks(
    polbooks_path: Path = DATA_DIR / "graphs" / "polbooks" / "polbooks.gml", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    G = nx.read_gml(polbooks_path, label="id")

    dists, idx = _top_cc_dists(G, bypassed=bypassed)
    labels_unique = ["c", "l", "n"]
    labels = [labels_unique.index(G.nodes[i]["value"]) for i in idx]

    return (torch.tensor(dists), torch.tensor(labels), torch.tensor(nx.to_numpy_array(G.subgraph(idx))))


def _load_network_repository(
    edges_path: Path, labels_path: Path, bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    # Edges
    G = nx.read_edgelist(edges_path, delimiter=",", data=[("weight", int)], nodetype=int)

    # Node labels
    with open(labels_path) as f:
        for line in f:
            node, label = line.strip().split(",")
            G.nodes[int(node)]["label"] = int(label)

    dists, idx = _top_cc_dists(G, bypassed=bypassed)

    labels = [G.nodes[i]["label"] for i in idx]
    return (torch.tensor(dists), torch.tensor(labels), torch.tensor(nx.to_numpy_array(G.subgraph(idx))))


def load_cora(
    cora_edges_path: Path = DATA_DIR / "graphs" / "cora" / "cora.edges",
    cora_labels_path: Path = DATA_DIR / "graphs" / "cora" / "cora.node_labels",
    bypassed: bool = False,
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    return _load_network_repository(cora_edges_path, cora_labels_path, bypassed=bypassed)


def load_citeseer(
    citeseer_edges_path: Path = DATA_DIR / "graphs" / "citeseer" / "citeseer.edges",
    citeseer_labels_path: Path = DATA_DIR / "graphs" / "citeseer" / "citeseer.node_labels",
    bypassed: bool = False,
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    return _load_network_repository(citeseer_edges_path, citeseer_labels_path, bypassed=bypassed)


def load_pubmed(
    pubmed_edges_path: Path = DATA_DIR / "graphs" / "pubmed" / "pubmed.edges",
    pubmed_labels_path: Path = DATA_DIR / "graphs" / "pubmed" / "pubmed.node_labels",
    bypassed: bool = False,
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    return _load_network_repository(pubmed_edges_path, pubmed_labels_path, bypassed=bypassed)


def load_karate_club(
    karate_club_path: Path = DATA_DIR / "graphs" / "karate" / "karate.gml", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], None, Float[torch.Tensor, "nodes nodes"]]:
    G = nx.read_gml(karate_club_path, label="id")

    dists, idx = _top_cc_dists(G, bypassed=bypassed)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_lesmis(
    lesmis_path: Path = DATA_DIR / "graphs" / "lesmis" / "lesmis.gml", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], None, Float[torch.Tensor, "nodes nodes"]]:
    G = nx.read_gml(lesmis_path, label="id")

    dists, idx = _top_cc_dists(G, bypassed=bypassed)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_adjnoun(
    adjnoun_path: Path = DATA_DIR / "graphs" / "adjnoun" / "adjnoun.gml", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], None, Float[torch.Tensor, "nodes nodes"]]:
    G = nx.read_gml(adjnoun_path, label="id")

    dists, idx = _top_cc_dists(G, bypassed=bypassed)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_football(
    football_path: Path = DATA_DIR / "graphs" / "football" / "football.mtx", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], None, Float[torch.Tensor, "nodes nodes"]]:
    G = nx.from_scipy_sparse_array(mmread(football_path))
    dists, idx = _top_cc_dists(G, bypassed=bypassed)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_dolphins(
    dolphin_path: Path = DATA_DIR / "graphs" / "dolphins" / "dolphins.gml", bypassed: bool = False
) -> Tuple[Float[torch.Tensor, "nodes nodes"], None, Float[torch.Tensor, "nodes nodes"]]:
    G = nx.read_gml(dolphin_path, label="id")

    dists, idx = _top_cc_dists(G, bypassed=bypassed)

    return torch.tensor(dists), None, torch.tensor(nx.to_numpy_array(G.subgraph(idx)))


def load_blood_cells(
    blood_cell_anndata_path: Path = DATA_DIR / "blood_cell_scrna" / "adata.h5ad.gz",
) -> Tuple[Float[torch.Tensor, "nodes dims"], Float[torch.Tensor, "nodes,"], None]:
    with gzip.open(blood_cell_anndata_path, "rb") as f:
        adata = anndata.read_h5ad(f)
    X = torch.tensor(adata.X.todense()).float()
    X = X / X.sum(dim=1, keepdim=True)
    y = torch.tensor([int(x) for x in adata.obs["cell_type"].values])

    return X, y, None


def load_lymphoma(
    lymphoma_anndata_path: Path = DATA_DIR / "lymphoma" / "adata.h5ad.gz",
) -> Tuple[Float[torch.Tensor, "nodes dims"], Float[torch.Tensor, "nodes,"], None]:
    """https://www.10xgenomics.com/resources/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0"""
    with gzip.open(lymphoma_anndata_path, "rb") as f:
        adata = anndata.read_h5ad(f)
    X = torch.tensor(adata.X.todense()).float()
    X = X / X.sum(dim=1, keepdim=True)
    y = torch.tensor([int(x) for x in adata.obs["cell_type"].values])

    return X, y, None


def load_cifar_100(
    cifar_data_path: Path = DATA_DIR / "cifar_100" / "cifar-100-python", coarse: bool = True, train: bool = True
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], None]:
    # Load data
    split = "train" if train else "test"
    with open(cifar_data_path / split, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    X = torch.tensor(data[b"data"]).float()
    X = X.reshape(-1, 3, 32, 32)  # .permute(0, 2, 3, 1)
    X = X / 255.0

    labels = data[b"coarse_labels"] if coarse else data[b"fine_labels"]

    return X, torch.tensor(labels), None


def load_mnist(
    mnist_data_path: Path = DATA_DIR / "mnist", train: bool = True
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], None]:
    split = "train" if train else "t10k"

    # Load data
    digits = []
    with open(mnist_data_path / f"{split}-images-idx3-ubyte", "rb") as f:
        f.read(16)
        while True:
            digit = f.read(28 * 28)
            if not digit:
                break
            digits.append(list(digit))

    X = torch.tensor(digits).reshape(-1, 28, 28).float()
    X /= 255.0

    # Get labels
    labels = []
    with open(mnist_data_path / f"{split}-labels-idx1-ubyte", "rb") as f:
        f.read(8)
        while True:
            label = f.read(1)
            if not label:
                break
            labels.append(int.from_bytes(label, byteorder="big"))

    return X, torch.tensor(labels), None


def _month_to_unit_circle_point(month: str) -> Tuple[float, float]:
    """Convert month abbreviation to a point on the unit circle."""
    month_to_index = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
    }

    if month not in month_to_index:
        raise ValueError(f"Invalid month: {month}")

    index = month_to_index[month]
    angle = 2 * np.pi * index / 12

    # Return x and y coordinates on the unit circle
    return np.cos(angle), np.sin(angle)


def load_temperature(
    temperature_path: Path = DATA_DIR / "temperature" / "temperature.csv", seed: int = 42  # Not used
) -> Tuple[Float[torch.Tensor, "nodes n_dims"], Float[torch.Tensor, "nodes,"], None]:
    temperature_dataset = pd.read_csv(temperature_path)
    temperature_dataset = temperature_dataset.drop(columns=["Latitude", "Longitude", "Country", "City", "Year"])
    temperature_dataset = pd.melt(
        temperature_dataset,
        id_vars=["X", "Y", "Z"],
        var_name="Month",
        value_name="Temperature",
    )

    # Apply month_to_unit_circle_point to the 'Month' column to get x and y for each month
    temperature_dataset[["Month_X", "Month_Y"]] = temperature_dataset["Month"].apply(
        lambda month: pd.Series(_month_to_unit_circle_point(month))
    )

    return (
        torch.tensor(
            temperature_dataset[["X", "Y", "Z", "Month_X", "Month_Y"]].values,
            dtype=torch.float32,
        ),
        torch.tensor(temperature_dataset[["Temperature"]].values.flatten()),
        None,
    )


def load_landmasses(
    n_points: int = 400, seed: Optional[int] = None
) -> Tuple[Float[torch.Tensor, "nodes n_dims"], Float[torch.Tensor, "nodes,"], None]:
    # 1. Get inputs
    _x = np.linspace(-180, 180, n_points)
    _y = np.linspace(-90, 90, n_points)
    xx, yy = np.meshgrid(_x, _y)
    xx = xx.flatten()
    yy = yy.flatten()

    # 2. Get projection coordinates
    m = Basemap()
    mxx, myy = m(xx, yy)

    # 3. Convert grid to radians
    xx_rads = np.radians(xx).reshape(-1, 1)
    yy_rads = np.radians(yy).reshape(-1, 1)
    X = np.hstack(
        [
            np.cos(yy_rads) * np.cos(xx_rads),
            np.cos(yy_rads) * np.sin(xx_rads),
            np.sin(yy_rads),
        ]
    )

    # 4. Get y-values
    y = np.array([m.is_land(x, y) for x, y in zip(mxx, myy)])

    # For some reason we were getting double-precision floats here by default
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y), None


def _load_neuron(
    neuron_idx: int,
    n_coefficients: int = 10,
    path_to_data: Path = DATA_DIR / "electrophysiology" / "623474383_ephys.nwb",
) -> Tuple[Float[torch.Tensor, "nodes n_dims"], Float[torch.Tensor, "nodes,"], None]:
    with h5py.File(path_to_data, "r") as data:
        X = np.array(data[f"acquisition/timeseries/Sweep_{neuron_idx}/data"])

    # Training indices = first 80% of points
    train_idx = int(0.8 * len(X))

    # FFT on the full signal
    X_fft = fft(X[:train_idx])
    top_k_idx = np.argsort(np.abs(X_fft))[-n_coefficients - 1 : -1]  # Avoid division by zero

    # Get freqs (use size of trianing set for correct mapping)
    freqs = fftfreq(train_idx, d=1.0)
    top_freqs = freqs[top_k_idx]

    # For each point in X, compute its position on each "clock"
    data = []
    for idx in range(len(X)):
        periods = [idx / f for f in top_freqs]
        angles = [np.pi * 2 * p for p in periods]
        xs = [np.cos(theta) for theta in angles]
        ys = [np.sin(theta) for theta in angles]
        data.append([[x, y] for x, y in zip(xs, ys)])
    data = np.stack(data, axis=0).reshape(len(X), 2 * n_coefficients)

    # Get labels: top half versus bottom half
    threshold = np.percentile(X[:train_idx], 50)
    labels = (X > threshold).astype(np.float32)

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels), None


def load_neuron33(**kwargs: Any) -> Tuple[Float[torch.Tensor, "nodes n_dims"], Float[torch.Tensor, "nodes,"], None]:
    return _load_neuron(33, **kwargs)


def load_neuron46(**kwargs: Any) -> Tuple[Float[torch.Tensor, "nodes n_dims"], Float[torch.Tensor, "nodes,"], None]:
    return _load_neuron(46, **kwargs)


def load_traffic(
    traffic_path: Path = DATA_DIR / "traffic" / "traffic.csv", seed: int = 42  # Not used
) -> Tuple[Float[torch.Tensor, "nodes n_dims"], Float[torch.Tensor, "nodes,"], None]:
    df = pd.read_csv(traffic_path)
    df["datetime"] = pd.to_datetime(df["DateTime"])
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["minute"] = df["datetime"].dt.minute

    angle = lambda x, n: x / n * 2 * np.pi

    X, y = [], []
    for i, row in df.iterrows():
        _, jct, vehicles, _, _, day, hr, doy, minute = row
        X.append(
            [
                jct,
                np.cos(angle(doy, 365)),
                np.sin(angle(doy, 365)),
                np.cos(angle(day, 7)),
                np.sin(angle(day, 7)),
                np.cos(angle(hr, 24)),
                np.sin(angle(hr, 24)),
                np.cos(angle(minute, 60)),
                np.sin(angle(minute, 60)),
            ]
        )
        y.append(vehicles)

    return (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).log(), None)


def load(
    name: str, **kwargs: Any
) -> Tuple[Float[torch.Tensor, "nodes nodes"], Float[torch.Tensor, "nodes,"], Float[torch.Tensor, "nodes nodes"]]:
    """
    Driver function to load the specified dataset

    Args:
        name: The name of the dataset in string.
        **kwargs: Additional keyword argument.

    Returns:
        A tuple that contains the distance matrix, the labels, and the adjacency matrix
    """
    loaders: Dict[
        str,
        Callable[
            ...,
            Tuple[
                Optional[Float[torch.Tensor, "nodes n_dims"]],
                Optional[Float[torch.Tensor, "nodes,"]],
                Optional[Float[torch.Tensor, "nodes nodes"]],
            ],
        ],
    ] = {
        "cities": load_cities,
        "cs_phds": load_cs_phds,
        "facebook": load_facebook,
        "power": load_power,
        "polblogs": load_polblogs,
        "polbooks": load_polbooks,
        "cora": load_cora,
        "citeseer": load_citeseer,
        "pubmed": load_pubmed,
        "karate_club": load_karate_club,
        "lesmis": load_lesmis,
        "adjnoun": load_adjnoun,
        "football": load_football,
        "dolphins": load_dolphins,
        "blood_cells": load_blood_cells,
        "lymphoma": load_lymphoma,
        "cifar_100": load_cifar_100,
        "mnist": load_mnist,
        "temperature": load_temperature,
        "landmasses": load_landmasses,
        "neuron_33": load_neuron33,
        "neuron_46": load_neuron46,
        "traffic": load_traffic,
    }
    if name in loaders:
        return loaders[name](**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
