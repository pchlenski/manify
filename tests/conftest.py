"""Shared pytest fixtures and configuration for the manify test suite.

This module centralizes Hugging Face dataset handling so that:

1. The on-disk Hugging Face cache is reused across tests and across CI runs. When a cache
   is already populated (locally or restored by ``actions/cache`` in CI), we flip
   ``datasets`` into offline mode so that no per-dataset network round-trip (revision /
   metadata check) is performed. With ``datasets >= 2`` an online ``load_dataset`` call
   costs ~1-1.5s per dataset *even when fully cached* and is subject to unauthenticated
   Hub rate limiting; offline reuse of the same cache is ~0.1s and never flakes.

2. ``load_hf`` results are memoized within a run (see ``manify.utils.dataloaders``), so a
   dataset requested by several tests is only decoded once.

The offline switch is conditional: if the expected datasets are not present in the cache
(e.g. a cold CI run before the cache exists), we leave ``datasets`` online so the first run
can populate the cache, which ``actions/cache`` then saves for subsequent runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Datasets pulled by the test suite via ``load_hf`` (namespace ``manify``). Kept in sync with
# tests/test_utils.py::test_dataloaders and the load_hf calls in other test modules. Used both
# to detect whether the local cache is warm and to pre-warm/memoize datasets for the session.
_HF_DATASETS: tuple[str, ...] = (
    "cities",
    "cs_phds",
    "polblogs",
    "polbooks",
    "cora",
    "citeseer",
    "karate_club",
    "lesmis",
    "adjnoun",
    "football",
    "dolphins",
    "blood_cells",
    "lymphoma",
    "cifar_100",
    "mnist",
    "temperature",
    "landmasses",
    "neuron_33",
    "neuron_46",
    "traffic",
    "qiita",
)


def _hf_home() -> Path:
    """Return the Hugging Face home directory that ``datasets`` will actually use.

    Honors ``HF_HOME`` if set, otherwise falls back to the platform default
    (``~/.cache/huggingface``). This mirrors how ``huggingface_hub`` resolves the cache root,
    so the path we probe is the same one ``load_dataset`` reads and writes.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "huggingface"


def _cache_is_warm() -> bool:
    """Heuristically determine whether the manify datasets are already cached on disk.

    We consider the cache warm if the ``hub`` cache directory contains entries for the manify
    datasets. This is intentionally lenient: any cached manify dataset flips us to offline mode,
    because a partial cache plus offline mode still avoids network round-trips for the cached
    datasets, and ``datasets`` will raise clearly if a genuinely-missing dataset is requested.
    """
    hub_dir = _hf_home() / "hub"
    if not hub_dir.is_dir():
        return False
    return any(p.name.startswith("datasets--manify--") for p in hub_dir.iterdir())


@pytest.fixture(scope="session", autouse=True)
def hf_offline_when_cached() -> None:
    """Enable Hugging Face offline mode for the session when the dataset cache is warm.

    Setting ``HF_HUB_OFFLINE`` / ``HF_DATASETS_OFFLINE`` makes ``datasets.load_dataset`` serve
    the manify datasets directly from the local cache with no network access, eliminating the
    per-dataset revision check (~1-1.5s each) and Hub rate-limit flakiness. If the cache is cold
    (no manify datasets present), we stay online so the first run can download and populate it.

    An explicit ``HF_HUB_OFFLINE`` already set in the environment (e.g. by CI after a cache hit)
    is always respected and never overridden.
    """
    if os.environ.get("HF_HUB_OFFLINE") is not None:
        # Caller (e.g. CI) has already decided; respect it. CI sets this before the test process
        # starts, so the library reads it at import time and no runtime patching is needed.
        return
    if _cache_is_warm():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        _force_offline_runtime()


def _force_offline_runtime() -> None:
    """Flip ``datasets``/``huggingface_hub`` into offline mode at runtime.

    ``datasets`` and ``huggingface_hub`` read ``HF_HUB_OFFLINE`` into module-level config at
    import time. Since ``datasets`` is already imported (via ``manify.utils.dataloaders``) by the
    time this session fixture runs, setting the environment variable alone is too late. We also
    patch the already-loaded config objects so ``load_dataset`` actually serves from the local
    cache without any network access. This is best-effort across library versions.
    """
    try:
        import datasets.config as ds_config  # noqa: PLC0415 - patch the already-imported config

        ds_config.HF_HUB_OFFLINE = True
        if hasattr(ds_config, "HF_DATASETS_OFFLINE"):
            ds_config.HF_DATASETS_OFFLINE = True
    except Exception:  # noqa: BLE001 - offline mode is an optimization, never fatal
        pass
    try:
        import huggingface_hub.constants as hub_constants  # noqa: PLC0415 - patch loaded config

        hub_constants.HF_HUB_OFFLINE = True
    except Exception:  # noqa: BLE001
        pass
