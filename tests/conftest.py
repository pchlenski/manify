"""Shared pytest fixtures and configuration for the manify test suite.

Centralizes Hugging Face dataset handling so that:

1. The on-disk HF cache is reused across tests and CI runs. When *every* manify
   dataset is already cached (locally or restored by ``actions/cache``), we flip
   ``datasets`` into offline mode to skip the ~1-1.5s per-dataset Hub round-trip
   (revision/metadata check) and the unauthenticated rate-limit flakiness.

2. ``load_hf`` results are memoized within a run (see ``manify.utils.dataloaders``).

Offline mode is only safe when the cache is **fully** warm: under offline mode a
requested-but-missing dataset hard-fails (``ConnectionError: OfflineModeIsEnabled``).
So we require ALL expected datasets to be present before going offline; otherwise we
force ONLINE — even if CI exported ``HF_HUB_OFFLINE`` on an incomplete cache — so the
run can fetch (and populate) whatever is missing. Correctness over speed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Datasets pulled by the suite via ``load_hf`` (namespace ``manify``). Keep in sync with
# tests/test_utils.py::test_dataloaders and load_hf calls elsewhere.
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
    """HF home dir that ``datasets`` will actually use (honors ``HF_HOME``)."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "huggingface"


def _cache_is_fully_warm() -> bool:
    """True only if EVERY manify dataset is present in the on-disk hub cache.

    Strict on purpose: a partial cache + offline mode hard-fails on the first
    missing dataset, so offline is only safe once all of them are cached.
    """
    hub_dir = _hf_home() / "hub"
    if not hub_dir.is_dir():
        return False
    cached = {p.name for p in hub_dir.iterdir() if p.is_dir()}
    return all(f"datasets--manify--{name}" in cached for name in _HF_DATASETS)


def _set_offline_runtime(offline: bool) -> None:
    """Flip ``datasets``/``huggingface_hub`` offline flags at runtime (both directions).

    These libs read ``HF_HUB_OFFLINE`` into module-level config at import time, and
    ``datasets`` is already imported (via ``manify.utils.dataloaders``) before this
    session fixture runs, so setting the env var alone is too late in either direction.
    Best-effort across library versions.
    """
    try:
        import datasets.config as ds_config  # noqa: PLC0415 - patch already-imported config

        ds_config.HF_HUB_OFFLINE = offline
        if hasattr(ds_config, "HF_DATASETS_OFFLINE"):
            ds_config.HF_DATASETS_OFFLINE = offline
    except Exception:  # noqa: BLE001 - offline mode is an optimization, never fatal
        pass
    try:
        import huggingface_hub.constants as hub_constants  # noqa: PLC0415 - patch loaded config

        hub_constants.HF_HUB_OFFLINE = offline
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="session", autouse=True)
def hf_offline_when_cached() -> None:
    """Enable HF offline mode for the session only when the cache is fully warm.

    Fully warm -> go offline (fast, no Hub round-trips). Otherwise force ONLINE,
    overriding any ``HF_HUB_OFFLINE`` the environment/CI may have set on an incomplete
    cache, so missing datasets can be downloaded instead of hard-failing.
    """
    offline = _cache_is_fully_warm()
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    _set_offline_runtime(offline)
