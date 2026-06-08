"""Performance benchmark for the vectorized vs legacy O(n^2) CART split-finding.

This is the section-8 gate for the CART refactor (see docs/design/cart_refactor.md):
before relying on the vectorized info-gain path, we demonstrate that it is no
slower than the legacy O(n^2) double-loop path for large n (and record any small-n
crossover), in both wall-clock and peak memory.

* "vectorized" = the live ``manify.predictors.decision_tree.ProductSpaceDT``
  (info-gain computed via sort + circular window + cumulative sums).
* "legacy"     = the frozen ``tests.legacy.decision_tree_legacy.ProductSpaceDT``
  run with ``batch_size=1``, which forces the O(n^2) Python double loop.

Each fit runs in a fresh subprocess so peak memory is measured accurately
(torch-inclusive) via ``ru_maxrss``, which ``tracemalloc`` cannot see.

Run from the worktree root with the worktree on PYTHONPATH:

    PYTHONPATH=$PWD /data/repos/manify/venv/bin/python scripts/bench_cart.py

Emits a markdown results table on stdout.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time

import torch

from manify.manifolds import ProductManifold
from manify.predictors.decision_tree import ProductSpaceDT
from tests.legacy.decision_tree_legacy import ProductSpaceDT as LegacyDT

# Sizes to sweep. 20000 is included but the legacy O(n^2) loop is skipped past a
# cap, where it becomes impractically slow (many minutes).
N_POINTS = [50, 200, 1000, 5000, 20000]
LEGACY_MAX_N = 5000
MAX_DEPTH = 4
SIGNATURES = {
    "E^3": [(0.0, 3)],
    "H^2xS^2": [(-1.0, 2), (1.0, 2)],
}


def _make_data(sig, n, seed=0):
    pm = ProductManifold(signature=sig)
    X, y = pm.gaussian_mixture(num_points=n, num_classes=3, seed=seed, task="classification")
    return pm, X, y


def _worker(sig_name: str, impl: str, n: int) -> tuple[float, float]:
    """Run one fit in a fresh subprocess; return (wall_seconds, peak_RSS_MiB)."""
    out = subprocess.run(
        [sys.executable, __file__, "--worker", sig_name, impl, str(n)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        check=True,
    )
    elapsed, peak_mib = out.stdout.strip().split()
    return float(elapsed), float(peak_mib)


def _run_worker(sig_name: str, impl: str, n: int) -> None:
    """Child entry point: fit one tree, print `elapsed peak_rss_MiB` to stdout."""
    torch.manual_seed(0)
    pm, X, y = _make_data(SIGNATURES[sig_name], n)
    kw = dict(task="classification", max_depth=MAX_DEPTH, n_features="d")
    t0 = time.perf_counter()
    if impl == "vec":
        ProductSpaceDT(pm=pm, **kw).fit(X, y)
    else:
        LegacyDT(pm=pm, batch_size=1, **kw).fit(X, y)
    elapsed = time.perf_counter() - t0
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is KiB on Linux
    print(f"{elapsed} {peak_mib}")


def main() -> None:
    rows = []
    for sig_name in SIGNATURES:
        for n in N_POINTS:
            vec_t, vec_m = _worker(sig_name, "vec", n)
            if n <= LEGACY_MAX_N:
                leg_t, leg_m = _worker(sig_name, "leg", n)
                speedup = f"{leg_t / vec_t:.1f}x"
                leg_ts, leg_ms = f"{leg_t:.3f}", f"{leg_m:.1f}"
            else:
                speedup = "n/a"
                leg_ts, leg_ms = "---", "---"
            rows.append((sig_name, n, f"{vec_t:.3f}", leg_ts, speedup, f"{vec_m:.1f}", leg_ms))
            print(f"done: {sig_name} n={n}", flush=True)

    print()
    print("| signature | n_points | vec fit (s) | legacy fit (s) | speedup | vec peak (MiB) | legacy peak (MiB) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for sig_name, n, vt, lt, su, vm, lm in rows:
        print(f"| {sig_name} | {n} | {vt} | {lt} | {su} | {vm} | {lm} |")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        torch.manual_seed(0)
        main()
