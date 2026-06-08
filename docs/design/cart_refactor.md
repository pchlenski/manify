# Design Doc: Product-Space Decision Tree — CART refactor

Status: **Draft** · Branch: `refactor/cart` (stacked on `fix/review-bugs`)

## 1. Motivation

`ProductSpaceDT` / `ProductSpaceRF` currently carry two parallel split-finding
implementations and an O(n²)-memory data structure that is both a scaling
ceiling and the root cause of a correctness bug.

Specific problems in `manify/predictors/decision_tree.py`:

1. **The `comparisons` tensor.** `_preprocess` materializes
   `comparisons` of shape `(n, d, n)` (all-pairs angular-greater), re-sliced on
   every split in `_get_split`. That is O(n²d) memory — worse than a sorted-CART
   approach for the large-n case it was meant to accelerate. The
   `batched`/`batch_size` toggle exists only to paper over this.
2. **Two code paths.** `_get_info_gains` (matmul over the comparisons tensor)
   vs `_get_info_gains_nobatch` (a Python `for d: for j:` double loop, O(n²d)
   *compute*). The no-batch path doesn't even implement MSE
   (`raise NotImplementedError`), so regression silently requires the batched
   path.
3. **Train/inference mismatch (the bug).** The training partition is defined by a
   *data point's* angle `θ_pos` (`comparisons[n,d]`), but the node stores and
   routes inference by the *midpoint* `m`. They agree only via a margin argument
   that breaks at the `atan2` ±π branch cut, because both the nearest-opposing
   selection (`_get_best_split`, numerical `.abs().argmin()`) and
   `spherical_midpoint` (`(u+v)/2`) treat a circular quantity numerically.

This is the "bug (4)" from the code review, plus the angular half of the
`arsin_k` family of issues. We want to dissolve it structurally, not patch it.

## 2. Goals / Non-goals

**Goals**
- One split-finding path. Delete `comparisons`, `_get_info_gains`,
  `_get_info_gains_nobatch`, `_get_split`, the matrix form of `_angular_greater`,
  and the `batched`/`batch_size` machinery.
- O(F·n log n) per node (sort + circular sliding window), no n² tensor.
- Classification (Gini) **and** regression (MSE) in the single path.
- Make the stored threshold == the evaluated threshold (kills the mismatch).
- Keep the public API, RF subsampling/`permutations`, `random_state`,
  the angular feature construction (`_preprocess`, `d` / `d_choose_2`), and the
  H/E/S `midpoint` geometry.

**Non-goals**
- No change to the product-manifold geometry or the feature-engineering scheme.
- Not adding new hyperparameters (beyond what is needed for parity).
- Not touching other predictors.

## 3. Target algorithm

### 3.1 Feature construction (unchanged)
`_preprocess(X, y)` still produces `angles ∈ (-π, π]` of shape `(n, F)`, one-hot
`labels` (classification) or raw `labels` (regression), and per-feature metadata
`angle2man`, `special_first`, `angle_dims`.

### 3.2 Split rule
A split on feature `f` at angle `θ` sends a point left iff its angle is in the
half-circle `[θ, θ+π) (mod 2π)` — i.e. `_angular_greater(θ, ·)` (kept as a
vector op for partitioning/inference). As `θ` sweeps the circularly-sorted
angles, "in `[θ, θ+π)`" is a **contiguous circular window of arc-length π**.

### 3.3 Fast evaluation (sort + circular two-pointer, vectorized over features)
Per node, for all candidate features at once:

1. `order = argsort(angles_node, dim=0)` → `sorted_ang (m, F)` (m = node size).
2. Build the circularly-extended array `ext = cat([sorted_ang, sorted_ang + 2π])`
   along the sample axis, and `right[i] = searchsorted(ext_f, sorted_ang[i] + π)`
   per feature (vectorized; no Python inner loop). The left-child window for
   candidate `i` is `[i, right[i])` in the doubled index space.
3. Gather labels into sorted order; take cumulative sums (class counts for Gini;
   sum and sum-of-squares for MSE). Window aggregates are `cum[right] - cum[i]`,
   giving impurity for every candidate in O(m) per feature.
4. Apply `min_samples_leaf` / `min_impurity_decrease`; pick the best
   `(feature f*, candidate i*)` by impurity decrease.

### 3.4 Threshold = stored threshold (the fix)
The chosen boundary sits in the **gap between two circularly-adjacent sorted
angles** `sorted_ang[i*-1]` and `sorted_ang[i*]`; the stored threshold is the
geometric `midpoint(...)` of *that adjacent pair* (H/E/S, honoring
`special_first`). Partition the node with `_angular_greater(m*, angles[:, f*])`,
and inference uses the identical call. Train == inference **by construction**, on
the correct arc, with no special-casing of the ±π cut. This removes the
`spherical_midpoint` wraparound issue and the `_get_best_split` numerical
nearest-neighbor issue together.

### 3.5 MSE
Variance reduction from window `sum`/`sumsq` cumulatives — fills the current
regression gap in the (former) no-batch path. Verified against sklearn on a
Euclidean-only signature (see §5).

### 3.6 Random Forest
Unchanged conceptually: bootstrap rows + feature subsample via `permutations`;
each tree calls the new single `_fit_node`. `_generate_subsample` stays.

### 3.7 Complexity
Per node: O(F·m log m) (dominated by the sort), O(F·m) memory. Whole tree:
≈ O(F·n log²n). Versus today's O(n²d) memory + O(n²d) (no-batch) compute.

## 4. Parity strategy (how we stay honest during migration)

The hard part: this refactor **also fixes a bug**, so the new tree cannot be
bitwise-identical to the old one in the wraparound cases — strict parity and the
bugfix are in tension. We resolve it by **splitting the work into two phases** so
"strict parity" holds precisely where it should:

### Phase A — Refactor (parity-gated)
Replace the internals with the sort/two-pointer implementation. During this
phase we enforce **equivalence to the legacy tree** via a frozen oracle:

- Snapshot today's implementation into `tests/_legacy/decision_tree_legacy.py`
  (precedent exists: `tests/legacy/tree_icml.py`). It is pinned, never imported
  by the library.
- `tests/test_cart_parity.py` runs old vs new across a grid of
  `(signature, task ∈ {classification, regression}, seed, n_points, max_depth,
  n_features ∈ {d, d_choose_2})` and asserts:
  - `new.predict(X) == old.predict(X)` exactly, and `predict_proba` `allclose`.
  - For RF: same with a fixed `random_state`.
- **Strictly bitwise (decided).** The new code reproduces legacy's candidate
  selection, tie-breaking (argmax order), and midpoint *exactly* — including the
  spherical `(u+v)/2` and the numerical nearest-neighbour — so the parity grid is
  **100% bit-exact on every signature, every commit**. No mismatches are tolerated
  in Phase A; all behaviour change is deferred to Phase B. This is the strongest
  guard that the data-structure surgery changes nothing we didn't intend.

Phase A lands only when the parity grid is green. This proves the data-structure
change introduces no unintended behavior change.

### Phase B — Bugfix (parity intentionally relaxed, in one diff)
A single focused commit switches the candidate/threshold logic to the circular
midpoints (§3.4). This **deliberately** diverges from legacy on wraparound
cases. In this diff we:
- Flip the affected parity expectations into explicit, documented divergence
  tests (old-vs-new with rationale), and
- Lean on the oracle-free behavioral tests (§5) as the new source of truth.

Net: a bisectable history where every refactor commit is provably equivalent and
the one behavior change is isolated and reviewable.

### Lifecycle of the legacy oracle (decided)
The frozen legacy implementation **lives permanently in `tests/legacy/`** and is
the oracle for a **permanent bit-exact parity test on noncircular signatures** —
Euclidean- and Hyperbolic-only products, which never exercise the spherical
circular-midpoint path. We enforce noncircular bit-exactness *forever*, not just
during migration: it's cheap and it pins the behaviour that should never change.

Only the **circular (spherical) parity** changes across phases:
- During Phase A it is bit-exact (new == legacy everywhere).
- In Phase B the spherical behaviour is deliberately fixed, so the spherical
  parity expectations are removed and replaced by the oracle-free invariants
  (§5). The noncircular (E/H) parity stays green throughout and afterward.

> Note: a hyperbolic/Euclidean component's *secondary* angles still route through
> the catch-all `spherical_midpoint`. "Noncircular" here means signatures with **no
> spherical factor**; if a secondary-dim midpoint ever causes an E/H parity break,
> that divergence is treated as part of Phase B and documented.

Also keep **golden/characterization** snapshots of the new model's predictions
for a fixed `(seed, signature)` set as defence against future drift.

## 5. Test plan (durable, oracle-free)
- **Train-consistency invariant**: a full-depth `ProductSpaceDT` reproduces its
  own training partition → `train_acc == 1.0` on spherical data (incl. points
  straddling the ±π cut). This is the bug-(4) regression guard.
- **sklearn parity (Euclidean)**: on a `[(0.0, k)]` signature the angular CART
  degenerates to ordinary sorted-threshold CART; assert split/accuracy parity
  with `sklearn.tree.DecisionTreeClassifier` / `DecisionTreeRegressor`.
- **Golden tests**: fixed-seed prediction snapshots.
- **MSE correctness**: regression `score`/splits sane vs sklearn.
- **Perf/memory smoke**: peak memory and wall-clock on the loader datasets to
  demonstrate the O(n²)→O(n log n) memory win and no accuracy regression.

## 6. Migration steps (diff sequence)
1. (test-only) Add frozen legacy oracle + `test_cart_parity.py` + invariants.
2. New `_fit_node` (sort/two-pointer, Gini) behind the existing API; delete
   `comparisons`/batched paths once parity passes. **Phase A, parity green.**
3. Add MSE to the single path; extend parity to regression.
4. RF onto the new `_fit_node`; parity for ensembles.
5. **Phase B**: circular midpoints; flip wraparound expectations; invariants
   become source of truth.
6. Cleanup: remove dead helpers. Keep the legacy oracle + the **noncircular**
   bit-exact parity test permanently; only the spherical pure-parity tests are
   retired in Phase B (replaced by invariants + golden tests).

## 7. Decisions (resolved)
- **`n_features="d_choose_2"`: keep.**
- **Tie-handling: sklearn-style, consistent between fit and inference.** Phase A
  must match legacy's tie-breaking bit-exactly, so if the sklearn-style rule
  diverges from legacy it is applied as the **final diff(s)**, after spherical
  pure-parity is gone. Preferred: pick a tie rule that keeps E/H bit-exact so the
  permanent noncircular test stays strict; if not possible, that step is explicit,
  documented, and downgrades the permanent test to "bit-exact modulo tie order".
- **Vectorize across features** (torch `argsort`/`cumsum`/`searchsorted`), gated
  on the §8 benchmark proving it's faster across most input sizes.
- **Phase A is strictly bitwise** (see §4).

## 8. Performance benchmark (gate for the vectorized path)
Before deleting the legacy path, benchmark fit (and predict) wall-clock + peak
memory of new vs legacy across `n_points ∈ {50, 200, 1k, 5k, 20k}`, `n_features`,
`max_depth`, and a few signatures. Require new ≤ legacy for most sizes (especially
large n); record the crossover if small-n regresses. Commit the script + a results
table.

### Results (vectorized info-gain vs legacy O(n²) loop)

Benchmark: `scripts/bench_cart.py`. Compares the live `ProductSpaceDT.fit`
(vectorized info-gain: sort + circular window + cumulative sums) against the
frozen legacy `ProductSpaceDT(batch_size=1).fit` (the O(n²) Python double loop),
`max_depth=4`, classification, `n_features="d"`, 3 classes. Each fit runs in a
fresh subprocess; peak memory is process peak RSS via `ru_maxrss` (the ~650 MiB
floor is the torch/library import baseline, identical for both paths — at these
sizes neither path allocates a large tensor on the nobatch path, so the win is
all wall-clock). The legacy O(n²) loop is skipped past n=5000 (impractically
slow). Measured on CPU.

| signature | n_points | vec fit (s) | legacy fit (s) | speedup | vec peak (MiB) | legacy peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| E^3 | 50 | 0.009 | 0.037 | 4.2x | 652.8 | 651.3 |
| E^3 | 200 | 0.013 | 0.135 | 10.4x | 651.4 | 651.2 |
| E^3 | 1000 | 0.019 | 0.743 | 38.3x | 652.7 | 652.9 |
| E^3 | 5000 | 0.037 | 5.083 | 136.7x | 656.5 | 657.8 |
| E^3 | 20000 | 0.096 | --- | n/a | 673.4 | --- |
| H^2xS^2 | 50 | 0.010 | 0.041 | 4.2x | 659.5 | 656.6 |
| H^2xS^2 | 200 | 0.016 | 0.176 | 11.3x | 659.3 | 659.2 |
| H^2xS^2 | 1000 | 0.019 | 0.923 | 48.0x | 659.6 | 659.4 |
| H^2xS^2 | 5000 | 0.044 | 7.046 | 159.5x | 661.5 | 662.9 |
| H^2xS^2 | 20000 | 0.113 | --- | n/a | 681.6 | --- |

The vectorized path is faster at **every** size — no small-n crossover (even
n=50 is ~4x faster) — and the gap widens with n (≈137–160x at n=5000), as
expected from O(n log n) vs O(n²). At n=20000 the vectorized fit completes in
~0.1 s while the legacy loop is infeasible. Gate satisfied: vectorized ≤ legacy
for all measured sizes.

## 9. Risks
- `searchsorted`/doubled-array indexing off-by-one at the wrap — covered by the
  invariant + parity tests.
- Reproducing legacy's argmax tie-breaking exactly in the vectorized path (the
  Phase A bitwise requirement) is the main implementation risk; the parity grid
  catches any divergence immediately.
- RF determinism: keep the exact `random_state` seeding/order so ensembles stay
  reproducible.
