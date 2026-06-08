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
- Because the wraparound pathology is rare on random data, we expect ~100%
  match. **Any mismatch is triaged, not silently allowed**: it must be shown to
  be a wraparound case where the *new* tree satisfies the train-consistency
  invariant (§5) and the old one does not. (If we want zero mismatches in Phase
  A, the alternative is to have the new code reproduce the legacy midpoint
  selection exactly and defer *all* behavior change to Phase B — heavier, but
  available if you want truly bitwise parity per-commit.)

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

### Lifecycle of the legacy oracle (answering "move to tests or drop?")
**Both, sequenced.** Keep the frozen legacy module + parity grid *during* the
migration. Once Phase B has merged and baked (a release or so), **delete** the
legacy oracle and the pure-equivalence tests — a frozen copy of internal code
is maintenance rot. But first distill the guarantees worth keeping into
**oracle-free** tests so nothing is lost:
- **Golden/characterization tests**: snapshot the new model's predictions on a
  fixed `(seed, signature)` set (inline arrays or a tiny committed fixture).
- **Permanent invariants** (§5), including sklearn-as-oracle on the Euclidean
  case — that one stays forever because sklearn isn't going anywhere.

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
6. Cleanup: remove dead helpers; after bake, remove legacy oracle + pure-parity
   tests, keep golden + invariants.

## 7. Open questions
- Keep `n_features="d_choose_2"`, or fold into "just more features"? (Lean keep.)
- Tie handling for duplicate angles in the window edges (define `<` vs `≤`
  consistently between fit and inference).
- Vectorize across features in torch (preferred) vs per-feature loop.
- Do we want strictly-bitwise Phase A (reproduce legacy midpoint) or the
  randomized-parity-with-triage approach above? (Lean the latter.)

## 8. Risks
- `searchsorted`/doubled-array indexing off-by-one at the wrap — covered by the
  invariant + parity tests.
- Float ties producing different sort orders than legacy → benign prediction
  differences; parity test tolerances/triage handle it.
- RF determinism: keep the exact `random_state` seeding/order so ensembles stay
  reproducible.
