# Sweep artifacts

Produced by the five-replicate four-point sweep with figures pruned:

```bash
python scripts/run_sweep.py --config configs/sweep.json --backend numpy \
    --out runs --prune-figures
```

Deterministic: each replicate derives its data from
`point_key(root_seed=67, noise_scale, horizon, replicate)` and its weights from
`model.seed + replicate`, so re-running reproduces these numbers exactly.

| Setting | Value |
|---|---|
| Replicates | 5 per sweep point (20 runs) |
| Model backend | NumPy |
| Array backend | NumPy (`LORENZ63_BACKEND=numpy`) |
| Epochs / batch | 50 / 128 |
| Training windows | 18 900 (17 900 for horizon 20) |
| Evaluation | 5000 steps, `dt = 0.01`, burn-in 1000 |
| Sampler | `manifold` (24 short trajectories from manifold points) |
| Loss | `mse + 0.1 * L1` (FTLE term disabled) |
| Diagnostics | 8, including `local_contraction` |

## Layout

Each `test_<noise>_<horizon>/replicate<N>/` contains:

| File | Contents |
|---|---|
| `model.npz`, `model.json` | Trained MLP weights and architecture |
| `metadata.json` | Sweep point, replicate, seeds, shapes, test IC, parameter count |
| `history.json` | Per-epoch total / mse / l1 |
| `results.json` | Every diagnostic's `gt`, `emu`, `control_gt_gt`, `ratios`, `meta` |
| 8 `*.png` (replicate 0 only) | One figure per diagnostic, verified non-blank at write time. Figures for replicates 1-4 were pruned as near-duplicates: within a point the ground truth is identical, so they differ only in the emulator overlay. All numeric results are retained. |

At the top level:

| File | Contents |
|---|---|
| `summary.json`, `summary_table.txt` | mean ± std over the 5 replicates, per metric |
| `resolvable.json`, `resolvable_table.txt` | all 198 pairwise point comparisons + verdict |
| `results_all.json`, `results_table.txt` | first replicate only (single-run view) |
| `sweep.log` | full stdout, including training curves |

## Headline numbers (emulator/ground-truth, mean ± std over 5 seeds)

| metric | 0.05/10 | 0.1/10 | 0.1/20 | 0.4/10 |
|---|---|---|---|---|
| λ₁ | 0.68 ± 0.25 | 0.68 ± 0.25 | −0.15 ± 1.54 | 0.03 ± 0.25 |
| **λ₃** | **2.66 ± 0.34** | **5.14 ± 0.64** | **4.48 ± 0.59** | **20.39 ± 1.83** |
| Σλ | 2.82 ± 0.35 | 5.50 ± 0.70 | 4.99 ± 0.83 | 22.13 ± 1.96 |
| `error_growth` rate | 1.45 ± 0.27 | 0.41 ± 0.14 | 2.48 ± 2.07 | 0.67 ± 1.09 |
| `energy_mean` | 1.01 ± 0.04 | 0.99 ± 0.04 | 0.98 ± 0.16 | 1.08 ± 0.06 |
| `spectrum_corr` | 0.98 ± 0.01 | 0.98 ± 0.01 | 0.94 ± 0.01 | 0.83 ± 0.18 |
| `flip_rate` | 0.96 ± 0.38 | 1.13 ± 0.45 | 0.76 ± 0.39 | 0.36 ± 0.63 |
| `d2` | 0.84 ± 0.11 | 0.92 ± 0.16 | 0.74 ± 0.35 | 0.49 ± 0.22 |
| **`local_contraction` ratio** | **2.94 ± 0.26** | **5.24 ± 0.56** | **4.84 ± 0.40** | **22.0 ± 0.34** |
| `local_contraction` p95 ratio | 1.16 ± 0.09 | 1.42 ± 0.18 | 1.51 ± 0.04 | 5.78 ± 0.47 |
| `local_contraction` p05 ratio | 6.42 ± 0.82 | 13.8 ± 3.0 | 11.1 ± 1.3 | 38.5 ± 0.9 |

Ground-truth Lyapunov spectra are stable to three decimals across all points, and Σλ matches the
divergence `-(sigma + 1 + beta) = -13.667` every time.

## Local contraction

Lorenz-63 has constant divergence, so the true flow map contracts volume by exactly
`-(sigma+1+beta)*dt` at every state. `local_contraction` measures `ln|det Dphi|` per state for both
maps at the same states, so any state-dependence in the emulator is pure model error. The
finite-difference Jacobian used for the emulator agrees with an exact analytic MLP Jacobian to
~5e-9, so these are not numerical artefacts.

* The over-contraction is a **tail** phenomenon: at noise 0.05 the bulk contracts 1.16x too fast
  while the worst 5% contracts 6.4x too fast. Fixing the mean would not fix the failure.
* The emulator's contraction has a spread of 62 (up to 378) per unit time against a true spread of
  ~0.4 (the RK4 discretisation floor). It invents state-dependence the system does not have.
* Stratified by position the excess is ~-8 near the saddle (`|x|<2`) and ~-40 in the mid-lobes
  (`6<=|x|<10`), and falls monotonically with altitude `z`.
* The correlation between true and emulated leading expansion is only 0.52 at noise 0.05, rising
  to 0.94 at 0.4 where the model has collapsed toward a compressible fixed point.

## Resolvability

137 of 198 pairwise point comparisons exceed the sum of the two standard errors (78 of 126 before
this diagnostic was added). Ranked by how many point-pairs a metric separates:

| metric | pairs resolved |
|---|---|
| **λ₃, `divergence_error`, `contraction_rate_std`, `contraction_rate_p05`** | **6 / 6** |
| Σλ, `trace_invariant` | 5 / 6 |
| λ₂, `t_sat`, `d2_hi`, `spectrum_corr`, `contraction_rate_mean` | 5 / 6 |
| `growth_rate`, `t_double`, λ₁ | 4 / 6 |
| `energy_mean`, `mean_dwell`, `d2`, `flip_rate` | 2–3 / 6 |

**Conclusions.**

1. **The contraction is the reproducible finding**, and `local_contraction` localises it: the error
   is concentrated in a tail of states and in the mid-lobes.
2. **Shape metrics are stable but insensitive** — they resolve only 2–3 of 6 pairs.
3. **λ₁ and `flip_rate` are dominated by seed noise.** The single-run claim that λ₁ was off by 13x
   is retracted: the 5-seed mean is 0.68 ± 0.25, consistent with recovering λ₁ to within ~30%.
4. **Noise 0.4 is a real collapse** — 18 of 21 metrics resolve against 0.1/10, while one-step MSE
   is still 0.112.
5. **Horizon 10 vs 20 at equal noise is barely distinguishable** (8 of 21 metrics).

## Caveat

Five replicates give a rough standard deviation (≈35% relative uncertainty on the spread itself),
and the resolvability screen is a crude ~1-sigma test comparing `std / sqrt(n)` — it is a screen,
not a hypothesis test with a stated false-positive rate. Treat "within noise" as "not
distinguishable by this study", not as evidence of equivalence.
