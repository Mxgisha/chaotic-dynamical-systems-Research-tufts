# Chaotic dynamical systems — Research@Tufts

Research into machine-learning emulation of chaotic dynamical systems, focusing on the
**failure modes** of the emulators rather than on trajectory MSE.

The objective is a suite of tests and graphics that analyse whether an emulator preserves the
*dynamical* character of the system it was trained on — its Lyapunov spectrum, invariant measure,
correlation dimension, symbolic dynamics, and attractor topology — and to say how far each has
drifted, with controls that make the numbers interpretable.

---

## Quick start

```bash
pip install -r requirements.txt          # numpy + matplotlib + jax.random (keys) + pytest
python -m pytest -q -m "not slow"        # ~1 minute, CPU only

# a fast end-to-end verification run (one sweep point, a few minutes)
python scripts/run_sweep.py --smoke --out runs

# the full four-point sweep
python scripts/run_sweep.py --config configs/sweep.json --out runs
```

Everything runs under NumPy on a CPU. The JAX/diffrax/equinox stack
([`requirements-jax.txt`](requirements-jax.txt)) is optional and selectable per run:

```bash
python scripts/run_sweep.py --smoke --backend jax --out runs   # equinox + optax trainer
```

The `--backend` flag chooses the *model/training* backend and is independent of
`LORENZ63_BACKEND`, which selects the array namespace for the data and diagnostic
code. Both trainers optimise the same objective; `tests/test_jax_trainer.py`
checks that they agree numerically (identical parameter counts, gradients
matching to 1e-8, weight export exact to 1e-12), which is what stops the two
implementations drifting apart the way the three copies of the old code did.

> **float64 is mandatory.** `lorenz63.backend` and `lorenz63.jax_trainer` set
> `jax_enable_x64` at import time. In float32 the Jacobian products lose
> precision and JAX training diverges to a gradient norm of ~1e6 within a few
> steps instead of converging.

## Layout

| Path | Purpose |
|---|---|
| [`lorenz63/system.py`](lorenz63/system.py) | The ODE, parameters, RK4/Tsit5 integration, analytic Jacobian |
| [`lorenz63/data.py`](lorenz63/data.py) | Dataset construction, manifold sampling, RNG discipline |
| [`lorenz63/models.py`](lorenz63/models.py) | NumPy MLP (run anywhere) and the identity null model |
| [`lorenz63/jax_trainer.py`](lorenz63/jax_trainer.py) | Optional equinox+optax trainer for the same objective |
| [`lorenz63/losses.py`](lorenz63/losses.py) | MSE and a differentiable distributional (L1) term |
| [`lorenz63/train.py`](lorenz63/train.py) | One training loop, one stepper contract, Adam |
| [`lorenz63/controls.py`](lorenz63/controls.py) | GT-vs-GT and identity baselines |
| [`lorenz63/aggregate.py`](lorenz63/aggregate.py) | Replicate mean ± spread and resolvability tests |
| [`lorenz63/diagnostics/`](lorenz63/diagnostics) | Pure measurement functions — **no plotting** |
| [`lorenz63/report.py`](lorenz63/report.py) | The single figure contract, `results.json` |
| [`scripts/run_sweep.py`](scripts/run_sweep.py) | Train → evaluate → emit, end to end |
| [`configs/sweep.json`](configs/sweep.json) | The sweep definition plus a small `smoke` block |
| [`tests/`](tests) | 46 tests, CPU-only |
| [`legacy/`](legacy) | Superseded notebook and helpers, kept for the record |
| [`IMPLEMENTATION_SPEC.md`](IMPLEMENTATION_SPEC.md) | The spec this package implements, with the defect list |

## How each diagnostic is made trustworthy

A diagnostic is a pure function `(gt, emu, ctx) -> (gt_metrics, emu_metrics)`. It never creates a
figure, and it never prints `TEST PASSED`. Pass/fail is decided in
[`lorenz63/report.py`](lorenz63/report.py) by comparison against the controls, because a "PASS"
that an identity map also achieves measures nothing.

**Controls.** Every metric is also computed for (a) a second *true* trajectory from a different
initial condition — the ceiling on achievable agreement — and (b) the identity map
`x_{n+1} = x_n` — the floor. Without those two references, "L1 = 0.03" or "flip ratio 0.667" has
no scale.

**Self-checks.** The Lyapunov spectrum reports both invariants that must hold:
`sum(lambda) == ln|det A| / dt` exactly, and to leading order that equals the constant divergence
`-(sigma + 1 + beta) = -13.667`. It also reports the slope of the running exponent over the last
20% of the window, so a value that has not converged is visible instead of silently quoted.

## Verified ground truth

`tests/test_system.py` and `tests/test_diagnostics.py` check the ground-truth system against the
analytic Lorenz-63 values. From a `--smoke` run (2000-step test trajectory, `dt = 0.01`):

| Quantity | Measured | Reference |
|---|---|---|
| `lambda_1` | 1.086 | 0.906 (textbook) |
| `lambda_2` | 0.035 | 0.0 |
| `lambda_3` | -14.908 | -14.572 (textbook) |
| `sum(lambda)` | -13.788 | -13.667 (the divergence) |
| `trace_invariant` | -13.788 | must equal `sum(lambda)` exactly |
| convergence slope of `lambda_1` | 0.00095 | 0.0 (converged) |
| `d2` (correlation dimension) | 2.26, CI [2.05, 2.65] | ~2.06 |

The sum matching the trace invariant to five decimals, and the convergence slope being ~1e-3,
are what establish that the normalisation is right and the value has plateaued. The test asserts
both invariants, so a future regression in the Jacobian or the QR step fails the suite rather
than silently shifting the numbers.

## What a run looks like

`--smoke` on one core (NumPy backend, 15 epochs) ends with training MSE 63.6 → 0.0082. That model
is deliberately under-fitted, and the diagnostics say so plainly — `lambda_1` at 0.09 of truth
while the invariant measure is within 11%:

```
test_0.05_10  lyapunov_spectrum     lambda_1        0.087
test_0.05_10  symbolic_dynamics     flip_rate       0.333
test_0.05_10  correlation_dimension d2              0.501
test_0.05_10  invariant_measure     energy_mean     1.114
test_0.05_10  dmd_spectrum          spectrum_corr   0.764
```

The full sweep above shows the same signature in properly trained models, with error bars:
**shape statistics pass while time-parameterisation statistics fail.** Cherry-picking the passing
metrics is exactly the failure mode this suite exists to prevent, which is why the controls are
stored alongside every number, `ratios()` refuses to report a ratio when the ground-truth value is
numerically zero, and `resolvable_table.txt` marks which differences survive the seed-to-seed
spread.

## Full sweep results (5 seed replicates per point)

`python scripts/run_sweep.py --config configs/sweep.json --backend numpy --out runs`
(50 epochs, 18 900 training windows, 5000-step evaluation, one core). Every number is
emulator/ground-truth, **mean ± std over 5 seed replicates**, from
[`runs/summary.json`](runs/summary.json). 140/140 figures verified non-blank.

Replicates vary both the sampled initial conditions (`point_key(..., replicate)`) and the model
initialisation, so the spread below is genuine run-to-run variation, not just weight noise.

| metric | 0.05/10 | 0.1/10 | 0.1/20 | 0.4/10 |
|---|---|---|---|---|
| λ₁ | 0.68 ± 0.25 | 0.68 ± 0.25 | −0.15 ± 1.54 | 0.03 ± 0.25 |
| **λ₃** | **2.66 ± 0.34** | **5.14 ± 0.64** | **4.48 ± 0.59** | **20.39 ± 1.83** |
| Σλ | 2.82 ± 0.35 | 5.50 ± 0.70 | 4.99 ± 0.83 | 22.13 ± 1.96 |
| `error_growth` rate | 1.45 ± 0.27 | 0.41 ± 0.14 | 2.48 ± 2.07 | 0.67 ± 1.09 |
| `invariant_measure` energy_mean | 1.01 ± 0.04 | 0.99 ± 0.04 | 0.98 ± 0.16 | 1.08 ± 0.06 |
| `dmd_spectrum` spectrum_corr | 0.98 ± 0.01 | 0.98 ± 0.01 | 0.94 ± 0.01 | 0.83 ± 0.18 |
| `symbolic_dynamics` flip_rate | 0.96 ± 0.38 | 1.13 ± 0.45 | 0.76 ± 0.39 | 0.36 ± 0.63 |
| `correlation_dimension` d2 | 0.84 ± 0.11 | 0.92 ± 0.16 | 0.74 ± 0.35 | 0.49 ± 0.22 |

### Which differences are real?

The sweep also writes [`resolvable_table.txt`](runs/resolvable_table.txt): all 126
pairwise point comparisons, each flagged when the difference exceeds the sum of the two standard
errors. **78 of 126 are resolvable.** Ranking metrics by how many point-pairs they separate:

| metric | point-pairs resolved |
|---|---|
| **λ₃** | **6 / 6** |
| Σλ, `trace_invariant` | 5 / 6 |
| λ₂, `t_sat`, `d2_hi`, `spectrum_corr` | 5 / 6 |
| `growth_rate`, `t_double`, λ₁ | 4 / 6 |
| `energy_mean`, `mean_dwell`, `d2`, `flip_rate` | 2–3 / 6 |

**This substantially revises what a single run suggests.** With error bars attached:

* **The contraction is the finding.** λ₃ separates *every* pair of sweep points, and Σλ separates
  five of six. The emulated attractor is 2.7–20x too dissipative, consistently and beyond
  seed noise. This is the one unambiguous, reproducible failure mode.
* **The "shape" metrics are stable but insensitive.** `energy_mean` sits within 8% of truth at
  every noise level, and `spectrum_corr` stays ≥ 0.83. But they resolve only 2–3 of 6 pairs, so
  they are poor discriminators — they say "the attractor is broadly right", not "this emulator
  is better than that one".
* **λ₁ and the lobe-flip rate are dominated by seed noise.** The earlier single-run claim that
  λ₁ was off by 13x was selecting one replicate: the 5-seed mean is **0.68 ± 0.25**, which is
  consistent with recovering λ₁ to within ~30%. `flip_rate` (0.96 ± 0.38 at the best point)
  likewise cannot support a "3x wrong" claim. Those specific numbers from the previous
  single-run analysis are **retracted**.
* **Noise 0.4 is a genuine collapse, not noise.** 18 of 21 metrics resolve against 0.1/10,
  including `mean_dwell` at 19.7 ± 11.0 against 1.5 ± 1.5 — the emulator stops switching lobes.
  Its one-step MSE is still only 0.112, so the headline conclusion stands: **low one-step MSE
  does not imply correct dynamics.**
* **Horizon 20 vs horizon 10 at the same noise is barely distinguishable** (8 of 21 metrics),
  so horizon is not a strong lever over this range; noise is.

`error_growth` is an independent estimator built from twin trajectories, and it agrees with the
QR spectrum on the well-behaved points — a useful cross-check of the Lyapunov code rather than a
second opinion from the same numbers.

## What the sweep produces

Each sweep point writes `runs/test_<noise>_<horizon>/replicate<N>/` — one directory **per seed
replicate**, always, so the on-disk layout does not change shape between a one-replicate run and a
full sweep. Each replicate directory contains `model.npz`, `metadata.json`, `history.json`,
`results.json`, and one PNG per diagnostic: `lyapunov_spectrum`, `invariant_measure`,
`correlation_dimension`, `symbolic_dynamics`, `dmd_spectrum`, `error_growth`,
`vector_field_slice`.

At the top level:

| file | contents |
|---|---|
| `results_all.json`, `results_table.txt` | first replicate of every point (single-run view) |
| `summary.json`, `summary_table.txt` | **mean ± std over replicates**, per metric |
| `resolvable.json`, `resolvable_table.txt` | all pairwise point comparisons with a verdict |
| `sweep.log` | full stdout of the run |

`emit()` **saves before showing** and then re-decodes every written PNG to assert it is not blank
(see `is_blank_png`). That check exists because 28 of the 32 PNGs previously committed to
`test/` were fully transparent 640x480 canvases containing no content at all; the regression test
`tests/test_end_to_end.py::test_smoke_sweep_produces_non_blank_artifacts` would have caught them.

## Notes on the headline failure modes

The pointwise trajectory error saturates and is not a skill measure: with `lambda_1 ~ 0.9` the
error grows e-folded ~65x over 50 time units, so after roughly one Lyapunov time the two
trajectories are uncorrelated and the error simply fluctuates at the size of the attractor. The
informative diagnostics are the structural ones, and among those the **contraction** (λ₃, Σλ) is
the only one that separates every pair of sweep points beyond seed noise. The lobe-flip rate and
λ₁ look dramatic in a single run but are dominated by run-to-run variation at 5 replicates.

Run `python scripts/run_sweep.py --smoke --replicates 2 --out runs` for a quick check, or the full
command at the top for the committed numbers.

## Reproducing the replicate study

```bash
python scripts/run_sweep.py --config configs/sweep.json --backend numpy --out runs
# or override the config's "replicates": 5
python scripts/run_sweep.py --replicates 5 --out runs
```

The run is deterministic: each replicate derives its data from
`point_key(root_seed=67, noise_scale, horizon, replicate)` and its weights from
`model.seed + replicate`. Re-running reproduces the committed numbers exactly.
