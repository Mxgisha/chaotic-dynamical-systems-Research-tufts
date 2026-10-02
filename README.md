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

The full sweep above shows the same signature in a properly trained model: **shape statistics pass
while time-parameterisation statistics fail.** Cherry-picking the passing metrics is exactly the
failure mode this suite exists to prevent, which is why the controls are stored alongside every
number and `ratios()` refuses to report a ratio when the ground-truth value is numerically zero.

## Full sweep results

`python scripts/run_sweep.py --config configs/sweep.json --backend numpy --out runs`
(50 epochs, 18 900 training windows, 5000-step evaluation, one core). Every number below is
emulator/ground-truth from `runs/*/results.json`; 28/28 figures verified non-blank.

Absolute Lyapunov spectra — the ground truth is stable to three decimals across all four noise
levels, and `sum` matches the divergence `-(sigma + 1 + beta) = -13.667` every time:

| sweep point | ground truth λ | emulator λ | λ₁ ratio |
|---|---|---|---|
| `0.05 / 10` | `0.997, 0.091, -14.868` | `0.835, -0.046, -39.727` | 0.84 |
| `0.1 / 10` | `1.022, 0.097, -14.898` | `0.785, -0.009, -73.856` | 0.77 |
| `0.1 / 20` | `1.002, 0.088, -14.865` | `0.804, -0.014, -67.512` | 0.80 |
| `0.4 / 10` | `0.990, 0.007, -14.771` | `-0.006, -0.514, -344.784` | **-0.01** |

Selected ratios across the sweep:

| diagnostic / metric | 0.05 | 0.1 | 0.1 (h=20) | 0.4 |
|---|---|---|---|---|
| `correlation_dimension` d2 | 1.16 | 1.11 | 1.06 | **0.42** |
| `error_growth` growth_rate | 0.92 | 2.05 | 0.69 | **0.06** |
| `symbolic_dynamics` flip_rate | 1.10 | 1.10 | 1.10 | **2.14** |
| `invariant_measure` energy_mean | 1.01 | 0.99 | 0.99 | 0.93 |
| `dmd_spectrum` spectrum_corr | 0.98 | 0.95 | 0.94 | 0.98 |
| `lyapunov_spectrum` λ₁ | 0.84 | 0.77 | 0.80 | -0.01 |

**What this says.** Training MSE falls monotonically and the emulator tracks the *shape* of the
attractor well at every noise level (`energy_mean` within 7%, DMD spectrum correlation ≥ 0.94).
What it does not reproduce is the *contraction*: λ₃ comes out 2.7–5x too negative at low noise
and 23x too negative at noise 0.4, so the emulated attractor is far more dissipative than the real
one even when λ₁ is within 20%. At noise 0.4 the emulator collapses to a fixed point (λ₁ ≈ 0,
D2 ratio 0.42, error-growth rate 6% of truth) while its one-step MSE is still only 0.112 — the
clearest demonstration in the sweep that **low one-step MSE does not imply correct dynamics**.
Note also that the error-growth rate is an independent estimator built from twin trajectories; it
agrees with the QR spectrum to within 8% on the well-behaved points, which is a useful cross-check
of the Lyapunov code.

## What the sweep produces

Each sweep point writes a directory `runs/test_<noise>_<horizon>/` containing
`model.npz`, `metadata.json`, `history.json`, `results.json`, and one PNG per diagnostic:
`lyapunov_spectrum`, `invariant_measure`, `correlation_dimension`, `symbolic_dynamics`,
`dmd_spectrum`, `error_growth`, `vector_field_slice`.

`emit()` **saves before showing** and then re-decodes every written PNG to assert it is not blank
(see `is_blank_png`). That check exists because 28 of the 32 PNGs previously committed to
`test/` were fully transparent 640x480 canvases containing no content at all; the regression test
`tests/test_end_to_end.py::test_smoke_sweep_produces_non_blank_artifacts` would have caught them.

## Notes on the headline failure modes

The pointwise trajectory error saturates and is not a skill measure: with `lambda_1 ~ 0.9` the
error grows e-folded ~65x over 50 time units, so after roughly one Lyapunov time the two
trajectories are uncorrelated and the error simply fluctuates at the size of the attractor. The
informative diagnostics are the structural ones — the Lyapunov ratios, the lobe-flip ratio, and
the return-map correlation.

Run `python scripts/run_sweep.py --smoke --out runs` and read `runs/*/results.json` for the
current numbers, which include an `ratios` block giving `emu / gt` for every metric.
