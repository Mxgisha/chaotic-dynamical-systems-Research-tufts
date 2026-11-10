# Steps 1 and 2: making the divergence learnable, or proving it is not

Two experiments that follow the open question in [`PAPER.md`](../PAPER.md) §5.4:
the repair works, but only because the divergence `−13.667` was computed
analytically. Can the same result be reached without being handed the answer?

Both experiments are small on purpose. They are pilots: two seeds, one system, a
4.6k-parameter model. The point is to find out which question is the right one
before spending compute on it.

```bash
python experiments/step1_scaling.py
python experiments/step2_data_divergence.py
```

Results are written to `results/*.json`, figures to `figures/`.

---

## Step 1 — does the timestep control the divergence error?

**Hypothesis.** The volume-change rate is the log-determinant of the one-step
Jacobian divided by `Δt`. An error `ε` in that Jacobian therefore produces an
error `ε/Δt` in the learned rate, so the relative divergence error should scale
like **`Δt⁻¹`**.

**Design.** Train a map and a field emulator at `Δt = 0.01, 0.02, 0.05`, holding
the data quantity at ~2,500 pairs and the model at 33k parameters / 80 epochs.
Measure the learned rate against the exact rate of the reference RK4 map at that
`Δt` (not against the ODE divergence — see below).

**The reference has to be checked first.** The RK4 step map's own volume-change
rate is only equal to `−13.667` while the integrator is accurate:

| Δt | step-map rate | drift from the ODE divergence |
|---|---|---|
| 0.01 | −13.667 | 0.0% |
| 0.02 | −13.665 | 0.0% |
| 0.05 | −13.569 | 0.7% |
| 0.10 | −10.847 | **20.6%** |

At `Δt = 0.1` the "ground truth" is not the Lorenz system any more, so that
timestep is excluded from the scaling fit. This bounds the testable range: you
cannot push `Δt` up without changing the integrator.

![Scaling](figures/step1_scaling.png)

**Result: the hypothesis fails.** Fitted log-log slopes are **+0.31 (map)** and
**+0.05 (field)**, not `−1`.

| Δt | map error | field error | map learned rate (2 seeds) |
|---|---|---|---|
| 0.01 | 37.9% | 77.9% | −16.6, −21.1 |
| 0.02 | 14.4% | 81.6% | −10.2, −13.2 |
| 0.05 | 56.0% | 85.0% | −6.2, −5.8 |

Two things are more interesting than the failed fit:

1. **The map's divergence is a lottery, not a bias.** Across seeds and timesteps
   the learned rate ranges from **−21.1 to −5.8** against a true `−13.67`. One
   run lands within 3.5%; the next is 56% off. There is no systematic error to
   correct — the objective simply does not constrain the quantity.
2. **The field is systematically wrong in one direction.** Every field run
   under-contracts, landing between **−1.4 and −3.7** (10–27% of the true rate),
   at every timestep. That is the paper's "the field alone does not help",
   reproduced, and it is a *scale* problem: the field is free to be small
   because a small field fits the increments almost as well.

**Conclusion.** The `1/Δt` law is a statement about a lower bound, not about
what training does in practice. In this regime the divergence error is set by
optimisation, not by the timestep. Testing the law properly requires matching
the fit quality across `Δt`, which is the design lesson for anyone repeating it.

---

## Step 2 — can the divergence be estimated from data?

**Design.** Estimate the local Jacobian from `(x_n, x_{n+1})` pairs by a linear
fit over the `k` nearest neighbours, and take `ln|det Ĵ|/Δt`. Then use that
per-state estimate as the penalty target instead of `−13.667`.

![Estimator and repair](figures/step2_data_divergence.png)

### The estimator depends entirely on which data it sees

| neighbourhood cloud | clean data (k=10 → 200) | noise 0.4 |
|---|---|---|
| on-attractor pairs | 0.53 → 0.45 | 39× → 29× |
| transient pairs (off-attractor) | 0.83 → **0.95** | 27× → 12× |

**This is the finding.** From pairs sampled on the attractor, the divergence is
**not identifiable**: the estimate recovers less than half the true rate at every
neighbourhood size. From transients — states still approaching the attractor —
it recovers 83–95%.

The reason is structural rather than numerical. The divergence is dominated by
the strongly contracting direction, `λ₃ ≈ −14.5` out of a total `−13.67`. A
cloud of points on the attractor is locally a two-dimensional sheet, so a local
fit never sees how fast the *transverse* direction contracts. The transverse
derivative is not in the data.

This sharpens the paper's central claim. It is not only that the *loss* cannot
see the divergence; for on-attractor training data, the divergence is **not
determined by the data at all**. The model is free to invent it because the
information is absent, not merely under-weighted.

### A bad target is worse than no target

Repair at input noise 0.4, all measured on the same probe states:

| arm | learned / true contraction |
|---|---|
| map, no penalty | 8.97 |
| field, no penalty | 0.69 |
| field + **analytic** target | **0.998** |
| field + estimate from **transient** pairs | **0.930** |
| field + estimate from **on-attractor** pairs | 0.498 |

The transient estimate has a mean absolute error of 10.3% and drives the repair
to 0.93 — close enough to the analytic result to call the circularity broken, in
a qualified way. The attractor estimate has 52.3% error and makes the model
*worse than leaving it alone* (0.50 vs 0.69). Penalising toward a wrong invariant
is actively harmful.

Under observation noise the estimator collapses (10–27× at `ε = 0.4`): local
linear regression is destroyed by noise in the inputs. So the qualified claim is:
**the repair can be driven from data when the data are clean and include
off-attractor states.**

---

## What this changes for the next step

1. **The open question is data-side, not architecture-side.** "Estimate the
   invariant from data" turns out to be "get data that identifies it". For a
   simulator that is easy; for observations it is the whole problem.
2. **On-attractor-only training is underdetermined.** Any method that claims to
   learn contraction from a long trajectory on the attractor should say how it
   breaks the transverse identifiability — otherwise it is guessing.
3. **The noise failure is the next real obstacle.** The estimator needs an
   errors-in-variables treatment (total least squares, or denoising first), and
   that is a well-posed, self-contained experiment.

## What would falsify this

* A local estimator that recovers the divergence from on-attractor data alone
  would break finding (2). We have not found one, but only one family was tried.
* If a *corrected* estimator (total least squares) recovers the rate at
  `ε = 0.4`, the noise failure is a fixable bias rather than a limit.
* If the same experiment on Rössler behaves differently, the identifiability
  argument is specific to constant-divergence systems, not general.

## Caveats

Two seeds, one system, one estimator family, a 4.6k-parameter model, and a
resolvability test that has not been applied here. These are pilots. The
headline numbers (0.93 vs 0.998, 0.45 vs 0.95) are far larger than the seed
spread in step 1, but they should be repeated with five seeds before being
quoted.
