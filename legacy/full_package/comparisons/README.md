# Comparison: map vs vector-field parameterisation

A head-to-head test of whether learning the **vector field** `dx/dt = g(x)` and
integrating it controls the emulator's volume contraction better than learning
the **one-step map** `x_{n+1} = f(x_n)`.

The motivation was structural. The map emulator's Jacobian is whatever the fit
produces, and `local_contraction` measured it over-contracting by up to 6.4x in a
tail of states while the true Lorenz-63 divergence is *constant*. A field
emulator's flow map is a composition of RK4 stages, so its volume change is tied
to `div g` rather than being free — so the hypothesis was that the field
parameterisation would inherit the right contraction.

**The hypothesis is falsified, and the reason is instructive.**

## 1. The field parameterisation alone does *not* fix the divergence

Trained on one-step increments with identical architecture, optimiser, data and
budget, the field emulator fits the increments *better* than the map (training
MSE 0.0023 against 0.0029 at the smoke size) while learning a field with a
divergence of **−2 to −5 instead of −13.67**.

The reason is a scaling argument. The divergence enters the one-step flow map only
at `O(dt^2)`:

    phi_dt(x) = x + dt g(x) + (dt^2/2) J g(x) + ...

so at `dt = 0.01` a field with several times the true divergence still reproduces
one-step increments to within the fitting tolerance. **The training signal simply
does not see the divergence.** The field parameterisation does not make the
emulator's contraction structural; it only relocates where the freedom lives.

## 2. Adding the known divergence as a penalty fixes it

Lorenz-63's divergence is the known constant `-(sigma + 1 + beta) = -13.667`, so
it can be added as a physics-informed penalty. Scanning its weight on a reduced
dataset (15 epochs, 2 replicates):

| weight | train MSE | contraction ratio | learned field divergence | divergence spread |
|---|---|---|---|---|
| 0 (none) | 0.218 | 0.159 | −2.19 | 3.74 |
| 0.003 | 0.231 | 0.501 | −6.90 | 0.60 |
| 0.01 | 0.250 | 0.531 | −7.30 | 0.40 |
| 0.1 | 0.353 | 0.665 | −9.15 | 0.60 |
| 0.3 | 0.437 | 0.826 | −11.36 | 0.92 |
| 0.6 | 0.532 | 0.958 | −13.18 | 1.06 |
| **1.0** | **0.610** | **0.987** | **−13.58** | **0.82** |
| 2.0 | 0.701 | 0.984 | −13.54 | 0.56 |
| 4.0 | 0.804 | 0.994 | −13.68 | 0.43 |

Three things to read off this:

1. **The penalty works, monotonically and controllably.** At weight 1.0 the field
   reproduces the true divergence to within 1% (−13.58 against −13.67) and the
   contraction ratio is 0.99 — where the map emulator over-contracts by 2–5x.
2. **It also tightens the spread.** The un-penalised field's divergence spread
   across states is 3.7; with the penalty it falls to 0.4–0.8. The emulator stops
   inventing state-dependence.
3. **The cost is real but modest.** Training MSE rises from 0.22 to 0.61 on the
   reduced dataset — a price in one-step accuracy for correct volume behaviour,
   and a dial that can be set explicitly rather than left to chance.

## 3. The map parameterisation has no equivalent lever

The asymmetry is the point. A diagnostic can measure `ln|det Df|` for a map
emulator, but there is no local quantity whose penalty would control it — the
volume change is an emergent property of the whole network. For the field, the
divergence is a cheap first derivative of the output with respect to the input,
so it can be measured and penalised directly.

That is a genuine and transferable conclusion: **the field parameterisation is
better not because it imposes the structure for free, but because it exposes the
structure to the loss.**

## Reproducing

```bash
# the weight scan
python scripts/scan_divergence_weight.py --weights 0 0.003 0.01 0.03 0.1 0.3

# the three-arm comparison (map, field, field + divergence penalty)
python scripts/compare_parameterisations.py --out comparisons/map_vs_field --replicates 2
```

| file | contents |
|---|---|
| `comparison.txt`, `comparison.json` | per-metric map vs field vs field+penalty |
| `map/summary.json`, `field/…`, `field_div/…` | replicate statistics per arm |
| `<arm>/test_<noise>_<horizon>/replicate<N>/` | models, histories, `results.json`, figures |

## Caveats

* The scan above uses a reduced dataset and 15 epochs to stay quick; the absolute
  MSE values are larger than the full-configuration numbers. The *trend* is what
  it establishes, and the full comparison re-runs at the committed configuration.
* `field_div` reaches divergence fidelity by construction on a *known* constant
  divergence. For a system whose divergence is unknown — or genuinely
  state-dependent — this penalty is not available, and the field parameterisation
  is back to being weakly constrained.
* The comparison is 2 replicates per point, so the resolvability screen on it is
  weaker than the 5-replicate study in `../runs`.
