# Sweep artifacts

Produced by the full four-point sweep:

```bash
python scripts/run_sweep.py --config configs/sweep.json --backend numpy --out runs
```

Reproduce exactly with the committed [`configs/sweep.json`](../configs/sweep.json) — the run is
deterministic, since every sweep point derives its data from
`point_key(root_seed=67, noise_scale, horizon)`.

| Setting | Value |
|---|---|
| Model backend | NumPy |
| Array backend | NumPy (`LORENZ63_BACKEND=numpy`) |
| Epochs / batch | 50 / 128 |
| Training windows | 18 900 (17 900 for horizon 20) |
| Evaluation | 5000 steps, `dt = 0.01`, burn-in 1000 |
| Sampler | `manifold` (24 short trajectories from manifold points) |
| Loss | `mse + 0.1 * L1` (FTLE term disabled) |

## Layout

Each `test_<noise>_<horizon>/` contains:

| File | Contents |
|---|---|
| `model.npz`, `model.json` | Trained MLP weights and architecture |
| `metadata.json` | Sweep point, seed, shapes, test IC, parameter count |
| `history.json` | Per-epoch total / mse / l1 |
| `results.json` | Every diagnostic's `gt`, `emu`, `control_gt_gt`, `ratios`, `meta` |
| 7 `*.png` | One figure per diagnostic, each verified non-blank at write time |

At the top level: `results_all.json` (all 28 diagnostic records), `results_table.txt`
(emulator/ground-truth ratios), and `sweep.log` (stdout of the run, including training curves).

## Headline numbers

Ground-truth Lyapunov spectra are stable to three decimals across all four points, and `sum`
matches the divergence `-(sigma + 1 + beta) = -13.667` every time:

| sweep | ground truth λ | emulator λ | λ₁ ratio |
|---|---|---|---|
| 0.05 / 10 | `0.997, 0.091, -14.868` | `0.835, -0.046, -39.727` | 0.84 |
| 0.1 / 10 | `1.022, 0.097, -14.898` | `0.785, -0.009, -73.856` | 0.77 |
| 0.1 / 20 | `1.002, 0.088, -14.865` | `0.804, -0.014, -67.512` | 0.80 |
| 0.4 / 10 | `0.990, 0.007, -14.771` | `-0.006, -0.514, -344.784` | -0.01 |

The emulator reproduces the attractor's *shape* at every noise level (`energy_mean` within 7%,
DMD spectrum correlation ≥ 0.94) but not its *contraction*: λ₃ is 2.7–5x too negative even at the
best point. At noise 0.4 it collapses to a fixed point while one-step MSE is still 0.112 — the
clearest evidence that low MSE does not imply correct dynamics.

## Caveat

These are single-trajectory, single-initialisation results. There are, as yet, **no seed-replicate
error bars**, so the differences between the 0.05, 0.1 and 0.1/h=20 points are not yet
distinguishable from run-to-run variation. Only the noise-0.4 collapse is unambiguous.
