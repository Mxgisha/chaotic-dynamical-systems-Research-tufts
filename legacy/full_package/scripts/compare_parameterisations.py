#!/usr/bin/env python3
"""Head-to-head: map emulator vs vector-field emulator.

Both learn the same object -- one step of the Lorenz-63 flow -- from identical
data, with identical architecture, optimiser, loss and epoch budget.  The only
difference is the parameterisation:

* **map**:      ``x_{n+1} = f(x_n)``, a direct network evaluation.  Nothing
  constrains ``det Df``, so the learned Jacobian contracts volume however the fit
  happens to produce.
* **field**:    ``dx/dt = g(x)``, integrated with RK4.  The flow map is a
  composition of solver stages, so its volume change is structurally tied to
  ``div g`` rather than being free.

The question is whether that structural difference closes the tail
over-contraction that :mod:`lorenz63.diagnostics.contraction` measured, at
matched capacity and matched training loss.

Usage::

    python scripts/compare_parameterisations.py --config configs/sweep.json \\
        --out comparisons/default --replicates 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lorenz63 import controls  # noqa: E402
from lorenz63.aggregate import format_table, summarise_sweep, write_summary  # noqa: E402
from lorenz63.data import make_dataset, point_key  # noqa: E402
from lorenz63.diagnostics import DiagnosticContext, get  # noqa: E402
from lorenz63.field_models import VectorFieldMLP, train_vector_field  # noqa: E402
from lorenz63.losses import LossConfig  # noqa: E402
from lorenz63.models import NumpyMLP  # noqa: E402
from lorenz63.report import DiagnosticResult, emit, write_results  # noqa: E402
from lorenz63.train import TrainConfig, save_history, train  # noqa: E402

#: The arms of the experiment.  ``map`` is the incumbent; ``field`` learns the
#: vector field; ``field_div`` adds the system's known divergence as a
#: physics-informed penalty, since the data alone constrains it only weakly.
ARMS = {
    "map": {"backend": "map", "loss": {}},
    "field": {"backend": "field", "loss": {}},
    # The weight comes from scripts/scan_divergence_weight.py: at 1.0 the learned
    # field reproduces the true divergence (-13.58 against -13.67) and the
    # contraction ratio is 0.99, while a larger weight only buys a tighter spread
    # at the cost of one-step accuracy.
    "field_div": {
        "backend": "field",
        "loss": {"divergence_weight": 1.0, "divergence_target": -13.666666666666666},
    },
}

#: The metrics the comparison turns on.  `contraction_rate_mean` is the primary
#: endpoint; the rest are reported so a win cannot come from a degenerate fit.
PRIMARY = ("contraction_rate_mean", "contraction_rate_std", "contraction_rate_p05",
           "contraction_rate_p95", "divergence_error")
SECONDARY = ("expansion_rate_mean", "corr_expansion_gt_vs_emu", "corr_contraction_gt_vs_emu")
SANITY = ("lambda_1", "lambda_3", "sum_lambda")


def build(backend, cfg, dt, replicate=0):
    m = cfg["model"]
    seed = int(m["seed"]) + int(replicate)
    if backend == "field":
        return VectorFieldMLP(width=m["width"], depth=m["depth"], seed=seed, dt=dt)
    return NumpyMLP(width=m["width"], depth=m["depth"], seed=seed)


def train_one(backend, model, ds, tcfg):
    if backend == "field":
        return train_vector_field(model, ds, tcfg)
    return train(model, ds, tcfg)


def run_arm(arm, cfg, entry, out_dir, replicate, data_cfg, diag_options,
            diag_names):
    """Train one arm and evaluate the diagnostics that matter here.

    ``arm`` is a key of :data:`ARMS`, which names the *backend* and any loss
    overrides -- passing the arm name where a backend was expected silently
    trained the map for every arm.
    """
    spec = ARMS[arm]
    backend = spec["backend"]
    arm_loss = spec.get("loss")
    dt = cfg["dt"]
    noise_scale, horizon = entry["noise_scale"], entry["horizon"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import jax.random as jrandom

    root = jrandom.PRNGKey(cfg["root_seed"])
    key = point_key(root, noise_scale=noise_scale, horizon=horizon, replicate=replicate)
    ds = make_dataset(key, noise_scale=noise_scale, horizon=horizon, **data_cfg)

    tcfg_raw = dict(cfg["train"])
    loss_kwargs = dict(tcfg_raw.pop("loss"))
    loss_kwargs.update(arm_loss or {})
    loss = LossConfig(**loss_kwargs)
    tcfg = TrainConfig(loss=loss, seed=int(tcfg_raw.pop("seed", 67)) + replicate, **tcfg_raw)

    model = build(backend, cfg, dt, replicate)
    model, history = train_one(backend, model, ds, tcfg)
    save_history(history, out_dir / "history.json")
    model.save(out_dir / "model.npz")
    (out_dir / "metadata.json").write_text(json.dumps({
        "arm": arm, "backend": backend, "replicate": replicate,
        "noise_scale": noise_scale, "horizon": horizon,
        "n_params": model.n_params, "final_train_loss": history[-1],
        "field_scale": getattr(model, "field_scale", None),
    }, indent=2))

    gt = np.asarray(ds.test_trajectory, dtype=np.float64)
    emu = controls.emulator_trajectory(model, ds.test_ic, gt.shape[0])
    ctx = DiagnosticContext(
        dt=dt, model=model, ic=ds.test_ic,
        gt_gt_trajectory=None, identity_trajectory=None,
        rng=np.random.default_rng(0), options=dict(diag_options or {}),
    )

    results = []
    for name in diag_names:
        gt_m, emu_m = get(name)(gt, emu, ctx)
        r = DiagnosticResult(
            name=name, gt=gt_m, emu=emu_m,
            meta={"arm": arm, "backend": backend, "replicate": replicate,
                  "noise_scale": noise_scale, "horizon": horizon,
                  "final_train_loss": history[-1]["mse"],
                  "n_params": model.n_params},
        )
        emit(r, out_dir)
        results.append(r)
    write_results(results, out_dir)
    return results, history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep.json"))
    ap.add_argument("--out", default=str(ROOT / "comparisons" / "map_vs_field"))
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    args = ap.parse_args(argv)

    arms = [a for a in args.arms if a in ARMS]
    for a in args.arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(ARMS)}")
    cfg = json.loads(Path(args.config).read_text())
    data_cfg = dict(cfg["data"])
    diag_options = dict(cfg.get("diagnostics", {}))
    if args.smoke:
        s = cfg["smoke"]
        data_cfg.update({k: s[k] for k in
                         ("train_steps", "eval_steps", "burn_in", "n_short", "short_steps")})
        cfg["train"]["epochs"] = s["epochs"]
        diag_options.update(s.get("diagnostics", {}))
        cfg["sweep"] = cfg["sweep"][:1]

    # The comparison is about the Jacobian, so only the contraction and Lyapunov
    # diagnostics are run -- they are the endpoints, and this keeps the sweep cheap.
    diag_names = ["local_contraction", "lyapunov_spectrum"]
    out_root = Path(args.out)
    summary_rows = []

    for entry in cfg["sweep"]:
        point = f"test_{entry['noise_scale']}_{entry['horizon']}"
        print(f"\n=== {point} ===")
        summaries = {}
        for arm in arms:
            arm_root = out_root / arm
            print(f"  --- arm: {arm} ---")
            for rep in range(args.replicates):
                rep_dir = arm_root / point / f"replicate{rep}"
                res, hist = run_arm(arm, cfg, entry, rep_dir, rep, data_cfg,
                                    diag_options, diag_names)
                final = hist[-1]["mse"]
                lc = next(r for r in res if r.name == "local_contraction")
                print(f"    replicate {rep}: train mse {final:.6f} "
                      f"| contraction ratio {lc.emu.get('contraction_ratio', float('nan')):.3f}")
            summaries[arm] = summarise_sweep(arm_root)

        # per-metric comparison at this point
        for metric in PRIMARY + SECONDARY:
            row = {"point": point, "metric": metric}
            for arm in arms:
                s = summaries[arm].get(point, {}).get("ratio", {}).get("local_contraction", {})
                v = s.get(metric)
                row[f"{arm}_mean"] = v["mean"] if v else float("nan")
                row[f"{arm}_std"] = v["std"] if v else float("nan")
            if all(f"{a}_mean" in row for a in arms):
                a = row[f"{arms[0]}_mean"]
                b = row[f"{arms[-1]}_mean"]
                se = (row[f"{arms[0]}_std"] + row[f"{arms[-1]}_std"]) / max(
                    np.sqrt(args.replicates), 1
                )
                row["difference"] = abs(a - b)
                row["combined_se"] = se
                row["resolvable"] = bool(abs(a - b) > se)
            summary_rows.append(row)

        for arm in arms:
            write_summary(summaries[arm], out_root / arm / "summary.json")
            (out_root / arm / "summary_table.txt").write_text(
                format_table(summaries[arm])
            )

    table_path = out_root / "comparison.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(summary_rows, indent=2, default=str))

    # human-readable
    lines = [f"{'metric':28s}" + "".join(f"{a:>20s}" for a in arms)
             + f"{'diff':>10s}{'SE':>10s}  verdict"]
    for row in summary_rows:
        vals = "".join(
            f"{row.get(f'{a}_mean', float('nan')):12.3f}+/-{row.get(f'{a}_std', float('nan')):<7.3f}"
            for a in arms
        )
        verdict = ""
        if "resolvable" in row:
            verdict = "RESOLVABLE" if row["resolvable"] else "within noise"
        lines.append(
            f"{row['metric']:28s}{vals}"
            f"{row.get('difference', float('nan')):10.3f}"
            f"{row.get('combined_se', float('nan')):10.3f}  {verdict}"
        )
    report = "\n".join(lines)
    (out_root / "comparison.txt").write_text(report)
    print("\n=== map vs field (emulator/ground-truth ratios) ===")
    print(report)
    print(f"\nwrote {out_root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
