#!/usr/bin/env python3
"""Scan the divergence-penalty weight for the vector-field emulator.

The data constrains the field's divergence only weakly -- it enters the one-step
map at ``O(dt^2)``, so at ``dt = 0.01`` a field with several times the true
divergence still fits the increments well.  Adding the known divergence as a
penalty fixes that, but too large a weight destroys the fit.  This finds the
weight that buys divergence fidelity without paying much in one-step error.

Usage::

    python scripts/scan_divergence_weight.py --weights 0 0.001 0.01 0.1 1.0
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

from lorenz63 import controls, system as S  # noqa: E402
from lorenz63.data import make_dataset, point_key  # noqa: E402
from lorenz63.diagnostics import DiagnosticContext, get  # noqa: E402
from lorenz63.field_models import VectorFieldMLP, train_vector_field  # noqa: E402
from lorenz63.losses import LossConfig  # noqa: E402
from lorenz63.train import TrainConfig  # noqa: E402


def measure_field_divergence(model, states, eps=1e-6):
    """Mean and spread of the learned field's divergence at ``states``."""
    divs = []
    for x in states:
        J = np.zeros((3, 3))
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            J[:, i] = (model.field(x + e) - model.field(x - e)) / (2 * eps)
        divs.append(np.trace(J))
    divs = np.asarray(divs)
    return float(divs.mean()), float(divs.std())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep.json"))
    ap.add_argument("--out", default=str(ROOT / "comparisons" / "divergence_scan"))
    ap.add_argument("--weights", nargs="*", type=float,
                    default=[0.0, 1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    dt = cfg["dt"]
    data_cfg = dict(cfg["data"])
    diag_options = dict(cfg.get("diagnostics", {}))
    entry = cfg["sweep"][0]

    import jax.random as jrandom

    root = jrandom.PRNGKey(cfg["root_seed"])
    key = point_key(root, noise_scale=entry["noise_scale"], horizon=entry["horizon"])
    ds = make_dataset(key, noise_scale=entry["noise_scale"], horizon=entry["horizon"],
                      **data_cfg)
    gt = np.asarray(ds.test_trajectory, dtype=np.float64)
    probe = gt[np.linspace(0, gt.shape[0] - 1, 200).astype(int)]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for w in args.weights:
        mses, ratios, divs, spreads = [], [], [], []
        for rep in range(args.replicates):
            tcfg_raw = dict(cfg["train"])
            tcfg_raw.pop("loss")
            # a target with zero weight is rejected by LossConfig by design
            loss = LossConfig(
                l1_weight=0.0,
                divergence_weight=w,
                divergence_target=(S.DIVERGENCE if w > 0 else None),
            )
            tcfg = TrainConfig(loss=loss, seed=int(tcfg_raw.pop("seed", 67)) + rep,
                               **tcfg_raw)
            if args.epochs is not None:
                tcfg.epochs = args.epochs
            model = VectorFieldMLP(width=cfg["model"]["width"],
                                   depth=cfg["model"]["depth"],
                                   seed=int(cfg["model"]["seed"]) + rep, dt=dt)
            model, hist = train_vector_field(model, ds, tcfg, verbose=False)
            emu = controls.emulator_trajectory(model, ds.test_ic, gt.shape[0])
            ctx = DiagnosticContext(dt=dt, model=model, ic=ds.test_ic,
                                   options=dict(diag_options))
            _, em = get("local_contraction")(gt, emu, ctx)
            mses.append(hist[-1]["mse"])
            ratios.append(em["contraction_ratio"])
            d, s = measure_field_divergence(model, probe)
            divs.append(d)
            spreads.append(s)

        row = {
            "divergence_weight": w,
            "train_mse": float(np.mean(mses)),
            "train_mse_std": float(np.std(mses)),
            "contraction_ratio": float(np.mean(ratios)),
            "contraction_ratio_std": float(np.std(ratios)),
            "field_divergence": float(np.mean(divs)),
            "field_divergence_std": float(np.std(divs)),
            "field_divergence_spread": float(np.mean(spreads)),
            "true_divergence": float(S.DIVERGENCE),
        }
        rows.append(row)
        print(f"w={w:<7g} mse {row['train_mse']:10.6f}  "
              f"contraction ratio {row['contraction_ratio']:7.3f}  "
              f"field div {row['field_divergence']:8.3f} "
              f"(true {S.DIVERGENCE:.2f})")

    (out_root / "scan.json").write_text(json.dumps(rows, indent=2))

    lines = [f"{'weight':>9s} {'train mse':>12s} {'contraction':>13s} "
             f"{'field div':>11s} {'div spread':>11s}   (true div = {S.DIVERGENCE:.2f})"]
    for r in rows:
        lines.append(f"{r['divergence_weight']:9g} {r['train_mse']:12.6f} "
                     f"{r['contraction_ratio']:13.3f} {r['field_divergence']:11.3f} "
                     f"{r['field_divergence_spread']:11.3f}")
    report = "\n".join(lines)
    (out_root / "scan.txt").write_text(report)
    print("\n" + report)
    print(f"\nwrote {out_root}/scan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
