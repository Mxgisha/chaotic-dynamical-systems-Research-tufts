#!/usr/bin/env python3
"""End-to-end sweep: train, evaluate every diagnostic, emit figures + results.json.

This replaces the notebook's cells 17/18.  Differences that matter:

* One key per sweep point (:func:`lorenz63.data.point_key`), so the noise axis
  actually varies the data instead of re-using one frozen realisation.
* One ground-truth and one emulator rollout per point, shared by every
  diagnostic, instead of each diagnostic re-rolling the system.
* ``model.eqx`` / ``model.json`` and ``metadata.json`` are always written, so the
  evaluation step can never silently skip every folder as it did before.
* Figures are written by :func:`lorenz63.report.emit`, which saves before
  showing and verifies the file is not blank.

Usage::

    python scripts/run_sweep.py --config configs/sweep.json --out runs
    python scripts/run_sweep.py --smoke          # small CPU verification run
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

from lorenz63 import controls, jax_trainer, system  # noqa: E402
from lorenz63.backend import BACKEND  # noqa: E402
from lorenz63.data import make_dataset, point_key  # noqa: E402
from lorenz63.diagnostics import DiagnosticContext, available, get  # noqa: E402
from lorenz63.losses import LossConfig  # noqa: E402
from lorenz63.models import NumpyMLP  # noqa: E402
from lorenz63.report import DiagnosticResult, emit, results_table, write_results  # noqa: E402
from lorenz63.train import TrainConfig, save_history, train  # noqa: E402


def load_config(path):
    return json.loads(Path(path).read_text())


def point_dirname(noise_scale, horizon):
    return f"test_{noise_scale}_{horizon}"


def build_model(cfg, backend):
    """Build the emulator for the requested backend.

    ``backend`` is the *model* backend, independent of ``LORENZ63_BACKEND`` (which
    only selects the array namespace for the data/diagnostics code).
    """
    m = cfg["model"]
    if backend in jax_trainer.ACCELERATED_BACKENDS:
        jax_trainer.require_jax()
        adapter = jax_trainer.EquinoxModelAdapter(
            jax_trainer.make_equinox_model(width=m["width"], depth=m["depth"], seed=m["seed"])
        )
        print(
            f"  model backend: jax/equinox ({adapter.n_params} params, x64="
            f"{jax_trainer.jax.config.jax_enable_x64})"
        )
        return adapter
    return NumpyMLP(width=m["width"], depth=m["depth"], seed=m["seed"])


def train_model(model, ds, tcfg, backend):
    """Dispatch to the right trainer; both return ``(model, history)``."""
    if backend in jax_trainer.ACCELERATED_BACKENDS:
        return jax_trainer.train_jax(model, ds, tcfg)
    return train(model, ds, tcfg)


def run_point(cfg, entry, out_root, backend="numpy", diag_options=None, diag_names=None):
    """Train and evaluate one sweep point.  Returns the list of results."""
    noise_scale, horizon = entry["noise_scale"], entry["horizon"]
    dt = cfg["dt"]
    data_cfg = dict(cfg["data"])
    smoke = cfg.get("_smoke_diag", {})

    out_dir = Path(out_root) / point_dirname(noise_scale, horizon)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data (independent per point: fixes the frozen-noise defect) --------
    import jax.random as jrandom

    root = jrandom.PRNGKey(cfg["root_seed"])
    key = point_key(root, noise_scale=noise_scale, horizon=horizon)
    ds = make_dataset(key, noise_scale=noise_scale, horizon=horizon, **data_cfg)
    print(f"  dataset: {ds.n_samples} windows, eval {ds.test_trajectory.shape[0]} steps")

    # --- train --------------------------------------------------------------
    tcfg_raw = dict(cfg["train"])
    loss = LossConfig(**tcfg_raw.pop("loss"))
    tcfg = TrainConfig(loss=loss, **tcfg_raw)
    model = build_model(cfg, backend)
    model, history = train_model(model, ds, tcfg, backend)
    save_history(history, out_dir / "history.json")

    # --- persist the model (so eval can never skip silently) ---------------
    model.save(out_dir / "model.npz")
    meta = dict(ds.meta)
    meta.update(
        {
            "root_seed": cfg["root_seed"],
            "model": cfg["model"],
            "backend": backend,
            "test_ic": np.asarray(ds.test_ic).tolist(),
            "n_params": model.n_params,
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # --- rollouts (one each, shared by every diagnostic) -------------------
    gt = np.asarray(ds.test_trajectory, dtype=np.float64)
    emu = controls.emulator_trajectory(model, ds.test_ic, gt.shape[0])
    ctl = controls.build_controls(ds, dt=dt)

    # --- diagnostics -------------------------------------------------------
    opts = dict(diag_options or {})
    ctx = DiagnosticContext(
        dt=dt,
        model=model,
        ic=ds.test_ic,
        gt_gt_trajectory=ctl["gt_gt_trajectory"],
        identity_trajectory=ctl["identity_trajectory"],
        rng=np.random.default_rng(0),
        options=opts,
    )

    names = diag_names or available()
    results = []
    for name in names:
        fn = get(name)
        print(f"  diagnostic: {name}")
        gt_m, emu_m = fn(gt, emu, ctx)
        # Null control: how does a second TRUE trajectory score?
        try:
            ctrl_m, _ = fn(gt, np.asarray(ctl["gt_gt_trajectory"]), ctx)
        except Exception as exc:  # a control failing must not lose the result
            ctrl_m = {"_control_error": str(exc)}
        r = DiagnosticResult(
            name=name,
            gt=gt_m,
            emu=emu_m,
            control_gt_gt=ctrl_m,
            meta={
                "noise_scale": noise_scale,
                "horizon": horizon,
                "n_steps": int(gt.shape[0]),
                "dt": dt,
                "backend": BACKEND,
            },
        )
        emit(r, out_dir)
        results.append(r)

    write_results(results, out_dir)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep.json"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--backend", default="numpy",
                    choices=["numpy", "jax", "jax_mlp", "jax_tracked"],
                    help="model/training backend (jax needs requirements-jax.txt)")
    ap.add_argument("--smoke", action="store_true", help="use the small CPU config")
    ap.add_argument("--only", nargs="*", help="subset of diagnostics to run")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    diag_options = dict(cfg.get("diagnostics", {}))

    if args.smoke:
        smoke = cfg["smoke"]
        cfg["data"].update(
            {
                "train_steps": smoke["train_steps"],
                "eval_steps": smoke["eval_steps"],
                "burn_in": smoke["burn_in"],
                "n_short": smoke["n_short"],
                "short_steps": smoke["short_steps"],
            }
        )
        cfg["train"]["epochs"] = smoke["epochs"]
        diag_options.update(smoke["diagnostics"])
        # keep only the first sweep point for a fast check
        cfg["sweep"] = cfg["sweep"][:1]

    print(f"backend={BACKEND} model_backend={args.backend} out={args.out}")
    all_results = {}
    for entry in cfg["sweep"]:
        name = point_dirname(entry["noise_scale"], entry["horizon"])
        print(f"\n=== {name} ===")
        all_results[name] = run_point(
            cfg,
            entry,
            args.out,
            backend=args.backend,
            diag_options=diag_options,
            diag_names=args.only,
        )

    table = results_table(all_results)
    print("\n" + table)
    (Path(args.out) / "results_table.txt").write_text(table)
    summary = {k: [r.scalars() for r in v] for k, v in all_results.items()}
    (Path(args.out) / "results_all.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
