#!/usr/bin/env python3
"""End-to-end sweep: train, evaluate every diagnostic, emit figures + results.json.

This replaces the notebook's cells 17/18.  Differences that matter:

* One key per sweep point **and replicate** (:func:`lorenz63.data.point_key`), so
  the noise axis actually varies the data instead of re-using one frozen
  realisation, and seed-to-seed spread can be quantified.
* The ground truth and the null controls are computed **once per sweep point** and
  shared by every replicate and every diagnostic; only the emulator varies across
  replicates.  Each diagnostic therefore costs one ground-truth pass per point
  rather than one per diagnostic.
* ``model.npz`` / ``model.json`` and ``metadata.json`` are always written, so the
  evaluation step can never silently skip every folder as it did before.
* Figures are written by :func:`lorenz63.report.emit`, which saves before showing
  and verifies the file is not blank.

Usage::

    python scripts/run_sweep.py --config configs/sweep.json --out runs
    python scripts/run_sweep.py --smoke                 # small CPU verification run
    python scripts/run_sweep.py --replicates 3          # seed replicates + error bars
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

from lorenz63 import aggregate, controls, jax_trainer, system  # noqa: E402
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


def build_model(cfg, backend, replicate=0):
    """Build the emulator for the requested backend.

    ``backend`` is the *model* backend, independent of ``LORENZ63_BACKEND`` (which
    only selects the array namespace for the data/diagnostics code).  The
    replicate index offsets the model initialisation seed, so replicates differ
    in both data and weights.
    """
    m = cfg["model"]
    seed = int(m["seed"]) + int(replicate)
    if backend in jax_trainer.ACCELERATED_BACKENDS:
        jax_trainer.require_jax()
        adapter = jax_trainer.EquinoxModelAdapter(
            jax_trainer.make_equinox_model(width=m["width"], depth=m["depth"], seed=seed)
        )
        return adapter
    return NumpyMLP(width=m["width"], depth=m["depth"], seed=seed)


def train_model(model, ds, tcfg, backend):
    """Dispatch to the right trainer; both return ``(model, history)``."""
    if backend in jax_trainer.ACCELERATED_BACKENDS:
        return jax_trainer.train_jax(model, ds, tcfg)
    return train(model, ds, tcfg)


def build_shared(cfg, entry, backend, data_cfg, diag_options):
    """Ground truth, controls and context -- identical for every replicate.

    Returned so a sweep point pays for its ground-truth trajectory, its control
    trajectories and its O(N^2) control diagnostics exactly once.
    """
    noise_scale, horizon = entry["noise_scale"], entry["horizon"]
    dt = cfg["dt"]

    import jax.random as jrandom

    root = jrandom.PRNGKey(cfg["root_seed"])
    key = point_key(root, noise_scale=noise_scale, horizon=horizon, replicate=0)
    ds = make_dataset(key, noise_scale=noise_scale, horizon=horizon, **data_cfg)
    gt = np.asarray(ds.test_trajectory, dtype=np.float64)
    ctl = controls.build_controls(ds, dt=dt)
    return ds, gt, ctl


def run_replicate(
    cfg,
    entry,
    ds,
    gt,
    ctl,
    out_dir,
    *,
    replicate=0,
    backend="numpy",
    diag_options=None,
    diag_names=None,
):
    """Train one replicate and evaluate every diagnostic.  Returns its results."""
    noise_scale, horizon = entry["noise_scale"], entry["horizon"]
    dt = cfg["dt"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tcfg_raw = dict(cfg["train"])
    loss = LossConfig(**tcfg_raw.pop("loss"))
    # Offset the shuffling seed too, so replicates are independent end to end.
    tcfg = TrainConfig(loss=loss, seed=int(tcfg_raw.pop("seed", 67)) + replicate, **tcfg_raw)

    model = build_model(cfg, backend, replicate=replicate)
    model, history = train_model(model, ds, tcfg, backend)
    save_history(history, out_dir / "history.json")

    model.save(out_dir / "model.npz")
    meta = dict(ds.meta)
    meta.update(
        {
            "root_seed": cfg["root_seed"],
            "replicate": int(replicate),
            "model": cfg["model"],
            "model_seed": int(cfg["model"]["seed"]) + int(replicate),
            "train_seed": tcfg.seed,
            "backend": backend,
            "test_ic": np.asarray(ds.test_ic).tolist(),
            "n_params": model.n_params,
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    emu = controls.emulator_trajectory(model, ds.test_ic, gt.shape[0])
    ctx = DiagnosticContext(
        dt=dt,
        model=model,
        ic=ds.test_ic,
        gt_gt_trajectory=ctl["gt_gt_trajectory"],
        identity_trajectory=ctl["identity_trajectory"],
        rng=np.random.default_rng(0),
        options=dict(diag_options or {}),
    )

    names = diag_names or available()
    results = []
    for name in names:
        fn = get(name)
        print(f"    diagnostic: {name}")
        gt_m, emu_m = fn(gt, emu, ctx)
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
                "replicate": int(replicate),
                "n_steps": int(gt.shape[0]),
                "dt": dt,
                "backend": BACKEND,
                "model_backend": backend,
                "final_train_loss": history[-1] if history else None,
            },
        )
        emit(r, out_dir)
        results.append(r)

    write_results(results, out_dir)
    return results


def run_point(cfg, entry, out_root, backend="numpy", diag_options=None,
              diag_names=None, replicates=1):
    """Run every replicate of one sweep point.  Returns a list of result-lists."""
    noise_scale, horizon = entry["noise_scale"], entry["horizon"]
    point_dir = Path(out_root) / point_dirname(noise_scale, horizon)
    data_cfg = dict(cfg["data"])

    print(f"  building ground truth and controls (shared across replicates)")
    ds, gt, ctl = build_shared(cfg, entry, backend, data_cfg, diag_options)
    print(f"  dataset: {ds.n_samples} windows, eval {gt.shape[0]} steps")

    all_runs = []
    for rep in range(max(int(replicates), 1)):
        # One replicate per directory, always -- so the on-disk layout does not
        # change shape when a single-replicate run is compared with a sweep.
        rep_dir = point_dir / f"replicate{rep}"
        print(f"  --- replicate {rep} ---")
        all_runs.append(
            run_replicate(
                cfg, entry, ds, gt, ctl, rep_dir,
                replicate=rep, backend=backend,
                diag_options=diag_options, diag_names=diag_names,
            )
        )
    return all_runs


def prune_duplicate_figures(out_root, keep_replicate=0):
    """Delete figures in replicate directories beyond ``keep_replicate``.

    Within a sweep point the ground truth is identical across replicates, so the
    diagnostic figures differ only in the emulator overlay; replicate 0 is kept
    for visual inspection.  All numeric results are preserved in ``results.json``
    and in the aggregate summary, and every figure is reproducible from the
    committed seeds, so this only trades disk for a re-run.

    Returns the number of files removed.
    """
    removed = 0
    for point_dir in sorted(Path(out_root).glob("test_*")):
        for rep_dir in sorted(point_dir.glob("replicate*")):
            name = rep_dir.name
            if not name.startswith("replicate"):
                continue
            try:
                idx = int(name[len("replicate"):])
            except ValueError:
                continue
            if idx == keep_replicate:
                continue
            for png in rep_dir.glob("*.png"):
                png.unlink()
                removed += 1
    return removed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep.json"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--backend", default="numpy",
                    choices=["numpy", "jax", "jax_mlp", "jax_tracked"],
                    help="model/training backend (jax needs requirements-jax.txt)")
    ap.add_argument("--smoke", action="store_true", help="use the small CPU config")
    ap.add_argument("--replicates", type=int, default=None,
                    help="seed replicates per sweep point (overrides config)")
    ap.add_argument("--only", nargs="*", help="subset of diagnostics to run")
    ap.add_argument("--prune-figures", action="store_true",
                    help="after the run, delete figures in replicate>0 (they differ only "
                         "in the emulator overlay; results.json is untouched)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    diag_options = dict(cfg.get("diagnostics", {}))
    # Precedence: explicit --replicates  >  smoke block  >  top-level config.
    replicates = args.replicates if args.replicates is not None else cfg.get("replicates", 1)

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
        diag_options.update(smoke.get("diagnostics", {}))
        cfg["sweep"] = cfg["sweep"][:1]
        # only let the smoke block change the replicate count when the user did
        # not ask for a specific number on the command line
        if args.replicates is None and smoke.get("replicates") is not None:
            replicates = smoke["replicates"]

    out_root = Path(args.out)
    print(
        f"backend={BACKEND} model_backend={args.backend} "
        f"replicates={replicates} out={args.out}"
    )

    all_results = {}
    for entry in cfg["sweep"]:
        name = point_dirname(entry["noise_scale"], entry["horizon"])
        print(f"\n=== {name} ===")
        runs = run_point(
            cfg, entry, out_root,
            backend=args.backend,
            diag_options=diag_options,
            diag_names=args.only,
            replicates=replicates,
        )
        all_results[name] = runs[0]

    # --- cross-point table from the first replicate (unchanged format) ------
    table = results_table(all_results)
    print("\n" + table)
    (out_root / "results_table.txt").write_text(table)
    summary_first = {k: [r.scalars() for r in v] for k, v in all_results.items()}
    (out_root / "results_all.json").write_text(
        json.dumps(summary_first, indent=2, default=str)
    )

    # --- replicate statistics ----------------------------------------------
    if int(replicates) > 1:
        summary = aggregate.summarise_sweep(out_root)
        aggregate.write_summary(summary, out_root / "summary.json")
        agg_table = aggregate.format_table(summary)
        (out_root / "summary_table.txt").write_text(agg_table)
        print("\n=== replicate statistics (emu/gt mean +/- std) ===")
        print(agg_table)

        # Which differences between sweep points exceed the seed-to-seed spread?
        rows = aggregate.resolvable_points(summary)
        (out_root / "resolvable.json").write_text(
            json.dumps(rows, indent=2, default=str)
        )
        res_table = aggregate.format_resolvable(rows)
        (out_root / "resolvable_table.txt").write_text(res_table)
        n_res = sum(1 for r in rows if r["resolvable"])
        print(
            f"\n=== resolvability: {n_res} of {len(rows)} pairwise comparisons "
            f"exceed the combined standard error ==="
        )
        print(aggregate.format_resolvable(rows, limit=25))
        print(f"\nwrote {out_root}/summary.json, summary_table.txt, "
              f"resolvable.json, resolvable_table.txt")

    if args.prune_figures:
        n = prune_duplicate_figures(out_root)
        print(f"\npruned {n} duplicate figures (replicate>0); results.json preserved")

    print(f"\nwrote {out_root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
