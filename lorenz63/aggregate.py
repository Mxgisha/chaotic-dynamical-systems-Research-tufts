"""Aggregate replicate runs into mean +/- spread.

A single trajectory per sweep point cannot distinguish a real difference between
configurations from run-to-run variation.  With ``replicates > 1`` the sweep
writes one directory per replicate and this module collapses them into a summary
that carries the spread, so a difference can be compared against it.

Two levels are reported:

``ratio``
    ``mean(emu_replicates) / gt`` -- how far the emulator is from the truth on
    each metric.  Reported with the spread of the emulator across replicates.

``spread``
    ``std(replicates) / |mean(replicates)|`` -- the coefficient of variation.
    A difference between two sweep points smaller than this is not resolvable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Keys that carry no information across replicates: bookkeeping counts and
# quantities that are identical for every run of a point.  Imported so the
# replicate summary and the single-run ratio table agree on what is a metric;
# lorenz63.report does not import this module, so there is no cycle.
from lorenz63.report import NON_RATIO_KEYS

SKIP_KEYS = frozenset(NON_RATIO_KEYS) | frozenset(
    {"n_points", "n_steps", "n_modes", "n_scales_used", "n_maxima", "n_flips", "fixed_val"}
)


def load_replicates(point_dir, pattern="replicate*"):
    """Load ``results.json`` from every replicate directory of one sweep point."""
    point_dir = Path(point_dir)
    runs = []
    for d in sorted(point_dir.glob(pattern)):
        f = d / "results.json"
        if f.exists():
            runs.append((d.name, json.loads(f.read_text())))
    return runs


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summarise_point(runs, field="emu", min_replicates=2):
    """Mean/std/min/max/spread per diagnostic metric across replicates."""
    if not runs:
        return {}
    names = [r["name"] for r in runs[0][1]]
    out = {}
    for name in names:
        per_metric = {}
        for rep_label, payload in runs:
            rec = next((r for r in payload if r["name"] == name), None)
            if rec is None:
                continue
            for k, v in rec.get(field, {}).items():
                if k.startswith("_") or k in SKIP_KEYS or not _numeric(v):
                    continue
                per_metric.setdefault(k, []).append(float(v))
        if not per_metric:
            continue
        stats = {}
        for k, vals in per_metric.items():
            arr = np.asarray(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            # A metric may legitimately be non-finite for some replicates (a
            # trajectory with no lobe crossings has infinite mean dwell).
            # Averaging that in would poison the summary, so such replicates are
            # dropped and counted in n_used.
            if finite.size == 0:
                continue
            mean = float(finite.mean())
            std = float(finite.std(ddof=0))
            if abs(mean) > 1e-12:
                spread = abs(std / mean)
            else:
                spread = float("nan")
            stats[k] = {
                "mean": mean,
                "std": std,
                "min": float(finite.min()),
                "max": float(finite.max()),
                "n": int(arr.size),
                "n_used": int(finite.size),
                # coefficient of variation: the scale a real difference must beat
                "spread": spread,
            }
        out[name] = stats
    return out


def gt_values(runs):
    """Ground-truth metrics for a point, from the first replicate."""
    if not runs:
        return {}
    out = {}
    for rec in runs[0][1]:
        out[rec["name"]] = {
            k: float(v)
            for k, v in rec.get("gt", {}).items()
            if not k.startswith("_") and _numeric(v)
        }
    return out


def summarise_sweep(runs_root, pattern="replicate*"):
    """Summarise every sweep point under ``runs_root``.

    Returns ``{point: {"gt": ..., "emu": ..., "ratio": ..., "n_replicates": n}}``.
    """
    runs_root = Path(runs_root)
    out = {}
    for point_dir in sorted(p for p in runs_root.glob("test_*") if p.is_dir()):
        runs = load_replicates(point_dir, pattern)
        if not runs:
            continue
        gt = gt_values(runs)
        emu = summarise_point(runs, field="emu")
        ratio = {}
        for name, stats in emu.items():
            ratio[name] = {}
            for k, s in stats.items():
                g = gt.get(name, {}).get(k)
                if g is None or abs(g) < 1e-12:
                    continue
                if not math.isfinite(s["mean"]) or not math.isfinite(s["std"]):
                    # A metric whose replicate mean is not finite has no
                    # meaningful ratio; skip rather than emit inf/NaN.
                    continue
                ratio[name][k] = {
                    "mean": s["mean"] / g,
                    "std": s["std"] / abs(g),
                    "spread": s["spread"],
                    "n": s["n"],
                    "n_used": s.get("n_used", s["n"]),
                }
        out[point_dir.name] = {
            "gt": gt,
            "emu": emu,
            "ratio": ratio,
            "n_replicates": len(runs),
            "replicates": [label for label, _ in runs],
        }
    return out


def _fmt(value, width=9, decimals=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return " " * (width - 2) + "n/a"
    return f"{value:{width}.{decimals}f}"


def format_table(summary, metric_filter=None):
    """Plain-text table: per sweep point, metric, emu/gt mean +/- std and spread."""
    lines = []
    header = (
        f"{'sweep point':16s} {'diagnostic':22s} {'metric':20s} "
        f"{'ratio':>9s} {'+/-':>8s} {'spread':>8s} {'n':>3s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for point, data in summary.items():
        n_rep = data.get("n_replicates", 1)
        for name, metrics in sorted(data["ratio"].items()):
            for k, s in sorted(metrics.items()):
                if k in SKIP_KEYS:
                    continue
                if metric_filter and k not in metric_filter:
                    continue
                lines.append(
                    f"{point:16s} {name:22s} {k:20s} "
                    f"{_fmt(s['mean'])} {_fmt(s['std'], 8)} "
                    f"{_fmt(s['spread'], 8)} {n_rep:3d}"
                )
    return "\n".join(lines)


def resolvable(summary, point_a, point_b, metric, diagnostic=None, field="ratio"):
    """Whether two sweep points differ by more than their combined spread.

    A crude but honest screen: the difference must exceed the sum of the two
    standard errors (``std / sqrt(n)``), i.e. roughly a 1-sigma separation.
    Returns ``None`` if the metric is absent from either point.
    """
    def get(point):
        point_data = summary.get(point)
        if not point_data:
            return None
        for name, metrics in point_data[field].items():
            if diagnostic and name != diagnostic:
                continue
            if metric in metrics:
                return metrics[metric]
        return None

    a, b = get(point_a), get(point_b)
    if a is None or b is None:
        return None
    if not (math.isfinite(a["mean"]) and math.isfinite(b["mean"])):
        return None
    se_a = a["std"] / math.sqrt(max(a.get("n_used", a.get("n", 1)), 1))
    se_b = b["std"] / math.sqrt(max(b.get("n_used", b.get("n", 1)), 1))
    diff = abs(a["mean"] - b["mean"])
    return {
        "point_a": point_a,
        "point_b": point_b,
        "mean_a": a["mean"],
        "mean_b": b["mean"],
        "difference": diff,
        "combined_se": se_a + se_b,
        "resolvable": diff > (se_a + se_b),
    }


def resolvable_points(summary, field="ratio", only_metrics=None):
    """All pairwise point comparisons, with a resolvability verdict each.

    Returns a list of dicts sorted so the resolvable ones come first.  A
    comparison is flagged when the difference exceeds the sum of the two standard
    errors; ties and sub-threshold differences are reported too, so the table
    shows what the data *cannot* distinguish as explicitly as what it can.
    """
    points = sorted(summary)
    if len(points) < 2:
        return []
    rows = []
    for i, pa in enumerate(points):
        for pb in points[i + 1:]:
            shared = set(summary[pa].get(field, {})) & set(summary[pb].get(field, {}))
            for diag in sorted(shared):
                keys = set(summary[pa][field][diag]) & set(summary[pb][field][diag])
                for metric in sorted(keys):
                    if only_metrics and metric not in only_metrics:
                        continue
                    v = resolvable(summary, pa, pb, metric, diagnostic=diag, field=field)
                    if v is None:
                        continue
                    v["diagnostic"] = diag
                    v["metric"] = metric
                    rows.append(v)
    rows.sort(key=lambda r: (not r["resolvable"], -r["difference"]))
    return rows


def format_resolvable(rows, limit=None):
    """Plain-text table of pairwise comparisons."""
    lines = []
    header = (
        f"{'point A':14s} {'point B':14s} {'diagnostic':22s} {'metric':20s} "
        f"{'A':>9s} {'B':>9s} {'diff':>9s} {'SE':>9s}  verdict"
    )
    lines.append(header)
    lines.append("-" * len(header))
    shown = rows[:limit] if limit else rows
    for r in shown:
        lines.append(
            f"{r['point_a']:14s} {r['point_b']:14s} {r['diagnostic']:22s} "
            f"{r['metric']:20s} {_fmt(r['mean_a'])} {_fmt(r['mean_b'])} "
            f"{_fmt(r['difference'])} {_fmt(r['combined_se'])}  "
            f"{'RESOLVABLE' if r['resolvable'] else 'within noise'}"
        )
    if limit and len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more comparisons")
    return "\n".join(lines)


def write_summary(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


__all__ = [
    "load_replicates",
    "summarise_point",
    "summarise_sweep",
    "format_table",
    "resolvable",
    "write_summary",
    "gt_values",
]
