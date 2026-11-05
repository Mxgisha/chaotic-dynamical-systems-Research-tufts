"""Invariant measure and energy diagnostics.

Replaces ``run_distributional_distance_test`` (which computed ``emu_z`` and then
never used it, and hard-coded ``bins_x`` to +/-25) and
``run_spatial_energy_test`` (which compared a single scalar moment
``E[sum x_i^2]`` against a hand-picked threshold of 50.0 and printed PASS/FAIL).

Now: per-coordinate L1 **and** Wasserstein-1 (robust to binning choices), and
four moments of the energy ``sum x_i^2`` rather than one.  Significance is
decided in :mod:`lorenz63.report` against the GT-vs-GT control.
"""

from __future__ import annotations

import numpy as np

from lorenz63.diagnostics import register


def l1_distance(a, b, n_bins=100, span=None):
    """L1 distance between two 1-D empirical densities over a common range."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if span is None:
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
    else:
        lo, hi = span
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    width = (hi - lo) / n_bins
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    return float(np.sum(np.abs(ha - hb)) * width)


def wasserstein1(a, b, n_grid=512):
    """1-Wasserstein distance between two 1-D samples (quantile based)."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    n = min(a.size, b.size, n_grid)
    q = (np.arange(n) + 0.5) / n
    # compare interpolated quantile functions at the same probability levels
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float(np.mean(np.abs(qa - qb)))


def energy_moments(traj):
    """Mean/variance/skew/kurtosis of the energy ``sum x_i^2``."""
    e = np.sum(np.asarray(traj, dtype=np.float64) ** 2, axis=-1)
    m = e.mean()
    s = e.std()
    if s < 1e-12:
        return {"energy_mean": float(m), "energy_std": 0.0,
                "energy_skew": 0.0, "energy_kurtosis": 0.0}
    z = (e - m) / s
    return {
        "energy_mean": float(m),
        "energy_std": float(s),
        "energy_skew": float(np.mean(z**3)),
        "energy_kurtosis": float(np.mean(z**4)),
    }


def _pair_metrics(a_traj, b_traj, n_bins=100):
    """Metrics comparing trajectory ``a`` against reference trajectory ``b``."""
    out = {}
    spans = []
    for i in range(a_traj.shape[1]):
        lo = min(a_traj[:, i].min(), b_traj[:, i].min())
        hi = max(a_traj[:, i].max(), b_traj[:, i].max())
        spans.append((lo, hi))
    for i, name in enumerate("xyz"):
        out[f"l1_{name}"] = l1_distance(a_traj[:, i], b_traj[:, i], n_bins, spans[i])
        out[f"w1_{name}"] = wasserstein1(a_traj[:, i], b_traj[:, i])
    out["l1_mean"] = float(np.mean([out[f"l1_{c}"] for c in "xyz"]))
    out["w1_mean"] = float(np.mean([out[f"w1_{c}"] for c in "xyz"]))
    out.update(energy_moments(a_traj))
    return out


@register("invariant_measure")
def invariant_measure(gt, emu, ctx):
    """Invariant-measure mismatch of the emulator relative to ground truth."""
    gt = np.asarray(gt, dtype=np.float64)
    emu = np.asarray(emu, dtype=np.float64)
    n_bins = int(ctx.options.get("n_bins", 100))

    # The emulator is measured against the GT reference trajectory.
    emu_m = _pair_metrics(emu, gt, n_bins)
    gt_m = _pair_metrics(gt, gt, n_bins)

    # Null control: a second true trajectory.  If the emulator's mismatch is no
    # worse than this, the diagnostic has found nothing.
    if ctx.gt_gt_trajectory is not None:
        ctrl = _pair_metrics(np.asarray(ctx.gt_gt_trajectory), gt, n_bins)
        emu_m["l1_mean_vs_gtgt_control"] = ctrl["l1_mean"]
        emu_m["l1_excess_over_control"] = emu_m["l1_mean"] - ctrl["l1_mean"]
    if ctx.identity_trajectory is not None:
        ident = _pair_metrics(np.asarray(ctx.identity_trajectory), gt, n_bins)
        emu_m["l1_mean_vs_identity"] = ident["l1_mean"]
    return gt_m, emu_m
