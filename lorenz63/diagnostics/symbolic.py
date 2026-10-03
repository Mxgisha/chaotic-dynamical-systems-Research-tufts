"""Symbolic dynamics: lobe transitions and the Lorenz return map.

``run_lobe_transition_test`` was the strongest diagnostic in the original
pipeline -- it produced the one unambiguous failure signal (``GT Flips: 30 |
Emu Flips: 20``, ratio 0.667).  It is kept, with two additions:

* the numbers are returned rather than only printed, and
* the return map is turned into a correlation coefficient so "visibly fuzzier"
  becomes a number.
"""

from __future__ import annotations

import numpy as np

from lorenz63.diagnostics import register


def lobe_transitions(traj, threshold=0.0):
    """Lobe index series and transition statistics for the x coordinate."""
    x = np.asarray(traj, dtype=np.float64)[:, 0]
    lobe = np.where(x >= threshold, 1, -1)
    crossings = np.abs(np.diff(lobe)) > 0
    n_flips = int(np.sum(crossings))
    n = lobe.size
    # NB: a zero-flip trajectory is a real outcome (an emulator stuck in one
    # lobe), not a divide-by-zero to paper over with an epsilon.  Adding 1e-12
    # produced a "mean dwell" of 6.2e12, which then poisoned every downstream
    # mean.  Report inf explicitly; aggregation skips non-finite values.
    mean_dwell = float(n / n_flips) if n_flips > 0 else float("inf")
    return {
        "n_flips": n_flips,
        "flip_rate": float(n_flips / max(n - 1, 1)),
        "mean_dwell": mean_dwell,
        "lobe_balance": float(np.mean(lobe)),
        "cum_flips": np.cumsum(np.concatenate([[0], crossings])),
    }


def find_maxima_in_z(traj):
    """Local maxima of z, ignoring the first/last sample."""
    z = np.asarray(traj, dtype=np.float64)[:, 2]
    if z.size < 3:
        return np.array([])
    return z[1:-1][(z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])]


def return_map_correlation(traj):
    """Correlation of the successive-maxima map ``(z_n, z_{n+1})``.

    The Lorenz return map is a sharp tent-like curve; a smeared emulator map
    shows up as a correlation well below the ground truth's.
    """
    m = find_maxima_in_z(traj)
    if m.size < 3:
        return float("nan"), 0
    a, b = m[:-1], m[1:]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan"), int(m.size)
    return float(np.corrcoef(a, b)[0, 1]), int(m.size)


@register("symbolic_dynamics")
def symbolic_dynamics(gt, emu, ctx):
    """Lobe-transition and return-map comparison."""
    gt = np.asarray(gt, dtype=np.float64)
    emu = np.asarray(emu, dtype=np.float64)
    threshold = float(ctx.options.get("x_threshold", 0.0))

    gt_lobe = lobe_transitions(gt, threshold)
    emu_lobe = lobe_transitions(emu, threshold)

    gt_corr, gt_n = return_map_correlation(gt)
    emu_corr, emu_n = return_map_correlation(emu)

    gt_m = {
        "n_flips": float(gt_lobe["n_flips"]),
        "flip_rate": gt_lobe["flip_rate"],
        "mean_dwell": gt_lobe["mean_dwell"],
        "return_map_r": gt_corr,
        "n_maxima": float(gt_n),
        "flip_ratio": 1.0,
    }
    emu_m = {
        "n_flips": float(emu_lobe["n_flips"]),
        "flip_rate": emu_lobe["flip_rate"],
        "mean_dwell": emu_lobe["mean_dwell"],
        "return_map_r": emu_corr,
        "n_maxima": float(emu_n),
        # The headline number: the emulator switches lobes this fraction as often.
        "flip_ratio": float(
            emu_lobe["n_flips"] / gt_lobe["n_flips"] if gt_lobe["n_flips"] else np.nan
        ),
    }

    if ctx.gt_gt_trajectory is not None:
        ctl = lobe_transitions(np.asarray(ctx.gt_gt_trajectory), threshold)
        emu_m["flip_ratio_vs_gtgt_control"] = float(
            ctl["n_flips"] / gt_lobe["n_flips"] if gt_lobe["n_flips"] else np.nan
        )

    # Series for plotting.
    gt_m["_cum_flips"] = gt_lobe["cum_flips"]
    emu_m["_cum_flips"] = emu_lobe["cum_flips"]
    gt_m["_maxima"] = find_maxima_in_z(gt)
    emu_m["_maxima"] = find_maxima_in_z(emu)
    return gt_m, emu_m
