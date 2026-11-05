"""Correlation dimension.

The previous ``calculate_fractal_dimension`` built a full ``N x N`` distance
matrix for ``N = 5000`` (25M entries) and subtracted ``N`` from the pair count to
remove self-pairs -- correct in effect, but O(N^2) memory and silently
inconsistent between the ground truth and the emulator if their trajectories had
different lengths.

This version uses a Theiler window (so temporally adjacent points, which are
correlated by construction, do not masquerade as small-scale structure),
compares both trajectories at a **matched subsample size**, and reports a
bootstrap confidence interval so two values can actually be distinguished.
"""

from __future__ import annotations

import numpy as np

from lorenz63.diagnostics import register


def correlation_integral(points, radii, theiler=10):
    """Grassberger-Procaccia ``C(r)`` with a Theiler window."""
    n = points.shape[0]
    c = np.zeros(len(radii))
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1)
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        d[lo:hi] = np.inf  # exclude the temporal neighbourhood of i
        for k, r in enumerate(radii):
            c[k] += np.sum(d < r)
    norm = n * (n - 1)
    return c / norm if norm > 0 else c


def fit_dimension(radii, c, lo_frac=0.1, hi_frac=0.9):
    """Slope of log C(r) vs log r over the scaling region.

    Returns ``(d2, intercept, mask)``.  The window excludes the extreme radii
    where finite-sample saturation and empty-bin noise dominate.  A saturated
    window (where every C(r) is the same, i.e. below the 1/N floor) yields NaN
    rather than a fabricated slope.
    """
    mask = (c > 0) & np.isfinite(c)
    if mask.sum() < 4:
        return float("nan"), float("nan"), mask
    lr = np.log(radii[mask])
    lc = np.log(c[mask])
    order = np.argsort(lr)
    lr, lc = lr[order], lc[order]
    n = len(lr)
    k0 = max(1, int(n * lo_frac))
    k1 = max(k0 + 2, int(n * hi_frac))
    x, y = lr[k0:k1], lc[k0:k1]
    full = np.zeros_like(mask)
    idx = np.flatnonzero(mask)[order][k0:k1]
    full[idx] = True
    if x.size < 3 or np.ptp(y) < 1e-9:
        # no measurable scaling region (e.g. a collapsed point cloud)
        return float("nan"), float("nan"), full
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept), full


def dimension_with_ci(points, radii, theiler=10, n_boot=20, rng=None):
    """D2 with a bootstrap CI over contiguous trajectory blocks."""
    rng = rng or np.random.default_rng(0)
    d2, intercept, mask = fit_dimension(radii, correlation_integral(points, radii, theiler))
    n = points.shape[0]
    boots = []
    for _ in range(n_boot):
        start = rng.integers(0, max(1, n // 2))
        sub = points[start : start + n // 2]
        if sub.shape[0] < 50:
            continue
        d, _, _ = fit_dimension(radii, correlation_integral(sub, radii, theiler))
        if np.isfinite(d):
            boots.append(d)
    lo = float(np.percentile(boots, 5)) if boots else float("nan")
    hi = float(np.percentile(boots, 95)) if boots else float("nan")
    return d2, lo, hi, mask, intercept


@register("correlation_dimension")
def correlation_dimension(gt, emu, ctx):
    """Correlation dimension of the true and emulated attractors."""
    gt = np.asarray(gt, dtype=np.float64)
    emu = np.asarray(emu, dtype=np.float64)

    n = int(ctx.options.get("n_points", 600))
    theiler = int(ctx.options.get("theiler", 10))
    n_boot = int(ctx.options.get("n_boot", 20))
    rng = ctx.rng if ctx.rng is not None else np.random.default_rng(0)

    def subsample(t):
        if t.shape[0] <= n:
            return t
        idx = np.linspace(0, t.shape[0] - 1, n).astype(int)
        return t[idx]

    a, b = subsample(gt), subsample(emu)

    # Common radius ladder spanning both attractors.  The span must be large
    # enough to resolve a scaling region: for a degenerate point-cloud (an
    # emulator that has collapsed to a fixed point) a span derived from the
    # attractor diameter alone yields radii below the inter-point spacing, where
    # C(r) sits at its 1/N floor and the fitted slope is meaningless.
    span = float(np.linalg.norm(a.max(0) - a.min(0)))
    span = max(span, 1e-3)
    radii = np.logspace(np.log10(span * 1e-3), np.log10(span * 0.3), 24)
    if radii[0] <= 0 or radii[-1] <= radii[0]:
        radii = np.logspace(-3, 1, 24)

    d_gt, lo_gt, hi_gt, mask_gt, _ = dimension_with_ci(a, radii, theiler, n_boot, rng)
    d_emu, lo_emu, hi_emu, mask_emu, _ = dimension_with_ci(b, radii, theiler, n_boot, rng)

    gt_m = {
        "d2": d_gt,
        "d2_lo": lo_gt,
        "d2_hi": hi_gt,
        "n_points": int(a.shape[0]),
    }
    emu_m = {
        "d2": d_emu,
        "d2_lo": lo_emu,
        "d2_hi": hi_emu,
        "n_points": int(b.shape[0]),
        "d2_ratio": d_emu / d_gt if d_gt and np.isfinite(d_gt) and d_gt != 0 else float("nan"),
    }

    # Empirical C(r) curves are kept for plotting (never computed in a renderer).
    emu_m["_radii"] = radii
    emu_m["_corr_emu"] = correlation_integral(b, radii, theiler)
    gt_m["_radii"] = radii
    gt_m["_corr_gt"] = correlation_integral(a, radii, theiler)
    return gt_m, emu_m
