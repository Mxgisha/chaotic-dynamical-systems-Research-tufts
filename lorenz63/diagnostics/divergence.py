"""Error growth rate -- replaces the trajectory-uniqueness test.

The original ``run_traj_uniqueness_test`` perturbed the initial condition by
``1e-5``, counted distinct final states, and printed

    TEST PASSED -> Variations successfully exploded into unique paths.

for 40/40 distinct endpoints.  That is a statement about *chaos*, not about
emulator fidelity: an identity map perturbs to 40 distinct endpoints too, and so
would a randomly initialised network.  It cannot fail for any model with a
non-degenerate output, which makes it uninformative.

What actually matters is whether the emulator's **error-doubling** behaviour
matches the true system: same Lyapunov time, same saturation floor.  This module
measures

* ``growth_rate`` -- the slope of ``log ||delta(t)||`` before saturation, i.e. an
  independent estimate of ``lambda_1`` from twin trajectories,
* ``t_double`` -- the error-doubling time ``ln 2 / growth_rate``,
* ``t_sat`` -- the time at which the separation reaches the attractor scale and
  the growth stops being informative,

each across several perturbation scales so the estimate can be checked for
scale-independence.  The distinct-endpoint count is retained only as a
descriptive field, never as pass/fail.
"""

from __future__ import annotations

import numpy as np

from lorenz63.diagnostics import register


def twin_separation(model, ic, scales, steps, dt):
    """Mean pairwise separation vs time for several perturbation scales."""
    ic = np.asarray(ic, dtype=np.float64)
    rng = np.random.default_rng(0)
    out = {}
    for eps in scales:
        # A few random directions per scale; average the log-separation.
        dirs = rng.normal(size=(5, ic.size))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        seps = np.zeros(steps)
        for d in dirs:
            base = ic[None, :]
            pert = ic[None, :] + eps * d[None, :]
            states = np.concatenate([base, pert], axis=0)
            logs = np.zeros(steps)
            for k in range(steps):
                states = np.asarray(model(states), dtype=np.float64)
                sep = np.linalg.norm(states[1] - states[0])
                logs[k] = sep
            seps += logs
        out[float(eps)] = seps / len(dirs)
    return out


def growth_rate(seps, dt, sat_frac=0.5):
    """Slope of log separation over the pre-saturation window.

    Returns ``(rate, t_sat_steps)``.  Saturation is detected as the first time the
    separation stalls (its growth rate over a trailing window falls below a
    quarter of the initial rate), which is where trajectory-level skill ends.
    """
    seps = np.asarray(seps, dtype=np.float64)
    valid = seps > 0
    if valid.sum() < 8:
        return float("nan"), 0
    log_s = np.log(np.where(valid, seps, np.nan))
    n = seps.size
    # initial rate from the first 10% of the run
    head = max(4, n // 10)
    t0 = np.arange(head) * dt
    r0 = np.polyfit(t0, log_s[:head], 1)[0] if np.all(np.isfinite(log_s[:head])) else np.nan
    # saturation: where a trailing estimate drops well below r0
    t_sat = n
    for k in range(head, n):
        w0, w1 = max(0, k - head), k
        seg = log_s[w0:w1]
        if not np.all(np.isfinite(seg)):
            continue
        r = np.polyfit(np.arange(seg.size) * dt, seg, 1)[0]
        if np.isfinite(r0) and r0 > 0 and r < 0.25 * r0:
            t_sat = k
            break
    end = max(head + 2, int(t_sat * sat_frac))
    seg = log_s[:end]
    if not np.all(np.isfinite(seg)) or seg.size < 4:
        return float("nan"), int(t_sat)
    rate = float(np.polyfit(np.arange(seg.size) * dt, seg, 1)[0])
    return rate, int(t_sat)


def _metrics_for(model, ic, scales, steps, dt):
    seps = twin_separation(model, ic, scales, steps, dt)
    rates, tsats = [], []
    for eps, s in seps.items():
        r, ts = growth_rate(s, dt)
        if np.isfinite(r):
            rates.append(r)
            tsats.append(ts)
    rate = float(np.median(rates)) if rates else float("nan")
    t_sat = int(np.median(tsats)) if tsats else 0
    return {
        "growth_rate": rate,
        "t_double": float(np.log(2.0) / rate) if rate and np.isfinite(rate) and rate > 0 else float("nan"),
        "t_sat": t_sat,
        "n_scales_used": len(rates),
        "_seps": seps,
    }


@register("error_growth")
def error_growth(gt, emu, ctx):
    """Error-growth rate of the emulator vs the true flow."""
    scales = ctx.options.get("scales", (1e-4, 1e-5, 1e-6, 1e-7, 1e-8))
    steps = int(ctx.options.get("steps", 2000))
    dt = ctx.dt
    ic = np.asarray(ctx.ic if ctx.ic is not None else gt[0], dtype=np.float64)

    from lorenz63 import system as S

    # Ground truth as a map: one RK4 step per unit step.
    class _Flow:
        def __call__(self, states):
            return np.stack([S.flow_map(s, dt) for s in np.asarray(states)])

    gt_m = _metrics_for(_Flow(), ic, scales, steps, dt)

    if ctx.model is not None:
        emu_m = _metrics_for(ctx.model, ic, scales, steps, dt)
    else:
        emu_m = dict(gt_m)

    if np.isfinite(gt_m["growth_rate"]) and gt_m["growth_rate"] != 0:
        emu_m["growth_rate_ratio"] = emu_m["growth_rate"] / gt_m["growth_rate"]
    else:
        emu_m["growth_rate_ratio"] = float("nan")
    gt_m["growth_rate_ratio"] = 1.0

    # Descriptive only -- explicitly NOT a pass/fail criterion.
    if ctx.identity_trajectory is not None:
        ident = np.asarray(ctx.identity_trajectory)
        emu_m["identity_n_unique_endpoints"] = float(len(np.unique(ident[-1:], axis=0)))
    return gt_m, emu_m
