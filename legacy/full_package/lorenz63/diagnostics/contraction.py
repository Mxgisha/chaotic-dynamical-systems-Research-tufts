"""Local volume contraction: why is the emulated flow over-dissipative?

The Lyapunov study established that ``lambda_1 + lambda_2 + lambda_3`` for the
emulator is far too negative -- the learned map destroys volume much too fast --
while the attractor's *shape* statistics match.  This module localises that error
in state space.

The lever is that Lorenz-63 has **constant divergence**,

    div f = -(sigma + 1 + beta) = -13.6667  everywhere,

so the true one-step flow map contracts volume by ``-(sigma+1+beta)*dt`` at
*every* state.  There is no state dependence in the ground truth, so any state
dependence in the emulator's contraction is model error with no confound.

Measured per state (the emulator is probed at the same states as the truth):

* ``ln|det D phi_emu|`` from finite differences of the emulator's step map
  (checked against an exact analytic MLP Jacobian: agreement to ~5e-9);
* the same for the true RK4 flow map, as a control -- this is *not* exactly
  constant, because RK4 does not preserve the divergence identity exactly; that
  finite spread is the numerical floor against which the emulator is judged;
* the leading expansion rate ``ln||D phi||_2``;
* stratification by ``|x|`` (distance from the saddle) and by ``z``, which is
  what turns "too dissipative" into "too dissipative *here*".

The cross-check is ``contraction_rate_mean``: it is the per-step, state-resolved
form of the ``trace_invariant`` reported by the Lyapunov diagnostic, so the two
should agree.
"""

from __future__ import annotations

import numpy as np

from lorenz63 import system as S
from lorenz63.diagnostics import register

#: The continuous-time contraction rate implied by the divergence.
TRUE_LOG_DET_PER_STEP = S.DIVERGENCE

#: Bins used to stratify the error in state space.
X_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 17.0)
Z_EDGES = (0.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0)


def fixed_points(sigma=S.SIGMA, rho=S.RHO, beta=S.BETA):
    """The three equilibria of Lorenz-63.

    ``C = sqrt(beta * (rho - 1))``: the origin plus the two lobe centres, which
    are the states the trajectory spirals around.
    """
    c = np.sqrt(beta * (rho - 1.0))
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [c, c, rho - 1.0],
            [-c, -c, rho - 1.0],
        ]
    )


def distance_to_fixed_points(state, points=None):
    """Euclidean distance from ``state`` to the nearest equilibrium."""
    points = fixed_points() if points is None else points
    return float(np.min(np.linalg.norm(points - np.asarray(state), axis=1)))


def local_contraction(jac):
    """``(ln|det J|, ln||J||_2)`` for one Jacobian.

    The determinant gives the volume change; the largest singular value gives
    the leading expansion.  Together they separate "contracts too much" from
    "fails to expand".
    """
    sign, logdet = np.linalg.slogdet(jac)
    if sign == 0:
        logdet = np.inf
    sv = np.linalg.svd(jac, compute_uv=False)
    return float(logdet), float(np.log(sv[0]))


def emulator_jacobian(model, x, eps=1e-6):
    """Central-difference Jacobian of the emulator's one-step map."""
    from lorenz63.diagnostics.lyapunov import model_jacobian_fd

    return model_jacobian_fd(model, x, eps=eps)


def true_jacobian(x, dt):
    from lorenz63.diagnostics.lyapunov import flow_jacobian

    return flow_jacobian(x, dt)


def measure_states(jac_at, states, dt):
    """Per-state ``(contraction_rate, expansion_rate)`` for a given Jacobian."""
    rates = np.empty(len(states))
    exps = np.empty(len(states))
    for i, x in enumerate(states):
        ld, ex = local_contraction(jac_at(x))
        rates[i] = ld / dt
        exps[i] = ex / dt
    return rates, exps


def _safe_nanmean(values):
    """``nanmean`` that returns NaN instead of warning on an all-NaN input."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _stratified(values, coordinate, edges):
    """Mean of ``values`` within bins of ``coordinate``, plus the bin counts."""
    means, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (coordinate >= lo) & (coordinate < hi)
        counts.append(int(m.sum()))
        means.append(_safe_nanmean(values[m]) if m.any() else float("nan"))
    return np.asarray(means), np.asarray(counts)


def _summarise(rate, expansion, excess, states, reference_rate):
    """Summarise one map's per-state contraction.

    ``reference_rate`` is the ground-truth contraction rate to compare against;
    it is passed in rather than derived from the map being summarised, because
    ``rate - excess`` is only the truth for the emulator's own excess.
    """
    ax = np.abs(states[:, 0])
    z = states[:, 2]
    x_mean, x_n = _stratified(rate, ax, X_EDGES)
    z_mean, z_n = _stratified(rate, z, Z_EDGES)
    x_excess, _ = _stratified(excess, ax, X_EDGES)
    z_excess, _ = _stratified(excess, z, Z_EDGES)

    # ratio = emulator contraction / true contraction; >1 means over-dissipative.
    if reference_rate != 0:
        ratio = float(np.mean(rate) / reference_rate)
        per_state = rate / reference_rate
    else:
        ratio = float("nan")
        per_state = np.full_like(rate, np.nan)

    return {
        "contraction_rate_mean": float(np.mean(rate)),
        "contraction_rate_std": float(np.std(rate)),
        "contraction_rate_p05": float(np.quantile(rate, 0.05)),
        "contraction_rate_p50": float(np.quantile(rate, 0.50)),
        "contraction_rate_p95": float(np.quantile(rate, 0.95)),
        "expansion_rate_mean": float(np.mean(expansion)),
        "expansion_rate_std": float(np.std(expansion)),
        "true_divergence": float(S.DIVERGENCE),
        "divergence_error": float(abs(np.mean(rate) - S.DIVERGENCE)),
        "excess_mean": float(np.mean(excess)),
        "excess_std": float(np.std(excess)),
        "contraction_ratio": ratio,
        "contraction_ratio_std": float(np.nanstd(per_state)) if np.any(np.isfinite(per_state)) else float("nan"),
        # The key state-dependence numbers.  A bin with no samples yields NaN
        # (possible on a short trajectory that visits only part of the attractor),
        # which propagates rather than silently becoming 0.  The one exception is
        # an excess that is identically zero -- the ground-truth control -- which
        # has no state dependence by definition.
        "excess_near_saddle": _safe_nanmean(x_excess[:1]),  # |x| < 2
        "excess_mid_lobe": _safe_nanmean(x_excess[3:5]),  # 6 <= |x| < 10
        "state_dependence": (
            0.0
            if (not np.any(np.asarray(excess) != 0)) or float(np.std(excess)) < 1e-12
            else _safe_nanmean(x_excess[3:5]) - _safe_nanmean(x_excess[:1])
        ),
        # plot state
        "_rate": rate,
        "_expansion": expansion,
        "_excess": excess,
        "_ax": ax,
        "_z": z,
        "_states": states,
        "_x_edges": np.asarray(X_EDGES, dtype=float),
        "_z_edges": np.asarray(Z_EDGES, dtype=float),
        "_x_mean": x_mean,
        "_z_mean": z_mean,
        "_x_excess": x_excess,
        "_z_excess": z_excess,
        "_x_counts": x_n,
        "_z_counts": z_n,
    }


@register("local_contraction")
def local_contraction_diagnostic(gt, emu, ctx):
    """Per-state contraction of the emulator against the constant truth.

    ``gt`` supplies the states to probe; both maps are evaluated at those same
    states, so the comparison is paired.
    """
    dt = ctx.dt
    states = np.asarray(gt, dtype=np.float64)
    n = int(ctx.options.get("n_states", 600))
    if states.shape[0] > n:
        idx = np.linspace(0, states.shape[0] - 1, n).astype(int)
        states = states[idx]

    gt_rate, gt_exp = measure_states(lambda x: true_jacobian(x, dt), states, dt)
    reference_rate = float(np.mean(gt_rate))
    zero = np.zeros_like(gt_rate)
    gt_m = _summarise(gt_rate, gt_exp, zero, states, reference_rate)

    if ctx.model is not None:
        emu_rate, emu_exp = measure_states(
            lambda x: emulator_jacobian(ctx.model, x), states, dt
        )
        excess = emu_rate - gt_rate
        emu_m = _summarise(emu_rate, emu_exp, excess, states, reference_rate)
    else:
        emu_m = dict(gt_m)

    # Does the error track the true local expansion?  A negative correlation means
    # the emulator damps least where the truth stretches fastest.
    if (gt_m["expansion_rate_std"] > 1e-9
            and emu_m["excess_std"] > 1e-9
            and emu_m["expansion_rate_std"] > 1e-9):
        emu_m["corr_expansion_vs_excess"] = float(
            np.corrcoef(gt_m["_expansion"], emu_m["_excess"])[0, 1]
        )
        emu_m["corr_expansion_gt_vs_emu"] = float(
            np.corrcoef(gt_m["_expansion"], emu_m["_expansion"])[0, 1]
        )
    else:
        # a constant expansion (e.g. an identity or scaled-identity map) has no
        # correlation to measure
        emu_m["corr_expansion_vs_excess"] = float("nan")
        emu_m["corr_expansion_gt_vs_emu"] = float("nan")
    gt_m["corr_expansion_vs_excess"] = 0.0
    gt_m["corr_expansion_gt_vs_emu"] = 1.0

    # How well does the emulator's per-state contraction correlate with the
    # truth's?  For a constant truth this is ~0 by construction, which is itself
    # the point: the emulator invents state dependence the system does not have.
    if gt_m["contraction_rate_std"] > 1e-12 and emu_m["contraction_rate_std"] > 1e-12:
        emu_m["corr_contraction_gt_vs_emu"] = float(
            np.corrcoef(gt_m["_rate"], emu_m["_rate"])[0, 1]
        )
    else:
        emu_m["corr_contraction_gt_vs_emu"] = float("nan")
    gt_m["corr_contraction_gt_vs_emu"] = 1.0
    return gt_m, emu_m


__all__ = [
    "local_contraction_diagnostic",
    "local_contraction",
    "fixed_points",
    "distance_to_fixed_points",
    "measure_states",
    "TRUE_LOG_DET_PER_STEP",
    "X_EDGES",
    "Z_EDGES",
]
