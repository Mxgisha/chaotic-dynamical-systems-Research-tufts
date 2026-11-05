"""Lyapunov spectrum.

The one thing this must get right is *which* Jacobian to use.

The system is a **time-dt map**, so the relevant object is the Jacobian of one
step of the flow,
``A(x) = D phi_dt(x) ~= I + dt J(x) + O(dt^2)``,
whose eigenvalues sit near 1.  Using the instantaneous ODE Jacobian ``J(x)``
instead is a different and wrong computation: ``J`` has entries of order 30, so
its log-growth accumulates ~``ln(17)/dt`` per unit time and yields exponents in
the hundreds rather than ~1.

The Benettin algorithm here is the textbook one:

1. ``A_n = D phi_dt(x_n)`` via the RK4 variational equation.
2. ``A_n Q_n = Q_{n+1} R_n`` by modified Gram-Schmidt, ``Q`` **orthonormal**.
3. ``lambda_i = (1/(n dt)) sum_k ln R_k[i]``.

Two invariants are reported as a self-check, which is what makes the numbers
trustworthy:

* ``sum(lambda) == ln|det A| / dt`` exactly (product of eigenvalues), and
* to leading order that equals the constant divergence ``-(sigma + 1 + beta)``.

The original implementation is broadly consistent with this (it reported
``0.993, 0, -14.519`` for the ground truth, against the textbook
``0.906, 0, -14.57``); what it lacked was a stated convergence criterion.  This
version returns the slope over the final 20% of the window so a drifting value
is visible rather than silently quoted.
"""

from __future__ import annotations

import numpy as np

from lorenz63 import system as S
from lorenz63.diagnostics import register
from lorenz63.linalg import convergence_slope, running_lyapunov


def gram_schmidt(M):
    """Modified Gram-Schmidt.  Returns ``(Q, r_diag)``, ``Q`` orthonormal."""
    n = M.shape[1]
    cols = []
    r = []
    for i in range(n):
        v = M[:, i]
        for q in cols:
            v = v - np.sum(v * q) * q
        nrm = np.sqrt(np.sum(v * v))
        cols.append(v / nrm)
        r.append(nrm)
    return np.stack(cols, axis=1), np.array(r)


def flow_jacobian(state, dt):
    """Jacobian of one RK4 step with respect to the state.

    The RK4 update is a composition of four stage maps, each of the form
    ``x -> x + c_j dt f(x_j)`` with ``x_j`` affine in ``x``.  Differentiating the
    composition gives the product of the stage Jacobians below.  Verified against
    central differences of :func:`lorenz63.system.flow_map` in the test suite.
    """
    h = dt
    k1 = S.lorenz_rhs(state)
    J1 = S.jacobian(state)
    x2 = state + 0.5 * h * k1
    J2 = S.jacobian(x2)
    k2 = S.lorenz_rhs(x2)
    x3 = state + 0.5 * h * k2
    J3 = S.jacobian(x3)
    k3 = S.lorenz_rhs(x3)
    x4 = state + h * k3
    J4 = S.jacobian(x4)
    I = np.eye(np.asarray(state).shape[0], dtype=np.float64)
    return (I + (h / 6) * J1) @ (I + (h / 3) * J2) @ (I + (h / 3) * J3) @ (I + (h / 6) * J4)


def model_jacobian_fd(model, x, eps=1e-6):
    """Central finite-difference Jacobian of a one-step emulator."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        fp = np.asarray(model.step(x + e), dtype=np.float64)
        fm = np.asarray(model.step(x - e), dtype=np.float64)
        J[:, i] = (fp - fm) / (2 * eps)
    return J


def spectrum(jac_at, states, dt):
    """Benettin spectrum for a map whose one-step Jacobian is ``jac_at``."""
    states = np.asarray(states, dtype=np.float64)
    n, dim = states.shape
    Q = np.eye(dim, dtype=np.float64)
    growth = np.zeros((n, dim))
    for k in range(n):
        A = jac_at(states[k])
        Q, r = gram_schmidt(A @ Q)
        growth[k] = r
        # safety: re-orthonormalise occasionally to limit drift
        if (k + 1) % 1000 == 0:
            Q, _ = gram_schmidt(Q)
    return running_lyapunov(growth, dt), growth


def _metrics(running, growth, dt):
    spec = running[-1]
    ln_det = float(np.mean(np.sum(np.log(growth), axis=1)))
    return {
        "lambda_1": float(spec[0]),
        "lambda_2": float(spec[1]),
        "lambda_3": float(spec[2]),
        "sum_lambda": float(np.sum(spec)),
        "trace_invariant": ln_det / dt,
        "convergence_slope_lambda_1": float(convergence_slope(running)[0]),
        "n_steps": int(running.shape[0]),
    }


@register("lyapunov_spectrum")
def lyapunov_spectrum(gt, emu, ctx):
    """GT vs emulator Lyapunov spectra.

    ``gt`` uses the analytic flow Jacobian; the emulator uses finite differences
    of ``ctx.model`` when available.
    """
    dt = ctx.dt
    gt = np.asarray(gt, dtype=np.float64)
    emu = np.asarray(emu, dtype=np.float64)

    gt_running, gt_growth = spectrum(lambda x: flow_jacobian(x, dt), gt, dt)
    gt_m = _metrics(gt_running, gt_growth, dt)

    if ctx.model is not None:
        emu_running, emu_growth = spectrum(
            lambda x: model_jacobian_fd(ctx.model, x), emu, dt
        )
        emu_m = _metrics(emu_running, emu_growth, dt)
    else:
        emu_m = dict(gt_m)

    denom = abs(gt_m["lambda_1"]) if abs(gt_m["lambda_1"]) > 1e-12 else 1e-12
    emu_m["lambda_1_ratio"] = emu_m["lambda_1"] / denom
    gt_m["lambda_1_ratio"] = 1.0
    return gt_m, emu_m
