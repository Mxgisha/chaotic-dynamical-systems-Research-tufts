"""Eight measurements of an emulator, all in JAX.

Each function takes a one-step function (or a trajectory) and returns a dict of
numbers.  Nothing here plots, and nothing here decides pass or fail -- the
notebook does that, with the controls in view.

The reference for most of them is a *known constant*: Lorenz-63 has divergence
``-(sigma+1+beta)`` everywhere, so the volume-change rate has an exact target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import lorenz

DT = lorenz.DT


# --- 1. Lyapunov spectrum -------------------------------------------------
def lyapunov_spectrum(step, y0, steps=3000, dt=DT):
    """Benettin/QR spectrum of a one-step map, plus the running estimate.

    Uses ``jax.linalg.qr`` and autodiff of ``step``, so there is no hand-derived
    Jacobian to get wrong.  Returns ``(summary, running)`` where ``running`` is
    (steps, 3) and can be plotted to show convergence.

    Self-check: ``sum(lambda)`` must equal ``ln|det D step| / dt``, which for the
    true flow is the constant divergence.
    """
    jac = jax.jacrev(step)

    def body(carry, _):
        y, Q = carry
        M = jac(y) @ Q
        Qn, R = jnp.linalg.qr(M)
        # fix the sign so Q is deterministic (otherwise it flips arbitrarily)
        s = jnp.where(jnp.diag(R) < 0, -1.0, 1.0)
        Qn = Qn * s
        grow = jnp.abs(jnp.diag(R))
        yn = step(y)
        return (yn, Qn), grow

    (_, _), grow = jax.lax.scan(body, (y0, jnp.eye(3)), None, length=steps)
    t = jnp.arange(1, steps + 1)[:, None] * dt
    running = jnp.cumsum(jnp.log(grow), axis=0) / t
    final = running[-1]

    # slope of the last 20%: near zero means the estimate has converged
    k = int(0.8 * steps)
    tail = running[k:]
    x = jnp.arange(tail.shape[0], dtype=tail.dtype)
    x = x - x.mean()
    slope = jnp.sum(x[:, None] * (tail - tail.mean(axis=0)), axis=0) / jnp.sum(x * x)

    return {
        "lambda_1": final[0],
        "lambda_2": final[1],
        "lambda_3": final[2],
        "sum": jnp.sum(final),
        "divergence": jnp.asarray(lorenz.DIVERGENCE),
        "sum_error": jnp.abs(jnp.sum(final) - lorenz.DIVERGENCE),
        "slope_lambda_1": slope[0],
    }, running


# --- 2. Local contraction -------------------------------------------------
def local_contraction(step, states, dt=DT, reference=lorenz.DIVERGENCE,
                      edges=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 17.0)):
    """Volume-change rate at each state, and where it departs from the truth.

    The true value is constant, so ``excess = rate - reference`` is pure model
    error.  The by-``|x|`` breakdown shows whether the error is uniform or lives
    in particular regions.  Returns ``(summary, arrays)``.
    """
    J = jax.vmap(jax.jacrev(step))(states)
    logdet = jax.vmap(lambda m: jnp.linalg.slogdet(m)[1])(J)
    rate = logdet / dt
    excess = rate - reference
    ax = jnp.abs(states[:, 0])

    binned = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ax >= lo) & (ax < hi)
        binned.append(jnp.where(m.sum() > 0, jnp.mean(jnp.where(m, excess, 0.0)), jnp.nan))

    return {
        "rate_mean": jnp.mean(rate),
        "rate_std": jnp.std(rate),
        "rate_p05": jnp.quantile(rate, 0.05),
        "rate_p50": jnp.quantile(rate, 0.50),
        "rate_p95": jnp.quantile(rate, 0.95),
        "excess_mean": jnp.mean(excess),
        "excess_std": jnp.std(excess),
        "ratio": jnp.mean(rate) / reference,
    }, {"rate": rate, "excess": excess, "abs_x": ax, "binned_excess": jnp.stack(binned),
        "edges": jnp.asarray(edges)}


# --- 3. Error growth ------------------------------------------------------
def error_growth(step, y0, steps=1200, dt=DT, eps=1e-6):
    """How fast two nearly-identical states separate under ``step``.

    An independent route to the largest Lyapunov exponent: no Jacobian, just two
    trajectories.  Returns ``(summary, separation_curve)``.
    """
    y_a, y_b = y0, y0 + jnp.array([eps, 0.0, 0.0])

    def body(carry, _):
        a, b = carry
        an, bn = step(a), step(b)
        return (an, bn), jnp.linalg.norm(bn - an)

    (_, _), sep = jax.lax.scan(body, (y_a, y_b), None, length=steps)
    sep = jnp.maximum(sep, 1e-30)

    # least-squares slope of log(separation) over the first third, before the
    # separation saturates at the size of the attractor
    n = max(steps // 3, 8)
    t = jnp.arange(n) * dt
    y = jnp.log(sep[:n])
    t0, y0c = t - t.mean(), y - y.mean()
    rate = jnp.sum(t0 * y0c) / jnp.sum(t0 * t0)

    return {
        "growth_rate": rate,
        "t_double": jnp.log(2.0) / rate,
        "separation_final": sep[-1],
    }, sep


# --- 4. Invariant measure -------------------------------------------------
def invariant_measure(reference, other, bins=60):
    """Does the emulator spend its time in the same regions?

    Per-coordinate L1 distance between histograms over a shared range, plus
    moments of the energy ``sum x_i^2``.
    """
    out = {}
    for i, name in enumerate("xyz"):
        a, b = reference[:, i], other[:, i]
        lo, hi = jnp.minimum(a.min(), b.min()), jnp.maximum(a.max(), b.max())
        edges = jnp.linspace(lo, hi, bins + 1)
        ha, _ = jnp.histogram(a, bins=edges, density=True)
        hb, _ = jnp.histogram(b, bins=edges, density=True)
        out[f"l1_{name}"] = jnp.sum(jnp.abs(ha - hb)) * (hi - lo) / bins
    out["l1_mean"] = (out["l1_x"] + out["l1_y"] + out["l1_z"]) / 3.0

    e = jnp.sum(other**2, axis=-1)
    e_ref = jnp.sum(reference**2, axis=-1)
    out["energy_mean"] = jnp.mean(e)
    out["energy_mean_ref"] = jnp.mean(e_ref)
    out["energy_error"] = jnp.abs(jnp.mean(e) - jnp.mean(e_ref)) / jnp.mean(e_ref)
    return out


# --- 5. Correlation dimension --------------------------------------------
def correlation_dimension(traj, n_points=600, n_r=18, theiler=10):
    """Grassberger-Procaccia slope: the attractor's geometric dimension.

    A point counts as a neighbour if it is within ``r`` *and* more than
    ``theiler`` samples away in time.  Without that exclusion, temporally
    adjacent points -- which are spatially close by construction -- dominate at
    small radii and flatten the slope, biasing the estimate low.
    """
    idx = jnp.linspace(0, traj.shape[0] - 1, min(n_points, traj.shape[0])).astype(int)
    p = traj[idx]
    n = p.shape[0]
    dist = jnp.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    near = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :]) <= theiler
    dist = jnp.where(near, jnp.inf, dist)

    span = jnp.linalg.norm(p.max(0) - p.min(0))
    radii = jnp.logspace(jnp.log10(span * 1e-3), jnp.log10(span * 0.3), n_r)
    c = jax.vmap(lambda r: jnp.mean(dist < r))(radii)

    ok = (c > 1.0 / n) & (c < 0.5)
    lr, lc = jnp.log(radii[ok]), jnp.log(c[ok])
    if lr.size < 4:
        return {"d2": jnp.nan, "n_radii_used": 0}
    x, y = lr - lr.mean(), lc - lc.mean()
    d2 = jnp.sum(x * y) / jnp.sum(x * x)
    return {"d2": d2, "n_radii_used": lr.size}


# --- 6. Symbolic dynamics -------------------------------------------------
def symbolic_dynamics(traj, threshold=0.0):
    """Lobe switching: how often the x coordinate crosses the saddle, and how
    predictable the successive maxima of z are."""
    x = traj[:, 0]
    lobe = jnp.where(x >= threshold, 1.0, -1.0)
    flips = jnp.abs(jnp.diff(lobe)) > 0
    n_flips = jnp.sum(flips)

    z = traj[:, 2]
    inner = z[1:-1]
    maxima = inner[(inner > z[:-2]) & (inner > z[2:])]
    if maxima.size > 2:
        a, b = maxima[:-1], maxima[1:]
        r = jnp.corrcoef(a, b)[0, 1]
    else:
        r = jnp.nan

    return {
        "n_flips": n_flips,
        "flip_rate": n_flips / (traj.shape[0] - 1),
        "return_map_r": r,
        "n_maxima": maxima.size,
    }


# --- 7. Dynamic mode decomposition ---------------------------------------
def dmd_spectrum(traj, delays=8):
    """Eigenvalues of the best-fit linear propagator of a time-delay embedding.

    A three-row snapshot matrix gives at most three modes; stacking ``delays``
    shifted copies gives a spectrum with structure to compare.
    """
    n = traj.shape[0] - (delays - 1)
    h = jnp.concatenate([traj[d : d + n] for d in range(delays)], axis=1)
    A, B = h[:-1].T, h[1:].T
    U, S, Vh = jnp.linalg.svd(A, full_matrices=False)
    if S[0] <= 1e-12:
        # a constant trajectory has no dynamics to decompose
        return {"mag_top": jnp.nan, "mag_mean": jnp.nan,
                "spectral_gap": jnp.nan, "n_modes": 0}, jnp.zeros(1)
    keep = S > S[0] * 1e-8
    U, S, Vh = U[:, keep], S[keep], Vh[keep]
    atilde = U.T.conj() @ B @ Vh.conj().T @ jnp.diag(1.0 / S)
    mags = jnp.sort(jnp.abs(jnp.linalg.eigvals(atilde)))[::-1]
    return {
        "mag_top": mags[0],
        "mag_mean": jnp.mean(mags),
        "spectral_gap": mags[0] - mags[1] if mags.size > 1 else jnp.nan,
        "n_modes": mags.size,
    }, mags


# --- 8. Vector-field slice ------------------------------------------------
def vector_field_slice(field_fn, traj, res=12, fixed_z=None):
    """Compare the emulator's implied velocity with the true one on a 2-D slice.

    ``field_fn`` maps a state to a velocity.  For a map emulator, pass
    ``lambda y: (step(y) - y) / dt``; for a field emulator, the model itself.
    The trajectory need not visit this slice, so it probes off-manifold.
    """
    x_lo, x_hi = traj[:, 0].min(), traj[:, 0].max()
    y_lo, y_hi = traj[:, 1].min(), traj[:, 1].max()
    pad = 0.1
    gx = jnp.linspace(x_lo - pad * (x_hi - x_lo), x_hi + pad * (x_hi - x_lo), res)
    gy = jnp.linspace(y_lo - pad * (y_hi - y_lo), y_hi + pad * (y_hi - y_lo), res)
    z0 = jnp.mean(traj[:, 2]) if fixed_z is None else fixed_z
    pts = jnp.stack([g.ravel() for g in jnp.meshgrid(gx, gy)] + [jnp.full(res * res, z0)], 1)

    pred = jax.vmap(field_fn)(pts)
    true = jax.vmap(lorenz.rhs)(pts)
    err = jnp.linalg.norm(pred - true, axis=1)
    rel = err / jnp.linalg.norm(true, axis=1)
    return {
        "mean_error": jnp.mean(err),
        "max_error": jnp.max(err),
        "mean_relative_error": jnp.mean(rel),
        "fixed_z": z0,
    }, {"grid_x": gx, "grid_y": gy, "error": err.reshape(res, res)}


# --- controls -------------------------------------------------------------
def identity_step(y):
    """The floor: a model with no dynamics at all."""
    return y


def as_velocity(step, dt=DT):
    """Turn a one-step map into the velocity it implies, for the slice check."""
    return lambda y: (step(y) - y) / dt
