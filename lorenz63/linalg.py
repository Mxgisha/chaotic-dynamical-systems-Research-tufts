"""Numerical helpers shared by the diagnostics.

The original code carried a perturbation frame ``Q`` scaled by ``DELTA`` and
divided the Gram-Schmidt column norms by ``DELTA`` to recover growth factors.
That is algebraically self-cancelling *when applied consistently*, but it makes
the routine very easy to break and impossible to unit test against a known
answer.  The functions here implement the textbook form instead: ``Q`` is kept
orthonormal and the growth factors are the diagonal of ``R`` with **no**
rescaling.
"""

from __future__ import annotations

from lorenz63.backend import xp


def gram_schmidt_qr(M):
    """Reduced QR of ``M`` by modified Gram-Schmidt.

    Returns ``(Q, r)`` where ``Q`` has orthonormal columns and ``r`` holds the
    diagonal entries of ``R`` (the local growth factors).  No ``DELTA`` scaling
    is applied anywhere.  Built from a column list so the same code runs under
    both NumPy and JAX without in-place mutation.
    """
    n = M.shape[1]
    cols = []
    r = []
    for i in range(n):
        v = M[:, i]
        for q in cols:
            v = v - xp.sum(v * q) * q
        nrm = xp.sqrt(xp.sum(v * v))
        cols.append(v / nrm)
        r.append(nrm)
    Q = xp.stack(cols, axis=1)
    r = xp.stack(r)
    return Q, r


def running_lyapunov(growth, dt):
    """Cumulative running exponents from per-step growth factors.

    ``lambda_i(n) = (1 / (n*dt)) * sum_{k<=n} log r_i(k)``, shape ``(n_steps, n)``.
    """
    logs = xp.log(growth)
    cum = xp.cumsum(logs, axis=0)
    steps = xp.arange(1, growth.shape[0] + 1, dtype=growth.dtype)[:, None]
    return cum / (steps * dt)


def convergence_slope(running):
    """Slope of each running exponent over the last 20% of the window.

    A slope near zero means the spectrum has plateaued and the reported value is
    trustworthy; the original pipeline reported values that were still drifting.
    """
    n = running.shape[0]
    start = int(n * 0.8)
    tail = running[start:]
    t = xp.arange(tail.shape[0], dtype=tail.dtype)
    t = t - xp.mean(t)
    num = xp.sum(t[:, None] * (tail - xp.mean(tail, axis=0)), axis=0)
    den = xp.sum(t * t)
    return num / den
