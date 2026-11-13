"""Estimating local expansion from data alone.

Given pairs ``(x_n, x_{n+1})`` and no equations, the local Jacobian of the step
map can be estimated by a linear fit over nearest neighbours.  Its
log-determinant divided by the timestep is the local volume-change rate.

This is what makes the divergence penalty usable when the answer is *not* known
analytically.  Step 2 shows the catch: on-attractor pairs do not identify it,
because the transverse direction is never sampled.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def estimate_at_points(X, Y, points, k=50, dt=0.01):
    """Local volume-change rate at ``points``, using ``(X, Y)`` as the cloud.

    For each point the ``k`` nearest members of the cloud are collected, centred,
    and one 3x3 linear map is fitted by least squares.  The cloud and the
    evaluation points need not be the same set: the estimate at a state on the
    attractor can use neighbours from transient data, which is the difference
    between a biased and an almost unbiased estimate.
    """
    X = jnp.asarray(X, jnp.float64)
    Y = jnp.asarray(Y, jnp.float64)
    points = jnp.asarray(points, jnp.float64)

    def one(p):
        d = jnp.linalg.norm(X - p, axis=1)
        idx = jnp.argsort(d)[:k]
        Xc = X[idx] - jnp.mean(X[idx], axis=0)
        Yc = Y[idx] - jnp.mean(Y[idx], axis=0)
        M = jnp.linalg.lstsq(Xc, Yc)[0]      # Yc ~= Xc @ M
        return jnp.linalg.slogdet(M.T)[1]    # map is y = (M^T) x

    return jax.vmap(one)(points) / dt


def estimate_at_points_chunked(X, Y, points, k=50, dt=0.01, chunk=100):
    """As :func:`estimate_at_points`, in chunks to bound memory.

    A full vmap over thousands of points materialises ``(points, n, 3)``
    distances, which is gigabytes at this size; chunking keeps it flat.
    """
    points = jnp.asarray(points)
    out = []
    for start in range(0, len(points), chunk):
        out.append(estimate_at_points(X, Y, points[start:start + chunk], k=k, dt=dt))
    return jnp.concatenate(out)


def spread_anchors(n, stride=10):
    """A deterministic, evenly spaced subset of indices."""
    return jnp.arange(0, n, stride)
