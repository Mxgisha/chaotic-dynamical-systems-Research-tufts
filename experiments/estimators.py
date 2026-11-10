"""Estimating local expansion from data alone.

Given pairs ``(x_n, x_{n+1})`` and no equations, the local Jacobian of the step
map can be estimated by a local linear fit over nearest neighbours.  Its
log-determinant divided by the timestep is the local volume-change rate.

This is what makes the divergence penalty usable when the answer is *not* known
analytically.  It is also, as step 2 shows, a small model of the emulator itself:
both are local smoothers, and both over-contract when the inputs are noisy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def knn_logdet(X, Y, anchors, k=50):
    """``ln|det A_i|`` for a local linear fit around each anchor state.

    ``X`` and ``Y`` are the input and target clouds, ``anchors`` indices into
    ``X``.  For each anchor the ``k`` nearest inputs are collected, centred, and
    a single 3x3 linear map is fitted by least squares.
    """
    X = jnp.asarray(X, jnp.float64)
    Y = jnp.asarray(Y, jnp.float64)

    def one(i):
        d = jnp.linalg.norm(X - X[i], axis=1)
        idx = jnp.argsort(d)[:k]
        Xc = X[idx] - jnp.mean(X[idx], axis=0)
        Yc = Y[idx] - jnp.mean(Y[idx], axis=0)
        M = jnp.linalg.lstsq(Xc, Yc)[0]          # Yc ~= Xc @ M
        return jnp.linalg.slogdet(M.T)[1]        # map is y = (M^T) x

    return jax.vmap(one)(jnp.asarray(anchors))


def estimate_divergence(X, Y, anchors, k=50, dt=0.01):
    """Per-anchor estimate of ``ln|det D step| / dt``, anchors indexing ``X``."""
    return knn_logdet(X, Y, anchors, k=k) / dt


def estimate_at_points(X, Y, points, k=50, dt=0.01):
    """Estimate the local rate at arbitrary states, using ``(X, Y)`` as the cloud.

    The cloud and the evaluation points need not be the same set: the estimate at
    a state on the attractor can use neighbours drawn from transient data, which
    is the difference between a biased and an almost unbiased estimate.
    """
    X = jnp.asarray(X, jnp.float64)
    Y = jnp.asarray(Y, jnp.float64)
    points = jnp.asarray(points, jnp.float64)

    def one(p):
        d = jnp.linalg.norm(X - p, axis=1)
        idx = jnp.argsort(d)[:k]
        Xc = X[idx] - jnp.mean(X[idx], axis=0)
        Yc = Y[idx] - jnp.mean(Y[idx], axis=0)
        M = jnp.linalg.lstsq(Xc, Yc)[0]
        return jnp.linalg.slogdet(M.T)[1]

    return jax.vmap(one)(points) / dt


def estimate_at_points_chunked(X, Y, points, k=50, dt=0.01, chunk=100):
    points = jnp.asarray(points)
    out = []
    for start in range(0, len(points), chunk):
        out.append(estimate_at_points(X, Y, points[start:start + chunk], k=k, dt=dt))
    return jnp.concatenate(out)


def estimate_divergence_chunked(X, Y, anchors, k=50, dt=0.01, chunk=100):
    """As :func:`estimate_divergence`, but in chunks to bound memory.

    A full vmap over thousands of anchors materialises ``(anchors, n, 3)``
    distances, which is gigabytes at this size; chunking keeps it flat.
    """
    anchors = jnp.asarray(anchors)
    out = []
    for start in range(0, len(anchors), chunk):
        block = anchors[start : start + chunk]
        out.append(estimate_divergence(X, Y, block, k=k, dt=dt))
    return jnp.concatenate(out)


def spread_anchors(n, stride=10):
    """A deterministic, evenly spaced subset of indices."""
    return jnp.arange(0, n, stride)
