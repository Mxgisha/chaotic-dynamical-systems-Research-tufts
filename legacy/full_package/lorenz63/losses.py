"""Loss terms.

Three terms, each returning a scalar plus everything needed to differentiate it.

* :func:`mse_loss` -- the one-step (or rollout) mean squared error.
* :func:`l1_distribution_distance` -- distributional mismatch between the true
  and predicted next-state clouds.  Computed with a *soft* histogram so it is
  differentiable with respect to the predictions; a hard ``jnp.histogram`` has no
  useful gradient.
* :func:`ftle_regularizer` -- **disabled by default**.  The original
  implementation perturbed along ``ones_like(x) * eps`` (an arbitrary direction,
  not the leading Lyapunov vector) and compared a per-step expansion rate against
  a hard-coded continuous-time exponent target.  Following the spec this is off
  unless a target is supplied, and the target comes from config rather than being
  baked in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LossConfig:
    l1_weight: float = 0.1
    ftle_weight: float = 0.0  # spec default: off
    ftle_target: float | None = None  # must be set explicitly to enable
    n_bins: int = 30
    #: weight on a penalty that drives the learned vector field's divergence
    #: toward ``divergence_target``.  Measured at the training samples; used only
    #: by the vector-field parameterisation.
    divergence_weight: float = 0.0
    divergence_target: float | None = None

    def __post_init__(self):
        if self.ftle_target is not None and self.ftle_weight == 0.0:
            raise ValueError(
                "ftle_target was set but ftle_weight is 0; set ftle_weight > 0 "
                "or leave ftle_target as None"
            )
        if self.divergence_target is not None and self.divergence_weight == 0.0:
            raise ValueError(
                "divergence_target was set but divergence_weight is 0; set "
                "divergence_weight > 0 or leave divergence_target as None"
            )


def mse_loss(pred, target):
    d = pred - target
    return float(np.mean(d * d))


def mse_grad(pred, target):
    """d/dpred of :func:`mse_loss`."""
    n = pred.size
    return 2.0 * (pred - target) / n


def l1_distribution_distance(real, emu, n_bins=30):
    """Average per-coordinate L1 distance between two empirical distributions.

    Averages over coordinates, using a common bin range spanning both samples.
    Non-finite samples are dropped so this agrees exactly with
    :func:`soft_l1_distribution_distance` (``np.histogram`` ignores NaN, while
    ``np.digitize`` clips it into the last bin; the two must not diverge).
    """
    real = np.asarray(real, dtype=np.float64)
    emu = np.asarray(emu, dtype=np.float64)
    total = 0.0
    for i in range(real.shape[-1]):
        r = real[..., i].ravel()
        e = emu[..., i].ravel()
        r = r[np.isfinite(r)]
        e = e[np.isfinite(e)]
        if r.size == 0 or e.size == 0:
            continue
        lo = min(r.min(), e.min())
        hi = max(r.max(), e.max())
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        width = (hi - lo) / n_bins
        rh, _ = np.histogram(r, bins=edges, density=True)
        eh, _ = np.histogram(e, bins=edges, density=True)
        total += np.sum(np.abs(rh - eh)) * width
    return float(total / real.shape[-1])


def soft_l1_distribution_distance(real, emu, n_bins=30, edges=None, bandwidth=None):
    """Differentiable form of :func:`l1_distribution_distance`.

    Returns ``(value, backward)`` where ``backward(grad_value)`` gives
    ``d value / d emu`` with **the same shape as** ``emu``.

    Two things had to be right, and the first attempt got both wrong:

    1. **One histogram per coordinate**, matching
       :func:`l1_distribution_distance`.  Flattening all coordinates into a
       single histogram trains a different quantity than the one reported.
    2. **Differentiable bin assignment.**  Placing each sample in a single bin
       with ``digitize`` gives a piecewise-constant objective whose gradient is
       zero almost everywhere and infinite at bin edges.  Here each sample
       contributes to bin ``b`` through a triangular kernel of width
       ``bandwidth`` centred on the bin, so the objective is piecewise linear in
       the predictions and its gradient is the finite difference you would
       measure.
    """
    real = np.asarray(real, dtype=np.float64)
    emu = np.asarray(emu, dtype=np.float64)
    dim = real.shape[-1]
    r2 = real.reshape(-1, dim)
    e2 = emu.reshape(-1, dim)
    r2 = np.where(np.isfinite(r2), r2, np.nan)
    e2 = np.where(np.isfinite(e2), e2, np.nan)

    if edges is None:
        edges = []
        for i in range(dim):
            ri = r2[:, i][np.isfinite(r2[:, i])]
            ei = e2[:, i][np.isfinite(e2[:, i])]
            if ri.size == 0 or ei.size == 0:
                edges.append(np.array([0.0, 1.0]))
                continue
            lo = min(ri.min(), ei.min())
            hi = max(ri.max(), ei.max())
            if hi <= lo:
                hi = lo + 1.0
            edges.append(np.linspace(lo, hi, n_bins + 1))

    total = 0.0
    grads = np.zeros_like(e2)
    for i in range(dim):
        eg = edges[i]
        w = bandwidth if bandwidth is not None else (eg[1] - eg[0])
        ri = r2[:, i]
        ei = e2[:, i]
        r_ok = np.isfinite(ri)
        e_ok = np.isfinite(ei)
        if not r_ok.any() or not e_ok.any():
            continue
        centers = 0.5 * (eg[:-1] + eg[1:])
        W_r = _tri_weights(ri[r_ok], centers, w)
        W_e = _tri_weights(ei[e_ok], centers, w)
        h_r = W_r.T @ ri[r_ok] / r_ok.sum()
        h_e = W_e.T @ ei[e_ok] / e_ok.sum()
        # keep these as (n_bins,) row-shaped vectors: (n_samples,) ops below
        # broadcast against the weight matrix, not against the bin axis
        diff = (h_r - h_e)[None, :]
        sign = np.sign(diff)
        total += float(np.sum(np.abs(diff)))

        # d|diff_b| / d h_e,b = -sign_b ; h_e,b = sum_j W_e[j,b] e_j / n_e
        # sign is (1, n_bins) so this yields (n_e,)
        coef = -(sign @ W_e.T)[0]
        y = ei[e_ok]
        # W_e[j,b] = relu(1 - |y_j - c_b| / w): dW/dy = -sign(y_j - c_b)/w when active
        delta = y[:, None] - centers[None, :]
        active = (np.abs(delta) / w) < 1.0
        dW_dy = -np.sign(delta) / w * active
        # d/dy_j [ sum_b coef_b W_e[j,b] y_j ] = sum_b coef_b (W_e[j,b] + y_j dW_dy[j,b])
        # coef is per-sample: it must be a column vector to scale rows of W_e
        # and of dW_dy (both (n_samples, n_bins)).
        coef_col = coef[:, None]
        g_local = (coef_col * W_e).sum(axis=1) + y * (coef_col * dW_dy).sum(axis=1)
        g_local = g_local / e_ok.sum()
        grads[e_ok, i] = g_local

    return total / dim, (lambda g: g * grads.reshape(emu.shape) / dim)


def _tri_weights(y, centers, width):
    """Triangular kernel weights of shape ``(len(y), len(centers))``."""
    d = np.abs(y[:, None] - centers[None, :]) / width
    return np.clip(1.0 - d, 0.0, None)


def ftle_regularizer(model, x, target, dt, eps=1e-4):
    """Finite-difference FTLE estimator (off by default -- see module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    base = model(x)
    direction = np.ones_like(x)
    pert = model(x + eps * direction)
    d0 = np.linalg.norm(eps * direction, axis=-1) + 1e-30
    d1 = np.linalg.norm(pert - base, axis=-1) + 1e-30
    est = float(np.mean(np.log(d1 / d0)) / dt)
    return (est - target) ** 2, est


def divergence_penalty(field_fn, x, target, eps=1e-5):
    """Mean-squared deviation of a vector field's divergence from ``target``.

    ``field_fn`` maps a batch ``(n, d)`` to a field ``(n, d)``.  The divergence is
    obtained by central differences of the *field*, which is exact enough here and
    keeps this independent of any autodiff framework.

    This is a physics-informed constraint, not a fitted parameter: Lorenz-63's
    divergence is the known constant ``-(sigma + 1 + beta)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    div = np.zeros(n)
    for i in range(d):
        e = np.zeros(d)
        e[i] = eps
        div += (field_fn(x + e)[:, i] - field_fn(x - e)[:, i]) / (2 * eps)
    return float(np.mean((div - target) ** 2))


__all__ = [
    "LossConfig",
    "mse_loss",
    "mse_grad",
    "l1_distribution_distance",
    "soft_l1_distribution_distance",
    "ftle_regularizer",
    "divergence_penalty",
]
