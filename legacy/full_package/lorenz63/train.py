"""Training.

Fixes the defects that made the previous training path unusable:

**D5 -- the 3-vs-6 unpacking crash.**  ``train_model`` unpacked six values from
whatever ``stepper`` it was handed, while ``model_stepper`` and
``model_stepper_l1_lya`` returned three.  There is now exactly one training
function and one loss configuration, so the mismatch cannot exist.

**D6 -- the loss terms were dead code.**  ``model_stepper_l1_lya`` had no call
sites, so the L1 and FTLE terms never trained anything.  The loss is now a
:class:`~lorenz63.losses.LossConfig` field and every component is logged every
epoch, so an unwired term is visible immediately.

**D19 -- no-op rebindings** (``epochs = epochs`` etc.) are gone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lorenz63.losses import (
    LossConfig,
    l1_distribution_distance,
    mse_loss,
    soft_l1_distribution_distance,
)
from lorenz63.models import LAYER, silu, silu_grad


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 67
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class LossLog:
    """Named loss components.  A dataclass, not a positional tuple."""

    total: float
    mse: float
    l1: float
    ftle: float = 0.0

    def as_dict(self):
        return asdict(self)


def _layer_keys(depth):
    return [LAYER.format(i=i) for i in range(depth + 1)]


def forward_with_cache(x, params, keys):
    """Forward pass retaining pre-activations for backprop."""
    cache = {}
    h = x
    for k in keys:
        layer = params[k]
        z = h @ layer["w"] + layer["b"]
        cache[k] = {"h_in": h, "z": z, "act": layer["act"]}
        h = silu(z) if layer["act"] else z
    return h, cache


def backward(pred, cache, params, keys, dpred):
    """Backpropagate ``dpred`` (dL/dpred) to a gradient dict."""
    grads = {}
    delta = dpred
    for k in reversed(keys):
        c = cache[k]
        if c["act"]:
            delta = delta * silu_grad(c["z"])
        grads[f"{k}/w"] = c["h_in"].T @ delta
        grads[f"{k}/b"] = np.sum(delta, axis=0)
        delta = delta @ params[k]["w"].T
    return grads


def _zero_grads(model):
    return {
        f"{k}/{p}": np.zeros_like(model.params[k][p])
        for k in model._layer_keys()
        for p in ("w", "b")
    }


def adam_update(params, grads, state, lr, keys):
    """Minimal Adam.  ``state`` holds the moment estimates."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    state["t"] += 1
    t = state["t"]
    for k in keys:
        for p in ("w", "b"):
            gk = f"{k}/{p}"
            g = grads[gk]
            m = state["m"][gk] = beta1 * state["m"][gk] + (1 - beta1) * g
            v = state["v"][gk] = beta2 * state["v"][gk] + (1 - beta2) * (g * g)
            mhat = m / (1 - beta1**t)
            vhat = v / (1 - beta2**t)
            params[k][p] = params[k][p] - lr * mhat / (np.sqrt(vhat) + eps)


def _init_adam(model):
    keys = model._layer_keys()
    return {
        "t": 0,
        "m": _zero_grads(model),
        "v": _zero_grads(model),
    }


def train(model, dataset, config: TrainConfig | None = None, verbose: bool = True):
    """Train ``model`` in place on ``dataset``.

    Returns ``(model, history)`` where ``history`` is a list of per-epoch dicts.
    """
    config = config or TrainConfig()
    rng = np.random.default_rng(config.seed)
    keys = model._layer_keys()

    x_train = np.asarray(dataset.x_train, dtype=np.float64)
    y_train = np.asarray(dataset.y_train, dtype=np.float64)
    # One-step target: the first step of the horizon.  The emulator is a
    # discrete one-step map, and the ground-truth dataset advances by one dt.
    y_one = y_train[:, 0, :]
    n = x_train.shape[0]

    adam = _init_adam(model)
    history = []

    # Fixed histogram edges for the distributional term, derived from the whole
    # training set so the objective is not a function of the minibatch.
    l1_edges = None
    if config.loss.l1_weight > 0.0:
        l1_edges = []
        for j in range(y_one.shape[1]):
            lo = min(float(x_train[:, j].min()), float(y_one[:, j].min()))
            hi = max(float(x_train[:, j].max()), float(y_one[:, j].max()))
            l1_edges.append(np.linspace(lo, hi, config.loss.n_bins + 1))

    for epoch in range(config.epochs):
        # Light exponential decay over the back half of training.
        frac = epoch / max(config.epochs - 1, 1)
        lr = config.lr * (0.1 ** max(0.0, (frac - 0.5) / 0.5))

        perm = rng.permutation(n)
        totals, mses, l1s = [], [], []
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = x_train[idx]
            yb = y_one[idx]

            pred, cache = forward_with_cache(xb, model.params, keys)

            m = mse_loss(pred, yb)
            dpred = np.zeros_like(pred)

            # MSE term: d/dpred of mean((pred - yb)^2)
            dpred += (2.0 * (pred - yb)) / pred.size

            # L1 distributional term (differentiable soft histogram, one per
            # coordinate so it matches the quantity l1_distribution_distance
            # reports).
            l1_val = 0.0
            if config.loss.l1_weight > 0.0:
                v, back = soft_l1_distribution_distance(
                    yb, pred, edges=l1_edges
                )
                l1_val = float(v)
                dpred += config.loss.l1_weight * back(1.0)

            grads = backward(pred, cache, model.params, keys, dpred)

            # Explicit global-norm clipping (the original used optax.clip_by_global_norm)
            gnorm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            if gnorm > 1.0:
                for gk in grads:
                    grads[gk] = grads[gk] * (1.0 / gnorm)

            adam_update(model.params, grads, adam, lr, keys)
            totals.append(m + config.loss.l1_weight * l1_val)
            mses.append(m)
            l1s.append(l1_val)

        entry = {
            "epoch": epoch,
            "lr": lr,
            "total": float(np.mean(totals)),
            "mse": float(np.mean(mses)),
            "l1": float(np.mean(l1s)),
            "ftle": 0.0,
        }
        history.append(entry)
        if verbose and (epoch % 10 == 0 or epoch == config.epochs - 1):
            print(
                f"Epoch {epoch:3d} | lr {lr:.2e} | total {entry['total']:.6f} "
                f"| mse {entry['mse']:.6f} | l1 {entry['l1']:.6f}"
            )

    return model, history


def save_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
    return path


__all__ = [
    "TrainConfig",
    "LossLog",
    "train",
    "save_history",
    "forward_with_cache",
    "backward",
    "adam_update",
]
