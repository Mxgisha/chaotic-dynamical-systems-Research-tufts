"""The emulator.

The research stack is an equinox MLP trained with optax (see
:mod:`lorenz63.jax_trainer`).  This module provides a pure-NumPy MLP with the
*same architecture and the same loss*, so the whole pipeline -- training,
rollout, every diagnostic, and the test suite -- can be exercised on a CPU-only
machine, and so results do not depend on a GPU being present.

Architecture matches the JAX trainer's equinox MLP (see
:func:`lorenz63.jax_trainer.make_equinox_model`): ``depth`` hidden layers plus a
final linear layer, with a bias on every layer.  The original notebook used
``use_final_bias=False``; the two backends now agree parameter-for-parameter so
weights can be exported between them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAYER = "layer_{i}"


def silu(x):
    return x / (1.0 + np.exp(-x))


def silu_grad(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


class NumpyMLP:
    """SiLU MLP with an explicit parameter tree (a dict of arrays)."""

    def __init__(self, in_size=3, out_size=3, width=128, depth=3, seed=67):
        self.in_size = in_size
        self.out_size = out_size
        self.width = width
        self.depth = depth
        self.rng_seed = seed
        self.params = self._init_params(np.random.default_rng(seed))

    def _init_params(self, rng):
        params = {}
        sizes = [self.in_size] + [self.width] * self.depth
        for i in range(self.depth):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            params[LAYER.format(i=i)] = {
                "w": rng.normal(0.0, scale, size=(fan_in, fan_out)),
                "b": np.zeros(fan_out),
                "act": True,
            }
        fan_in, fan_out = self.width, self.out_size
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        params[LAYER.format(i=self.depth)] = {
            "w": rng.normal(0.0, scale, size=(fan_in, fan_out)),
            "b": np.zeros(fan_out),
            "act": False,
        }
        return params

    @property
    def n_params(self) -> int:
        return sum(int(v["w"].size + v["b"].size) for v in self.params.values())

    def forward(self, x, params=None):
        """Apply the map to a batch ``x`` of shape ``(n, in_size)``.

        The output layer carries a bias, matching
        :func:`lorenz63.jax_trainer.make_equinox_model` (``use_final_bias=True``)
        so the two backends are parameter-for-parameter identical and weights
        can be exported between them.
        """
        params = self.params if params is None else params
        h = x
        for key in self._layer_keys():
            layer = params[key]
            h = h @ layer["w"] + layer["b"]
            if layer["act"]:
                h = silu(h)
        return h

    def __call__(self, x, params=None):
        return self.forward(x, params)

    def step(self, state, params=None):
        """One-step map for a single state vector."""
        return self.forward(np.asarray(state)[None, :], params)[0]

    def _layer_keys(self):
        return [LAYER.format(i=i) for i in range(self.depth + 1)]

    # --- parameters -------------------------------------------------------
    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    # --- persistence ------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for key, layer in self.params.items():
            arrays[f"{key}/w"] = layer["w"]
            arrays[f"{key}/b"] = layer["b"]
        np.savez(path, **arrays)
        meta = {
            "in_size": self.in_size,
            "out_size": self.out_size,
            "width": self.width,
            "depth": self.depth,
            "rng_seed": self.rng_seed,
            "n_params": self.n_params,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = cls(
            in_size=meta.get("in_size", 3),
            out_size=meta.get("out_size", 3),
            width=meta.get("width", 128),
            depth=meta.get("depth", 3),
            seed=meta.get("rng_seed", 67),
        )
        with np.load(path) as data:
            params = {}
            for i in range(model.depth + 1):
                key = LAYER.format(i=i)
                params[key] = {
                    "w": data[f"{key}/w"],
                    "b": data[f"{key}/b"],
                    "act": i < model.depth,
                }
        model.set_params(params)
        return model


class IdentityModel:
    """Null baseline: ``x_{n+1} = x_n``.

    Any diagnostic that cannot separate this from a trained emulator is not
    measuring what it claims to measure.
    """

    in_size = 3
    out_size = 3
    width = 0
    depth = 0

    @property
    def n_params(self) -> int:
        return 0

    def forward(self, x, params=None):
        return np.asarray(x)

    def __call__(self, x, params=None):
        return self.forward(x, params)

    def step(self, state, params=None):
        return np.asarray(state)

    def get_params(self):
        return {}

    def set_params(self, params):
        return None

    def save(self, path):  # pragma: no cover - trivial
        return Path(path)

    @classmethod
    def load(cls, path):  # pragma: no cover - trivial
        return cls()


def make_model(backend: str = "numpy", **kwargs):
    """Factory used by the sweep script.

    Both branches return an object with the same ``step``/``forward``/``save``
    interface, so the diagnostics and ``scripts/run_sweep.py`` are
    backend-agnostic.  The JAX branch returns an adapter around the equinox
    module for exactly that reason.
    """
    if backend == "numpy":
        return NumpyMLP(**kwargs)
    if backend in ("jax", "jax_mlp", "jax_tracked"):
        from lorenz63.jax_trainer import EquinoxModelAdapter, make_equinox_model

        return EquinoxModelAdapter(make_equinox_model(**kwargs))
    raise ValueError(f"unknown model backend {backend!r}")


__all__ = ["NumpyMLP", "IdentityModel", "make_model", "silu", "silu_grad"]
