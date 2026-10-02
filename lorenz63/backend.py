"""Array-backend selection.

The diagnostics and dataset code are written against a single array namespace so
they can run under JAX (the research stack) or NumPy (so the whole test suite runs
on a CPU-only machine, and so results are reproducible without a GPU).

``LORENZ63_BACKEND`` may be set to ``jax`` or ``numpy`` to force a choice.  JAX is
preferred when it is importable; otherwise NumPy is used.
"""

from __future__ import annotations

import os

_FORCE = os.environ.get("LORENZ63_BACKEND", "").strip().lower()


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


if _FORCE in ("jax", "numpy"):
    BACKEND = _FORCE
elif _has("jax"):
    BACKEND = "jax"
else:
    BACKEND = "numpy"

if BACKEND == "jax":
    import jax
    import jax.numpy as xp

    # float64 is required: the pipeline differentiates a chaotic flow over 10^4
    # steps, and float32 silently destroys the Lyapunov spectrum and makes
    # training diverge.  Set at import time, before any array exists.
    jax.config.update("jax_enable_x64", True)
else:
    import numpy as xp

    # Even on the NumPy backend, jax.random supplies the PRNG keys, so x64 must
    # be enabled here too -- otherwise jnp.asarray(..., float64) silently
    # truncates and the two training paths disagree.
    import importlib.util as _ilu

    if _ilu.find_spec("jax") is not None:
        import jax as _jax

        _jax.config.update("jax_enable_x64", True)


def to_numpy(a):
    """Return a host NumPy array (no-op under the NumPy backend)."""
    if BACKEND == "jax":
        import numpy as _np

        return _np.asarray(a)
    return xp.asarray(a)


def asarray(a, dtype=None):
    if dtype is None:
        dtype = xp.float64
    return xp.asarray(a, dtype=dtype)


__all__ = ["xp", "BACKEND", "to_numpy", "asarray"]
