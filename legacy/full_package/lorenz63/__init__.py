"""Chaotic dynamical systems emulator failure-mode analysis (Lorenz-63).

The package is split so that the scientific content is testable without a GPU:

* :mod:`lorenz63.system`      -- the ODE, its parameters, and integration
* :mod:`lorenz63.data`        -- dataset construction and RNG discipline
* :mod:`lorenz63.models`      -- the emulator (numpy and equinox backends)
* :mod:`lorenz63.losses`      -- MSE / L1 / FTLE loss terms
* :mod:`lorenz63.train`       -- one training loop, one stepper contract
* :mod:`lorenz63.controls`    -- GT-vs-GT and identity null baselines
* :mod:`lorenz63.diagnostics` -- pure measurement functions, no plotting
* :mod:`lorenz63.report`      -- the single figure contract and results.json
"""

__version__ = "0.2.0"

from lorenz63.backend import xp, BACKEND, to_numpy  # noqa: F401

__all__ = ["xp", "BACKEND", "to_numpy", "__version__"]
