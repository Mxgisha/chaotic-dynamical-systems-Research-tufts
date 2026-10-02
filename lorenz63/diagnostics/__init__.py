"""Diagnostics package.

Every diagnostic is a **pure measurement function** with the signature::

    fn(gt, emu, ctx) -> (dict[str, float], dict[str, float])

returning ``(ground_truth_metrics, emulator_metrics)`` over the same metric keys.
None of them import or touch matplotlib, and none of them print
``TEST PASSED``/``TEST FAILED``: pass/fail is decided in
:mod:`lorenz63.report` by comparing against the null controls in
:class:`DiagnosticContext`, so a "PASS" can never again be emitted for a
property (such as explosive divergence) that an identity map also exhibits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class DiagnosticContext:
    """Everything a diagnostic may need beyond the two trajectories.

    The control trajectories are produced once by the driver and reused, so the
    sweep no longer re-rolls trajectories four times per diagnostic.
    """

    dt: float = 0.01
    model: Any = None  # emulator, for spectral/RHS-based diagnostics
    ic: Any = None
    gt_gt_trajectory: Any = None  # second TRUE trajectory, different IC
    identity_trajectory: Any = None  # x_{n+1} = x_n rollout
    rng: Any = None  # numpy Generator
    options: dict = field(default_factory=dict)


class Diagnostic(Protocol):
    name: str
    keys: tuple[str, ...]

    def __call__(self, gt, emu, ctx: DiagnosticContext): ...


_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        fn.name = name
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"no diagnostic {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available():
    return sorted(_REGISTRY)


def run_registered(names=None, **kwargs):
    """Run several diagnostics, returning a dict name -> (gt, emu)."""
    names = names or available()
    out = {}
    for n in names:
        out[n] = get(n)(**kwargs)
    return out


# Import each submodule for its registration side effect.  Explicit module
# imports (rather than `from ... import name`) so no module is dropped.
import importlib as _importlib  # noqa: E402

for _mod in (
    "dimension",
    "divergence",
    "dmd",
    "field",
    "lyapunov",
    "measure",
    "symbolic",
):
    _importlib.import_module(f"{__name__}.{_mod}")
del _importlib, _mod

__all__ = [
    "DiagnosticContext",
    "Diagnostic",
    "register",
    "get",
    "available",
    "run_registered",
]
