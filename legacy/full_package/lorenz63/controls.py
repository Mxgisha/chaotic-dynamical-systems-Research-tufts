"""Null baselines.

This is the addition that makes the other diagnostics interpretable.  Without a
baseline there is no scale on which "L1 = 0.03", "flip ratio 0.667" or
"energy bias 33" means anything.

* :func:`gt_gt_trajectory` -- a second, genuinely true trajectory started from a
  different initial condition.  It answers *"how different do two true
  trajectories look under this metric?"*, i.e. the ceiling on achievable
  agreement.  A diagnostic that cannot separate the emulator from this control is
  not measuring fidelity.
* :func:`identity_trajectory` -- the rollout of ``x_{n+1} = x_n``.  It answers
  *"what does a model with no dynamics score?"*, i.e. the floor.
"""

from __future__ import annotations

import numpy as np

from lorenz63 import system as S
from lorenz63.models import IdentityModel


def gt_gt_trajectory(gt_trajectory, dt=0.01, burn_in=0, seed=1234):
    """A second true trajectory, integrated from a different initial condition."""
    rng = np.random.default_rng(seed)
    # Start from a point on the same attractor but a genuinely different IC.
    shift = rng.normal(0.0, 1.0, size=3)
    ic = np.asarray(gt_trajectory[0], dtype=np.float64) + shift
    steps = int(gt_trajectory.shape[0])
    return S.generate_trajectory(ic, steps, dt, burn_in=burn_in)


def identity_trajectory(ic, steps):
    """Rollout of the identity map: constant, equal to the initial condition."""
    ic = np.asarray(ic, dtype=np.float64)
    return np.repeat(ic[None, :], int(steps), axis=0)


def emulator_trajectory(model, ic, steps):
    """Roll the emulator forward ``steps`` times from ``ic``."""
    ic = np.asarray(ic, dtype=np.float64)
    out = np.zeros((int(steps), ic.size), dtype=np.float64)
    x = ic
    for k in range(int(steps)):
        x = np.asarray(model.step(x), dtype=np.float64)
        out[k] = x
    return out


def build_controls(dataset, model=None, dt=0.01, seed=1234):
    """Bundle the control trajectories for one sweep point."""
    gt = np.asarray(dataset.test_trajectory, dtype=np.float64)
    return {
        "gt_gt_trajectory": gt_gt_trajectory(gt, dt=dt, seed=seed),
        "identity_trajectory": identity_trajectory(dataset.test_ic, gt.shape[0]),
    }


__all__ = [
    "gt_gt_trajectory",
    "identity_trajectory",
    "emulator_trajectory",
    "build_controls",
    "IdentityModel",
]
