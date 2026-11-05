"""Vector-field slice error.

The original function plotted three panels (GT field, emulated field, error)
and returned the error grid with no summary scalar.  It is kept -- it is the only
diagnostic that probes the *local* map away from the attractor -- with a scalar
``mean_error`` emitted so it participates in the results table.
"""

from __future__ import annotations

import numpy as np

from lorenz63 import system as S
from lorenz63.diagnostics import register


def slice_grid(gt_trajectory, axis="z", fixed_val=None, res=25):
    """A 2-D slice through the attractor bounding box."""
    g = np.asarray(gt_trajectory, dtype=np.float64)
    lo = g.min(axis=0)
    hi = g.max(axis=0)
    pad = 0.1 * (hi - lo)
    axis = axis.lower()
    if fixed_val is None:
        fixed_val = float(g[:, "xyz".index(axis)].mean())

    if axis == "z":
        u = np.linspace(lo[0] - pad[0], hi[0] + pad[0], res)
        v = np.linspace(lo[1] - pad[1], hi[1] + pad[1], res)
        U, V = np.meshgrid(u, v)
        pts = np.stack([U.ravel(), V.ravel(), np.full(U.size, fixed_val)], axis=1)
        labels = ("X", "Y")
    elif axis == "x":
        u = np.linspace(lo[1] - pad[1], hi[1] + pad[1], res)
        v = np.linspace(lo[2] - pad[2], hi[2] + pad[2], res)
        U, V = np.meshgrid(u, v)
        pts = np.stack([np.full(U.size, fixed_val), U.ravel(), V.ravel()], axis=1)
        labels = ("Y", "Z")
    else:
        raise ValueError("axis must be 'x' or 'z'")
    return U, V, pts, labels, float(fixed_val)


@register("vector_field_slice")
def vector_field_slice(gt, emu, ctx):
    """Compare the true and emulated one-step maps on a slice."""
    axis = ctx.options.get("axis", "z")
    res = int(ctx.options.get("res", 25))
    U, V, pts, labels, fixed_val = slice_grid(gt, axis, ctx.options.get("fixed_val"), res)

    gt_vel = np.stack([S.lorenz_rhs(p) for p in pts])
    if ctx.model is not None:
        # The emulator is a step map, so compare the *increment* it implies.
        dt = ctx.dt
        emu_next = np.asarray(ctx.model(pts), dtype=np.float64)
        emu_vel = (emu_next - pts) / dt
    else:
        emu_vel = gt_vel.copy()

    err = np.linalg.norm(gt_vel - emu_vel, axis=1).reshape(U.shape)
    rel = err / (np.linalg.norm(gt_vel, axis=1).reshape(U.shape) + 1e-30)

    gt_m = {
        "mean_error": 0.0,
        "max_error": 0.0,
        "mean_relative_error": 0.0,
        "fixed_val": fixed_val,
        "_U": U,
        "_V": V,
        "_gt_vel": gt_vel,
        "_labels": labels,
    }
    emu_m = {
        "mean_error": float(err.mean()),
        "max_error": float(err.max()),
        "mean_relative_error": float(rel.mean()),
        "fixed_val": fixed_val,
        "_err": err,
        "_emu_vel": emu_vel,
        "_U": U,
        "_V": V,
        "_gt_vel": gt_vel,
        "_labels": labels,
    }
    return gt_m, emu_m
