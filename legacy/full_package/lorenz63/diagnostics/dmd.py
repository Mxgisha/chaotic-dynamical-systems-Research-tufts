"""Dynamic Mode Decomposition.

The original applied DMD to the raw ``3 x N`` state matrix.  With three rows
there are at most three DMD eigenvalues, and for a chaotic trajectory they all
cluster at ``Re ~ 1`` with no structure -- the "spectrum comparison" could not
discriminate anything (confirmed in the committed ``dmd_spectrum.png``).

This version applies DMD to a **Hankel / time-delay embedding** of the
trajectory, which is the standard fix: the delay coordinates lift the scalar
observations into a space where the Koopman operator's spectrum is resolved, so
the comparison has content.  Reported metrics are the sorted eigenvalue
magnitudes, the spectral gap, and the correlation between the true and emulated
mode-magnitude profiles.
"""

from __future__ import annotations

import numpy as np

from lorenz63.diagnostics import register


def hankel(traj, delays, stride=1):
    """Stack ``delays`` time-shifted copies of the state into one snapshot matrix."""
    traj = np.asarray(traj, dtype=np.float64)
    n = traj.shape[0] - (delays - 1) * stride
    if n <= 1:
        raise ValueError("trajectory too short for the requested delay embedding")
    rows = [traj[k * stride : k * stride + n] for k in range(delays)]
    return np.concatenate(rows, axis=1)  # (n, delays * dim)


def dmd(X):
    """Standard exact DMD.  ``X`` is ``(n_snapshots, n_features)``.

    Returns ``(eigenvalues, modes)``.
    """
    X = np.asarray(X, dtype=np.float64)
    A = X[:-1].T
    B = X[1:].T
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    keep = s > (s.max() * 1e-10 if s.size else 1.0)
    U, s, Vh = U[:, keep], s[keep], Vh[keep]
    Atilde = U.conj().T @ B @ Vh.conj().T @ np.diag(1.0 / s)
    eigvals, W = np.linalg.eig(Atilde)
    modes = B @ Vh.conj().T @ np.diag(1.0 / s) @ W
    return eigvals, modes


def spectrum_metrics(traj, delays=10):
    """Sorted ``|lambda|`` and derived scalars for one trajectory."""
    X = hankel(traj, delays)
    eigvals, modes = dmd(X)
    mags = np.sort(np.abs(eigvals))[::-1]
    profile = np.abs(modes)
    # Normalise the mode-magnitude profile so two systems can be compared.
    prof = profile / (np.linalg.norm(profile) + 1e-30)
    gap = float(mags[0] - mags[1]) if mags.size > 1 else 0.0
    return {
        "n_modes": int(mags.size),
        "mag_top": float(mags[0]) if mags.size else float("nan"),
        "mag_mean": float(np.mean(mags)) if mags.size else float("nan"),
        "spectral_gap": gap,
        "_mags": mags,
        "_profile": prof,
    }


@register("dmd_spectrum")
def dmd_spectrum(gt, emu, ctx):
    """Hankel-DMD comparison of the two attractors."""
    delays = int(ctx.options.get("delays", 10))
    gt = np.asarray(gt, dtype=np.float64)
    emu = np.asarray(emu, dtype=np.float64)

    gt_m = spectrum_metrics(gt, delays)
    emu_m = spectrum_metrics(emu, delays)

    # Compare the sorted spectra on their common length.
    k = min(gt_m["n_modes"], emu_m["n_modes"])
    if k > 1:
        a, b = gt_m["_mags"][:k], emu_m["_mags"][:k]
        if a.std() > 1e-12 and b.std() > 1e-12:
            emu_m["spectrum_corr"] = float(np.corrcoef(a, b)[0, 1])
        else:
            emu_m["spectrum_corr"] = float("nan")
        emu_m["mag_top_ratio"] = float(b[0] / a[0]) if a[0] else float("nan")
    else:
        emu_m["spectrum_corr"] = float("nan")
        emu_m["mag_top_ratio"] = float("nan")
    gt_m["spectrum_corr"] = 1.0
    gt_m["mag_top_ratio"] = 1.0
    return gt_m, emu_m
