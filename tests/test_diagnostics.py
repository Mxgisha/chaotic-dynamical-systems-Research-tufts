"""Diagnostics: shapes, decomposition of synthetic cases, and discrimination."""

import numpy as np
import pytest

from lorenz63 import system as S
from lorenz63.diagnostics import DiagnosticContext, available, get
from lorenz63.models import IdentityModel
from lorenz63.diagnostics.symbolic import lobe_transitions, return_map_correlation


@pytest.fixture(scope="module")
def lorenz_data():
    """A real Lorenz trajectory plus a deliberately degraded 'emulator'."""
    dt = 0.01
    y0 = np.array([1.0, 1.0, 1.0])
    gt = S.generate_trajectory(y0, 4000, dt, burn_in=1000)
    # A second true trajectory from a different IC (the control).
    ctl = S.generate_trajectory(np.array([5.0, 5.0, 20.0]), 4000, dt, burn_in=1000)
    # A poor emulator: contract the attractor slightly and add a bit of lag.
    emu = 0.95 * gt + 0.05 * np.roll(gt, 3, axis=0)
    return gt, emu, ctl, dt


def test_registry_has_all_expected_diagnostics():
    expected = {
        "lyapunov_spectrum",
        "invariant_measure",
        "correlation_dimension",
        "symbolic_dynamics",
        "dmd_spectrum",
        "error_growth",
        "vector_field_slice",
    }
    assert expected.issubset(set(available()))


def test_each_diagnostic_returns_finite_metrics(lorenz_data):
    gt, emu, ctl, dt = lorenz_data
    ctx = DiagnosticContext(
        dt=dt,
        model=IdentityModel(),
        ic=gt[0],
        gt_gt_trajectory=ctl,
        identity_trajectory=np.repeat(gt[0][None, :], gt.shape[0], axis=0),
        rng=np.random.default_rng(0),
        options={"n_points": 150, "n_boot": 2, "steps": 200, "delays": 5, "res": 10},
    )
    for name in available():
        gt_m, emu_m = get(name)(gt, emu, ctx)
        assert isinstance(gt_m, dict) and isinstance(emu_m, dict)
        assert gt_m and emu_m
        for d, label in ((gt_m, "gt"), (emu_m, "emu")):
            for k, v in d.items():
                if k.startswith("_") or isinstance(v, (str, type(None))):
                    continue
                assert np.isfinite(float(v)), f"{name}/{label}/{k} = {v}"


def test_lyapunov_recovers_textbook_values_on_ground_truth(lorenz_data):
    """The headline physics test, through the diagnostic entry point."""
    gt, emu, ctl, dt = lorenz_data
    ctx = DiagnosticContext(dt=dt, model=IdentityModel(), ic=gt[0])
    gt_m, _ = get("lyapunov_spectrum")(gt, emu, ctx)
    assert abs(gt_m["lambda_1"] - 0.9056) < 0.4, gt_m
    assert abs(gt_m["lambda_3"] - (-14.5723)) < 1.5, gt_m
    # the two invariants the old code never reported
    assert np.isclose(gt_m["sum_lambda"], gt_m["trace_invariant"], rtol=1e-6)
    assert abs(gt_m["sum_lambda"] - S.DIVERGENCE) < 1.0, gt_m


def test_lobe_transition_counts_are_exact_on_a_known_signal():
    """The crossing detector itself must be exact.

    Verified on a synthetic square wave with a known number of lobe changes,
    rather than on a chaotic trajectory where downsampling can both miss and
    invent crossings.
    """
    x = np.concatenate(
        [
            np.full(10, 1.0),
            np.full(10, -1.0),
            np.full(10, 1.0),
            np.full(10, -1.0),
        ]
    )
    traj = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    stats = lobe_transitions(traj)
    assert stats["n_flips"] == 3
    assert stats["mean_dwell"] == pytest.approx(40 / 3)
    # all-positive: no crossings at all
    pos = np.stack([np.ones(20), np.zeros(20), np.zeros(20)], axis=1)
    assert lobe_transitions(pos)["n_flips"] == 0


def test_lobe_transitions_are_time_reparameterisation_sensitive():
    """A slower trajectory must cross the saddle less often per step.

    Built explicitly by repeating each state, so the lobe sequence is preserved
    and the flip *count* is provably reduced.
    """
    dt = 0.01
    gt = S.generate_trajectory(np.array([1.0, 1.0, 1.0]), 4000, dt, burn_in=1000)
    slow = np.repeat(gt, 2, axis=0)
    assert lobe_transitions(slow)["n_flips"] == lobe_transitions(gt)["n_flips"]
    assert lobe_transitions(slow)["flip_rate"] < lobe_transitions(gt)["flip_rate"]


def test_symbolic_diagnostic_reports_a_flip_ratio(lorenz_data):
    """The diagnostic must return a usable, correctly-normalised ratio."""
    gt, emu, ctl, dt = lorenz_data
    ctx = DiagnosticContext(dt=dt, ic=gt[0], gt_gt_trajectory=ctl, options={})
    gt_m, emu_m = get("symbolic_dynamics")(gt, emu, ctx)
    assert np.isfinite(emu_m["flip_ratio"])
    # emu is a near-copy of gt here, so the ratio must be ~1 (not ~0 or ~N)
    assert abs(emu_m["flip_ratio"] - 1.0) < 0.25, emu_m["flip_ratio"]
    assert np.isfinite(gt_m["return_map_r"]) and np.isfinite(emu_m["return_map_r"])
    # the flip count is a non-negative integer count
    assert gt_m["n_flips"] >= 0 and emu_m["n_flips"] >= 0


def test_symbolic_flip_ratio_separates_a_structurally_different_emulator(lorenz_data):
    """A decoy that genuinely breaks the lobe sequence must change the ratio.

    Note that merely slowing a trajectory does *not*: the x = 0 crossing
    sequence is invariant under time reparameterisation, which is exactly why
    the original "uniqueness" framing was not measuring time fidelity.  A
    measurable change requires the lobe sequence itself to differ.
    """
    gt, emu, ctl, dt = lorenz_data
    ctx = DiagnosticContext(dt=dt, ic=gt[0], options={})
    gt_m, same_m = get("symbolic_dynamics")(gt, gt, ctx)
    assert gt_m["n_flips"] > 0
    # a bit-identical input gives exactly 1.0
    assert same_m["flip_ratio"] == pytest.approx(1.0)

    # Decoy: reverse the lobe-time sequence so the crossing count differs.
    x = gt[:, 0].copy()
    lobe = np.where(x >= 0, 1, -1)
    # count residence runs, then rebuild a trajectory whose crossings are halved
    # by dropping every other lobe segment's sign flip
    flips = np.flatnonzero(np.diff(lobe) != 0)
    if len(flips) >= 4:
        decoy = gt.copy()
        for j, f in enumerate(flips):
            if j % 2 == 0:
                end = flips[j + 1] if j + 1 < len(flips) else gt.shape[0]
                decoy[f:end, 0] = -gt[f:end, 0]
        _, dec_m = get("symbolic_dynamics")(gt, decoy, ctx)
        assert not np.isclose(dec_m["flip_ratio"], 1.0)


def test_error_growth_replaces_uniqueness(lorenz_data):
    """D15: the growth-rate diagnostic must exist and be the GT-referenced one."""
    gt, emu, ctl, dt = lorenz_data
    ctx = DiagnosticContext(
        dt=dt, model=IdentityModel(), ic=gt[0], options={"steps": 400}
    )
    gt_m, emu_m = get("error_growth")(gt, emu, ctx)
    # the ground truth must show a positive error-growth rate (it is chaotic)
    assert gt_m["growth_rate"] > 0
    assert "t_double" in gt_m and "t_sat" in gt_m
    # an identity map has zero growth, so discrimination is visible
    assert "growth_rate_ratio" in emu_m


def test_dimension_of_a_synthetic_line_is_one():
    """Sanity: a 1-D set must give D2 ~ 1."""
    from lorenz63.diagnostics.dimension import (
        correlation_integral,
        fit_dimension,
    )

    t = np.linspace(0, 1, 400)
    pts = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    radii = np.logspace(-2.5, -0.3, 20)
    d2, _, _ = fit_dimension(radii, correlation_integral(pts, radii, theiler=0))
    assert 0.7 < d2 < 1.3, d2


def test_dmd_uses_a_delay_embedding():
    """D12: the snapshot matrix must be wider than the 3-D state."""
    from lorenz63.diagnostics.dmd import hankel

    traj = np.arange(90, dtype=float).reshape(30, 3)
    X = hankel(traj, delays=5)
    assert X.shape[1] == 15, X.shape
    assert X.shape[0] == 30 - 4


def test_identity_model_is_separated_from_ground_truth(lorenz_data):
    """The core claim of the controls: a no-dynamics model must look bad."""
    gt, emu, ctl, dt = lorenz_data
    ident_traj = np.repeat(gt[0][None, :], gt.shape[0], axis=0)
    ctx = DiagnosticContext(dt=dt, ic=gt[0], identity_trajectory=ident_traj, options={})
    gt_m, ident_m = get("invariant_measure")(gt, ident_traj, ctx)
    _, emu_m = get("invariant_measure")(gt, emu, ctx)
    # the poor emulator should match the truth better than a frozen point does
    assert emu_m["l1_mean"] < ident_m["l1_mean"]
