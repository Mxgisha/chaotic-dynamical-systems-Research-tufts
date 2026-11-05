"""Local contraction: the state-dependence of the emulator's volume error."""

import numpy as np
import pytest

from lorenz63 import system as S
from lorenz63.diagnostics import DiagnosticContext, available, get
from lorenz63.diagnostics.contraction import (
    TRUE_LOG_DET_PER_STEP,
    distance_to_fixed_points,
    fixed_points,
    local_contraction,
)
from lorenz63.diagnostics.lyapunov import flow_jacobian, model_jacobian_fd
from lorenz63.models import IdentityModel, NumpyMLP


def test_fixed_points_match_the_analytic_equilibria():
    """Equilibria are (0,0,0) and (+-C, +-C, rho-1) with C = sqrt(beta(rho-1))."""
    c = np.sqrt(S.BETA * (S.RHO - 1.0))
    pts = fixed_points()
    assert pts.shape == (3, 3)
    assert np.allclose(pts[0], [0, 0, 0])
    assert np.allclose(pts[1], [c, c, S.RHO - 1])
    assert np.allclose(pts[2], [-c, -c, S.RHO - 1])
    # the non-trivial equilibria really are equilibria
    for p in pts[1:]:
        assert np.allclose(S.lorenz_rhs(p), 0.0, atol=1e-10)


def test_rhs_vanishes_at_the_origin_too():
    assert np.allclose(S.lorenz_rhs(np.zeros(3)), 0.0)


def test_distance_to_fixed_points_picks_the_nearest():
    pts = fixed_points()
    assert distance_to_fixed_points(pts[1]) == pytest.approx(0.0)
    assert distance_to_fixed_points(pts[2]) == pytest.approx(0.0)
    mid = 0.5 * (pts[1] + pts[2])
    assert distance_to_fixed_points(mid) == pytest.approx(np.linalg.norm(pts[1] - mid))


def test_true_flow_contraction_is_the_constant_divergence():
    """The whole diagnostic rests on this: the truth has *no* state dependence.

    Checked as a mean over many attractor states.  The spread is the RK4
    discretisation floor (RK4 does not preserve the divergence identity
    exactly), not physics, so it is bounded loosely here and reported as a
    ``divergence_error`` in the results.
    """
    dt = 0.01
    traj = S.generate_trajectory(np.array([1.0, 1.0, 1.0]), 400, dt, burn_in=1000)
    rates = np.array(
        [local_contraction(flow_jacobian(x, dt))[0] / dt for x in traj]
    )
    assert np.mean(rates) == pytest.approx(S.DIVERGENCE, abs=0.5)
    # and it is genuinely near-constant
    assert np.std(rates) < 1.0


def test_flow_jacobian_determinant_is_positive():
    """The true flow map is orientation preserving; det must be > 0."""
    dt = 0.01
    for x in (np.array([1.0, 1.0, 1.0]), np.array([-8.0, -8.0, 27.0])):
        sign, _ = np.linalg.slogdet(flow_jacobian(x, dt))
        assert sign > 0


def test_identity_model_looks_exactly_like_the_truth():
    """A no-op map is the degenerate case the control catches.

    The identity map has ``D phi = I`` everywhere, so its contraction rate is
    exactly zero rather than the divergence -- a difference the diagnostic must
    show, not hide.
    """
    ctx = DiagnosticContext(dt=0.01, model=IdentityModel(), options={"n_states": 40})
    gt = S.generate_trajectory(np.array([1.0, 1.0, 1.0]), 200, 0.01, burn_in=500)
    gt_m, emu_m = get("local_contraction")(gt, gt, ctx)
    assert gt_m["contraction_rate_mean"] == pytest.approx(S.DIVERGENCE, abs=0.5)
    assert emu_m["contraction_rate_mean"] == pytest.approx(0.0, abs=1e-6)
    assert emu_m["divergence_error"] == pytest.approx(abs(S.DIVERGENCE), abs=1e-6)
    # an identity map expands nothing either
    assert emu_m["expansion_rate_mean"] == pytest.approx(0.0, abs=1e-5)


def test_emulator_jacobian_is_accurate():
    """The finite-difference Jacobian must be trustworthy before it is believed."""
    model = NumpyMLP(width=16, depth=2, seed=0)
    x = np.array([5.0, 5.0, 20.0])
    for eps in (1e-4, 1e-5, 1e-6):
        J = model_jacobian_fd(model, x, eps=eps)
        assert J.shape == (3, 3)
        assert np.all(np.isfinite(J))
    # the determinant must be stable across step sizes
    dets = [np.linalg.slogdet(model_jacobian_fd(model, x, eps=e))[1] for e in (1e-4, 1e-5, 1e-6)]
    assert max(dets) - min(dets) < 1e-3


def test_diagnostic_is_registered_and_returns_the_expected_keys():
    assert "local_contraction" in available()
    gt = S.generate_trajectory(np.array([1.0, 1.0, 1.0]), 200, 0.01, burn_in=500)
    ctx = DiagnosticContext(dt=0.01, model=IdentityModel(), options={"n_states": 40})
    gt_m, emu_m = get("local_contraction")(gt, gt, ctx)
    for key in (
        "contraction_rate_mean",
        "contraction_ratio",
        "excess_mean",
        "excess_near_saddle",
        "excess_mid_lobe",
        "state_dependence",
        "corr_expansion_gt_vs_emu",
    ):
        assert key in emu_m, key
        assert key in gt_m, key


def test_ground_truth_ratio_is_one_and_excess_is_zero():
    """The control must be self-consistent, or nothing downstream is."""
    gt = S.generate_trajectory(np.array([1.0, 1.0, 1.0]), 200, 0.01, burn_in=500)
    ctx = DiagnosticContext(dt=0.01, model=IdentityModel(), options={"n_states": 40})
    gt_m, _ = get("local_contraction")(gt, gt, ctx)
    assert gt_m["contraction_ratio"] == pytest.approx(1.0)
    assert gt_m["excess_mean"] == pytest.approx(0.0)
    assert gt_m["excess_std"] == pytest.approx(0.0)
    assert gt_m["state_dependence"] == pytest.approx(0.0)


def test_constant_scale_is_measured_exactly():
    """A model ``x -> c*x`` has ``D phi = c*I``, so ``ln|det|/dt = 3 ln(c)/dt``.

    Checked for a contracting scalar map, and for an expanding one: the
    contraction rate is the same formula in both cases, so the diagnostic
    measures the actual log-volume change rather than assuming contraction.
    """

    class ScaledIdentity:
        def __init__(self, c):
            self.c = c

        def step(self, state, params=None):
            return self.c * np.asarray(state)

        def __call__(self, x, params=None):
            return self.c * np.asarray(x)

    dt = 0.01
    # long enough that the trajectory visits the |x| < 2 region near the saddle,
    # otherwise the state-dependence bins are empty and correctly report NaN
    gt = S.generate_trajectory(np.array([1.0, 1.0, 1.0]), 4000, dt, burn_in=1000)
    for c in (0.99, 1.01):
        ctx = DiagnosticContext(dt=dt, model=ScaledIdentity(c), options={"n_states": 20})
        gt_m, emu_m = get("local_contraction")(gt, gt, ctx)
        expected = 3.0 * np.log(c) / dt
        assert emu_m["contraction_rate_mean"] == pytest.approx(expected, rel=1e-6)
        # a scalar map has no state dependence at all
        assert emu_m["contraction_rate_std"] == pytest.approx(0.0, abs=1e-6)
        assert emu_m["expansion_rate_mean"] == pytest.approx(np.log(c) / dt, rel=1e-6)
        # a scalar map's *excess* is the same everywhere (the truth is nearly
        # constant), so the difference between bins is ~0
        assert emu_m["state_dependence"] == pytest.approx(0.0, abs=0.5)


def test_contraction_ratio_convention():
    """``contraction_ratio > 1`` means the emulator contracts *more* than truth.

    Both rates are negative in the contracting regime, so the ratio is positive;
    an over-dissipative emulator has the more negative rate and therefore the
    larger ratio.  This pins the sign, which is easy to get backwards.
    """
    c = 0.99  # the emulator contracts at 3*ln(0.99)/dt, slower than the truth
    dt = 0.01
    gt = S.generate_trajectory(np.array([1.0, 1.0, 1.0]), 100, dt, burn_in=500)
    ctx = DiagnosticContext(dt=dt, model=_ScaledIdentity(c), options={"n_states": 20})
    gt_m, emu_m = get("local_contraction")(gt, gt, ctx)
    ratio = emu_m["contraction_ratio"]
    assert ratio == pytest.approx(
        emu_m["contraction_rate_mean"] / gt_m["contraction_rate_mean"], rel=1e-9
    )
    # the emulator under-contracts here relative to the true divergence
    assert ratio < 1.0
    assert abs(emu_m["contraction_rate_mean"]) < abs(gt_m["contraction_rate_mean"])


class _ScaledIdentity:
    def __init__(self, c):
        self.c = c

    def step(self, state, params=None):
        return self.c * np.asarray(state)

    def __call__(self, x, params=None):
        return self.c * np.asarray(x)
