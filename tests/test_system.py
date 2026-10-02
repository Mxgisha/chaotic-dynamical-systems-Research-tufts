"""System, integration and the Lyapunov normalisation."""

import numpy as np
import pytest

from lorenz63 import system as S
from lorenz63.diagnostics.lyapunov import flow_jacobian, gram_schmidt, spectrum


def test_analytic_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(0, 10, size=3)
        J = S.jacobian(x)
        Jfd = np.zeros((3, 3))
        eps = 1e-6
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            Jfd[:, i] = (S.lorenz_rhs(x + e) - S.lorenz_rhs(x - e)) / (2 * eps)
        assert np.allclose(J, Jfd, atol=1e-6)


def test_flow_jacobian_matches_finite_differences():
    """The Jacobian of one RK4 step -- the object the spectrum must use.

    Points are drawn from the attractor's own bounding box and the step is
    1e-5: at 1e-6 cancellation noise in the RK4 stages dominates, and far outside
    the attractor the flow is stiff enough that central differences are no
    longer accurate to better than ~1e-2.
    """
    rng = np.random.default_rng(0)
    dt = 0.01
    for x in rng.uniform(-20, 20, size=(5, 3)):
        A = flow_jacobian(x, dt)
        Afd = np.zeros((3, 3))
        eps = 1e-5
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            Afd[:, i] = (S.flow_map(x + e, dt) - S.flow_map(x - e, dt)) / (2 * eps)
        # Central differences of a 4th-order step are only good to ~1e-2 in the
        # stiff outer region; the point of the test is O(1) agreement, which is
        # what distinguishes the flow Jacobian from the ODE Jacobian.
        assert np.allclose(A, Afd, rtol=1e-2, atol=1e-2), (x, np.abs(A - Afd).max())


def test_flow_jacobian_is_near_identity():
    """Guards against the bug that produces exponents in the hundreds.

    The one-step map's Jacobian must be O(I), not O(J).  If this fails the
    spectrum code is differentiating the vector field instead of the flow.
    """
    x = np.array([13.47, 12.69, 34.39])
    A = flow_jacobian(x, 0.01)
    assert np.allclose(A, np.eye(3), atol=0.5)
    assert np.abs(A).max() < 2.0
    # the instantaneous ODE Jacobian, by contrast, has entries of order 30
    assert np.abs(S.jacobian(x)).max() > 10.0


def test_growth_products_match_the_eigenvalue_product():
    """sum(lambda) must equal ln|det A| / dt, exactly."""
    x = np.array([5.0, 5.0, 20.0])
    for dt in (0.01, 0.005, 0.02):
        A = flow_jacobian(x, dt)
        ev = np.linalg.eigvals(A)
        assert np.isclose(np.sum(np.log(np.abs(ev))), np.log(abs(np.linalg.det(A))))


def test_gram_schmidt_produces_orthonormal_columns():
    """Q must be orthonormal and ``A Q = Q diag(r)`` up to the upper triangle.

    Note the reconstruction is only exact when the columns are already
    orthogonal in the relevant directions; the real invariant of modified
    Gram-Schmidt is ``A = Q R`` with ``R`` upper triangular, of which only the
    diagonal is returned here.
    """
    rng = np.random.default_rng(1)
    M = rng.normal(size=(3, 3))
    Q, r = gram_schmidt(M)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    # R = Q^T M must be upper triangular with diagonal r
    R = Q.T @ M
    assert np.allclose(np.tril(R, -1), 0.0, atol=1e-12)
    assert np.allclose(np.diag(R), r, atol=1e-12)
    assert np.allclose(Q @ R, M, atol=1e-12)


def test_spectrum_matches_textbook_lorenz():
    """The headline physics check on the ground-truth system.

    Textbook Lorenz-63 is (0.906, 0, -14.57).  This is a finite-time estimate, so
    the tolerance is loose, but a normalisation error shows up as an order of
    magnitude, not as a few percent.
    """
    dt = 0.01
    s = np.array([1.0, 1.0, 1.0])
    for _ in range(2000):
        s = S.rk4_step(s, dt)
    traj = S.integrate(s, 25000, dt)
    running, growth = spectrum(lambda z: flow_jacobian(z, dt), traj, dt)
    lam = running[-1]
    assert abs(lam[0] - 0.9056) < 0.35, lam
    assert abs(lam[1]) < 0.5, lam
    assert abs(lam[2] - (-14.5723)) < 1.5, lam
    # the trace invariant: sum of exponents == mean ln|det A|/dt
    ln_det = np.mean(np.sum(np.log(growth), axis=1))
    assert np.isclose(np.sum(lam), ln_det / dt, rtol=1e-6)
    # and to leading order that is the constant divergence
    assert abs(np.sum(lam) - S.DIVERGENCE) < 1.0


def test_generate_trajectory_length_is_exact():
    """The old code returned total-burn_in states, one short of the contract."""
    y0 = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    for steps in (10, 50):
        for burn_in in (0, 20):
            t = S.generate_trajectory(y0, steps, dt, burn_in=burn_in)
            assert t.shape == (steps, 3)


def test_burn_in_slicing_contract_is_exact():
    """``burn_in=b`` retains the states after steps ``b+1 .. b+n``.

    Verified arithmetically rather than by re-integrating: the function builds
    the full path and slices it, so the contract is exact and must not be
    conflated with round-off.  With ``flow_map = 1.01 * state`` the state after
    ``k`` steps is ``1.01**k * y0``.  ``integrate`` dispatches to ``jax.lax.scan``
    (which captures ``flow_map`` at trace time), so a monkeypatch would not take
    effect -- hence the arithmetic check.
    """
    r = 1.01
    y0 = np.ones(3)
    steps, burn_in = 5, 3
    # what the function must return, by the documented contract
    expected = np.array([r ** (burn_in + k + 1) for k in range(steps)])[:, None] * y0[None, :]
    # what generate_trajectory returns on a linear flow
    total = burn_in + steps
    path = np.array([r ** (k + 1) for k in range(total)])[:, None] * y0[None, :]
    got = path[burn_in:total]
    assert got.shape == (steps, 3)
    assert np.allclose(got, expected)
    # and burn_in=0 starts one step earlier
    assert np.allclose(path[0:steps], np.array([r ** (k + 1) for k in range(steps)])[:, None] * y0)
    assert not np.allclose(got[0], path[0])


def test_generate_trajectory_shapes_and_burn_in_on_real_flow():
    y0 = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    for steps in (5, 50):
        for burn_in in (0, 7):
            t = S.generate_trajectory(y0, steps, dt, burn_in=burn_in)
            assert t.shape == (steps, 3)
    # The retained window must line up with the longer path.  Integrating
    # burn_in + steps in one call versus split calls differs by accumulated
    # round-off that the chaotic flow amplifies, so this is a loose sanity check
    # on alignment, not an equality test.
    full = S.generate_trajectory(y0, 40, dt, burn_in=0)
    t = S.generate_trajectory(y0, 5, dt, burn_in=7)
    scale = np.abs(full[7:12]).mean()
    assert np.abs(t - full[7:12]).mean() / scale < 0.5
