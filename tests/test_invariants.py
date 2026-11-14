"""The checks that make the measurements trustworthy.

Each one is an identity that must hold if the code is right, so a failure means
the diagnostic is broken rather than that the physics changed.

    pip install pytest && python -m pytest -q
"""

import jax
import jax.numpy as jnp
import pytest

import diagnostics as diag
import lorenz
import train


@pytest.fixture(scope="module", autouse=True)
def _setup():
    train.setup()


def on_attractor():
    """A single state on the attractor, not the raw initial condition."""
    return lorenz.trajectory(jnp.array([1.0, 1.0, 1.0]), 1, burn_in=1000)[0]


def test_step_jacobian_has_the_right_volume_change():
    """ln|det D step| / dt must equal the divergence of the field."""
    rate = float(jnp.linalg.slogdet(lorenz.flow_jacobian(on_attractor()))[1] / lorenz.DT)
    assert abs(rate - lorenz.DIVERGENCE) < 1e-3


def test_lyapunov_sum_equals_the_divergence():
    lam, _ = diag.lyapunov_spectrum(
        lambda y: lorenz.rk4_step(y), on_attractor(), steps=2000
    )
    assert abs(float(lam["sum"]) - lorenz.DIVERGENCE) < 1e-2


def test_ground_truth_spectrum_is_the_textbook_one():
    lam, _ = diag.lyapunov_spectrum(
        lambda y: lorenz.rk4_step(y), on_attractor(), steps=3000
    )
    assert 0.7 < float(lam["lambda_1"]) < 1.1
    assert float(lam["lambda_2"]) == pytest.approx(0.0, abs=0.1)
    assert -15.0 < float(lam["lambda_3"]) < -14.0


def test_identity_model_does_not_grow():
    """The floor control: a model with no dynamics has no chaotic growth."""
    eg, _ = diag.error_growth(diag.identity_step, on_attractor(), steps=200)
    assert abs(float(eg["growth_rate"])) < 1e-6


def test_cloud_volume_change_is_exp_two_dt_divergence():
    """The identity ensemble training relies on."""
    members = on_attractor() + jax.random.normal(jax.random.PRNGKey(0), (32, 3)) * 0.5
    nxt = jax.vmap(lambda y: lorenz.rk4_step(y))(members)

    def log_volume(z):
        z = z - z.mean(axis=0)
        return jnp.linalg.slogdet(z.T @ z / (len(z) - 1))[1]

    change = float(log_volume(nxt) - log_volume(members))
    assert change == pytest.approx(2 * lorenz.DT * lorenz.DIVERGENCE, abs=0.02)
