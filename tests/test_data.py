"""Dataset construction: the RNG defect and the disjoint-test-set fix."""

import numpy as np
import pytest

from lorenz63.data import make_dataset, point_key, training_dataset, traing_data_set

SMOKE = dict(
    n_train=2,
    n_test=2,
    n_short=4,
    short_steps=60,
    train_steps=300,
    eval_steps=60,
    burn_in=100,
)


def test_point_key_differs_per_sweep_point():
    """D7: every sweep point must get independent data.

    The old code reused one frozen IC pool and one noise realisation for all four
    points, so the "noise sweep" only varied magnitude on a single pattern.
    """
    import jax.random as jrandom

    root = jrandom.PRNGKey(67)
    k1 = point_key(root, noise_scale=0.05, horizon=10)
    k2 = point_key(root, noise_scale=0.1, horizon=10)
    k3 = point_key(root, noise_scale=0.1, horizon=20)
    assert not np.array_equal(np.asarray(k1), np.asarray(k2))
    assert not np.array_equal(np.asarray(k2), np.asarray(k3))
    # deterministic
    assert np.array_equal(
        np.asarray(point_key(root, noise_scale=0.1, horizon=10)), np.asarray(k2)
    )


def test_different_noise_scales_give_different_training_data():
    import jax.random as jrandom

    root = jrandom.PRNGKey(67)
    a = make_dataset(point_key(root, noise_scale=0.05, horizon=10), noise_scale=0.05,
                     horizon=10, **SMOKE)
    b = make_dataset(point_key(root, noise_scale=0.4, horizon=10), noise_scale=0.4,
                     horizon=10, **SMOKE)
    # Same root seed, different point: the sampled initial conditions differ.
    assert not np.allclose(a.test_ic, b.test_ic)
    assert not np.allclose(np.asarray(a.x_train), np.asarray(b.x_train))


def test_same_seed_is_reproducible():
    import jax.random as jrandom

    root = jrandom.PRNGKey(67)
    kw = dict(noise_scale=0.1, horizon=10, **SMOKE)
    a = make_dataset(point_key(root, **{k: kw[k] for k in ("noise_scale", "horizon")}), **kw)
    b = make_dataset(point_key(root, **{k: kw[k] for k in ("noise_scale", "horizon")}), **kw)
    assert np.allclose(np.asarray(a.x_train), np.asarray(b.x_train))
    assert np.allclose(np.asarray(a.y_train), np.asarray(b.y_train))


def test_shapes_and_one_step_alignment():
    """Y[i,k] must be the state k+1 steps after X[i], matching the flow map."""
    import jax.random as jrandom

    ds = make_dataset(jrandom.PRNGKey(3), noise_scale=0.0, horizon=10, **SMOKE)
    assert ds.x_train.shape[1] == 3
    assert ds.y_train.shape[1] == 10
    assert ds.y_train.shape[2] == 3
    assert ds.test_trajectory.shape == (SMOKE["eval_steps"], 3)
    # with zero noise, X[i] must be a state on the trajectory and Y[i,0] its
    # successor under the flow map
    from lorenz63 import system as S

    x0 = np.asarray(ds.x_train[0])
    y0 = np.asarray(ds.y_train[0, 0])
    assert np.allclose(S.flow_map(x0, 0.01), y0, atol=1e-9)


def test_test_ic_is_disjoint_from_training():
    """D17: the test initial condition must not come from the training pool."""
    import jax.random as jrandom

    ds = make_dataset(jrandom.PRNGKey(11), noise_scale=0.0, horizon=10, **SMOKE)
    assert not np.allclose(ds.test_ic, ds.x_train[0])


def test_training_dataset_rejects_too_short_trajectory():
    traj = np.zeros((5, 3))
    with pytest.raises(ValueError):
        training_dataset(traj, horizon=10)


def test_training_dataset_window_count():
    traj = np.arange(21, dtype=float).reshape(-1, 1).repeat(3, axis=1)
    X, Y = training_dataset(traj, horizon=4)
    # windows start at 0 .. (21 - 5) - 1 = 15  -> 16 windows
    assert X.shape == (16, 3)
    assert Y.shape == (16, 4, 3)
    assert np.allclose(X[0], traj[0])
    assert np.allclose(Y[0, 0], traj[1])


def test_deprecated_alias_still_works():
    import jax.random as jrandom

    x, y, ic = traing_data_set(
        jrandom.PRNGKey(5), noise_scale=0.1, horizon=10, **SMOKE
    )
    assert x.shape[1] == 3 and y.shape[1] == 10 and ic.shape == (3,)
