"""Model, loss-gradient and training-loop checks."""

import numpy as np
import pytest

from lorenz63.losses import (
    LossConfig,
    l1_distribution_distance,
    mse_loss,
    soft_l1_distribution_distance,
)
from lorenz63.models import IdentityModel, NumpyMLP
from lorenz63.train import TrainConfig, adam_update, backward, forward_with_cache


def test_mlp_shapes_and_determinism():
    m = NumpyMLP(width=16, depth=2, seed=1)
    x = np.zeros((5, 3))
    y = m(x)
    assert y.shape == (5, 3)
    m2 = NumpyMLP(width=16, depth=2, seed=1)
    assert np.allclose(y, m2(x))


def test_mlp_save_load_roundtrip(tmp_path):
    m = NumpyMLP(width=8, depth=2, seed=3)
    x = np.random.default_rng(0).normal(size=(4, 3))
    before = m(x)
    p = m.save(tmp_path / "model.npz")
    m2 = NumpyMLP.load(p)
    assert np.allclose(m2(x), before)
    assert m2.width == m.width and m2.depth == m.depth


def test_identity_model_has_no_dynamics():
    m = IdentityModel()
    x = np.random.default_rng(0).normal(size=(6, 3))
    assert np.allclose(m(x), x)


def test_mse_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(4, 3))
    target = rng.normal(size=(4, 3))
    from lorenz63.losses import mse_grad

    g = mse_grad(pred, target)
    eps = 1e-6
    for idx in [(0, 0), (2, 1), (3, 2)]:
        p = pred.copy()
        p[idx] += eps
        up = mse_loss(p, target)
        p[idx] -= 2 * eps
        dn = mse_loss(p, target)
        assert np.isclose((up - dn) / (2 * eps), g[idx], rtol=1e-5, atol=1e-8)


def test_soft_l1_gradient_matches_finite_differences():
    """The distributional term must be differentiable in the predictions.

    Bin edges are held fixed across the perturbation, matching how the trainer
    uses the function: with varying edges the comparison would be measuring the
    edges moving rather than the objective's gradient.
    """
    rng = np.random.default_rng(0)
    real = rng.normal(size=(30, 2))
    emu = rng.normal(size=(30, 2))
    edges = [np.linspace(-3.5, 3.5, 9), np.linspace(-3.5, 3.5, 9)]
    val, back = soft_l1_distribution_distance(real, emu, edges=edges)
    g = back(1.0)
    assert g.shape == emu.shape
    eps = 1e-8
    # Points are chosen to sit strictly inside a kernel; on a kernel kink the
    # objective has a measure-zero derivative discontinuity and a central
    # difference straddling it is not the one-sided analytic gradient.
    bin_centres = [0.5 * (e[:-1] + e[1:]) for e in edges]
    checked = 0
    for r in range(emu.shape[0]):
        for c in range(emu.shape[1]):
            if np.min(np.abs(emu[r, c] - bin_centres[c])) < 0.05:
                continue
            e = emu.copy()
            e[r, c] += eps
            up = soft_l1_distribution_distance(real, e, edges=edges)[0]
            e[r, c] -= 2 * eps
            dn = soft_l1_distribution_distance(real, e, edges=edges)[0]
            fd = (up - dn) / (2 * eps)
            assert np.isclose(fd, g[r, c], rtol=1e-3, atol=1e-6), (r, c, fd, g[r, c])
            checked += 1
            if checked >= 6:
                return
    raise AssertionError("no well-inside sample found to check")


def test_soft_l1_approaches_the_hard_histogram():
    """The differentiable form must approximate the quantity it stands in for.

    This is the check the first implementation failed: it flattened all
    coordinates into a single histogram, so the trained objective (0.159) was not
    the reported one (0.240).  With one histogram per coordinate and a kernel of
    one bin width, the two agree to a few percent.
    """
    rng = np.random.default_rng(2)
    real = rng.normal(size=(500, 3))
    emu = rng.normal(size=(500, 3))
    soft = soft_l1_distribution_distance(real, emu, n_bins=30)[0]
    hard = l1_distribution_distance(real, emu, n_bins=30)
    assert np.isclose(soft, hard, rtol=0.15), (soft, hard)


def test_backward_matches_finite_difference_gradients():
    """Full backprop check through the MLP."""
    rng = np.random.default_rng(0)
    m = NumpyMLP(width=6, depth=2, seed=0)
    keys = m._layer_keys()
    x = rng.normal(size=(5, 3))
    y = rng.normal(size=(5, 3))

    pred, cache = forward_with_cache(x, m.params, keys)
    dpred = 2.0 * (pred - y) / pred.size
    grads = backward(pred, cache, m.params, keys, dpred)

    def loss_of(params):
        p, _ = forward_with_cache(x, params, keys)
        return mse_loss(p, y)

    eps = 1e-6
    for layer_key in keys:
        for pname in ("w", "b"):
            gk = f"{layer_key}/{pname}"
            arr = m.params[layer_key][pname]
            idx = tuple(0 for _ in arr.shape)
            saved = arr[idx]
            arr[idx] = saved + eps
            up = loss_of(m.params)
            arr[idx] = saved - eps
            dn = loss_of(m.params)
            arr[idx] = saved
            fd = (up - dn) / (2 * eps)
            assert np.isclose(fd, grads[gk][idx], rtol=1e-4, atol=1e-7), gk


def test_loss_config_rejects_ftle_target_without_weight():
    """The FTLE term is off by default and cannot be half-configured."""
    with pytest.raises(ValueError):
        LossConfig(ftle_target=0.9056, ftle_weight=0.0)
    cfg = LossConfig()
    assert cfg.ftle_weight == 0.0 and cfg.ftle_target is None


def test_adam_reduces_a_quadratic():
    params = {"layer_0": {"w": np.array([[3.0]]), "b": np.zeros(1)}}
    grads = {"layer_0/w": np.array([[6.0]]), "layer_0/b": np.zeros(1)}
    state = {"t": 0, "m": {"layer_0/w": np.zeros((1, 1)), "layer_0/b": np.zeros(1)},
             "v": {"layer_0/w": np.zeros((1, 1)), "layer_0/b": np.zeros(1)}}
    for _ in range(50):
        adam_update(params, grads, state, 0.1, ["layer_0"])
        grads["layer_0/w"] = 2.0 * params["layer_0"]["w"]
    assert abs(params["layer_0"]["w"][0, 0]) < 0.5


def test_training_reduces_loss():
    """End-to-end: the loop must actually fit the one-step map."""
    import jax.random as jrandom

    from lorenz63.data import make_dataset
    from lorenz63.train import train

    ds = make_dataset(
        jrandom.PRNGKey(1), noise_scale=0.0, horizon=5,
        n_train=2, n_test=1, n_short=3, short_steps=40,
        train_steps=200, eval_steps=30, burn_in=50,
    )
    model = NumpyMLP(width=32, depth=2, seed=0)
    model, history = train(model, ds, TrainConfig(epochs=8, batch_size=64), verbose=False)
    first = np.mean([h["mse"] for h in history[:2]])
    last = np.mean([h["mse"] for h in history[-2:]])
    assert last < first, (first, last)
    assert all(np.isfinite(h["mse"]) for h in history)
