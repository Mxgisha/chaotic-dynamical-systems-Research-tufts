"""The optional JAX trainer.

These tests compare the JAX path against the NumPy path on identical data, which
is the check that keeps the two implementations from drifting apart the way the
notebook, ``functions.py`` and ``derivaitve-lor63.py`` did.

They are skipped when JAX/equinox/optax are not installed, so the default
CPU-only suite still runs everywhere.
"""

import numpy as np
import pytest

from lorenz63.losses import LossConfig
from lorenz63.models import NumpyMLP
from lorenz63.train import TrainConfig, forward_with_cache, backward

jax_trainer = pytest.importorskip("lorenz63.jax_trainer")
if not jax_trainer.HAVE_JAX:  # pragma: no cover
    pytest.skip("JAX stack not installed", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import equinox as eqx  # noqa: E402

from lorenz63.data import make_dataset  # noqa: E402


WIDTH, DEPTH, SEED = 16, 2, 0


@pytest.fixture(scope="module")
def ds():
    return make_dataset(
        jrandom.PRNGKey(1),
        noise_scale=0.0,
        horizon=5,
        n_train=2,
        n_test=1,
        n_short=3,
        short_steps=40,
        train_steps=200,
        eval_steps=40,
        burn_in=50,
    )


def test_x64_is_enabled_at_import():
    """float32 makes the JAX path diverge; see the module docstring."""
    assert jax.config.jax_enable_x64 is True


def test_architectures_have_identical_parameter_counts():
    """The two backends must agree parameter-for-parameter.

    A mismatch here silently breaks weight export and makes the two trainers
    incomparable.
    """
    jax_model = jax_trainer.make_equinox_model(width=WIDTH, depth=DEPTH, seed=SEED)
    adapter = jax_trainer.EquinoxModelAdapter(jax_model)
    np_model = NumpyMLP(width=WIDTH, depth=DEPTH, seed=SEED)
    assert adapter.n_params == np_model.n_params


def test_exported_weights_give_identical_predictions(ds):
    """Exporting the equinox model to NumPy must be exact.

    This checks the (out, in) -> (in, out) transpose and the final bias in one go.
    """
    adapter = jax_trainer.EquinoxModelAdapter(
        jax_trainer.make_equinox_model(width=WIDTH, depth=DEPTH, seed=SEED)
    )
    np_mlp = adapter.to_numpy_mlp()
    x = np.asarray(ds.x_train)[:256]
    assert np.allclose(adapter.forward(x), np_mlp.forward(x), atol=1e-12)


def test_jax_and_numpy_gradients_agree(ds):
    """The two loss implementations must produce the same gradient.

    Compared through the exported model so the parameters are identical: the
    JAX gradient is taken w.r.t. the equinox model, the NumPy gradient through
    the hand-written backprop, and both are measured against the same batch.
    """
    adapter = jax_trainer.EquinoxModelAdapter(
        jax_trainer.make_equinox_model(width=WIDTH, depth=DEPTH, seed=SEED)
    )
    loss_cfg = LossConfig(l1_weight=0.0, n_bins=16)  # MSE only, exactly comparable
    x = np.asarray(ds.x_train)[:64]
    y = np.asarray(ds.y_train)[:, 0, :][:64]

    # JAX gradient
    loss_fn = jax_trainer.make_loss_fn(loss_cfg, None)
    (_, (mse_jax, _)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        adapter.model, jnp.asarray(x), jnp.asarray(y)
    )

    # NumPy gradient on the exported (identical-parameter) model
    np_mlp = adapter.to_numpy_mlp()
    keys = np_mlp._layer_keys()
    pred, cache = forward_with_cache(x, np_mlp.params, keys)
    dpred = 2.0 * (pred - y) / pred.size
    np_grads = backward(pred, cache, np_mlp.params, keys, dpred)

    jax_leaves = [
        np.asarray(v) for v in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    ]
    # equinox leaves are (w, b) per layer in order; the numpy grads use w then b
    for i, (w, b) in enumerate(zip(jax_leaves[0::2], jax_leaves[1::2])):
        assert np.allclose(w.T, np_grads[f"layer_{i}/w"], rtol=1e-8, atol=1e-8), i
        assert np.allclose(b, np_grads[f"layer_{i}/b"], rtol=1e-8, atol=1e-8), i

    from lorenz63.losses import mse_loss

    assert np.isclose(float(mse_jax), mse_loss(pred, y), rtol=1e-10)


def test_train_jax_reduces_loss(ds):
    adapter = jax_trainer.EquinoxModelAdapter(
        jax_trainer.make_equinox_model(width=32, depth=2, seed=0)
    )
    cfg = TrainConfig(epochs=30, batch_size=64, seed=0, loss=LossConfig(l1_weight=0.1, n_bins=16))
    adapter, history = jax_trainer.train_jax(adapter, ds, cfg, verbose=False)
    first = np.mean([h["mse"] for h in history[:3]])
    last = np.mean([h["mse"] for h in history[-3:]])
    assert last < first, (first, last)
    assert all(np.isfinite(h["mse"]) for h in history)
    assert len(history) == cfg.epochs


def test_train_jax_and_numpy_agree_on_final_loss(ds):
    """Both trainers must land in the same place given the same budget."""
    loss_cfg = LossConfig(l1_weight=0.1, n_bins=16)
    cfg = TrainConfig(epochs=200, batch_size=64, seed=0, loss=loss_cfg)

    adapter = jax_trainer.EquinoxModelAdapter(
        jax_trainer.make_equinox_model(width=32, depth=2, seed=0)
    )
    adapter, jax_hist = jax_trainer.train_jax(adapter, ds, cfg, verbose=False)

    from lorenz63.train import train

    np_model = NumpyMLP(width=32, depth=2, seed=0)
    np_model, np_hist = train(np_model, ds, cfg, verbose=False)

    # Different initialisers, so compare convergence quality rather than paths.
    assert jax_hist[-1]["mse"] < 1.0, jax_hist[-1]
    assert np_hist[-1]["mse"] < 1.0, np_hist[-1]
    assert jax_hist[-1]["mse"] < 0.2 * np_hist[0]["mse"]


def test_adapter_satisfies_the_interface_diagnostics_use(ds):
    """`step`, `forward` and `rollout` are what the diagnostics call."""
    adapter = jax_trainer.EquinoxModelAdapter(
        jax_trainer.make_equinox_model(width=WIDTH, depth=DEPTH, seed=SEED)
    )
    assert adapter.step(np.zeros(3)).shape == (3,)
    assert adapter.forward(np.zeros((5, 3))).shape == (5, 3)
    assert adapter(np.zeros((5, 3))).shape == (5, 3)
    assert adapter.rollout(ds.test_ic, 7).shape == (7, 3)
    # set_params / get_params roundtrip
    m = adapter.get_params()
    adapter.set_params(m)
    assert adapter.forward(np.zeros((2, 3))).shape == (2, 3)


def test_save_load_roundtrip(tmp_path, ds):
    adapter = jax_trainer.EquinoxModelAdapter(
        jax_trainer.make_equinox_model(width=WIDTH, depth=DEPTH, seed=SEED)
    )
    x = np.zeros((4, 3))
    before = adapter.forward(x)
    path = adapter.save(tmp_path / "model.eqx")
    assert path.exists()
    loaded = jax_trainer.EquinoxModelAdapter.load(path, adapter)
    assert np.allclose(loaded.forward(x), before, atol=1e-14)


def test_make_model_jax_factory():
    """`make_model('jax')` must reach the equinox path.

    It returns the adapter, not the bare equinox module, so the returned object
    satisfies the same interface as NumpyMLP and can be passed straight to the
    diagnostics.
    """
    from lorenz63.models import make_model

    m = make_model("jax", width=WIDTH, depth=DEPTH, seed=SEED)
    assert isinstance(m, jax_trainer.EquinoxModelAdapter)
    assert m.step(np.zeros(3)).shape == (3,)


def test_soft_histogram_matches_numpy_kernel():
    """The JAX histogram kernel must agree with the NumPy one it mirrors."""
    from lorenz63.losses import _tri_weights

    x = np.array([-1.0, 0.0, 0.5, 2.0])
    edges = np.linspace(-2.0, 3.0, 6)
    width = edges[1] - edges[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    # value-weighted: sum_j W_jb * x_j, matching soft_l1_distribution_distance
    np_value = _tri_weights(x, centres, width).T @ x
    jax_value = np.asarray(
        jax_trainer.soft_histogram(jnp.asarray(x), jnp.asarray(centres), width)
    )
    assert np.allclose(np_value, jax_value, atol=1e-12)


def test_l1_distribution_loss_matches_numpy_value(ds):
    """The JAX distributional loss must match the NumPy soft version."""
    from lorenz63.losses import soft_l1_distribution_distance

    rng = np.random.default_rng(0)
    real = rng.normal(size=(64, 3))
    emu = rng.normal(size=(64, 3))
    edges_np = [np.linspace(-3.5, 3.5, 17) for _ in range(3)]
    np_val = soft_l1_distribution_distance(real, emu, edges=edges_np)[0]
    edges_jax = [jnp.asarray(e) for e in edges_np]
    jax_val = float(
        jax_trainer.l1_distribution_loss(jnp.asarray(real), jnp.asarray(emu), edges_jax)
    )
    assert np.isclose(np_val, jax_val, rtol=2e-2), (np_val, jax_val)
