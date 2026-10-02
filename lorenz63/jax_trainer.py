"""Optional JAX trainer.

The research stack is JAX + diffrax + equinox + optax.  The NumPy trainer in
:mod:`lorenz63.train` exists so the diagnostics and the whole test suite run on a
CPU-only machine; this module is the **same objective** on the JAX stack, for
speed.

The two implementations share the definition of every loss term, and
``tests/test_jax_trainer.py`` checks that they agree numerically on identical
data and initial weights.  That is the check that keeps them from drifting apart
the way the notebook, ``functions.py`` and ``derivaitve-lor63.py`` did.

Importing this module is always safe: if JAX is missing, ``HAVE_JAX`` is False and
:func:`require_jax` raises a clear error, while :mod:`lorenz63.models` keeps using
the NumPy backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lorenz63.losses import LossConfig

try:  # pragma: no cover - the import path depends on the environment
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    # Must happen at import time, before any array is created or traced.  The
    # pipeline differentiates a chaotic flow over 10^4 steps; in float32 the
    # Jacobian products lose precision and training diverges to a gradient norm
    # of ~1e6 within a few steps instead of converging.
    jax.config.update("jax_enable_x64", True)

    HAVE_JAX = True
except ImportError:  # pragma: no cover
    HAVE_JAX = False


ACCELERATED_BACKENDS = ("jax", "jax_mlp", "jax_tracked")


def require_jax():
    if not HAVE_JAX:
        raise ImportError(
            "the JAX backend needs jax, equinox and optax. Install them with "
            "`pip install -r requirements-jax.txt`, or use the NumPy backend "
            "(LORENZ63_BACKEND=numpy / --backend numpy)."
        )


def enable_x64():
    """float64 is required: the pipeline differentiates a chaotic flow over 10^4
    steps, and float32 destroys the Lyapunov spectrum."""
    require_jax()
    jax.config.update("jax_enable_x64", True)


# --- model -----------------------------------------------------------------
def make_equinox_model(width=128, depth=3, seed=67, activation=None, key=None):
    """Equinox MLP with the same architecture as :class:`lorenz63.models.NumpyMLP`."""
    require_jax()
    enable_x64()
    import jax.nn as jnn

    act = activation or jnn.silu
    return eqx.nn.MLP(
        in_size=3,
        out_size=3,
        width_size=width,
        depth=depth,
        activation=act,
        use_bias=True,
        # A final bias is included so the architecture is parameter-for-parameter
        # identical to lorenz63.models.NumpyMLP, which is what lets the two
        # trainers be compared and the weights exported between them.
        use_final_bias=True,
        key=key if key is not None else jax.random.PRNGKey(seed),
    )


class EquinoxModelAdapter:
    """Expose an equinox MLP through the interface the diagnostics expect.

    ``step``/``forward`` match :class:`lorenz63.models.NumpyMLP` so
    :func:`lorenz63.controls.emulator_trajectory` and every diagnostic work
    unchanged; ``set_params`` accepts either an equinox module or a raw PyTree.
    """

    def __init__(self, model):
        self.model = model

    # --- inference --------------------------------------------------------
    def forward(self, x, params=None):
        require_jax()
        m = self.model if params is None else params
        return np.asarray(jax.vmap(m)(jnp.asarray(x, dtype=jnp.float64)))

    __call__ = forward

    def step(self, state, params=None):
        require_jax()
        m = self.model if params is None else params
        return np.asarray(m(jnp.asarray(state, dtype=jnp.float64)))

    def rollout(self, ic, steps):
        require_jax()
        m = self.model

        def body(carry, _):
            nxt = m(carry)
            return nxt, nxt

        _, out = jax.lax.scan(body, jnp.asarray(ic, dtype=jnp.float64), None, length=steps)
        return np.asarray(out)

    # --- parameters -------------------------------------------------------
    def get_params(self):
        return self.model

    def set_params(self, params):
        self.model = params

    @property
    def n_params(self) -> int:
        require_jax()
        leaves = jax.tree_util.tree_leaves(eqx.filter(self.model, eqx.is_array))
        return int(sum(np.size(np.asarray(x)) for x in leaves))

    # --- persistence ------------------------------------------------------
    def save(self, path):
        require_jax()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, self.model)
        return path

    @classmethod
    def load(cls, path, like):
        require_jax()
        return cls(eqx.tree_deserialise_leaves(path, like.model))

    def to_numpy_mlp(self, width=None, depth=None, seed=67):
        """Export to :class:`lorenz63.models.NumpyMLP` for backend-independent use.

        Equinox's MLP keeps its chain in ``.layers`` as a list of
        :class:`equinox.nn.Linear`, so the parameters are read from there rather
        than by walking every leaf of the tree (which would pick up the
        activation callable and the string key names).
        """
        require_jax()
        from lorenz63.models import NumpyMLP

        linears = list(self.model.layers)
        first = linears[0]
        n_linear = len(linears)
        # NumpyMLP builds in_size + depth hidden layers + out_size, so the
        # architecture is fully determined by the equinox layer shapes.  The
        # depth must be passed explicitly: leaving it to the default would build
        # a different number of layers and then load the weights into the wrong
        # shapes.
        in_size = int(first.weight.shape[1])
        hidden = int(first.weight.shape[0])
        out_size = int(linears[-1].weight.shape[0])
        np_mlp = NumpyMLP(
            in_size=in_size,
            out_size=out_size,
            width=width or hidden,
            depth=n_linear - 1,
            seed=seed,
        )
        params = {}
        for i, layer in enumerate(linears):
            # equinox stores Linear.weight as (out_features, in_features); the
            # NumPy MLP uses (in_features, out_features).
            w = np.asarray(layer.weight, dtype=np.float64).T
            if layer.bias is None:
                b = np.zeros(w.shape[1], dtype=np.float64)
            else:
                b = np.asarray(layer.bias, dtype=np.float64)
            params[f"layer_{i}"] = {"w": w, "b": b, "act": i < n_linear - 1}
        if len(params) != np_mlp.depth + 1:
            raise ValueError(
                f"layer count mismatch: equinox has {len(params)} linear layers, "
                f"the numpy MLP was built with depth={np_mlp.depth} ({np_mlp.depth + 1} layers)"
            )
        np_mlp.set_params(params)
        return np_mlp


# --- losses (JAX) ----------------------------------------------------------
def soft_histogram(x, centres, width):
    """Differentiable triangular-kernel density histogram, shape ``(n_bins,)``.

    Weights each sample by its **value** as well as its kernel weight, so the
    result is ``sum_j W_jb x_j`` -- the same convention as
    :func:`lorenz63.losses.soft_l1_distribution_distance`, which is then divided
    by ``n * width`` to form a density.  Using the raw counts ``sum_j W_jb``
    instead gives a different (and non-matching) objective.
    """
    d = jnp.abs(x[:, None] - centres[None, :]) / width
    return jnp.sum(jnp.clip(1.0 - d, 0.0, None) * x[:, None], axis=0)


def l1_distribution_loss(real, emu, edges):
    """Mean over coordinates of the histogram L1 distance.

    Matches :func:`lorenz63.losses.l1_distribution_distance` (one histogram per
    coordinate) using the differentiable kernel of
    :func:`lorenz63.losses.soft_l1_distribution_distance`.
    """
    total = 0.0
    dim = real.shape[1]
    for i in range(dim):
        eg = edges[i]
        width = eg[1] - eg[0]
        centers = 0.5 * (eg[:-1] + eg[1:])
        h_r = soft_histogram(real[:, i], centers, width) / (real.shape[0] * width)
        h_e = soft_histogram(emu[:, i], centers, width) / (emu.shape[0] * width)
        total = total + jnp.sum(jnp.abs(h_r - h_e)) * width
    return total / dim


def make_loss_fn(loss_cfg: LossConfig, edges):
    """Build ``fn(model, x, y) -> (total, (mse, l1))`` for equinox."""

    def loss_fn(model, x, y):
        pred = jax.vmap(model)(x)
        mse = jnp.mean((pred - y) ** 2)
        l1 = jnp.asarray(0.0)
        if loss_cfg.l1_weight > 0.0 and edges is not None:
            l1 = l1_distribution_loss(y, pred, edges)
        total = mse + loss_cfg.l1_weight * l1
        return total, (mse, l1)

    return loss_fn


def fixed_edges(x_train, y_one, n_bins):
    """Histogram edges fixed from the whole training set.

    Fixing them keeps the objective independent of which rows land in a
    minibatch, and keeps the bins constant across a JIT-traced step.
    """
    import numpy as _np

    edge_list = []
    for j in range(y_one.shape[1]):
        lo = min(float(_np.min(x_train[:, j])), float(_np.min(y_one[:, j])))
        hi = max(float(_np.max(x_train[:, j])), float(_np.max(y_one[:, j])))
        if hi <= lo:
            hi = lo + 1.0
        edge_list.append(jnp.asarray(_np.linspace(lo, hi, n_bins + 1), dtype=jnp.float64))
    return edge_list


# --- training --------------------------------------------------------------
def train_jax(
    adapter,
    dataset,
    config=None,
    loss=None,
    verbose=True,
    history_path=None,
):
    """Train an equinox MLP with Adam on the one-step MSE + L1 objective.

    ``config`` is a :class:`lorenz63.train.TrainConfig` (or None for defaults).
    Returns ``(adapter, history)`` where ``history`` is a list of per-epoch dicts
    with the same keys the NumPy trainer produces.
    """
    require_jax()
    enable_x64()
    from lorenz63.train import TrainConfig

    config = config or TrainConfig()
    loss_cfg = loss or config.loss

    x_train = np.asarray(dataset.x_train, dtype=np.float64)
    y_one = np.asarray(dataset.y_train, dtype=np.float64)[:, 0, :]
    n = x_train.shape[0]

    edges = fixed_edges(x_train, y_one, loss_cfg.n_bins) if loss_cfg.l1_weight > 0 else None
    loss_fn = make_loss_fn(loss_cfg, edges)

    model = adapter.model
    # `optax.adam(lr)` already includes the descent step, so no extra scale(-1).
    # The learning-rate schedule is applied to the final updates rather than to
    # the gradients: scaling the gradient before `clip_by_global_norm` would
    # defeat the clip (a large gradient divided down passes the threshold
    # untouched), which is what previously stalled the JAX path.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def lr_at(epoch):
        frac = epoch / max(config.epochs - 1, 1)
        return config.lr * (0.1 ** max(0.0, (frac - 0.5) / 0.5))

    @eqx.filter_jit
    def batch_step(model, opt_state, xb, yb, lr):
        (total, (mse, l1)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, xb, yb
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        updates = jax.tree_util.tree_map(lambda u: lr * u, updates)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, total, mse, l1

    def train_epoch(model, opt_state, x, y, lr):
        n_batches = x.shape[0] // config.batch_size
        totals, mses, l1s = [], [], []
        for b in range(n_batches):
            s = b * config.batch_size
            xb = x[s : s + config.batch_size]
            yb = y[s : s + config.batch_size]
            model, opt_state, total, mse, l1 = batch_step(model, opt_state, xb, yb, lr)
            totals.append(total)
            mses.append(mse)
            l1s.append(l1)
        return (
            model,
            opt_state,
            jnp.mean(jnp.stack(totals)),
            jnp.mean(jnp.stack(mses)),
            jnp.mean(jnp.stack(l1s)),
        )

    history = []
    rng = np.random.default_rng(config.seed)
    x_jax = jnp.asarray(x_train)
    y_jax = jnp.asarray(y_one)

    for epoch in range(config.epochs):
        lr = lr_at(epoch)
        perm = rng.permutation(n)
        model, opt_state, total, mse, l1 = train_epoch(
            model, opt_state, x_jax[perm], y_jax[perm], lr
        )
        entry = {
            "epoch": epoch,
            "lr": float(lr),
            "total": float(total),
            "mse": float(mse),
            "l1": float(l1),
            "ftle": 0.0,
        }
        history.append(entry)
        if verbose and (epoch % 10 == 0 or epoch == config.epochs - 1):
            print(
                f"Epoch {epoch:3d} | lr {lr:.2e} | total {entry['total']:.6f} "
                f"| mse {entry['mse']:.6f} | l1 {entry['l1']:.6f}"
            )

    adapter.set_params(model)
    if history_path is not None:
        import json

        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        Path(history_path).write_text(json.dumps(history, indent=2))
    return adapter, history


__all__ = [
    "HAVE_JAX",
    "ACCELERATED_BACKENDS",
    "require_jax",
    "enable_x64",
    "make_equinox_model",
    "EquinoxModelAdapter",
    "soft_histogram",
    "l1_distribution_loss",
    "make_loss_fn",
    "fixed_edges",
    "train_jax",
]
