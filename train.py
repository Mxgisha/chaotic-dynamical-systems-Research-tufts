"""Emulators: equinox MLPs, losses, and a training loop.

Two parameterisations of the same object -- one step of the Lorenz flow:

``map``    learn ``x_{n+1} = f(x_n)`` directly
``field``  learn ``dx/dt = g(x)`` and integrate it with RK4

Precision policy (important on GPU): **train in float32, diagnose in float64.**
Consumer GPUs run float64 at a fraction of float32 speed, but a float32 Lyapunov
spectrum is wrong in ways that look like a physics result.  So parameters are
float32 during training and upcast for the diagnostics, which are cheap.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import lorenz

DTYPE_TRAIN = jnp.float32
DTYPE_DIAG = jnp.float64


# --- setup ----------------------------------------------------------------
def setup():
    """Enable float64 and report the device.  Call once, before anything else."""
    jax.config.update("jax_enable_x64", True)
    backend, devices = jax.default_backend(), jax.devices()
    x64 = jax.config.read("jax_enable_x64")
    return backend, devices, x64


def x64_enabled():
    """Whether float64 is actually available in this process."""
    return bool(jax.config.jax_enable_x64)


def describe_device():
    backend, devices, x64 = setup()
    on_gpu = backend in ("gpu", "cuda", "rocm")
    msg = f"JAX backend: {backend} | devices: {devices} | x64 available: {x64}"
    if not on_gpu:
        msg += "\n(running on CPU -- the same code selects the GPU automatically if one is visible)"
    return msg


# --- models ---------------------------------------------------------------
def make_mlp(key, width=64, depth=2):
    """An MLP 3 -> width -> ... -> 3, initialised in the training dtype."""
    m = eqx.nn.MLP(
        in_size=3,
        out_size=3,
        width_size=width,
        depth=depth,
        activation=jax.nn.silu,
        use_bias=True,
        use_final_bias=True,
        key=key,
    )
    return cast(m, DTYPE_TRAIN)


def cast(model, dtype):
    """Return a copy of the model with every floating array in ``dtype``.

    Raises rather than silently truncating if float64 was asked for while x64 is
    disabled: jax would quietly hand back float32, and a float32 Lyapunov
    spectrum is wrong in a way that reads as a physics result.
    """
    if dtype == jnp.float64 and not x64_enabled():
        raise RuntimeError(
            "float64 requested but x64 is disabled -- call train.setup() first "
            "(jax silently truncates otherwise)"
        )
    return jax.tree_util.tree_map(
        lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, model
    )


def step_map(model, y):
    """One step of a map emulator."""
    return model(y)


def step_field(model, y, dt=lorenz.DT):
    """One RK4 step of a *vector field* emulator."""
    k1 = model(y)
    k2 = model(y + 0.5 * dt * k1)
    k3 = model(y + 0.5 * dt * k2)
    k4 = model(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_step(kind, model, dt=lorenz.DT):
    """Return the one-step function for ``kind`` in {'map', 'field'}."""
    if kind == "map":
        return lambda y: step_map(model, y)
    if kind == "field":
        return lambda y: step_field(model, y, dt)
    raise ValueError(f"unknown kind {kind!r}")


def rollout(step, y0, steps):
    """Apply ``step`` ``steps`` times, returning the states after each step."""

    def body(y, _):
        nxt = step(y)
        return nxt, nxt

    _, out = jax.lax.scan(body, y0, None, length=steps)
    return out


# --- data -----------------------------------------------------------------
def make_pairs(traj):
    """Consecutive pairs (x_n, x_{n+1}) from a trajectory."""
    return traj[:-1], traj[1:]


def make_dataset(key, *, n_traj=4, steps=4000, burn_in=1000, noise=0.0, dt=lorenz.DT):
    """Build (x, y) training pairs from ``n_traj`` trajectories, plus a test run.

    ``noise`` is added to the *inputs* only; the targets stay clean.  That models
    an imperfectly observed system whose true dynamics you still want to learn.
    """
    k_init, k_noise, k_test = jax.random.split(key, 3)
    ics = lorenz.initial_conditions(k_init, n_traj + 1)

    xs, ys = [], []
    for ic in ics[:n_traj]:
        traj = lorenz.trajectory(ic, steps, dt, burn_in=burn_in)
        x, y = make_pairs(traj)
        xs.append(x)
        ys.append(y)
    x = jnp.concatenate(xs, axis=0)
    y = jnp.concatenate(ys, axis=0)

    if noise:
        x = x + jax.random.normal(k_noise, x.shape) * noise

    full_test = lorenz.trajectory(ics[n_traj], steps, dt, burn_in=burn_in)
    # test_traj[0] is the state the emulator is rolled out from, so a rollout of
    # length len(test_traj) - 1 lines up with test_traj[1:].  Getting this wrong
    # silently compares the model against a *different* trajectory.
    return {
        "x": x.astype(DTYPE_TRAIN),
        "y": y.astype(DTYPE_TRAIN),
        "test_ic": full_test[0].astype(DTYPE_DIAG),
        "test_traj": full_test.astype(DTYPE_DIAG),
        "noise": noise,
    }


# --- loss and training ----------------------------------------------------
def divergence_at(model, y):
    """Divergence of the learned field at a single state, by autodiff."""
    return jnp.trace(jax.jacrev(model)(y))


def implied_divergence(step, y, dt=lorenz.DT):
    """Volume-change rate implied by a one-step map: ``ln|det D step| / dt``.

    This is the exact definition of the rate at which the step changes volume,
    and it is what the Lyapunov exponents sum to.  ``(trace - dim)/dt`` is *not*
    equivalent at finite dt: the flow Jacobian has O(dt^2) corrections to its
    trace that are large here, and using it understates the divergence by ~20%.
    """
    J = jax.jacrev(step)(y)
    return jnp.linalg.slogdet(J)[1] / dt


def mse_loss(kind, model, x, y, dt=lorenz.DT):
    pred = jax.vmap(make_step(kind, model, dt))(x)
    return jnp.mean((pred - y) ** 2)


def divergence_penalty(model, x, target=lorenz.DIVERGENCE):
    """Mean squared error of the learned field's divergence against a target.

    Only meaningful for the ``field`` parameterisation: the divergence is a
    first derivative of the field's output, so it can be evaluated and therefore
    penalised.  A map emulator has no equivalent local quantity.

    ``target`` may be a scalar (the analytically known divergence) or an array
    with one entry per row of ``x`` (a per-state estimate, e.g. from data).
    """
    div = jax.vmap(lambda s: divergence_at(model, s))(x)
    t = jnp.asarray(target, dtype=div.dtype)
    if t.ndim == 0:
        t = t * jnp.ones_like(div)
    return jnp.mean((div - t) ** 2)


def make_loss(kind, dt=lorenz.DT, div_weight=0.0, div_target=lorenz.DIVERGENCE):
    """Build ``fn(model, x, y, target=None) -> (total, (mse, penalty))``."""
    if div_weight and kind != "field":
        raise ValueError("the divergence penalty only applies to the 'field' kind")

    def loss_fn(model, x, y, target=None):
        mse = mse_loss(kind, model, x, y, dt)
        if not div_weight:
            return mse, (mse, jnp.zeros(()))
        t = div_target if target is None else target
        pen = divergence_penalty(model, x, t)
        return mse + div_weight * pen, (mse, pen)

    return loss_fn


def train(
    kind,
    model,
    data,
    key,
    *,
    epochs=40,
    lr=1e-3,
    batch=256,
    div_weight=0.0,
    div_target=lorenz.DIVERGENCE,
    dt=lorenz.DT,
    verbose=True,
):
    """Fit one emulator.  Returns ``(model, history)``."""
    x, y = data["x"], data["y"]
    n = x.shape[0]
    loss_fn = make_loss(kind, dt, div_weight, div_target)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # a per-state target lives in the dataset; a scalar lives in the closure
    per_state = data.get("div_target")
    if per_state is not None:
        per_state = jnp.asarray(per_state)

    @eqx.filter_jit
    def step(model, opt_state, xb, yb, tb):
        (total, (mse, pen)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, xb, yb, tb
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, total, mse, pen

    history = []
    for epoch in range(epochs):
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, n)
        totals, mses, pens = [], [], []
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            tb = None if per_state is None else per_state[idx]
            model, opt_state, total, mse, pen = step(
                model, opt_state, x[idx], y[idx], tb
            )
            totals.append(total)
            mses.append(mse)
            pens.append(pen)
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "total": float(jnp.mean(jnp.stack(totals))),
                "mse": float(jnp.mean(jnp.stack(mses))),
                "divergence_penalty": float(jnp.mean(jnp.stack(pens))),
            }
        )
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            h = history[-1]
            print(
                f"epoch {epoch:3d} | mse {h['mse']:.6f}"
                f" | divergence penalty {h['divergence_penalty']:.4f}"
            )
    return model, history


def final_mse(data, step_diag):
    """One-step MSE over the whole training set, evaluated in float64.

    ``step_diag`` must be built from a ``cast(model, DTYPE_DIAG)``: mixing a
    float64 input with float32 weights is a dtype error in JAX, which is the one
    sharp edge of the float32-train / float64-diagnose split.
    """
    x = data["x"].astype(DTYPE_DIAG)
    y = data["y"].astype(DTYPE_DIAG)
    pred = jax.vmap(step_diag)(x)
    return float(jnp.mean((pred - y) ** 2))
