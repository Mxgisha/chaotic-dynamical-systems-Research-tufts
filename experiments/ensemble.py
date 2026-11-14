"""Ensemble training: constrain contraction with a cloud instead of an answer.

A single trajectory constrains the divergence only at second order in ``dt``:
the covariance of a *cloud* of nearby states transforms as

    C_{n+1} = Dphi C_n Dphi^T,        det C_{n+1} / det C_n = det(Dphi)^2

so the volume ratio of one step is ``exp(2 dt div)`` — a first-order effect.
Matching the cloud therefore constrains the divergence without ever being told
what it is, and it supplies data *off the attractor*, where the transverse
contraction that dominates the divergence actually lives (see step 2).

    python experiments/step3_ensemble_training.py

This module holds the dataset, the loss and the training loop; the driver holds
the arms and the reporting.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import lorenz
import train


# --- data -----------------------------------------------------------------
def make_ensemble_dataset(key, *, n_base=1200, m=16, eps=0.5, dt=lorenz.DT,
                          n_traj=3, steps=4000, burn_in=1000, noise=0.0):
    """Base states on the attractor, a fixed perturbation cloud, and its image.

    ``bases``  (n_base, 3)          attractor states
    ``delta``  (n_base, m, 3)       fixed perturbations, isotropic
    ``true``   (n_base, m, 3)       the true step applied to each member
    """
    k_ic, k_delta, k_noise = jax.random.split(key, 3)
    ics = lorenz.initial_conditions(k_ic, n_traj)
    xs = []
    for ic in ics:
        tr = lorenz.trajectory(ic, steps, dt, burn_in=burn_in)
        xs.append(tr[:-1])
    pool = jnp.concatenate(xs, axis=0)
    idx = jax.random.choice(k_delta, pool.shape[0], (n_base,), replace=False)
    bases = pool[idx]

    delta = jax.random.normal(k_noise, (n_base, m, 3)) * eps
    members = bases[:, None, :] + delta
    true = jax.vmap(jax.vmap(lambda y: lorenz.rk4_step(y, dt)))(members)
    if noise:
        members = members + jax.random.normal(k_noise, members.shape) * noise
    return {
        "bases": bases.astype(train.DTYPE_TRAIN),
        "delta": delta.astype(train.DTYPE_TRAIN),
        "true": true.astype(train.DTYPE_TRAIN),
        "members": members.astype(train.DTYPE_TRAIN),
        "eps": eps, "m": m,
    }


# --- losses ---------------------------------------------------------------
def _cov(Z, m):
    """Per-sample covariance of centred members, Z is (n, m, 3)."""
    return jnp.einsum("nmi,nmj->nij", Z, Z) / (m - 1)


def ensemble_terms(kind, model, members, true, dt=lorenz.DT):
    """Return (mean_error, covariance_error).

    Matching the covariance already fixes the volume, because the determinant of
    the covariance is the square of the volume ratio.  A separate log-volume term
    was tried and dropped: it diverges while the predicted cloud is still
    singular.
    """
    step = train.make_step(kind, model, dt)
    pred = jax.vmap(jax.vmap(step))(members)
    m = members.shape[1]

    pred_mean = jnp.mean(pred, axis=1)
    true_mean = jnp.mean(true, axis=1)
    mean_err = jnp.mean(jnp.sum((pred_mean - true_mean) ** 2, axis=-1))

    Zp = pred - pred_mean[:, None, :]
    Zt = true - true_mean[:, None, :]
    Cp, Ct = _cov(Zp, m), _cov(Zt, m)
    cov_err = jnp.mean(jnp.sum((Cp - Ct) ** 2, axis=(-1, -2)))
    return mean_err, cov_err


def make_ensemble_loss(kind, dt=lorenz.DT, w_mean=1.0, w_cov=1.0):
    """Ensemble-only loss, with each term normalised by its own data scale.

    Without normalisation the position term is O(10^2) and the covariance term
    O(10^-4), so the covariance weight would be meaningless.  Each term is
    divided by the magnitude of its target computed from the batch.
    """

    def loss_fn(model, members, true):
        mean_err, cov_err = ensemble_terms(kind, model, members, true, dt)
        m = members.shape[1]
        true_mean = jnp.mean(true, axis=1)
        scale_mean = jnp.mean(jnp.sum(true_mean ** 2, axis=-1)) + 1e-8
        Ct = _cov(true - true_mean[:, None, :], m)
        scale_cov = jnp.mean(jnp.sum(Ct ** 2, axis=(-1, -2))) + 1e-12
        total = w_mean * mean_err / scale_mean + w_cov * cov_err / scale_cov
        return total, (mean_err, cov_err)

    return loss_fn


def pair_loss(kind, model, x, y, dt=lorenz.DT):
    return train.mse_loss(kind, model, x, y, dt)


# --- training -------------------------------------------------------------
def train_model(kind, model, key, *, pairs=None, ens=None, w_pairs=1.0,
                w_ens=1.0, epochs=40, batch=192, lr=1e-3, verbose=False):
    """Train on pairs, on ensembles, or both.

    When both are given, every step takes one batch of each and applies a single
    update, so the two objectives see the same number of updates.
    """
    ens_loss = make_ensemble_loss(kind)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n_pairs = 0 if pairs is None else pairs["x"].shape[0]
    n_ens = 0 if ens is None else ens["members"].shape[0]
    n = max(n_pairs, n_ens)
    n_batches = max(1, n // batch)

    use_pairs = pairs is not None
    use_ens = ens is not None

    @eqx.filter_jit
    def step(model, opt_state, xb, yb, mb, tb):
        def total_loss(model):
            tot = jnp.zeros(())
            aux = (jnp.zeros(()), jnp.zeros(()))
            if use_pairs:
                tot = tot + w_pairs * pair_loss(kind, model, xb, yb)
            if use_ens:
                l, aux = ens_loss(model, mb, tb)
                tot = tot + w_ens * l
            return tot, aux

        (total, aux), grads = eqx.filter_value_and_grad(total_loss, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, total, aux

    history = []
    for epoch in range(epochs):
        key, kp, ke = jax.random.split(key, 3)
        perm_p = (jax.random.permutation(kp, n_pairs) if n_pairs else None)
        perm_e = (jax.random.permutation(ke, n_ens) if n_ens else None)
        totals = []
        for b in range(n_batches):
            if perm_p is not None:
                i = jnp.arange(b * batch, (b + 1) * batch) % n_pairs
                idx = perm_p[i]
                xb, yb = pairs["x"][idx], pairs["y"][idx]
            else:
                xb = yb = jnp.zeros((batch, 3))
            if perm_e is not None:
                i = jnp.arange(b * batch, (b + 1) * batch) % n_ens
                j = perm_e[i]
                mb, tb = ens["members"][j], ens["true"][j]
            else:
                mb = tb = jnp.zeros((batch, ens["m"] if ens else 2, 3))
            model, opt_state, total, aux = step(
                model, opt_state, xb, yb, mb, tb)
            totals.append(total)
        history.append({"epoch": epoch, "total": float(jnp.mean(jnp.stack(totals)))})
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss {history[-1]['total']:.6f}")
    return model, history


# --- measurement ----------------------------------------------------------
def evaluate(model, kind, pairs, probe, test_ens, dt=lorenz.DT):
    """Contraction ratio, one-step MSE, and cloud spread ratio."""
    m64 = train.cast(model, train.DTYPE_DIAG)
    step = train.make_step(kind, m64, dt)
    div = jax.vmap(lambda s: train.implied_divergence(step, s, dt))(probe)
    learned = float(jnp.mean(div))
    mse = train.final_mse(pairs, step)

    members = test_ens["members"].astype(train.DTYPE_DIAG)
    true = test_ens["true"].astype(train.DTYPE_DIAG)
    pred = jax.vmap(jax.vmap(step))(members)
    m = members.shape[1]
    Zp = pred - pred.mean(1, keepdims=True)
    Zt = true - true.mean(1, keepdims=True)
    spread_p = jnp.sqrt(jnp.mean(jnp.sum(Zp**2, axis=-1)))
    spread_t = jnp.sqrt(jnp.mean(jnp.sum(Zt**2, axis=-1)))
    return {
        "learned_rate": learned,
        "ratio": learned / lorenz.DIVERGENCE,
        "mse": mse,
        "spread_ratio": float(spread_p / spread_t),
    }
