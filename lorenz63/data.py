"""Dataset construction and RNG discipline.

Two defects are fixed here.

**D7 -- the sweep was not a sweep.**  ``traing_data_set`` called
``jrandom.uniform(master_key, ...)`` and ``jrandom.normal(noise_key, ...)``
without ever splitting ``master_key``, so all four sweep points reused the
identical initial-condition pool *and* the identical noise realisation: only the
noise magnitude changed.  Every entry point here takes an explicit key and
splits it, and :func:`point_key` derives an independent key per sweep point.

**D8/D17 -- the wrong sampler was used and the test set was not disjoint.**
``traing_data_set_many`` (short windows from manifold points) existed but was
only called by a visualisation cell, while training and evaluation used the
single-trajectory ``traing_data_set``.  The manifold sampler is now the default,
and test initial conditions are drawn from a separate pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from lorenz63.backend import xp
from lorenz63.system import generate_trajectory


@dataclass(frozen=True)
class Dataset:
    """Explicit container replacing the old positional 3-tuple.

    The old ``return x_train, y_train, testing_initial_states`` made it easy to
    transpose ``y_train`` and ``testing_initial_states`` silently.
    """

    x_train: Any  # (n_samples, 3)
    y_train: Any  # (n_samples, horizon, 3)
    test_ic: Any  # (3,)
    test_trajectory: Any  # (eval_steps, 3) ground truth from test_ic
    train_trajectory: Any  # (train_steps, 3) reference trajectory
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.x_train.shape[0])


def _point_salt(noise_scale: float, horizon: int, replicate: int) -> int:
    """A stable 32-bit salt for ``(noise_scale, horizon, replicate)``.

    ``jax.random.fold_in`` requires a uint32 data argument, and the naive
    ``int(noise_scale * 1e6) * 1000 + ...`` encoding overflows it once a
    replicate index is folded in.  A BLAKE2b digest is stable across processes
    (unlike ``hash``, which is salted) and always fits.
    """
    import hashlib

    payload = f"{abs(float(noise_scale)):.9g}|{int(horizon)}|{int(replicate)}".encode()
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


def point_key(root_key, *, noise_scale: float, horizon: int, replicate: int = 0):
    """Derive an independent key for one sweep point (and replicate).

    Deterministic in ``(root_key, noise_scale, horizon, replicate)`` so a run is
    reproducible, but distinct across sweep points so the noise axis actually
    varies the data, and distinct across replicates so seed-to-seed spread can be
    quantified.  Fixes D7.
    """
    import jax.random as jrandom  # only for keys; works under either backend

    return jrandom.fold_in(root_key, _point_salt(noise_scale, horizon, replicate))


def training_dataset(trajectory, horizon: int, stride: int = 1):
    """Slide a window of ``horizon + 1`` states over ``trajectory``.

    Returns ``(X, Y)`` with ``X`` of shape ``(n, 3)`` and ``Y`` of shape
    ``(n, horizon, 3)``, where ``Y[i, k]`` is the state ``k+1`` steps after
    ``X[i]``.  This matches the one-step flow map: ``Y[i, 0] = phi_dt(X[i])``.
    """
    window_length = horizon + 1
    max_start = trajectory.shape[0] - window_length
    if max_start <= 0:
        raise ValueError(
            f"trajectory of length {trajectory.shape[0]} is too short for "
            f"horizon={horizon} (needs at least {window_length + 1} states)"
        )
    starts = xp.arange(0, max_start, stride)
    # Build windows by stacking slices: works under numpy and jax, unlike
    # lax.dynamic_slice inside a vmap over python ints.
    windows = xp.stack([trajectory[s : s + window_length] for s in _tolist(starts)])
    return windows[:, 0, :], windows[:, 1:, :]


def _tolist(a):
    return [int(v) for v in (a.tolist() if hasattr(a, "tolist") else a)]


def initial_conditions(key, n: int, *, low=-15.0, high=15.0, z_offset=25.0):
    """Draw ``n`` initial conditions, offset in z to sit near the attractor."""
    import jax.random as jrandom

    ic = jrandom.uniform(key, shape=(n, 3), minval=low, maxval=high)
    return ic.at[:, 2].add(z_offset)


def make_dataset(
    key,
    *,
    n_train: int = 6,
    n_test: int = 3,
    noise_scale: float = 0.1,
    horizon: int = 10,
    dt: float = 0.01,
    train_steps: int = 5000,
    eval_steps: int = 5000,
    burn_in: int = 1000,
    n_short: int = 100,
    short_steps: int = 200,
    sampler: Literal["manifold", "long_traj"] = "manifold",
) -> Dataset:
    """Build a dataset for one sweep point.

    ``sampler="manifold"`` (default) samples short windows starting from points
    drawn off long reference trajectories, which is what ``traing_data_set_many``
    did correctly.  ``sampler="long_traj"`` reproduces the legacy behaviour for
    backward comparison.
    """
    import jax.random as jrandom

    key, k_ic, k_noise, k_sel, k_test = jrandom.split(key, 5)

    # Training and test IC pools are disjoint: test seeds are never trained on.
    n_ic = n_train + n_test
    ic_all = initial_conditions(k_ic, n_ic)
    train_ics = ic_all[:n_train]
    test_ics = ic_all[n_train:]

    if sampler == "manifold":
        # One long reference trajectory; windows are drawn from its manifold.
        ref_steps = burn_in + train_steps
        ref = generate_trajectory(train_ics[0], ref_steps, dt, burn_in=0)[burn_in - 1 :]
        manifold = xp.reshape(ref, (-1, 3))
        n_avail = manifold.shape[0]
        idx = jrandom.choice(k_sel, n_avail, shape=(n_short,), replace=False)
        starts = manifold[idx]

        x_parts, y_parts, trajs = [], [], []
        for s in _tolist_rows(starts):
            traj = generate_trajectory(xp.asarray(s, dtype=xp.float64), short_steps, dt, burn_in=0)
            trajs.append(traj)
        trajs = xp.stack(trajs)
        for traj in trajs:
            x_d, y_d = training_dataset(traj, horizon=horizon)
            x_parts.append(x_d)
            y_parts.append(y_d)
        x_train_pre = xp.concatenate(x_parts, axis=0)
        y_train = xp.concatenate(y_parts, axis=0)
        train_trajectory = trajs[0]
    else:
        x_parts, y_parts, trajs = [], [], []
        for ic in train_ics:
            traj = generate_trajectory(ic, train_steps, dt, burn_in=burn_in)
            trajs.append(traj)
            x_d, y_d = training_dataset(traj, horizon=horizon)
            x_parts.append(x_d)
            y_parts.append(y_d)
        x_train_pre = xp.concatenate(x_parts, axis=0)
        y_train = xp.concatenate(y_parts, axis=0)
        train_trajectory = trajs[0]

    # Measurement noise on the training inputs only; targets stay clean.
    noise = jrandom.normal(k_noise, shape=x_train_pre.shape) * noise_scale
    x_train = x_train_pre + noise

    test_ic = test_ics[0]
    test_trajectory = generate_trajectory(test_ic, eval_steps, dt, burn_in=burn_in)

    meta = {
        "noise_scale": float(noise_scale),
        "horizon": int(horizon),
        "dt": float(dt),
        "sampler": sampler,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_samples": int(x_train.shape[0]),
        "n_short": int(n_short) if sampler == "manifold" else None,
        "short_steps": int(short_steps) if sampler == "manifold" else None,
        "eval_steps": int(eval_steps),
        "burn_in": int(burn_in),
    }
    return Dataset(
        x_train=x_train,
        y_train=y_train,
        test_ic=test_ic,
        test_trajectory=test_trajectory,
        train_trajectory=train_trajectory,
        meta=meta,
    )


def _tolist_rows(a):
    for row in (a.tolist() if hasattr(a, "tolist") else a):
        yield row


# --- Deprecated alias ------------------------------------------------------
def traing_data_set(*args, **kwargs):
    """Deprecated misspelled alias.  Use :func:`make_dataset`.

    Kept for one release so the old notebook keeps running during migration.
    Returns the legacy 3-tuple rather than a :class:`Dataset`.
    """
    ds = make_dataset(*args, **kwargs)
    return ds.x_train, ds.y_train, ds.test_ic


__all__ = [
    "Dataset",
    "make_dataset",
    "training_dataset",
    "initial_conditions",
    "point_key",
    "traing_data_set",
]
