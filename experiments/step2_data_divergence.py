"""Step 2: can the divergence penalty work without knowing the divergence?

The paper's repair (§4.4) uses ``-13.667`` because it can be computed
analytically for Lorenz-63.  That is partly circular.  This experiment replaces
the analytic target with one **estimated from the data**, using a local linear
fit over nearest neighbours (``experiments/estimators.py``), and asks three
questions:

1. Does the estimator recover the true rate from clean pairs?
2. How does it degrade as the inputs get noisy, and as the neighbourhood grows?
3. Does the repair still work when the target is the estimate rather than the
   answer?

There is a secondary hypothesis worth testing here.  A local linear fit is a
*smoother*: noisy inputs attenuate its slope, which shrinks the fitted Jacobian
and therefore over-estimates contraction.  The trained emulator is also a
smoother.  If the estimator's noise response tracks the emulator's, then the
emulator's over-contraction is not an arbitrary bug but the attenuation bias of
local regression — which would explain the noise dependence the paper reports
without explaining.

    python experiments/step2_data_divergence.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import estimators
import lorenz
import train

OUT = pathlib.Path(__file__).resolve().parent
(OUT / "results").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

DT = 0.01
BURN = 1000
STEPS = 3000
N_TRAJ = 3
SEED = 0
K_GRID = (10, 25, 50, 100, 200)
NOISE_GRID = (0.0, 0.05, 0.2, 0.4)
EPOCHS = 40


def transient_pairs(key, n_traj=60, steps=250, dt=DT):
    """Pairs from states that are still *approaching* the attractor.

    Locally, on-attractor data span a ~2-D sheet; the transverse contraction
    that carries most of the divergence is not sampled.  Transient data fill the
    volume, which is what a local linear fit needs.
    """
    ics = jax.random.uniform(key, (n_traj, 3), minval=-18.0, maxval=18.0)
    ics = ics.at[:, 2].add(25.0)
    xs, ys = [], []
    for ic in ics:
        tr = lorenz.trajectory(ic, steps, dt=dt, burn_in=0)
        xs.append(tr[:-1]); ys.append(tr[1:])
    return jnp.concatenate(xs), jnp.concatenate(ys)


def build_pairs(key, n_traj=N_TRAJ, steps=STEPS, burn=BURN, dt=DT):
    """Consecutive pairs from several clean trajectories."""
    ics = lorenz.initial_conditions(key, n_traj + 1)
    xs, ys = [], []
    for ic in ics[:n_traj]:
        tr = lorenz.trajectory(ic, steps, dt, burn_in=burn)
        xs.append(tr[:-1])
        ys.append(tr[1:])
    test = lorenz.trajectory(ics[n_traj], steps, dt, burn_in=burn)
    return (jnp.concatenate(xs), jnp.concatenate(ys),
            ics[n_traj], test)


def add_noise(x, key, noise):
    if not noise:
        return x
    return x + jax.random.normal(key, x.shape) * noise


def rate_of(step, probe, dt=DT):
    vals = jax.vmap(lambda s: train.implied_divergence(step, s, dt))(probe)
    return float(jnp.mean(vals)), float(jnp.std(vals))


def main():
    print(train.describe_device())
    key = jax.random.PRNGKey(SEED)
    k_ic, k_noise, k_split = jax.random.split(key, 3)
    X, Y, test_ic, test_traj = build_pairs(k_ic)
    print(f"pairs: {X.shape[0]}")
    true_rate = float(lorenz.DIVERGENCE)

    anchors = estimators.spread_anchors(X.shape[0], stride=10)
    print(f"anchors: {len(anchors)}")

    # --- 1/2. the estimator: which data are needed? -----------------------
    Xt, Yt = transient_pairs(k_split)
    est_rows = []
    for label, (Cx, Cy) in (("on-attractor", (X, Y)), ("transient", (Xt, Yt))):
        for noise in NOISE_GRID:
            Cn = add_noise(Cx, k_noise, noise)
            for k in K_GRID:
                est = estimators.estimate_at_points_chunked(
                    Cn, Cy, X[anchors], k=k, dt=DT)
                est = est[jnp.isfinite(est)]
                mean = float(jnp.mean(est))
                est_rows.append({"cloud": label, "noise": noise, "k": k,
                                 "mean": mean, "spread": float(jnp.std(est)),
                                 "ratio": mean / true_rate})
                print(f"  {label:13s} noise {noise:.2f} k {k:3d}  "
                      f"estimate {mean:+9.2f}  ratio {mean/true_rate:6.3f}")

    # a per-anchor target to train against, estimated from CLEAN pairs
    k_best = 50
    target_attr = estimators.estimate_at_points_chunked(X, Y, X, k=k_best, dt=DT)
    target_trans = estimators.estimate_at_points_chunked(Xt, Yt, X, k=k_best, dt=DT)

    # --- 3. repair arms at high noise -------------------------------------
    noise = 0.4
    Xn = add_noise(X, k_noise, noise)
    data = {
        "x": Xn.astype(train.DTYPE_TRAIN),
        "y": Y.astype(train.DTYPE_TRAIN),
        "test_ic": test_traj[0].astype(train.DTYPE_DIAG),
        "test_traj": test_traj.astype(train.DTYPE_DIAG),
    }
    probe = lorenz.trajectory(jnp.array([1.0, 1.0, 1.0]), 1500, burn_in=BURN)
    print(f"\nrepair arms at input noise {noise}")

    arms = {}

    def note(name, model, kind, div_target=None):
        step = train.make_step(kind, train.cast(model, train.DTYPE_DIAG), DT)
        mean, spread = rate_of(step, probe)
        mse = train.final_mse(data, step)
        arms[name] = {"learned_rate": mean, "spread": spread, "mse": mse,
                      "ratio": mean / true_rate}
        print(f"  {name:22s} rate {mean:+8.3f}  ratio {mean/true_rate:6.3f}"
              f"  spread {spread:7.2f}  mse {mse:.5f}")

    for kind in ("map", "field"):
        model = train.make_mlp(jax.random.PRNGKey(10), 64, 2)
        model, _ = train.train(kind, model, data, key, epochs=EPOCHS,
                               batch=256, dt=DT, verbose=False)
        note(kind, model, kind)

    # analytic target
    model = train.make_mlp(jax.random.PRNGKey(20), 64, 2)
    model, _ = train.train("field", model, data, key, epochs=EPOCHS, batch=256,
                           dt=DT, div_weight=1.0, div_target=true_rate, verbose=False)
    note("field + analytic", model, "field")

    # estimated target, neighbours from the attractor only
    model = train.make_mlp(jax.random.PRNGKey(21), 64, 2)
    d_attr = dict(data); d_attr["div_target"] = target_attr.astype(train.DTYPE_TRAIN)
    model, _ = train.train("field", model, d_attr, key, epochs=EPOCHS, batch=256,
                           dt=DT, div_weight=1.0, verbose=False)
    note("field + est (attractor)", model, "field")

    # estimated target, neighbours from transient data
    model = train.make_mlp(jax.random.PRNGKey(22), 64, 2)
    d_trans = dict(data); d_trans["div_target"] = target_trans.astype(train.DTYPE_TRAIN)
    model, _ = train.train("field", model, d_trans, key, epochs=EPOCHS, batch=256,
                           dt=DT, div_weight=1.0, verbose=False)
    note("field + est (transient)", model, "field")

    # how good is each target, on the same scale as the repair?
    dose = []
    for label, tgt in (("attractor", target_attr), ("transient", target_trans)):
        err = float(jnp.mean(jnp.abs(tgt - true_rate))) / abs(true_rate)
        dose.append({"target": label, "mean": float(jnp.mean(tgt)),
                     "target_error": err})
        print(f"  target from {label:10s}: mean {float(jnp.mean(tgt)):+8.3f}"
              f"  mean abs error {err*100:5.1f}%")

    results = {"true_rate": true_rate, "estimator": est_rows, "arms": arms,
               "dose_response": dose, "noise": noise, "k_used": k_best}
    (OUT / "results" / "step2_data_divergence.json").write_text(json.dumps(results, indent=2))

    # --- figures ----------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))

    for cloud, color in (("on-attractor", "tab:blue"), ("transient", "tab:green")):
        pts = [r for r in est_rows if r["cloud"] == cloud and r["noise"] == 0.0]
        axs[0].plot([p["k"] for p in pts], [p["ratio"] for p in pts],
                    marker="o", color=color, label=cloud)
    axs[0].axhline(1.0, color="black", lw=1)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("neighbours k"); axs[0].set_ylabel("estimated / true")
    axs[0].set_title("clean data: transients are needed")
    axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)

    for cloud, color in (("on-attractor", "tab:blue"), ("transient", "tab:green")):
        pts = [r for r in est_rows if r["cloud"] == cloud and r["k"] == 25]
        axs[1].plot([p["noise"] for p in pts], [p["ratio"] for p in pts],
                    marker="o", color=color, label=cloud)
    axs[1].axhline(1.0, color="black", lw=1)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("input noise"); axs[1].set_ylabel("estimated / true")
    axs[1].set_title("k = 25: noise breaks it")
    axs[1].legend(fontsize=7); axs[1].grid(alpha=.3, which="both")

    names = list(arms)
    cols = ["#C44E52", "#DD8452", "#4C72B0", "#8172B3", "#55A868"]
    axs[2].bar(range(len(names)), [arms[n]["ratio"] for n in names], color=cols)
    axs[2].axhline(1.0, color="black", lw=1, ls="--")
    axs[2].set_xticks(range(len(names)))
    axs[2].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    axs[2].set_ylabel("learned / true contraction")
    axs[2].set_title(f"repair at noise {noise}")
    axs[2].grid(alpha=.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "step2_data_divergence.png", dpi=180, facecolor="white")
    plt.close(fig)

    print("\n--- repair arms ---")
    for n, a in arms.items():
        print(f"  {n:22s} ratio {a['ratio']:6.3f}  mse {a['mse']:.5f}")
    print("wrote experiments/results/step2_data_divergence.json and "
          "figures/step2_data_divergence.png")


if __name__ == "__main__":
    main()
