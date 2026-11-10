"""Step 3: does an ensemble objective fix contraction without being told the answer?

Arms share data, architecture, optimiser and budget.  The single-trajectory arms
are the baseline; the ensemble arms never see `-13.667`.

    python experiments/step3_ensemble_training.py
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
import ensemble
import lorenz
import train

OUT = pathlib.Path(__file__).resolve().parent
(OUT / "results").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

DT = 0.01
NOISE = 0.05
EPOCHS = 40
SEEDS = (0, 1, 2)
BATCH = 192
EPS = 0.5
M = 16


def main():
    print(train.describe_device())
    key = jax.random.PRNGKey(0)
    k_data, k_ens, k_test, k_testens = jax.random.split(key, 4)

    pairs = train.make_dataset(k_data, n_traj=4, steps=3000, burn_in=1000,
                               noise=NOISE, dt=DT)
    ens = ensemble.make_ensemble_dataset(k_ens, n_base=1200, m=M, eps=EPS, dt=DT)
    test_ens = ensemble.make_ensemble_dataset(k_testens, n_base=300, m=M,
                                              eps=EPS, dt=DT)
    probe = lorenz.trajectory(jnp.array([1.0, 1.0, 1.0]), 1500,
                              dt=DT, burn_in=1000)

    print(f"pairs: {pairs['x'].shape[0]}   ensemble bases: {ens['bases'].shape[0]}"
          f" x {M} members   eps {EPS}")

    # sanity: how much does the true cloud change in one step?
    Z0 = ens["members"] - ens["members"].mean(1, keepdims=True)
    Z1 = ens["true"] - ens["true"].mean(1, keepdims=True)
    vol0 = jnp.linalg.slogdet(ensemble._cov(Z0, M) + jnp.eye(3) * 1e-8)[1]
    vol1 = jnp.linalg.slogdet(ensemble._cov(Z1, M) + jnp.eye(3) * 1e-8)[1]
    print(f"true cloud log-volume change per step: {float(jnp.mean(vol1 - vol0)):+.4f}"
          f"  (expected {2 * DT * lorenz.DIVERGENCE:+.4f})")

    spec = [
        ("map, pairs only", "map", dict(use_pairs=True, use_ens=False)),
        ("field, pairs only", "field", dict(use_pairs=True, use_ens=False)),
        ("field + analytic penalty", "field", dict(use_pairs=True, use_ens=False,
                                                   div_weight=1.0)),
        ("map, pairs + ensemble", "map", dict(use_pairs=True, use_ens=True)),
        ("field, pairs + ensemble", "field", dict(use_pairs=True, use_ens=True)),
        ("map, ensemble only", "map", dict(use_pairs=False, use_ens=True)),
    ]

    def run(kind, seed, use_pairs, use_ens, div_weight=0.0):
        model = train.make_mlp(jax.random.PRNGKey(100 + seed), width=64, depth=2)
        if div_weight:
            model, _ = train.train(kind, model, pairs, jax.random.PRNGKey(seed),
                                   epochs=EPOCHS, batch=BATCH, dt=DT,
                                   div_weight=div_weight, verbose=False)
        else:
            model, _ = ensemble.train_model(
                kind, model, jax.random.PRNGKey(seed),
                pairs=pairs if use_pairs else None,
                ens=ens if use_ens else None,
                epochs=EPOCHS, batch=BATCH)
        return ensemble.evaluate(model, kind, pairs, probe, test_ens, dt=DT)

    print(f"\narms (input noise {NOISE}, seeds {SEEDS})")
    arms = {}
    for name, kind, opts in spec:
        runs = [run(kind, sd, **opts) for sd in SEEDS]
        agg = {}
        for metric in ("ratio", "mse", "spread_ratio", "learned_rate"):
            vals = [r[metric] for r in runs]
            agg[metric] = sum(vals) / len(vals)
            agg[metric + "_spread"] = max(vals) - min(vals)
        arms[name] = agg
        print(f"  {name:28s} ratio {agg['ratio']:6.3f}+/-{agg['ratio_spread']/2:5.3f}"
              f"  mse {agg['mse']:.5f}+/-{agg['mse_spread']/2:.5f}"
              f"  spread {agg['spread_ratio']:6.3f}")

    (OUT / "results" / "step3_ensemble_training.json").write_text(
        json.dumps({"noise": NOISE, "eps": EPS, "m": M, "seeds": list(SEEDS),
                    "arms": arms}, indent=2))

    # --- figure -----------------------------------------------------------
    names = list(arms)
    colors = ["#C44E52", "#DD8452", "#4C72B0", "#8172B3", "#55A868", "#937860"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axs[0].bar(range(len(names)), [arms[n]["ratio"] for n in names], color=colors,
               yerr=[arms[n]["ratio_spread"] / 2 for n in names], capsize=3)
    axs[0].axhline(1.0, color="black", ls="--", lw=1)
    axs[0].set_xticks(range(len(names)))
    axs[0].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    axs[0].set_ylabel("learned / true contraction")
    axs[0].set_title("contraction accuracy (1.0 is correct)")
    axs[0].grid(alpha=.3, axis="y")

    axs[1].bar(range(len(names)), [arms[n]["mse"] for n in names], color=colors,
               yerr=[arms[n]["mse_spread"] / 2 for n in names], capsize=3)
    axs[1].set_yscale("log")
    axs[1].set_xticks(range(len(names)))
    axs[1].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    axs[1].set_ylabel("one-step MSE (log scale)")
    axs[1].set_title("accuracy cost of each repair")
    axs[1].grid(alpha=.3, axis="y", which="both")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "step3_ensemble_training.png", dpi=180,
                facecolor="white")
    plt.close(fig)

    print("\n--- summary ---")
    for n, r in arms.items():
        print(f"  {n:28s} ratio {r['ratio']:6.3f}+/-{r['ratio_spread']/2:5.3f}"
              f"  mse {r['mse']:.5f}+/-{r['mse_spread']/2:.5f}")
    print("wrote experiments/results/step3_ensemble_training.json and "
          "figures/step3_ensemble_training.png")


if __name__ == "__main__":
    main()
