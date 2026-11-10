"""Step 1: is the divergence error a function of the timestep, as predicted?

The paper's mechanism says the volume-change rate enters a one-step prediction
only at second order in ``dt``.  That is a claim about a *rate*, and it makes a
testable prediction: if the divergence is weakly constrained because it is a
derivative-level property, then the same map accuracy should buy a *worse*
divergence estimate as ``dt`` shrinks, because the divergence is the Jacobian
divided by ``dt``.

Hypothesis: relative divergence error scales like ``dt^-1``.

We train the same field and map emulators at four timesteps over the same total
simulated time, hold the data quantity roughly fixed, and measure the learned
volume-change rate against the exact rate of the reference step map at that dt.

    python experiments/step1_scaling.py
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
import lorenz
import train

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).resolve().parent
(OUT / "results").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

# RK4 stops being an accurate reference well before 0.1: the step map's own
# volume-change rate drifts from -13.667 to about -10.8 there.  We report that
# check and train only where the reference is valid.
TRAIN_DTS = (0.01, 0.02, 0.05)
CHECK_DTS = (0.01, 0.02, 0.05, 0.1)
SEEDS = (0, 1)
EPOCHS = 80
TARGET_PAIRS = 2500
T_TOTAL = 30.0


def measure(step, probe, dt):
    """Mean volume-change rate of a step function over probe states."""
    vals = jax.vmap(lambda s: train.implied_divergence(step, s, dt))(probe)
    return float(jnp.mean(vals)), float(jnp.std(vals))


def run_one(dt, seed, kind):
    """Train one emulator at one timestep, return its measured rates."""
    steps = int(round(T_TOTAL / dt))
    burn = int(round(10.0 / dt))
    n_traj = max(1, min(12, int(round(TARGET_PAIRS / steps))))

    key = jax.random.PRNGKey(1000 * seed + int(round(dt * 1000)))
    data = train.make_dataset(key, n_traj=n_traj, steps=steps, burn_in=burn,
                              noise=0.05, dt=dt)
    model = train.make_mlp(jax.random.PRNGKey(seed + 7), width=128, depth=3)
    model, _ = train.train(kind, model, data, key, epochs=EPOCHS, batch=256,
                           dt=dt, verbose=False)

    step = train.make_step(kind, train.cast(model, train.DTYPE_DIAG), dt)
    probe = lorenz.trajectory(jnp.array([1.0, 1.0, 1.0]), 1500,
                              dt=dt, burn_in=burn)
    learned, spread = measure(step, probe, dt)
    mse = train.final_mse(data, step)
    return {
        "dt": dt, "seed": seed, "kind": kind, "n_pairs": data["x"].shape[0],
        "learned_rate": learned, "rate_spread": spread,
        "mse": mse, "mse_per_dt2": mse / dt**2,
    }


def reference_rate(dt):
    """Exact volume-change rate of the reference RK4 step map at this dt."""
    burn = int(round(10.0 / dt))
    probe = lorenz.trajectory(jnp.array([1.0, 1.0, 1.0]), 1500, dt=dt, burn_in=burn)
    gt = lambda y: lorenz.rk4_step(y, dt)
    return measure(gt, probe, dt)


def main():
    print(train.describe_device())
    print("\n--- is the reference step map itself accurate at this dt? ---")
    refs = {}
    for dt in CHECK_DTS:
        ref_mean, ref_spread = reference_rate(dt)
        refs[dt] = {"mean": ref_mean, "spread": ref_spread}
        drift = abs(ref_mean - lorenz.DIVERGENCE) / abs(lorenz.DIVERGENCE)
        print(f"  dt {dt:.2f}  step-map rate {ref_mean:+9.3f}  "
              f"spread {ref_spread:7.3f}  drift from ODE divergence {drift*100:5.1f}%")

    rows = []
    for dt in TRAIN_DTS:
        ref_mean = refs[dt]["mean"]
        print(f"\ndt = {dt:.2f}   reference rate {ref_mean:+.4f} "
              f"(spread {refs[dt]['spread']:.2e}, ode divergence {lorenz.DIVERGENCE:+.4f})")
        for kind in ("map", "field"):
            for seed in SEEDS:
                r = run_one(dt, seed, kind)
                r["ref_rate"] = ref_mean
                r["rel_error"] = abs(r["learned_rate"] - ref_mean) / abs(ref_mean)
                rows.append(r)
                print(f"  {kind:5s} seed {seed}  learned {r['learned_rate']:+9.3f}"
                      f"  rel err {r['rel_error']*100:7.2f}%  mse/dt2 {r['mse_per_dt2']:.4f}")

    # aggregate per (dt, kind)
    agg = []
    for dt in TRAIN_DTS:
        for kind in ("map", "field"):
            sel = [r for r in rows if r["dt"] == dt and r["kind"] == kind]
            rel = [r["rel_error"] for r in sel]
            agg.append({
                "dt": dt, "kind": kind,
                "rel_error_mean": sum(rel) / len(rel),
                "rel_error_spread": (max(rel) - min(rel)) / 2,
                "learned_rate_mean": sum(r["learned_rate"] for r in sel) / len(sel),
            })

    # log-log slope of relative error against dt
    fits = {}
    for kind in ("map", "field"):
        pts = [a for a in agg if a["kind"] == kind and a["rel_error_mean"] > 0]
        x = jnp.log(jnp.array([p["dt"] for p in pts]))
        y = jnp.log(jnp.array([p["rel_error_mean"] for p in pts]))
        xc, yc = x - x.mean(), y - y.mean()
        slope = float(jnp.sum(xc * yc) / jnp.sum(xc * xc))
        fits[kind] = slope

    results = {"dt_values": list(TRAIN_DTS), "rows": rows, "aggregate": agg,
               "reference": {str(k): v for k, v in refs.items()},
               "loglog_slope": fits}
    (OUT / "results" / "step1_scaling.json").write_text(json.dumps(results, indent=2))

    # --- figure -----------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, mark in (("map", "o"), ("field", "s")):
        pts = [a for a in agg if a["kind"] == kind]
        axs[0].errorbar([p["dt"] for p in pts], [p["rel_error_mean"] for p in pts],
                        yerr=[p["rel_error_spread"] for p in pts],
                        marker=mark, capsize=3, label=kind)
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel(r"timestep $\Delta t$")
    axs[0].set_ylabel("relative divergence error")
    axs[0].set_title(f"error vs timestep (fitted slopes: "
                     f"map {fits['map']:+.2f}, field {fits['field']:+.2f})")
    axs[0].axhline(0.01, color="grey", ls=":", lw=1)
    axs[0].text(0.011, 0.011, "1%", fontsize=7, color="grey")
    axs[0].legend(); axs[0].grid(alpha=.3, which="both")

    axs[1].axhline(lorenz.DIVERGENCE, color="black", lw=1.5, label="true divergence")
    for kind, mark in (("map", "o"), ("field", "s")):
        pts = [r for r in rows if r["kind"] == kind]
        axs[1].scatter([p["dt"] for p in pts], [p["learned_rate"] for p in pts],
                       marker=mark, label=kind)
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"timestep $\Delta t$")
    axs[1].set_ylabel("learned volume-change rate")
    axs[1].set_title("learned rate vs the true rate")
    axs[1].legend(); axs[1].grid(alpha=.3)
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "step1_scaling.png", dpi=180, facecolor="white")
    plt.close(fig)

    print("\n--- aggregate ---")
    for a in agg:
        print(f"  dt {a['dt']:.2f} {a['kind']:5s}  rel err "
              f"{a['rel_error_mean']*100:7.2f}%  rate {a['learned_rate_mean']:+8.3f}")
    print(f"\nlog-log slope of relative error vs dt: "
          f"map {fits['map']:+.2f}, field {fits['field']:+.2f}  "
          f"(hypothesis: about -1)")
    print("wrote experiments/results/step1_scaling.json and figures/step1_scaling.png")


if __name__ == "__main__":
    main()
