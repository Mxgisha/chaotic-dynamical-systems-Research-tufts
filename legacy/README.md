# Legacy code

Superseded material, kept for the record. **Nothing in this directory is
imported by the package, the tests, or `scripts/run_sweep.py`.**

| Item | What it was | Why it was replaced |
|---|---|---|
| `Lorenze_63.ipynb` | The live Colab notebook (31 cells) | Contained the only copy of the pipeline; three copies of the same logic had drifted apart. Its run outputs are the reference for the old numbers. |
| `Lorenz_63.ipynb` | An earlier export (30 cells) | Calls eight `run_*_test` functions that are **defined nowhere in the file** — they only resolve if `functions.py` has been fetched and `exec`-ed into globals first. |
| `functions.py` | Helper library fetched over HTTP at runtime and `exec`-ed into the notebook's globals | An edit to `main` silently changed notebook behaviour. `lyaponove_stepper` also calls `jnp.linalg.norm` without the module importing `jnp`. Now [`lorenz63/system.py`](../lorenz63/system.py) and [`lorenz63/diagnostics/lyapunov.py`](../lorenz63/diagnostics/lyapunov.py). |
| `derivaitve-lor63.py` | An abandoned neural-ODE formulation (model as a vector field, integrated with Heun) | Never imported; a different learning problem from the discrete one-step map the notebook and tests use. |
| `test/` | 4 sweep directories x 8 PNGs | **28 of the 32 PNGs are fully transparent** 640x480 canvases with no content. They were produced by the `plt.ioff()` → `plt.show()` → `plt.savefig()` sequence, and never regenerated from committed models — no `model.eqx` / `metadata.json` was ever committed, so the evaluation loop short-circuited on every folder. |

## Accessing them

```bash
git show HEAD~1:functions.py          # or any earlier revision
```

The notebooks' embedded cell outputs are worth keeping: they contain the real
plots from the last Colab run, including the failure signal that motivated the
rewrite — a ground-truth lobe-flip count of 30 against the emulator's 20.

Regenerate equivalent figures with:

```bash
python scripts/run_sweep.py --smoke --out runs
```
