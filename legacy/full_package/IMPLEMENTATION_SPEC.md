# Implementation Spec — Lorenz-63 Emulator Failure-Mode Pipeline

Status: **draft for review** (no code written yet)
Scope: consolidate three drifting copies of the pipeline into one importable package, fix the
defects that currently invalidate the committed artifacts, and convert the diagnostics from
qualitative plots into quantified, controlled measurements.

---

## 0. TL;DR

| | |
|---|---|
| **Create** | `lorenz63/` package (8 modules), `configs/*.json`, `scripts/`, `tests/`, `requirements.txt` |
| **Rewrite** | `Lorenze_63.ipynb` -> thin driver notebook (~10 cells) |
| **Delete** | `Lorenz_63.ipynb`, `derivaitve-lor63.py`, `functions.py` (as an executable module — see D3), all 32 files in `test/` |
| **Root cause of blank artifacts** | `plt.ioff()` -> diagnostics call `plt.show()` -> loop calls `plt.savefig()`. Fix = one figure contract (2.8). |
| **Root cause of the "sweep" not being a sweep** | `traing_data_set` never splits `master_key`, so all 4 sweep points reuse identical ICs and identical noise (2.3). |
| **Root cause of the crash on re-run** | `train_model` hard-unpacks 6 values from `stepper`; the legacy steppers return 3 (2.5). |

---

## 1. Evidence for every defect

Each defect is traced to a concrete observation, so the fixes can be validated against it.

| ID | Defect | Evidence |
|----|--------|----------|
| D1 | 28 of 32 committed PNGs are **fully transparent** (640x480 RGBA, 1 opaque pixel, 0 variance) | Pixel-level decode of all `test/*/*.png` |
| D2 | Notebook-embedded outputs *do* contain real 35-300 KB plots | 29 `image/png` outputs in `Lorenze_63.ipynb` cell 18 |
| D3 | `functions.py` is only partly load-bearing; `lyaponove_stepper` uses `jnp.linalg.norm` but the file never imports `jnp` | `functions.py:1-41`; notebook `exec`s the fetched text into globals |
| D4 | `Lorenz_63.ipynb` calls 8 `run_*_test` functions **defined nowhere** in the file | Definition/call-site map; only resolved by exec'ing `functions.py` |
| D5 | `train_model` unpacks 6 values; `model_stepper` and `model_stepper_l1_lya` return **3** | `Lorenze_63.ipynb` cells 10 vs 12 |
| D6 | `model_stepper_l1_lya` has **zero call sites**; the sweep passes the legacy `model_stepper`, so the L1 + FTLE losses never train anything | Call-site map (both notebooks) |
| D7 | `traing_data_set` calls `jrandom.uniform(master_key, ...)` and `jrandom.normal(noise_key, ...)` **without splitting** `master_key`, so every sweep point reuses the same ICs and noise | `Lorenze_63.ipynb` cell 8 |
| D8 | `traing_data_set_many` (the correct sampler) is called only by the visualization cell 15, never by training or eval | Call-site map |
| D9 | No `model.eqx` / `metadata.json` exist in the repo, so the eval loop's `if not os.path.exists(model_path): continue` short-circuits — producing zero artifacts | `ls` of workspace; `test/` holds only PNGs |
| D10 | Cell 17 first uses `folder_name` inside its save block before assignment when *no* model exists, so it raises rather than skipping cleanly | Cell 17 source |
| D11 | `run_distributional_distance_test` computes `emu_z` and never uses it; `bins_x` is hard-coded to +/-25 | Cell 23 |
| D12 | `run_dmd_spectrum_test` applies DMD to a 3-row (state x time) snapshot matrix — 3 modes only, all clustered at Re~1, so the spectrum comparison cannot discriminate | Embedded `dmd_spectrum.png` |
| D13 | `run_lyapunov_spectrum_test` defines `SPEC_STEPS_A = 30_000` and never uses it; GT and emulator both run `test_steps`, with lambda_3 still drifting at cutoff | Cell 22 |
| D14 | `compute_ftle_loss` perturbs along `ones_like(x)*eps` (arbitrary direction, not the leading Lyapunov vector) and ignores `y_batch`; target hard-coded to `0.9056` | Cell 11 |
| D15 | `run_traj_uniqueness_test` calls "TEST PASSED" when 1e-5 perturbations *diverge* — that is chaos, which an identity map also exhibits | Cell 21; output `40/40 (100.0%) TEST PASSED` |
| D16 | `run_spatial_energy_test` thresholds a single scalar moment `E[sum x_i^2]` at a hand-picked 50.0 | Cell 26 |
| D17 | `traing_data_set` returns test ICs drawn from the same 9-IC pool as training; `test_0.1_20`'s embedded run is also incomplete | Cells 8, 18 |
| D18 | `traing_data_set`/`traing_data_set_many` are misspelled and used consistently — renaming is a breaking change across all call sites | Grep |
| D19 | `train_model` rebinds `epochs = epochs`, `batch_size = batch_size`, `x_train = x_train`, `y_train = y_train` (no-ops) | Cell 10 |
| D20 | No dependency manifest anywhere; installs are inline `!pip install -U "jax[cuda12]"` | Cell 1 |

**Genuine scientific signal already present** (what the fixes must preserve and sharpen):

- GT Lyapunov spectrum `0.993, -0.002, -14.519` ~ textbook `(0.906, 0, -14.57)`.
- Emulator `0.074, -0.301, -52.249` -> lambda_1 **13x too small**, lambda_3 **3.6x too negative**.
- Lobe flips `GT 30 | Emu 20` -> ratio **0.667**.
- Euclidean divergence mean 18.2 (attractor diameter), not a skill measure.

---

## 2. Target architecture

```
lorenz63/
  __init__.py
  system.py        RHS, params, Tsit5 stepper, trajectory generation      <- new, consolidates
  data.py          windowing, manifold sampling, RNG discipline           <- new, fixes D7/D8
  models.py        MLP factory, save/load, metadata sidecar               <- new
  losses.py        MSE, L1-distance, FTLE-consistent regularizer          <- new, fixes D14
  train.py         train loop, stepper protocol, loss history             <- new, fixes D5/D6/D19
  diagnostics/
    __init__.py    registry + Diagnostic protocol
    lyapunov.py    Benettin spectra for flow and map                      <- fixes D13
    measure.py     L1/Wasserstein on invariant measures, energy moments   <- fixes D11/D16
    dimension.py   GP-corrected correlation dimension                     <- fixes D11/D16
    symbolic.py    lobe transitions, dwell times, return map              <- keeps the good idea
    dmd.py         Hankel-DMD spectrum + mode correlation                 <- fixes D12
    divergence.py  error-growth rate vs lambda_1, with controls           <- replaces D15
    field.py       3D vector-field slice error
  controls.py      gt-gt and identity/untrained baselines                 <- new, the key addition
  report.py        results.json + figure emission, single figure contract
configs/
  sweep.json       the 4-point sweep, explicit and versioned
scripts/
  run_sweep.py     end-to-end: train -> save -> evaluate -> report
tests/
  test_diagnostics.py
requirements.txt / pyproject.toml                                         <- fixes D20
```

**Design rule:** a diagnostic is a pure function
`(gt_arrays, emu_arrays, ctx) -> DiagnosticResult` and **never touches matplotlib globals**.
Plotting is a separate `render(result) -> Figure` in `report.py`. This makes D1 structurally
impossible rather than fixed-by-patch.

### 2.1 `DiagnosticResult` contract (new)

```python
@dataclass(frozen=True)
class DiagnosticResult:
    name: str                      # e.g. "lyapunov_spectrum"
    value: dict[str, float]        # scalar metrics, JSON-serialisable
    gt: dict[str, float]           # same keys, measured on ground truth
    control_gt_gt: dict[str, float] | None   # null baseline, see 2.6
    control_identity: dict[str, float] | None
    meta: dict[str, Any]           # n_steps, dt, seed, sweeps
    figure: Figure | None = None   # populated only by render()
```

`ratio = value[k] / gt[k]` is then computable for every metric, which turns the existing plots
into a defensible table.

### 2.2 `lorenz63/system.py`

- Move `parms`, `lorenze_63`, `lorenz_rhs`, `lorenz_rhs_diffrax`, `stepper` here.
- **One** RHS definition. Keep `lorenz_rhs(state)` and add the `(t, state, args)` adapter.
- `generate_trajectory(y0, steps, dt, burn_in)` returns `(steps, 3)` float64.
- Use `SaveAt(t1=True)` in the scan stepper (current `stepper` omits `saveat` and indexes `ys[-1]`,
  which works only because `t1=True` is the diffeqsolve default — make it explicit).
- **Resolves D3.**

### 2.3 `lorenz63/data.py` — RNG discipline

```python
def make_dataset(key, *, n_train, n_test, noise_scale, horizon, burn_in, steps,
                 sampler: Literal["long_traj", "manifold"] = "manifold"):
    key, k_ic, k_noise, k_sel = jrandom.split(key, 4)
    ...
```

- **Always split.** The caller owns one root key per sweep point, derived from
  `(root_seed, noise_scale, horizon)` so runs are reproducible *and* independent. **Fixes D7.**
- Default sampler becomes `"manifold"` (current `traing_data_set_many` behaviour): sample short
  windows from manifold points of a few long trajectories. **Fixes D8.**
- Test ICs come from a **disjoint** IC pool (`n_test` seeds never used in training). **Fixes D17.**
- Rename to `make_dataset`; keep `traing_data_set` as a deprecated alias for one release.
  **Addresses D18.**
- Return an explicit `Dataset` dataclass rather than a 3-tuple, so `y_train` shape and
  `testing_initial_states` cannot be silently transposed.

### 2.4 `lorenz63/losses.py`

Keep: MSE. Keep with fixes: L1 distance. Redesign: FTLE.

- **`l1_distribution_distance(real, emu, n_bins)`** — average per-coordinate L1 over the **batch**
  dimension, not the horizon. Document that a 128-sample batch gives a weak distributional signal;
  offer an EMA of batch histograms across steps so the term sees more than 128 points. Record in
  `meta` that the signal is weak.
- **`ftle_regularizer`** — three options, pick one at review:
  1. **Drop it** (recommended default): MSE already pins the one-step map, and the current form is
     a biased, high-variance estimator.
  2. **Variance-reduced:** perturb along the top right-singular vector of the *true* one-step
     Jacobian at `x` (computable once, offline, cached), so the estimate targets lambda_1.
  3. **Keep as-is** but log it separately and default weight to 0.
  In all cases **remove the hard-coded `target_lyapunov=0.9056`**; read from config, default `None`
  (= disabled). **Fixes D14.**
- Every loss returns a `dict` of components; `train.py` logs them per epoch.

### 2.5 `lorenz63/train.py`

- Define **one** stepper:

```python
def make_stepper(loss_fn, optimizer) -> Stepper:
    """Returns fn(model, opt_state, x, y) -> (model, opt_state, LossLog)"""
```

  `LossLog` is a dataclass, not a positional tuple. **Removes the 3-vs-6 unpacking class of bug
  entirely (D5).**
- Delete the `epochs=epochs` etc. no-op block. **Fixes D19.**
- The `stepper` argument is **removed** from `train_model`; the loss becomes a config field. The
  notebook call site is updated accordingly. **Fixes D5.**
- Emit `history.json` (per-epoch total/mse/l1/ftle) — replaces the
  `# ! add functionality to plot both total loss and individual losses` TODO at cell 12. **Fixes D6**
  by making it impossible to train with loss terms that are not wired in.
- Save `model.eqx` **and** `metadata.json` through `models.save` (atomic write). **Fixes D9.**

### 2.6 `lorenz63/controls.py` — the key scientific addition

```python
def gt_gt_control(ic_a, ic_b, **kw) -> dict   # two true trajectories, different ICs
def identity_control(model, **kw) -> dict      # untrained model / x_{n+1} = x_n
```

Every diagnostic is run against both and stored in the result. Rationale: without a null baseline
there is no scale on which "0.667 flip ratio" or "L1 = 0.03" means anything. GT-GT answers *"how
different do two genuinely true trajectories look under this metric?"* — the ceiling. Identity
answers *"what does a model with no dynamics score?"* — the floor.

### 2.7 `lorenz63/diagnostics/`

| Module | Change | Fixes |
|---|---|---|
| `lyapunov.py` | **Rewrite normalisation.** Compute `lambda_i = cumsum(log r_i) / (n*dt)` with `Q` carried **orthonormal** and `r_i = ||residual_i||`; do not rescale `Q` by `DELTA` between steps. Expose `steps=` per system and run GT long enough for lambda_3 to plateau (use `SPEC_STEPS_A`, currently dead). Return lambda plus a convergence diagnostic (slope over the last 20%). | D13 |
| `measure.py` | Drop the unused `emu_z`; derive `bins_x` from the union of GT+emu ranges instead of hard-coded +/-25. Return per-coordinate L1 **and** Wasserstein-1. Keep the energy moment but report **several** moments (mean, variance, skew, kurtosis of `sum x_i^2`) and z-scores against the GT-GT control, not a hand-picked threshold. | D11, D16 |
| `dimension.py` | Add the standard finite-sample correction to `C(r)`; report D2 with a bootstrap CI over trajectory subsamples; compute on a fixed subsample size so GT and emulator are compared at equal N. | D11 |
| `symbolic.py` | Keep `run_lobe_transition_test` largely as-is — it is the best diagnostic in the repo. Add: return flip ratio, mean dwell times, **and** the GT-GT control ratio so 0.667 can be read as significant or not. | — |
| `dmd.py` | Apply DMD to a **Hankel/time-delayed** snapshot matrix, not the raw 3xN state matrix. Report sorted `|lambda|` spectrum, spectral gap, mode-magnitude correlation. | D12 |
| `divergence.py` | **Replace `run_traj_uniqueness_test`.** New metric: fit slope of `log||delta(t)||` over the pre-saturation window across >=5 perturbation scales (1e-4..1e-8), report the estimated lambda_1 against the Benettin value and the GT-GT control. Report `t_sat` and `t_double`. Keep the unique-endpoint count only as a descriptive field, never as pass/fail. | D15 |
| `field.py` | Keep as-is; add scalar `mean||f_gt - f_emu||` to `value` so it becomes a metric, not just a picture. | — |

**Removed from every diagnostic:** the `print("TEST PASSED/WARNING/FAILED")` pattern. Pass/fail
moves to `report.py` as a control-relative comparison, so a "PASS" can never again be emitted for
a property (explosive divergence) that an identity map also has.

### 2.8 `lorenz63/report.py` — the figure contract (fixes D1)

```python
def emit(result: DiagnosticResult, out_dir: Path, *, dpi=150) -> None:
    fig = render(result)                      # diagnostics never own a figure
    fig.savefig(out_dir / f"{result.name}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
```

Rules, enforced by a test:
1. **`savefig` before `show`.** Never the reverse.
2. Diagnostics **never** call `plt.show()`, `plt.ioff()`, or `plt.ion()`. The driver sets the
   backend once (`matplotlib.use("Agg")` in `scripts/`, inline in Colab).
3. Every emitted PNG is verified non-blank: decode, assert `>1` distinct colour and `alpha > 0`
   for at least 1% of pixels. **This is the regression test for D1** and would have caught the
   28 empty files.
4. `results.json` is written alongside, so the sweep yields a table.

### 2.9 `scripts/run_sweep.py`

Replaces cells 17/18. Structure:

```python
for point in load_config("configs/sweep.json"):
    key = derive_key(ROOT_SEED, point)
    ds = make_dataset(key, **point)          # independent data per point  (D7)
    model, history = train(...)              # single stepper            (D5/D6)
    models.save(model, out_dir, metadata)    # always writes model.eqx   (D9)
    emu = rollout(model, ds.test_ic, steps)
    results = [run_diagnostic(d, gt, emu, ctx) for d in REGISTRY]
    for r in results: emit(r, out_dir)
    write_json(out_dir / "results.json", results)
```

`test/` is renamed to `runs/` so it stops shadowing `tests/` (pytest would currently try to
collect `test/`). **Fixes D10** by removing the `folder_name`-before-assignment path: `out_dir` is
computed at the top of the loop body.

### 2.10 Driver notebook

`Lorenze_63.ipynb` is rewritten to ~10 cells: install -> import `lorenz63` -> choose config ->
`run_sweep(...)` -> display `results.json` as a DataFrame -> display selected figures. **All
diagnostic code leaves the notebook.** This is what makes the three-copy drift impossible.

---

## 3. Removal list (explicit)

| Item | Action | Reason |
|---|---|---|
| `Lorenz_63.ipynb` | **Delete** | Stale duplicate; calls 8 `run_*_test` functions defined nowhere in it (D4); no unique code |
| `derivaitve-lor63.py` | **Delete** | Abandoned neural-ODE formulation; imported by nothing; its `neural_prediction_func`/`model_stepper_der` exist only here and in `functions.py` |
| `functions.py` -> `lyaponove_stepper` (L22-41) | **Delete** | Dead; uses `jnp` without importing it (D3); superseded by `diagnostics/lyapunov.py` |
| `functions.py` -> `neural_prediction_func`, `compute_loss`, `model_stepper_der` (L194-239) | **Delete** | Duplicate of `derivaitve-lor63.py`; unused |
| `functions.py` -> `method_A_step`, `method_C_step`, `gram_schmidt_qr` | **Rewrite** in `diagnostics/lyapunov.py` | Normalisation under review (D13) |
| `functions.py` (whole file) | **Convert** to `lorenz63/system.py` + `losses.py`, then delete | Fetched over HTTP at runtime and `exec`'d into globals, so an edit to `main` silently changes notebook behaviour (D3) |
| `test/*` (32 files) | **Delete**, regenerate into `runs/` | 28 are blank (D1); the 4 real ones are not reproducible from committed state (D9) |
| `run_traj_uniqueness_test` | **Replace** with `divergence.error_growth_rate` | Tests for chaos, not fidelity (D15) |
| `run_spatial_energy_test` thresholds (`50.0`, pass/warn) | **Remove** | Hand-picked scalar threshold (D16) |
| `SPEC_STEPS_A = 30_000` | **Use it** (or delete) | Currently dead (D13) |
| `print("TEST PASSED/WARNING/FAILED")` in all diagnostics | **Remove** | Moves to `report.py` control-relative comparison |
| `train_model`'s `stepper` parameter | **Remove** | Source of the 3-vs-6 crash (D5) |
| `train_model`'s `epochs=epochs` etc. no-op block | **Remove** | D19 |
| Hard-coded `target_lyapunov=0.9056` | **Move to config, default off** | D14 |
| `plt.ioff()` / `plt.ion()` / per-diagnostic `plt.show()` | **Remove** | Direct cause of blank artifacts (D1) |
| Hard-coded `bins_x = linspace(-25, 25, ...)` | **Remove** | D11 |
| `userdata.get('GH_TOKEN')` + push cells (29/30) | **Remove from notebook**, move to `scripts/publish.sh` | Secrets and git side effects do not belong in an analysis notebook; the token is also interpolated into a URL that lands in cell output |
| `# ! add functionality...` TODO | **Remove** | Implemented as `history.json` |

---

## 4. Open questions (need your call)

| # | Question | Default if unanswered |
|---|----------|-----------------------|
| Q1 | Does `functions.py` need to stay fetchable at runtime (e.g. you edit it on GitHub from Colab)? If yes the package must be installed from the repo instead, and I need to know how Colab gets it. | Package committed in-repo; notebook does `sys.path.insert` / `pip install -e .` |
| Q2 | FTLE regularizer: drop, variance-reduce, or keep-at-weight-0? (2.4) | Drop |
| Q3 | Keep the 4-point sweep exactly as-is, or widen it (e.g. add 0.2 and horizon 40)? | Keep as-is, so before/after numbers stay comparable |
| Q4 | Target environment: Colab T4 only, or also the `/lulab` H200 cluster mentioned in the scratch comment? Affects backend and CUDA pinning. | Colab T4 primary; CPU-runnable for tests |
| Q5 | Can I get a short end-to-end run on real data to regenerate artifacts, or should I deliver code + tests only? | Deliver code + a CPU smoke config; you regenerate on GPU |

---

## 5. Phasing and acceptance criteria

**Phase 1 — stop the bleeding (no science changes).**
Land the package skeleton, `report.py` figure contract, `train.py` single-stepper, RNG splits, and
the non-blank-PNG test. Delete per section 3.
*Accept:* `pytest tests/` green; a CPU smoke run (2 epochs, 200 steps) produces 8 non-blank PNGs
and a `results.json`; re-running produces identical metrics.

**Phase 2 — controls and quantification.**
Add `controls.py`, convert every diagnostic to `DiagnosticResult`, replace the uniqueness test
with error-growth-rate, fix Lyapunov normalisation, Hankel-DMD.
*Accept:* `results.json` reports `ratio` against `gt` and both controls for every metric; lobe-flip
ratio and lambda_1 ratio appear as numbers; the GT-GT control shows near-1.0 ratios.

**Phase 3 — regenerate and document.**
Run the real 4-point sweep, commit `runs/`, write the README with the results table.
*Accept:* committed artifacts reproduce from committed config + seed; README states the three
headline findings (lambda_1 13x low, lambda_3 3.6x negative, flip ratio 0.667) with control context.

---

## 6. Risks

- **Renaming `traing_data_set` breaks the notebook and `functions.py` simultaneously.** Mitigated by
  keeping a deprecated alias for one release (2.3).
- **`runs/` vs `test/` rename** is a breaking path change for anything reading `test/`. Low risk
  (nothing else reads it), but call it out in the PR.
- **Changing the Lyapunov normalisation will move the GT numbers away from the currently
  correct-looking 0.993/0/-14.519.** If a corrected implementation still yields those values, the
  old code was fine and D13 downgrades to a documentation fix. *This is the one item I could not
  validate numerically in this environment* — I would treat Phase 2's Lyapunov work as "reimplement
  from the textbook with a unit test against the analytic Lorenz values (0.906, 0, -14.57)" rather
  than "fix a known bug".
- **Full sweep runtime**: diagnostics currently re-roll trajectories 4x. The refactor computes one
  GT and one emulator rollout per sweep point, but the 5000-step x 40-perturbation uniqueness test
  and the O(N^2) correlation dimension dominate. Budget for a subsample parameter.
