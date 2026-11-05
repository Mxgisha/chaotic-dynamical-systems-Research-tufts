"""End-to-end: the sweep driver produces non-blank artifacts and a results table."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_smoke_sweep_produces_non_blank_artifacts(tmp_path):
    """The full pipeline, small config, with seed replicates.

    Asserts the exact thing that was broken: every emitted PNG must contain
    visible pixels, and results.json must exist with ratios.  Also checks that
    replicates are laid out separately and that the aggregate summary is written.
    """
    out = tmp_path / "runs"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_sweep.py"),
         "--smoke", "--replicates", "2", "--out", str(out)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=3600,
    )
    if proc.returncode != 0:
        pytest.fail(f"run_sweep failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    from lorenz63.report import is_blank_png

    pngs = sorted(out.rglob("*.png"))
    assert pngs, "no figures were written"
    blanks = [str(p) for p in pngs if is_blank_png(p)]
    assert blanks == [], f"blank artifacts: {blanks}"

    results = sorted(out.rglob("results.json"))
    assert results, "no results.json written"
    payload = json.loads(results[0].read_text())
    names = {r["name"] for r in payload}
    assert "invariant_measure" in names
    assert any(r["ratios"] for r in payload), "no ratios computed"

    # the model must be persisted, otherwise evaluation could silently skip
    assert list(out.rglob("model.npz")), "model was not saved"
    assert list(out.rglob("metadata.json")), "metadata was not saved"

    # replicates: one directory each, plus an aggregate summary
    rep_dirs = sorted(out.glob("test_*/replicate*"))
    assert len(rep_dirs) == 2, [str(d) for d in rep_dirs]
    summary_path = out / "summary.json"
    assert summary_path.exists(), "no replicate summary written"
    summary = json.loads(summary_path.read_text())
    point = next(iter(summary))
    assert summary[point]["n_replicates"] == 2
    lam = summary[point]["ratio"]["lyapunov_spectrum"]["lambda_1"]
    assert lam["n"] == 2
    assert lam["std"] >= 0.0
    assert (out / "summary_table.txt").exists()


@pytest.mark.slow
def test_cli_replicate_override_beats_the_config(tmp_path):
    """``--replicates 1`` must win over the config, and still use replicate0/.

    A one-replicate run must keep the same on-disk shape as a sweep, otherwise
    every downstream path would need two code paths.  The smoke config itself
    sets ``replicates: 2``, so this also checks the CLI override.
    """
    out = tmp_path / "runs"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_sweep.py"),
         "--smoke", "--replicates", "1", "--out", str(out),
         "--only", "invariant_measure"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=3600,
    )
    if proc.returncode != 0:
        pytest.fail(f"run_sweep failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert (out / "test_0.05_10" / "replicate0" / "results.json").exists()
    # exactly one replicate directory: the CLI override beat the smoke config's 2
    assert len(sorted(out.glob("test_*/replicate*"))) == 1
    # with a single replicate there is nothing to aggregate, so no summary
    assert not (out / "summary.json").exists()
