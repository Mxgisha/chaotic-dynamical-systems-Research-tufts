"""End-to-end: the sweep driver produces non-blank artifacts and a results table."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_smoke_sweep_produces_non_blank_artifacts(tmp_path):
    """The full pipeline, small config.

    Asserts the exact thing that was broken: every emitted PNG must contain
    visible pixels, and a results.json must exist with ratios.
    """
    out = tmp_path / "runs"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_sweep.py"),
         "--smoke", "--out", str(out)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=1800,
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
