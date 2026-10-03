"""Replicate aggregation: mean, spread, and what counts as a real difference."""

import json
import math

import numpy as np
import pytest

from lorenz63 import aggregate


def _write_replicate(root, point, rep, emu_lambda_1, gt_lambda_1=1.0):
    d = root / point / f"replicate{rep}"
    d.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": "lyapunov_spectrum",
            "gt": {"lambda_1": gt_lambda_1, "n_steps": 5000.0},
            "emu": {"lambda_1": emu_lambda_1, "n_steps": 5000.0},
            "control_gt_gt": {"lambda_1": gt_lambda_1},
            # The aggregator recomputes ratios from gt/emu; this block exists in
            # real results.json but is not read, so keep the fixture trivial.
            "ratios": ({"lambda_1": emu_lambda_1 / gt_lambda_1} if gt_lambda_1 else {}),
            "meta": {},
        }
    ]
    (d / "results.json").write_text(json.dumps(payload))
    return d


def test_summarise_point_computes_mean_and_spread(tmp_path):
    for rep, val in enumerate([1.0, 2.0, 3.0]):
        _write_replicate(tmp_path, "test_0.1_10", rep, val)
    runs = aggregate.load_replicates(tmp_path / "test_0.1_10")
    assert len(runs) == 3
    stats = aggregate.summarise_point(runs, field="emu")
    s = stats["lyapunov_spectrum"]["lambda_1"]
    assert s["mean"] == pytest.approx(2.0)
    assert s["std"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert s["min"] == 1.0 and s["max"] == 3.0
    assert s["n"] == 3
    # spread is the coefficient of variation: std / |mean|
    assert s["spread"] == pytest.approx(s["std"] / 2.0)


def test_summarise_handles_a_single_replicate(tmp_path):
    _write_replicate(tmp_path, "test_0.1_10", 0, 0.8)
    runs = aggregate.load_replicates(tmp_path / "test_0.1_10")
    stats = aggregate.summarise_point(runs, field="emu")
    s = stats["lyapunov_spectrum"]["lambda_1"]
    assert s["mean"] == pytest.approx(0.8)
    assert s["std"] == 0.0
    assert s["n"] == 1


def test_ratio_is_emulator_mean_over_ground_truth(tmp_path):
    """The ratio must divide the emulator *mean* by the GT value."""
    for rep, val in enumerate([0.5, 1.0, 1.5]):
        _write_replicate(tmp_path, "test_0.1_10", rep, val, gt_lambda_1=1.0)
    summary = aggregate.summarise_sweep(tmp_path)
    r = summary["test_0.1_10"]["ratio"]["lyapunov_spectrum"]["lambda_1"]
    assert r["mean"] == pytest.approx(1.0)  # mean(0.5,1.0,1.5) / 1.0
    assert r["std"] == pytest.approx(0.40824829, rel=1e-6)
    assert r["n"] == 3


def test_ratio_skips_a_zero_ground_truth(tmp_path):
    """A GT metric that is legitimately zero must not produce an infinite ratio."""
    for rep in range(2):
        _write_replicate(tmp_path, "test_0.1_10", rep, 0.7, gt_lambda_1=0.0)
    summary = aggregate.summarise_sweep(tmp_path)
    assert "lambda_1" not in summary["test_0.1_10"]["ratio"]["lyapunov_spectrum"]


def test_code_variation_flags_a_noisy_metric(tmp_path):
    """Large relative scatter must show up as a large spread."""
    for rep, val in enumerate([0.1, 1.0, 1.9]):
        _write_replicate(tmp_path, "test_0.1_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    s = summary["test_0.1_10"]["emu"]["lyapunov_spectrum"]["lambda_1"]
    assert s["spread"] > 0.5


def test_resolvable_distinguishes_real_from_noise(tmp_path):
    """Two points differing by much more than their SE must be resolvable."""
    # point A: tight cluster around 1.0
    for rep, val in enumerate([1.0, 1.01, 0.99]):
        _write_replicate(tmp_path, "test_0.05_10", rep, val)
    # point B: tight cluster around 0.5 (clearly different)
    for rep, val in enumerate([0.5, 0.51, 0.49]):
        _write_replicate(tmp_path, "test_0.4_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    verdict = aggregate.resolvable(
        summary, "test_0.05_10", "test_0.4_10", "lambda_1",
        diagnostic="lyapunov_spectrum",
    )
    assert verdict is not None
    assert verdict["resolvable"] is True
    assert verdict["difference"] == pytest.approx(0.5, abs=0.02)


def test_resolvable_is_false_when_spread_swamps_the_difference(tmp_path):
    """Overlapping clusters must not be reported as different."""
    for rep, val in enumerate([0.2, 1.0, 1.8]):
        _write_replicate(tmp_path, "test_0.05_10", rep, val)
    for rep, val in enumerate([0.25, 1.05, 1.75]):
        _write_replicate(tmp_path, "test_0.4_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    verdict = aggregate.resolvable(
        summary, "test_0.05_10", "test_0.4_10", "lambda_1",
        diagnostic="lyapunov_spectrum",
    )
    assert verdict["resolvable"] is False


def test_format_table_lists_points_and_hides_ground_truth_only_keys(tmp_path):
    for rep, val in enumerate([0.9, 1.1]):
        _write_replicate(tmp_path, "test_0.1_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    table = aggregate.format_table(summary)
    assert "test_0.1_10" in table
    assert "lambda_1" in table
    assert "n_steps" not in table  # GT-only bookkeeping


def test_summarise_sweep_ignores_directories_without_results(tmp_path):
    (tmp_path / "test_empty").mkdir()
    _write_replicate(tmp_path, "test_0.1_10", 0, 1.0)
    summary = aggregate.summarise_sweep(tmp_path)
    assert list(summary) == ["test_0.1_10"]


def test_derived_ratios_and_bookkeeping_are_excluded(tmp_path):
    """Derived ratios must not be re-aggregated as if they were metrics.

    ``lambda_1_ratio`` is already emu/gt, so reporting a mean of it alongside
    ``lambda_1`` double-counts and confuses the table.  Bookkeeping counts
    (``n_steps``) are identical across replicates and are not metrics either.
    """
    for rep, val in enumerate([0.9, 1.1]):
        _write_replicate(tmp_path, "test_0.1_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    emu_keys = set(summary["test_0.1_10"]["emu"]["lyapunov_spectrum"])
    ratio_keys = set(summary["test_0.1_10"]["ratio"]["lyapunov_spectrum"])
    for k in ("n_steps", "lambda_1_ratio"):
        assert k not in emu_keys, k
        assert k not in ratio_keys, k
    assert "lambda_1" in emu_keys


def test_table_reports_the_replicate_count(tmp_path):
    for rep, val in enumerate([0.9, 1.1, 0.95]):
        _write_replicate(tmp_path, "test_0.1_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    table = aggregate.format_table(summary)
    assert summary["test_0.1_10"]["n_replicates"] == 3
    # the n column must show 3 on the metric rows
    metric_row = next(line for line in table.splitlines() if "lambda_1 " in line)
    assert metric_row.split()[-1] == "3"


def _write_raw_replicate(root, point, rep, metrics):
    """Write a results.json with an arbitrary emu/gt metric dict."""
    d = root / point / f"replicate{rep}"
    d.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": "symbolic_dynamics",
            "gt": {k: 1.0 for k in metrics},
            "emu": dict(metrics),
            "ratios": {},
            "meta": {},
        }
    ]
    (d / "results.json").write_text(json.dumps(payload))
    return d


def test_non_finite_replicates_are_dropped_not_averaged(tmp_path):
    """A replicate with no lobe crossings reports infinite mean dwell.

    Averaging that in produced a summary of 6.2e12, which then dominated every
    pairwise comparison.  Non-finite replicates must be dropped and counted.
    """
    _write_raw_replicate(tmp_path, "test_0.1_20", 0, {"mean_dwell": float("inf")})
    _write_raw_replicate(tmp_path, "test_0.1_20", 1, {"mean_dwell": 2.0})
    _write_raw_replicate(tmp_path, "test_0.1_20", 2, {"mean_dwell": 4.0})
    summary = aggregate.summarise_sweep(tmp_path)
    s = summary["test_0.1_20"]["emu"]["symbolic_dynamics"]["mean_dwell"]
    assert s["mean"] == pytest.approx(3.0)
    assert s["n"] == 3
    assert s["n_used"] == 2


def test_a_metric_that_is_never_finite_is_omitted(tmp_path):
    for rep in range(2):
        _write_raw_replicate(tmp_path, "test_0.1_20", rep, {"mean_dwell": float("inf")})
    summary = aggregate.summarise_sweep(tmp_path)
    assert "mean_dwell" not in summary["test_0.1_20"]["emu"]["symbolic_dynamics"]


def test_resolvable_points_lists_every_pair(tmp_path):
    for rep, val in enumerate([1.0, 1.01, 0.99]):
        _write_replicate(tmp_path, "test_0.05_10", rep, val)
    for rep, val in enumerate([0.5, 0.51, 0.49]):
        _write_replicate(tmp_path, "test_0.4_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    rows = aggregate.resolvable_points(summary)
    assert len(rows) == 1  # two points -> one pair
    row = rows[0]
    assert row["metric"] == "lambda_1"
    assert row["diagnostic"] == "lyapunov_spectrum"
    assert row["resolvable"] is True
    # and the formatted table carries the verdict
    table = aggregate.format_resolvable(rows)
    assert "RESOLVABLE" in table
    assert "test_0.05_10" in table and "test_0.4_10" in table


def test_resolvable_points_is_empty_for_a_single_point(tmp_path):
    _write_replicate(tmp_path, "test_0.1_10", 0, 1.0)
    summary = aggregate.summarise_sweep(tmp_path)
    assert aggregate.resolvable_points(summary) == []


def test_write_summary_roundtrips(tmp_path):
    for rep, val in enumerate([0.9, 1.1]):
        _write_replicate(tmp_path, "test_0.1_10", rep, val)
    summary = aggregate.summarise_sweep(tmp_path)
    path = aggregate.write_summary(summary, tmp_path / "summary.json")
    assert json.loads(path.read_text()) == json.loads(json.dumps(summary, default=str))
