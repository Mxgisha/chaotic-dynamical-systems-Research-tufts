"""The sweep driver's post-processing helpers."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sweep import prune_duplicate_figures  # noqa: E402


def _make_point(root, point, replicates, n_figs=2, prefix="a"):
    for rep in range(replicates):
        d = root / point / f"replicate{rep}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text("[]")
        for i in range(n_figs):
            (d / f"fig{i}.png").write_bytes(b"x")


def test_prune_keeps_replicate_zero_and_drops_the_rest(tmp_path):
    _make_point(tmp_path, "test_0.1_10", replicates=3, n_figs=2)
    removed = prune_duplicate_figures(tmp_path, keep_replicate=0)
    assert removed == 4  # 2 figures x 2 dropped replicates
    assert sorted(p.name for p in (tmp_path / "test_0.1_10" / "replicate0").glob("*.png")) == [
        "fig0.png",
        "fig1.png",
    ]
    for rep in (1, 2):
        assert list((tmp_path / "test_0.1_10" / f"replicate{rep}").glob("*.png")) == []


def test_prune_preserves_every_results_json(tmp_path):
    """The numeric results must survive pruning -- they are the evidence."""
    _make_point(tmp_path, "test_0.1_10", replicates=3)
    prune_duplicate_figures(tmp_path)
    jsons = sorted(tmp_path.rglob("results.json"))
    assert len(jsons) == 3


def test_prune_handles_multiple_points(tmp_path):
    _make_point(tmp_path, "test_0.1_10", replicates=2)
    _make_point(tmp_path, "test_0.4_10", replicates=2)
    removed = prune_duplicate_figures(tmp_path)
    assert removed == 4  # 2 figures x 1 dropped replicate x 2 points


def test_prune_is_a_noop_for_a_single_replicate(tmp_path):
    _make_point(tmp_path, "test_0.1_10", replicates=1)
    assert prune_duplicate_figures(tmp_path) == 0
    assert len(list((tmp_path / "test_0.1_10" / "replicate0").glob("*.png"))) == 2


def test_prune_can_keep_a_different_replicate(tmp_path):
    _make_point(tmp_path, "test_0.1_10", replicates=3)
    prune_duplicate_figures(tmp_path, keep_replicate=2)
    assert list((tmp_path / "test_0.1_10" / "replicate2").glob("*.png"))
    assert not list((tmp_path / "test_0.1_10" / "replicate0").glob("*.png"))


def test_prune_ignores_unexpected_directory_names(tmp_path):
    """A stray directory must not crash the pruner."""
    (tmp_path / "test_0.1_10" / "replicateX").mkdir(parents=True)
    (tmp_path / "test_0.1_10" / "notes").mkdir(parents=True)
    _make_point(tmp_path, "test_0.2_10", replicates=2)
    # should not raise; the valid point is still pruned
    assert prune_duplicate_figures(tmp_path) == 2
