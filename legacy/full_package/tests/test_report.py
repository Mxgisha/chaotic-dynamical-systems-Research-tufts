"""The report layer and the figure contract -- the regression test for D1."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from lorenz63.report import DiagnosticResult, emit, is_blank_png, render, write_results

matplotlib = pytest.importorskip("matplotlib")


def _toy_result(name="invariant_measure"):
    return DiagnosticResult(
        name=name,
        # a non-zero GT reference so ratios are well defined
        gt={"l1_mean": 0.20, "w1_mean": 0.18, "l1_x": 0.2, "l1_y": 0.25, "l1_z": 0.15},
        emu={"l1_mean": 0.42, "w1_mean": 0.31, "l1_x": 0.4, "l1_y": 0.5, "l1_z": 0.36},
        control_gt_gt={"l1_mean": 0.02},
    )


def test_is_blank_png_flags_an_empty_canvas(tmp_path):
    """A fully transparent canvas -- exactly what 28 committed files are.

    Mirrors the committed artifacts: 640x480 RGBA, one transparent pixel, no
    visible content at all.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.4, 4.8))
    fig.patch.set_alpha(0.0)
    p = tmp_path / "blank.png"
    fig.savefig(p, transparent=True)
    plt.close(fig)
    plt.close("all")
    assert p.exists()
    assert is_blank_png(p) is True


def test_is_blank_png_accepts_real_content(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    p = tmp_path / "real.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    plt.close("all")
    assert is_blank_png(p) is False


def test_emit_writes_a_non_blank_figure(tmp_path):
    """The contract: emit saves before showing and verifies content."""
    result = _toy_result()
    path = emit(result, tmp_path)
    assert path.exists()
    assert Path(path).stat().st_size > 1000
    assert is_blank_png(path) is False


def test_emit_raises_on_a_blank_figure(tmp_path, monkeypatch):
    """If a renderer ever produces nothing, emit must not stay silent."""
    import lorenz63.report as report
    import matplotlib.pyplot as plt

    def blank_renderer(result, fig):
        # An untouched figure with a transparent canvas and no artists.
        fig.patch.set_alpha(0.0)
        return fig

    monkeypatch.setitem(report.RENDERERS, "toy", blank_renderer)
    result = DiagnosticResult("toy", {"a": 1.0}, {"a": 2.0})
    with pytest.raises(RuntimeError, match="blank"):
        emit(result, tmp_path)
    plt.close("all")


def test_no_diagnostic_module_imports_matplotlib():
    """Diagnostics must be pure: they may not own figures.

    This is what makes the blank-artifact failure mode structurally impossible
    rather than merely fixed.  Docstrings are stripped before scanning so a
    module may *mention* matplotlib without importing it.
    """
    import ast
    import lorenz63.diagnostics as diag

    pkg = Path(diag.__file__).parent
    offenders = []
    for py in sorted(pkg.glob("*.py")):
        tree = ast.parse(py.read_text())
        # Drop every docstring so prose is not mistaken for an import.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    node.body = body[1:] or [ast.Pass()]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any("matplotlib" in (n or "") for n in names):
                offenders.append(py.name)
                break
    assert offenders == [], f"diagnostics importing plotting: {offenders}"


def test_ratios_are_computed_and_plot_state_is_stripped():
    result = DiagnosticResult(
        name="x",
        gt={"a": 2.0, "b": 0.0, "_hidden": np.zeros(3)},
        emu={"a": 1.0, "b": 0.0, "_hidden": np.ones(3)},
    )
    r = result.ratios()
    assert r["a"] == pytest.approx(0.5)
    assert "b" not in r  # divide by zero is excluded, not inf
    s = result.scalars()
    assert "_hidden" not in s["gt"] and "_hidden" not in s["emu"]


def test_write_results_roundtrips(tmp_path):
    import json

    write_results([_toy_result()], tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload[0]["name"] == "invariant_measure"
    assert payload[0]["ratios"]["l1_mean"] > 0


def test_render_smoke_for_every_registered_diagnostic():
    """Every registered diagnostic that has a renderer must render headlessly."""
    from lorenz63.diagnostics import available
    from lorenz63.report import RENDERERS
    import matplotlib.pyplot as plt

    for name in available():
        if name not in RENDERERS:
            continue
        result = DiagnosticResult(
            name=name,
            gt={"l1_mean": 1.0, "d2": 2.0, "lambda_1": 0.9, "flip_ratio": 1.0},
            emu={"l1_mean": 1.1, "d2": 2.1, "lambda_1": 0.5, "flip_ratio": 0.6},
        )
        fig = render(result)
        assert fig is not None
        plt.close(fig)
