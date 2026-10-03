"""The single figure contract and results emission.

This module exists to make the blank-artifact failure mode impossible.

The original evaluation loop did::

    plt.ioff()
    run_lyapunov_spectrum_test(..., plot=True)   # calls plt.show() internally
    plt.savefig(os.path.join(folder_name, "lyapunov_spectrum.png"))
    plt.close()

28 of the 32 committed PNGs are fully transparent 640x480 canvases, i.e. nothing
was drawn.  Two structural rules prevent a recurrence:

1. **Diagnostics never own a figure.**  They are pure measurement functions
   (:mod:`lorenz63.diagnostics`).  Rendering happens only here.
2. **``savefig`` happens on a figure this module created**, before anything is
   shown or closed, and every written file is verified to be non-blank.

Diagnostics emit their own state via the keys they attach to ``ctx``/
``DiagnosticResult``; the renderers below read those.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# The true divergence of Lorenz-63 -- a constant, which is what makes the
# local-contraction diagnostic interpretable.
from lorenz63.system import DIVERGENCE as S_DIVERGENCE

# Diagnostics may attach array-valued plotting state under keys starting with
# this prefix; they are stripped before JSON serialisation.
PLOT_PREFIX = "_"

# Diagnostic keys that are bookkeeping rather than measurements, or whose
# "ratio" is meaningless (a convergence slope or a fixed slice coordinate).
NON_RATIO_KEYS = frozenset(
    {
        "n_points",
        "n_steps",
        "n_modes",
        "n_scales_used",
        "n_maxima",
        "n_flips",
        "fixed_val",
        "convergence_slope_lambda_1",
        "lambda_1_ratio",
        "d2_ratio",
        "growth_rate_ratio",
        "flip_ratio",
        "l1_mean_vs_gtgt_control",
        "l1_excess_over_control",
        "l1_mean_vs_identity",
        "identity_n_unique_endpoints",
        "flip_ratio_vs_gtgt_control",
        "return_map_r",
        "contraction_ratio",
    }
)

# Values smaller than this are treated as zero, so `emu / gt` is not reported as
# a spectacular ratio when the denominator merely sits at numerical noise.
RATIO_DENOM_EPS = 1e-9


@dataclass
class DiagnosticResult:
    """A single diagnostic's outcome for one sweep point."""

    name: str
    gt: dict
    emu: dict
    control_gt_gt: dict | None = None
    control_identity: dict | None = None
    meta: dict = field(default_factory=dict)

    def ratios(self) -> dict:
        """``emu[k] / gt[k]`` for every metric where a ratio is meaningful.

        Keys in :data:`NON_RATIO_KEYS` are excluded (bookkeeping counts, derived
        ratios, and quantities that can legitimately be zero, where a ratio would
        be meaningless or unbounded), as are pairs whose ground-truth value is
        numerically zero.
        """
        out = {}
        for k, v in self.emu.items():
            if k.startswith(PLOT_PREFIX) or k in NON_RATIO_KEYS:
                continue
            g = self.gt.get(k)
            if not isinstance(v, (int, float)) or not isinstance(g, (int, float)):
                continue
            if abs(float(g)) < RATIO_DENOM_EPS:
                continue
            if np.isfinite(v) and np.isfinite(g):
                out[k] = float(v) / float(g)
        return out

    def scalars(self) -> dict:
        """JSON-safe view: drops array-valued plot state."""
        def clean(d):
            if d is None:
                return None
            return {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in d.items()
                if not k.startswith(PLOT_PREFIX)
            }

        return {
            "name": self.name,
            "gt": clean(self.gt),
            "emu": clean(self.emu),
            "control_gt_gt": clean(self.control_gt_gt),
            "control_identity": clean(self.control_identity),
            "ratios": self.ratios(),
            "meta": self.meta,
        }


def is_blank_png(path, min_distinct_colors=2, min_ink_fraction=1e-3) -> bool:
    """Return True if a PNG contains no visible content.

    Decodes the IHDR and IDAT chunks directly (no image library needed).  A file
    counts as blank when it is *essentially uniform*: fewer than
    ``min_distinct_colors`` distinct pixels, or one colour covering all but
    ``min_ink_fraction`` of the canvas.

    This is the regression test for the 28 empty artifacts.  An earlier version
    of this function required a fraction of non-transparent pixels, which is
    wrong for the normal case: matplotlib saves an *opaque* white background
    behind a sparse plot, so a legitimate one-line figure can have under 1% of
    pixels that are neither transparent nor background coloured.
    """
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return True
    width, height = struct.unpack(">II", data[16:24])
    bit_depth, color_type = data[24], data[25]
    if bit_depth != 8 or color_type not in (2, 6):
        # Only 8-bit RGB/RGBA supported; cannot verify, so do not claim blank.
        return False
    bpp = 4 if color_type == 6 else 3
    idat = b""
    i = 8
    while i < len(data):
        length = struct.unpack(">I", data[i : i + 4])[0]
        ctype = data[i + 4 : i + 8]
        if ctype == b"IDAT":
            idat += data[i + 8 : i + 8 + length]
        i += 12 + length
    raw = zlib.decompress(idat)
    stride = 1 + width * bpp
    counts: dict[bytes, int] = {}
    total = 0
    for y in range(height):
        row = raw[y * stride + 1 : y * stride + 1 + width * bpp]
        for x in range(width):
            px = bytes(row[x * bpp : (x + 1) * bpp])
            counts[px] = counts.get(px, 0) + 1
            total += 1
    if len(counts) < min_distinct_colors:
        return True
    dominant = max(counts.values())
    return (1.0 - dominant / max(total, 1)) < min_ink_fraction


# --- renderers -------------------------------------------------------------
def _ax(fig, *args):
    """Create a subplot and remember it, so legends are only drawn where there
    is actually something labelled."""
    ax = fig.add_subplot(*args)
    fig._dsh_axes = list(getattr(fig, "_dsh_axes", [])) + [ax]
    return ax


def _add_legends(fig):
    for ax in getattr(fig, "_dsh_axes", []):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()


def _render_contraction(result, fig):
    """Four panels: the contraction distribution and where the error lives.

    ``|x|`` and ``z`` are different quantities with different ranges, so they get
    their own panels rather than sharing an axis.
    """
    e = result.emu
    g = result.gt
    gt_rate = g.get(f"{PLOT_PREFIX}rate")
    emu_rate = e.get(f"{PLOT_PREFIX}rate")
    x_edges = e.get(f"{PLOT_PREFIX}x_edges")
    x_excess = e.get(f"{PLOT_PREFIX}x_excess")
    z_edges = e.get(f"{PLOT_PREFIX}z_edges")
    z_excess = e.get(f"{PLOT_PREFIX}z_excess")
    gt_exp = g.get(f"{PLOT_PREFIX}expansion")

    ax = _ax(fig, 2, 2, 1)
    if gt_rate is not None and emu_rate is not None:
        lo = float(min(gt_rate.min(), emu_rate.min()))
        hi = float(max(gt_rate.max(), emu_rate.max()))
        ax.hist(gt_rate, bins=np.linspace(lo, hi, 40), alpha=0.6,
                label="true (constant)", color="#1f77b4")
        ax.hist(emu_rate, bins=np.linspace(lo, hi, 40), alpha=0.6,
                label="emulator", color="#d62728")
        ax.axvline(S_DIVERGENCE, color="k", ls="--", lw=1, label="true divergence")
        ax.set_yscale("log")
    ax.set_xlabel("local contraction rate")
    ax.set_ylabel("count")
    ax.set_title("Contraction is spread out, not constant")
    if gt_rate is not None:
        ax.legend(fontsize=7)

    ax2 = _ax(fig, 2, 2, 2)
    if x_edges is not None and x_excess is not None:
        mids = 0.5 * (np.asarray(x_edges[:-1]) + np.asarray(x_edges[1:]))
        ax2.plot(mids, x_excess, "o-", color="#d62728")
        ax2.set_xticks(mids)
        ax2.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in zip(x_edges[:-1], x_edges[1:])],
                            rotation=45, fontsize=7)
    ax2.set_xlabel("$|x|$ (distance from the saddle)")
    ax2.set_ylabel("excess contraction")
    ax2.set_title("Worst in the mid-lobes, best at the saddle")

    ax3 = _ax(fig, 2, 2, 3)
    if z_edges is not None and z_excess is not None:
        mids = 0.5 * (np.asarray(z_edges[:-1]) + np.asarray(z_edges[1:]))
        ax3.plot(mids, z_excess, "s--", color="#1f77b4")
        ax3.set_xticks(mids)
        ax3.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in zip(z_edges[:-1], z_edges[1:])],
                            rotation=45, fontsize=7)
    ax3.set_xlabel("$z$")
    ax3.set_ylabel("excess contraction")
    ax3.set_title("Monotonic in altitude")

    ax4 = _ax(fig, 2, 2, 4)
    keys = ["contraction_rate_mean", "expansion_rate_mean"]
    x = np.arange(len(keys))
    ax4.bar(x - 0.2, [g.get(k, np.nan) for k in keys], width=0.4,
            label="true", color="#1f77b4")
    ax4.bar(x + 0.2, [e.get(k, np.nan) for k in keys], width=0.4,
            label="emulator", color="#d62728")
    ax4.axhline(S_DIVERGENCE, color="k", ls="--", lw=1, label="true divergence")
    ax4.set_xticks(x)
    ax4.set_xticklabels(["contraction", "expansion"])
    ax4.set_title(f"Deficit {e.get('contraction_ratio', float('nan')):.2f}x"
                  f"  (spread {e.get('contraction_ratio_std', float('nan')):.2f})")
    ax4.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _render_lyapunov(result, fig):
    # NB: mathtext supports \lambda but not \lambda_1, so the labels are built as
    # "$\\lambda_{1}$" -- writing "$\\lambda_1$" renders as literal text.
    labels = [r"$\lambda_{1}$", r"$\lambda_{2}$", r"$\lambda_{3}$"]
    ax = _ax(fig, 1, 2, 1)
    g = result.gt
    names = ["lambda_1", "lambda_2", "lambda_3"]
    vals = [g.get(n, np.nan) for n in names]
    ax.bar(range(3), vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.axhline(0, color="k", ls=":", alpha=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_title("Ground truth spectrum")
    ax.set_ylabel("Exponent")

    ax2 = _ax(fig, 1, 2, 2)
    e = result.emu
    vals_e = [e.get(n, np.nan) for n in names]
    ax2.bar(range(3), vals_e, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.7)
    ax2.axhline(0, color="k", ls=":", alpha=0.5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ratio = e.get("lambda_1_ratio", float("nan"))
    ax2.set_title(f"Emulator spectrum (ratio {ratio:.2f})")
    fig.suptitle("Lyapunov spectrum")
    return fig


def _render_scalar_bars(result, fig, keys, title):
    ax = _ax(fig, 1, 1, 1)
    x = np.arange(len(keys))
    g = [result.gt.get(k, np.nan) for k in keys]
    e = [result.emu.get(k, np.nan) for k in keys]
    ax.bar(x - 0.2, g, width=0.4, label="ground truth", color="#1f77b4")
    ax.bar(x + 0.2, e, width=0.4, label="emulator", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_title(title)
    return fig


def _render_curve(result, fig, key, title):
    ax = _ax(fig, 1, 1, 1)
    gx = result.gt.get(f"{PLOT_PREFIX}{key}_x")
    gy = result.gt.get(f"{PLOT_PREFIX}{key}")
    ex = result.emu.get(f"{PLOT_PREFIX}{key}_x")
    ey = result.emu.get(f"{PLOT_PREFIX}{key}")
    if gy is not None:
        ax.plot(gx if gx is not None else np.arange(len(gy)), gy, label="ground truth", color="#1f77b4")
    if ey is not None:
        ax.plot(ex if ex is not None else np.arange(len(ey)), ey, label="emulator", color="#d62728", ls="--")
    ax.set_title(title)
    return fig


def _render_correlation(result, fig):
    ax = _ax(fig, 1, 1, 1)
    r = result.emu.get(f"{PLOT_PREFIX}radii")
    c = result.emu.get(f"{PLOT_PREFIX}corr_emu")
    # dimension.py emits _corr_emu / _corr_gt (already prefixed).
    if r is not None and c is not None:
        ax.loglog(r, c, "x--", label="emulator")
    r2 = result.gt.get(f"{PLOT_PREFIX}radii")
    c2 = result.gt.get(f"{PLOT_PREFIX}corr_gt")
    if r2 is not None and c2 is not None:
        ax.loglog(r2, c2, "o-", label="ground truth")
    ax.set_xlabel("r")
    ax.set_ylabel("C(r)")
    ax.set_title(f"Correlation integral (D2 ratio {result.emu.get('d2_ratio', float('nan')):.2f})")
    return fig


def _render_return_map(result, fig):
    ax = _ax(fig, 1, 2, 1)
    for key, color, label in (
        (f"{PLOT_PREFIX}maxima", "#1f77b4", "ground truth"),
    ):
        m = result.gt.get(key)
        if m is not None and len(m) > 1:
            ax.scatter(m[:-1], m[1:], s=12, alpha=0.6, color=color, label=label)
    m = result.emu.get(f"{PLOT_PREFIX}maxima")
    if m is not None and len(m) > 1:
        ax.scatter(m[:-1], m[1:], s=12, alpha=0.6, color="#d62728", marker="x", label="emulator")
    ax.set_xlabel("$Z_n$")
    ax.set_ylabel("$Z_{n+1}$")
    ax.set_title("Successive maxima return map")

    ax2 = _ax(fig, 1, 2, 2)
    gc = result.gt.get(f"{PLOT_PREFIX}cum_flips")
    ec = result.emu.get(f"{PLOT_PREFIX}cum_flips")
    if gc is not None:
        ax2.plot(gc, color="#1f77b4", label="ground truth")
    if ec is not None:
        ax2.plot(ec, color="#d62728", ls="--", label=f"emulator (ratio {result.emu.get('flip_ratio', float('nan')):.2f})")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cumulative lobe flips")
    ax2.set_title("Symbolic dynamics")
    return fig


def _render_divergence(result, fig):
    ax = _ax(fig, 1, 2, 1)
    for label, d in (("ground truth", result.gt), ("emulator", result.emu)):
        seps = d.get(f"{PLOT_PREFIX}seps")
        if isinstance(seps, dict):
            for eps, s in list(seps.items())[:2]:
                ax.semilogy(s, label=f"{label} eps={eps:g}")
    ax.set_xlabel("step")
    ax.set_ylabel("separation")
    ax.set_title("Twin-trajectory divergence")

    ax2 = _ax(fig, 1, 2, 2)
    keys = ["growth_rate", "t_double", "t_sat"]
    x = np.arange(len(keys))
    ax2.bar(x - 0.2, [result.gt.get(k, np.nan) for k in keys], width=0.4, label="GT", color="#1f77b4")
    ax2.bar(x + 0.2, [result.emu.get(k, np.nan) for k in keys], width=0.4, label="emu", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, rotation=15)
    ax2.set_title("Error-growth summary")
    return fig


def _render_measure(result, fig):
    fig = _render_scalar_bars(
        result, fig, ["l1_mean", "w1_mean", "l1_x", "l1_y", "l1_z"],
        "Invariant-measure mismatch (lower is better)",
    )
    return fig


def _render_dmd(result, fig):
    ax = _ax(fig, 1, 1, 1)
    g = result.gt.get(f"{PLOT_PREFIX}mags")
    e = result.emu.get(f"{PLOT_PREFIX}mags")
    if g is not None:
        ax.semilogy(g, "o-", label="ground truth", color="#1f77b4")
    if e is not None:
        ax.semilogy(e, "x--", label="emulator", color="#d62728")
    ax.set_xlabel("mode index")
    ax.set_ylabel("|eigenvalue|")
    ax.set_title(f"Hankel-DMD spectrum (corr {result.emu.get('spectrum_corr', float('nan')):.3f})")
    return fig


def _render_field(result, fig):
    U = result.emu.get(f"{PLOT_PREFIX}U")
    V = result.emu.get(f"{PLOT_PREFIX}V")
    gt_vel = result.emu.get(f"{PLOT_PREFIX}gt_vel")
    emu_vel = result.emu.get(f"{PLOT_PREFIX}emu_vel")
    err = result.emu.get(f"{PLOT_PREFIX}err")
    labels = result.emu.get(f"{PLOT_PREFIX}labels", ("X", "Y"))
    if U is None or err is None:
        return fig
    ax = _ax(fig, 1, 2, 1)
    if gt_vel is not None and emu_vel is not None:
        ax.quiver(U, V, gt_vel[:, 0].reshape(U.shape), gt_vel[:, 1].reshape(U.shape),
                  color="black", alpha=0.6, label="GT")
        ax.quiver(U, V, emu_vel[:, 0].reshape(U.shape), emu_vel[:, 1].reshape(U.shape),
                  color="crimson", alpha=0.6, label="emulator")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("Vector field slice")
    ax2 = _ax(fig, 1, 2, 2)
    im = ax2.contourf(U, V, err, levels=20, cmap="inferno")
    fig.colorbar(im, ax=ax2, label="||f_gt - f_emu||")
    ax2.set_title(f"mean error {result.emu.get('mean_error', float('nan')):.2f}")
    return fig


RENDERERS = {
    "lyapunov_spectrum": _render_lyapunov,
    "local_contraction": _render_contraction,
    "invariant_measure": _render_measure,
    "correlation_dimension": _render_correlation,
    "symbolic_dynamics": _render_return_map,
    "error_growth": _render_divergence,
    "dmd_spectrum": _render_dmd,
    "vector_field_slice": _render_field,
}


def render(result: DiagnosticResult):
    """Build the figure for a diagnostic.  Only this module creates figures."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))
    fn = RENDERERS.get(result.name)
    if fn is None:
        _ax(fig, 1, 1, 1).text(0.5, 0.5, f"no renderer for {result.name}", ha="center")
        return fig
    fig = fn(result, fig)
    _add_legends(fig)
    return fig


def emit(result: DiagnosticResult, out_dir, dpi=150, verify=True):
    """Render, **save**, close, and verify one diagnostic figure.

    ``savefig`` is called on a figure this function created, before any
    ``show()``, and the written file is checked for actual content.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = render(result)
    path = out_dir / f"{result.name}.png"
    # bbox_inches='tight' trims the blank margin; facecolor keeps it opaque so the
    # non-blank check has something to see.
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if verify and is_blank_png(path):
        raise RuntimeError(
            f"{path} was written blank; the figure contract has been violated"
        )
    return path


def write_results(results, out_dir, filename="results.json"):
    """Write the scalar view of every result plus a cross-point table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [r.scalars() for r in results]
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def results_table(all_results: dict) -> str:
    """Render a plain-text table of emulator/ground-truth ratios."""
    lines = []
    header = f"{'sweep point':22s} {'diagnostic':24s} {'metric':24s} {'ratio':>9s}"
    lines.append(header)
    lines.append("-" * len(header))
    for point, results in all_results.items():
        for r in results:
            for k, v in sorted(r.ratios().items()):
                lines.append(f"{point:22s} {r.name:24s} {k:24s} {v:9.3f}")
    return "\n".join(lines)


__all__ = [
    "DiagnosticResult",
    "render",
    "emit",
    "write_results",
    "results_table",
    "is_blank_png",
]
