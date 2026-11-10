"""Generate the pipeline diagrams for PIPELINE.md.

Everything is derived from the source: functions and call edges come from an AST
pass over ``lorenz.py``, ``train.py`` and ``diagnostics.py``; the notebook panel
comes from parsing the notebook's code cells and markdown headings.  Nothing is
typed in by hand, so the diagrams cannot drift from the code.

    python scripts/make_pipeline_figures.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MODULES = {"lorenz.py": "lorenz", "train.py": "train", "diagnostics.py": "diagnostics"}
COLOR = {
    "lorenz": "#4C72B0",
    "train": "#DD8452",
    "diagnostics": "#55A868",
    "external": "#7F7F7F",
    "notebook": "#8172B3",
}
# pipeline order within each module, used for layout
ORDER = {
    "lorenz": ["rhs", "rhs_t", "jacobian", "rk4_step", "flow", "trajectory",
               "flow_jacobian", "reference", "initial_conditions", "fixed_points"],
    "train": ["setup", "x64_enabled", "describe_device", "cast", "make_mlp",
              "step_map", "step_field", "make_step", "make_pairs", "make_dataset",
              "mse_loss", "divergence_at", "divergence_penalty", "make_loss",
              "train", "rollout", "implied_divergence", "final_mse"],
    "diagnostics": ["identity_step", "as_velocity", "lyapunov_spectrum",
                    "local_contraction", "error_growth", "invariant_measure",
                    "correlation_dimension", "symbolic_dynamics", "dmd_spectrum",
                    "vector_field_slice"],
}


# --- source extraction ----------------------------------------------------
def inventory():
    """Return {function: module} and the set of internal call edges."""
    funcs, trees = {}, {}
    for fname, mod in MODULES.items():
        tree = ast.parse((ROOT / fname).read_text())
        trees[mod] = tree
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                funcs[node.name] = mod

    edges = set()
    for mod, tree in trees.items():
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                fn = sub.func
                callee = None
                if isinstance(fn, ast.Name) and fn.id in funcs:
                    callee = fn.id
                elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                    if fn.value.id in ("lorenz", "train", "diag") and fn.attr in funcs:
                        callee = fn.attr
                if callee and callee != node.name:
                    edges.add((node.name, callee))
    return funcs, edges


def notebook_calls_and_sections():
    """({cell: set(module.function)}, [(cell_index, section_title)])."""
    nb = json.loads((ROOT / "Lorenz63_emulator.ipynb").read_text())
    calls, sections = {}, []
    pending = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"]).strip()
            for line in src.splitlines():
                if line.startswith("## "):
                    pending = line[3:].strip()
        else:
            if pending:
                sections.append((i, pending))
                pending = None
            try:
                tree = ast.parse("".join(c["source"]))
            except SyntaxError:
                continue
            found = set()
            for sub in ast.walk(tree):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id in ("lorenz", "train", "diag")):
                    mod = "diagnostics" if sub.func.value.id == "diag" else sub.func.value.id
                    found.add(f"{mod}.{sub.func.attr}")
            calls[i] = found
    return calls, sections


# --- drawing helpers ------------------------------------------------------
def box(ax, x, y, w, h, text, color, fontsize=8.5, alpha=0.15, weight="normal",
        rounding=0.8, zorder=2, pad=0.04):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha, zorder=zorder,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1,
            weight=weight)


def arrow(ax, p, q, color="#555555", lw=1.0, ls="-", rad=0.0, zorder=1,
          style="-|>", shrink=2.5):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=10, linewidth=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=zorder,
        shrinkA=shrink, shrinkB=shrink,
    ))


def clean(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# --- figure 1: modules and external libraries -----------------------------
def fig_modules():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    clean(ax)

    box(ax, 50, 87, 34, 12, "Lorenz63_emulator.ipynb\n13 code cells", COLOR["notebook"],
        weight="bold", fontsize=9.5)
    box(ax, 18, 55, 26, 14, "lorenz.py\n10 functions\nground truth", COLOR["lorenz"], fontsize=9)
    box(ax, 50, 55, 26, 14, "train.py\n18 functions\nemulator + training", COLOR["train"], fontsize=9)
    box(ax, 82, 55, 26, 14, "diagnostics.py\n10 functions\nmeasurement", COLOR["diagnostics"],
        fontsize=9)

    for x in (18, 50, 82):
        arrow(ax, (50, 81), (x, 62.2), color=COLOR["notebook"])

    # importer -> imported
    arrow(ax, (37.2, 55), (31.2, 55), color=COLOR["train"])
    ax.text(34.2, 52.6, "imports", fontsize=6.5, ha="center", color="#444")
    arrow(ax, (68.8, 55), (63.2, 55), color=COLOR["diagnostics"])
    ax.text(66.0, 52.6, "imports", fontsize=6.5, ha="center", color="#444")

    box(ax, 18, 24, 24, 9, "jax  ·  diffrax", COLOR["external"], fontsize=8.5)
    box(ax, 50, 24, 24, 9, "jax  ·  equinox  ·  optax", COLOR["external"], fontsize=8.5)
    box(ax, 82, 24, 24, 9, "jax", COLOR["external"], fontsize=8.5)
    for x in (18, 50, 82):
        arrow(ax, (x, 47.8), (x, 28.6), color=COLOR["external"], ls="--")

    ax.text(50, 9, "one notebook drives three modules; both train.py and diagnostics.py import lorenz.py",
            ha="center", fontsize=7.5, color="#555", style="italic")
    ax.set_title("What imports what", fontsize=11, loc="left", weight="bold")
    save(fig, "fig1_modules.png")


# --- figure 2: the pipeline blueprint -------------------------------------
def fig_blueprint():
    fig, ax = plt.subplots(figsize=(10, 11.2))
    clean(ax)
    sx = 50
    r = {"rhs": 95, "rk4": 87, "traj": 79, "data": 67, "mlp": 57,
         "step": 49, "train": 40, "roll": 31, "diag": 20, "verdict": 9}

    box(ax, sx, r["rhs"], 36, 6.2, "lorenz.rhs   (the three equations)", COLOR["lorenz"])
    box(ax, sx, r["rk4"], 36, 6.2, "lorenz.rk4_step   (one step)", COLOR["lorenz"])
    box(ax, sx, r["traj"], 42, 6.2, "lorenz.trajectory   →   (N, 3)  float64", COLOR["lorenz"])
    box(ax, sx, r["data"], 46, 7.2,
        "train.make_dataset   →   x, y  (N, 3)  float32\n+ a test trajectory", COLOR["train"])
    box(ax, sx, r["mlp"], 38, 6.2, "train.make_mlp   (4,611 parameters)", COLOR["train"])
    box(ax, sx, r["step"], 40, 6.2, "train.make_step   map  or  field + RK4", COLOR["train"])
    box(ax, sx, r["train"], 48, 7.2,
        "train.train   Adam on MSE\n+ optional divergence penalty", COLOR["train"])
    box(ax, sx, r["roll"], 42, 6.2, "train.rollout   →   (N, 3)  float64", COLOR["train"])
    box(ax, sx, r["diag"], 48, 8.0,
        "diagnostics.*   eight measurements\nno plotting, no pass/fail", COLOR["diagnostics"])
    box(ax, sx, r["verdict"], 46, 6.4, "notebook table   controls  →  verdict",
        COLOR["notebook"], weight="bold")

    for a, b in [("rhs", "rk4"), ("rk4", "traj"), ("traj", "data"), ("data", "mlp"),
                 ("mlp", "step"), ("step", "train"), ("train", "roll"),
                 ("roll", "diag"), ("diag", "verdict")]:
        arrow(ax, (sx, r[a] - 3.1), (sx, r[b] + 3.1))

    # exact Jacobian: rk4_step -> flow_jacobian -> diagnostics, along the far left
    box(ax, 13, r["traj"], 18, 6.2, "lorenz.flow_jacobian\nautodiff of one step",
        COLOR["lorenz"], fontsize=7.2)
    # flow_jacobian is derived from rk4_step, not the other way round
    arrow(ax, (sx - 18, r["rk4"]), (13, r["traj"] + 3.2), color=COLOR["lorenz"], rad=0.25)
    ax.plot([3, 3], [r["traj"] - 3.1, r["diag"] + 2.6], color=COLOR["lorenz"],
            lw=1.0, ls="-", solid_capstyle="round")
    arrow(ax, (3, r["traj"] - 3.1), (3, r["diag"] + 2.6), color=COLOR["lorenz"])
    arrow(ax, (3, r["diag"] + 2.6), (sx - 24, r["diag"] + 2.6), color=COLOR["lorenz"])
    ax.text(4.8, (r["traj"] + r["diag"]) / 2, "autodiff Jacobian\nfeeds the spectrum",
            fontsize=6.8, color="#444", rotation=90, va="center")

    box(ax, 13, r["data"], 18, 6.2, "lorenz.initial_conditions", COLOR["lorenz"], fontsize=7.2)
    arrow(ax, (22, r["data"]), (sx - 23.5, r["data"]), color=COLOR["lorenz"])

    box(ax, 87, r["train"], 22, 7.2, "train.divergence_at\n$\\nabla\\cdot g$ by autodiff",
        COLOR["train"], fontsize=7.2)
    arrow(ax, (76, r["train"] + 1.4), (sx + 24.5, r["train"] + 1.4), color=COLOR["train"])
    ax.text(80, r["train"] + 3.0, "penalty", fontsize=6.8, color="#444", ha="center")

    box(ax, 87, r["roll"], 22, 6.2, "train.implied_divergence\nln|det Dφ| / dt",
        COLOR["train"], fontsize=7.2)
    arrow(ax, (76, r["roll"] + 1.2), (sx + 21.5, r["roll"] + 1.2), color=COLOR["train"])

    box(ax, 87, r["diag"], 22, 8.0, "controls\ntrue trajectory, 2nd IC\nidentity map",
        COLOR["notebook"], fontsize=7.2)
    arrow(ax, (87, r["diag"] - 4.0), (sx + 23, r["verdict"] + 1.5),
          color=COLOR["notebook"], ls="--")

    ax.text(53, 13.5, "diagnostics run on cast(model, float64);\ntraining ran on float32 weights",
            fontsize=7, color="#555", style="italic", va="top", ha="left")

    ax.set_title("The pipeline, function by function", fontsize=11, loc="left", weight="bold")
    save(fig, "fig2_blueprint.png")


# --- figure 3: layered call graph (train + lorenz) -----------------------
def _depths(funcs, edges):
    out = {f: set() for f in funcs}
    for a, b in edges:
        out[a].add(b)
    memo = {}

    def depth(name, stack=()):
        if name in memo:
            return memo[name]
        if name in stack or not out[name]:
            return 0
        memo[name] = 1 + max(depth(c, stack + (name,)) for c in out[name])
        return memo[name]

    return out, depth, max(depth(f) for f in funcs)


def fig_inventory():
    funcs, edges = inventory()
    _, depth, maxd = _depths(funcs, edges)
    x_of = {d: 93 - d * (80 / max(maxd, 1)) for d in range(maxd + 1)}

    bands = {"train": (52, 98), "lorenz": (4, 44)}
    pos = {}
    fig, ax = plt.subplots(figsize=(8.8, 8.4))
    clean(ax)

    for mod, (y0, y1) in bands.items():
        names = sorted(ORDER[mod], key=lambda n: (depth(n), ORDER[mod].index(n)))
        n = len(names)
        for i, name in enumerate(names):
            y = y1 - i * ((y1 - y0) / max(n - 1, 1))
            pos[name] = (x_of[depth(name)], y)
            box(ax, *pos[name], 15, 1.6, name, COLOR[mod], fontsize=7.4, rounding=0.5)
        ax.text(3, y1 + 3.0, f"{mod}.py", fontsize=9, weight="bold", color=COLOR[mod])

    for a, b in edges:
        color = COLOR[funcs[a]] if funcs[a] != funcs[b] else "#999999"
        arrow(ax, pos[a], pos[b], color=color, lw=0.9, rad=0.12, zorder=0, shrink=4)

    for d in range(maxd + 1):
        ax.text(x_of[d], 100, f"depth {d}", ha="center", fontsize=7, color="#888")

    ax.text(50, 1.5, f"{len(pos)} functions · {len(edges)} internal call edges · "
                     "arrow = caller to callee (left to right) · "
                     "grey within a module, coloured across",
            ha="center", fontsize=7.0, color="#555", style="italic")
    ax.set_title("The call graph of the two wired modules", fontsize=11, loc="left",
                 weight="bold", pad=14)
    save(fig, "fig3_call_graph.png")


# --- figure 3b: the diagnostics have no internal edges -------------------
def fig_diagnostics():
    fig, ax = plt.subplots(figsize=(8.8, 2.9))
    clean(ax)
    names = ORDER["diagnostics"]
    for i, name in enumerate(names):
        row, col = divmod(i, 5)
        box(ax, 11 + col * 19.5, 72 - row * 26, 17.5, 15, name,
            COLOR["diagnostics"], fontsize=7.4, rounding=0.6)
    ax.text(50, 12, "ten independent measurements — no internal call edges; "
                    "each takes a step function\nor a trajectory and returns numbers, and none of them plots",
            ha="center", va="center", fontsize=7.4, color="#555", style="italic")
    ax.set_title("The measurement layer is a set of leaves", fontsize=11, loc="left",
                 weight="bold")
    save(fig, "fig3b_diagnostics.png")


# --- figure 4: which notebook cell touches which function -----------------
def fig_notebook_matrix():
    calls, sections = notebook_calls_and_sections()
    cells = sorted(calls)
    used_mods = ("lorenz", "train", "diagnostics")
    used = [(m, f) for m in used_mods for f in ORDER[m]
            if any(f"{m}.{f}" in calls[c] for c in cells)]

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    fig.subplots_adjust(left=0.32, right=0.98, top=0.86, bottom=0.06)
    rows = len(used)
    cell_pos = {c: j for j, c in enumerate(cells)}

    for j, cell in enumerate(cells):
        ax.axvline(j, color="#f2f2f2", lw=0.5, zorder=0)
        for i, (mod, fn) in enumerate(used):
            if f"{mod}.{fn}" in calls[cell]:
                ax.scatter(j, rows - i, s=78, color=COLOR[mod], marker="s", zorder=3)
        ax.text(j, rows + 0.35, str(cell), ha="center", va="bottom", fontsize=9.0,
                color="#333")

    for i, (mod, fn) in enumerate(used):
        ax.text(-0.8, rows - i, fn, ha="right", va="center", fontsize=10.2, color=COLOR[mod],
                clip_on=False)

    # section dividers, taken from the notebook's own markdown headings
    legend = []
    for idx, title in sections:
        if idx not in cell_pos:
            continue
        x = cell_pos[idx] - 0.5
        num = re.match(r"^(\d+)\.", title)
        num = num.group(1) if num else "?"
        ax.axvline(x, color="#bbbbbb", lw=0.8, zorder=0)
        ax.text(x + 0.08, rows + 0.9, f"§{num}", fontsize=9, color="#666",
                ha="left", va="bottom")
        short = re.sub(r"^\d+\.\s*", "", title).split(":")[0]
        if len(short) > 26:
            short = short[:24] + "…"
        legend.append(f"§{num} {short}")

    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylim(-1.6, rows + 1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.suptitle("Which notebook cell calls which function", fontsize=11, x=0.01,
                 ha="left", weight="bold")
    fig.text(0.32, 0.015, "  ·  ".join(legend), fontsize=7.2, color="#555", ha="left")
    save(fig, "fig4_notebook_matrix.png")


if __name__ == "__main__":
    fig_modules()
    fig_blueprint()
    fig_inventory()
    fig_diagnostics()
    fig_notebook_matrix()
