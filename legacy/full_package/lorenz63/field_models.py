"""Vector-field emulator: learn ``dx/dt`` and integrate it.

The map emulator learns one step of the flow, ``x_{n+1} = f(x_n)``.  Nothing in
the training objective constrains ``det Df``, so the learned Jacobian is free to
contract volume however it likes -- and a local-contraction study shows it
over-contracts by up to 6.4x in a tail of states, while the true Lorenz-63
divergence is *constant*.

This module learns the **vector field** ``g(x) ~= f(x)`` and integrates it, a
different parameterisation of the same problem.  The flow map is then a
composition of RK4 stages,

    phi_dt = x + h * sum_j c_j * g(x_j),   x_j = x + a_j h g(x_{j-1})

so its Jacobian is a product of stage Jacobians and

    det D phi_dt = prod_j det(stage_j)

is structural rather than something the loss has to discover.  For a linear field
it is exactly ``exp(dt * tr Dg)``.

Architecture, optimiser, data and loss are held identical to the map emulator so
the comparison isolates the parameterisation.  ``VectorFieldMLP.step`` integrates
one RK4 step, making it a drop-in replacement wherever the diagnostics call
``model.step`` / ``model.forward``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lorenz63.models import NumpyMLP, silu_grad
from lorenz63.train import (
    TrainConfig,
    _init_adam,
    adam_update,
    backward,
    forward_with_cache,
)

#: RK4 stage offsets and weights (the ``a`` and ``c`` of the Butcher tableau)
RK4_A = (0.0, 0.5, 0.5, 1.0)
RK4_C = (1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0)


class VectorFieldMLP:
    """A neural vector field ``g(x)`` with the map emulator's architecture."""

    #: the diagnostics' finite differences differentiate an integration here
    is_vector_field = True

    def __init__(self, in_size=3, out_size=3, width=128, depth=3, seed=67,
                 dt=0.01, field_scale=1.0):
        self.net = NumpyMLP(in_size=in_size, out_size=out_size, width=width,
                            depth=depth, seed=seed)
        self.in_size = in_size
        self.out_size = out_size
        self.width = width
        self.depth = depth
        self.dt = dt
        self.field_scale = float(field_scale)

    # --- field and map ----------------------------------------------------
    def field(self, x, params=None):
        x = np.asarray(x, dtype=np.float64)
        if params is None:
            return self.field_scale * self.net.forward(x)
        scale = float(np.asarray(params["field_scale"]["w"]).ravel()[0])
        return scale * self.net.forward(x, params)

    def flow_map(self, x, params=None):
        """One RK4 step of the learned field, for a batch ``(n, in_size)``."""
        x = np.asarray(x, dtype=np.float64)
        h = self.dt
        k_prev = None
        total = 0.0
        for j in range(4):
            stage_in = x if j == 0 else x + RK4_A[j] * h * k_prev
            k = self.field(stage_in, params)
            total = total + RK4_C[j] * k
            k_prev = k
        return x + h * total

    def forward(self, x, params=None):
        return self.flow_map(x, params)

    __call__ = forward

    def step(self, state, params=None):
        return self.flow_map(np.asarray(state, dtype=np.float64)[None, :], params)[0]

    # --- parameters -------------------------------------------------------
    @property
    def params(self):
        """Network parameters plus the scalar field scale."""
        p = {k: dict(v) for k, v in self.net.params.items()}
        p["field_scale"] = {"w": np.asarray([[self.field_scale]]),
                            "b": np.zeros(1), "act": False}
        return p

    def _layer_keys(self):
        return self.net._layer_keys()

    @property
    def n_params(self) -> int:
        # the scale is one shared scalar and is not counted, so the parameter
        # count matches the map emulator exactly
        return self.net.n_params

    def get_params(self):
        return self.params

    def set_params(self, params):
        if "field_scale" in params:
            self.field_scale = float(np.asarray(params["field_scale"]["w"]).ravel()[0])
        self.net.set_params({k: v for k, v in params.items() if k in self.net.params})

    # --- persistence ------------------------------------------------------
    def save(self, path):
        path = self.net.save(path)
        path.with_suffix(".field.json").write_text(
            json.dumps({"field_scale": self.field_scale, "dt": self.dt}, indent=2)
        )
        return path

    @classmethod
    def load(cls, path, dt=0.01):
        net = NumpyMLP.load(path)
        side = Path(path).with_suffix(".field.json")
        scale = 1.0
        if side.exists():
            meta = json.loads(side.read_text())
            scale = meta.get("field_scale", 1.0)
            dt = meta.get("dt", dt)
        m = cls(in_size=net.in_size, out_size=net.out_size, width=net.width,
                depth=net.depth, seed=net.rng_seed, dt=dt, field_scale=scale)
        m.net = net
        return m


def init_field_scale(net, x_train, y_one, dt):
    """Least-squares scale for the field so training starts near the right size.

    A freshly initialised network outputs O(1) while the true field is O(30), so
    fitting one scalar removes a large constant from the first epochs.  It is a
    single shared parameter, not a change of architecture.
    """
    h = dt
    k_prev = None
    total = 0.0
    for j in range(4):
        stage_in = x_train if j == 0 else x_train + RK4_A[j] * h * k_prev
        k = net.forward(stage_in)
        total = total + RK4_C[j] * k
        k_prev = k
    inc = h * total
    target = y_one - x_train
    denom = float(np.sum(inc * inc))
    return 1.0 if denom <= 1e-30 else float(np.sum(inc * target) / denom)


# --- gradients through the integration ------------------------------------
def flow_with_cache(x, params, keys, scale, dt):
    """One RK4 step with every stage cached.

    Returns ``(out, cache)`` where ``cache["stages"]`` holds, per stage,
    ``{"stage_in", "net_out", "net_cache", "c"}``.
    """
    h = dt
    stages = []
    k_prev = None
    total = 0.0
    for j in range(4):
        stage_in = x if j == 0 else x + RK4_A[j] * h * k_prev
        net_out, net_cache = forward_with_cache(stage_in, params, keys)
        k = scale * net_out
        stages.append({"stage_in": stage_in, "net_out": net_out,
                       "net_cache": net_cache, "c": RK4_C[j]})
        total = total + RK4_C[j] * k
        k_prev = k
    return x + h * total, {"stages": stages, "h": h, "scale": scale}


def flow_backward(dout, cache, keys, net_params):
    """Backpropagate ``dout`` = dL/d(flow output) through the RK4 stages.

    ``D phi`` is a composition of the stage maps, so contributions accumulate
    through the stages in reverse.  Returns ``(grads, dscale)`` where ``grads`` is
    keyed ``"<layer>/w"`` / ``"<layer>/b"``.
    """
    h = cache["h"]
    stages = cache["stages"]
    scale = cache["scale"]
    grads = {f"{k}/{p}": np.zeros_like(net_params[k][p])
             for k in keys for p in ("w", "b")}
    dscale = 0.0
    d_prev = np.zeros_like(dout)  # dL/d(stage input), from *later* stages

    for j in reversed(range(4)):
        st = stages[j]
        c = st["c"]
        # out += h * c * scale * net_out_j
        dL_dnetout = h * c * scale * dout + d_prev
        # the scalar scale multiplies every stage
        dscale += float(np.sum(h * c * dout * st["net_out"]))
        g = backward(st["net_out"], st["net_cache"], net_params, keys, dL_dnetout)
        for key in keys:
            grads[f"{key}/w"] += g[f"{key}/w"]
            grads[f"{key}/b"] += g[f"{key}/b"]
        if j > 0:
            # stage_in_j = x + a_j h k_{j-1}, and k_{j-1} = scale * net_out_{j-1}
            d = dL_dnetout
            for key in reversed(keys):
                ncache = st["net_cache"][key]
                if ncache["act"]:
                    d = d * silu_grad(ncache["z"])
                d = d @ net_params[key]["w"].T
            d_prev = RK4_A[j] * h * scale * d
        else:
            d_prev = np.zeros_like(dout)
    return grads, dscale


def divergence_penalty_and_grad(net, x, target, scale, eps=1e-5):
    """Divergence penalty for the learned field, plus its gradients.

    The field is ``scale * net(x)``, so its divergence is ``scale * div(net)(x)``.
    The penalty is ``mean((scale * div_net - target)^2)`` and returns

    ``(value, weight_grads, scale_grad)``

    where ``scale_grad`` is ``2 * mean(residual * div_net)``.  That term is not
    optional: the divergence is linear in ``scale``, so a penalty that ignores it
    cannot control the divergence at all, and omitting it let the scale grow
    unchecked and drove the divergence far past the target.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape

    # divergence of the *unscaled* network, from central differences of the field
    div_net = np.zeros(n)
    probes = []
    for i in range(d):
        e = np.zeros(d)
        e[i] = eps
        plus, minus = x + e, x - e
        div_net += (net.forward(plus) - net.forward(minus))[:, i] / (2 * eps)
        probes.append((plus, minus, i))

    div = scale * div_net
    residual = div - target
    value = float(np.mean(residual ** 2))
    dscale = 2.0 * float(np.mean(residual * div_net))
    # An exact analytic field (used in the tests, and a useful reference) has no
    # parameters to descend; report the value and a zero weight gradient.
    keys = list(getattr(net, "_layer_keys", lambda: [])())
    if not np.isfinite(value) or value == 0.0 or not keys:
        return value, None, dscale
    grads = {f"{k}/{p}": np.zeros_like(net.params[k][p])
             for k in keys for p in ("w", "b")}
    # d value / d theta, scaled by the field scale
    weight = 2.0 * scale * residual / n
    for plus, minus, i in probes:
        for side, sign in ((plus, 1.0), (minus, -1.0)):
            out, cache = forward_with_cache(side, net.params, keys)
            dout = np.zeros_like(out)
            dout[:, i] = sign * weight / (2 * eps)
            g = backward(out, cache, net.params, keys, dout)
            for key in keys:
                grads[f"{key}/w"] += g[f"{key}/w"]
                grads[f"{key}/b"] += g[f"{key}/b"]
    return value, grads, dscale


def train_vector_field(model, dataset, config=None, verbose=True):
    """Train the vector field with the same objective and optimiser as the map.

    One RK4 step of the learned field predicts the first state of the horizon,
    exactly as the map emulator does, so the two are compared like for like.

    If ``config.loss.divergence_weight`` is non-zero, the system's known
    divergence is added as a physics-informed penalty.  Without it the divergence
    is only weakly constrained by the data: it enters the one-step map at
    ``O(dt^2)``, so at ``dt = 0.01`` a field with several times the true
    divergence still fits the increments very well.
    """
    config = config or TrainConfig()
    loss_cfg = config.loss
    rng = np.random.default_rng(config.seed)
    net = model.net
    keys = net._layer_keys()

    x_train = np.asarray(dataset.x_train, dtype=np.float64)
    y_one = np.asarray(dataset.y_train, dtype=np.float64)[:, 0, :]
    n = x_train.shape[0]

    model.field_scale = init_field_scale(net, x_train, y_one, model.dt)
    use_div = (loss_cfg.divergence_weight > 0.0
               and loss_cfg.divergence_target is not None)

    adam = _init_adam(net)
    history = []
    for epoch in range(config.epochs):
        frac = epoch / max(config.epochs - 1, 1)
        lr = config.lr * (0.1 ** max(0.0, (frac - 0.5) / 0.5))
        perm = rng.permutation(n)
        totals, mses, divs = [], [], []
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = x_train[idx]
            yb = y_one[idx]

            pred, cache = flow_with_cache(
                xb, net.params, keys, model.field_scale, model.dt
            )
            mse = float(np.mean((pred - yb) ** 2))
            dpred = 2.0 * (pred - yb) / pred.size
            grads, dscale = flow_backward(dpred, cache, keys, net.params)

            div_val = 0.0
            dscale_div = 0.0
            if use_div:
                div_val, div_grads, dscale_div = divergence_penalty_and_grad(
                    net, xb, loss_cfg.divergence_target, model.field_scale
                )
                if div_grads is not None:
                    w = loss_cfg.divergence_weight
                    for gk in grads:
                        grads[gk] += w * div_grads[gk]

            gnorm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            if gnorm > 1.0:
                for gk in grads:
                    grads[gk] = grads[gk] * (1.0 / gnorm)
            adam_update(net.params, grads, adam, lr, keys)
            total_dscale = dscale + loss_cfg.divergence_weight * dscale_div
            if np.isfinite(total_dscale):
                model.field_scale = float(
                    np.clip(model.field_scale - lr * float(total_dscale), -1e4, 1e4)
                )
            totals.append(mse + loss_cfg.divergence_weight * div_val)
            mses.append(mse)
            divs.append(div_val)

        entry = {
            "epoch": epoch,
            "lr": lr,
            "total": float(np.mean(totals)),
            "mse": float(np.mean(mses)),
            "l1": 0.0,
            "ftle": 0.0,
            "divergence_penalty": float(np.mean(divs)),
            "field_scale": float(model.field_scale),
        }
        history.append(entry)
        if verbose and (epoch % 10 == 0 or epoch == config.epochs - 1):
            print(f"Epoch {epoch:3d} | lr {lr:.2e} | mse {entry['mse']:.6f} "
                  f"| div_pen {entry['divergence_penalty']:.3f} "
                  f"| scale {entry['field_scale']:.3f}")
    return model, history


__all__ = [
    "VectorFieldMLP",
    "divergence_penalty_and_grad",
    "train_vector_field",
    "flow_with_cache",
    "flow_backward",
    "init_field_scale",
    "RK4_A",
    "RK4_C",
]
