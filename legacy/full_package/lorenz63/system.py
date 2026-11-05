"""The Lorenz-63 system and its integration.

Single source of truth for the ODE, its parameters, and stepping.  Previously
this lived in the notebook *and* in ``functions.py`` (which was fetched over HTTP
at runtime and ``exec``-ed into globals), so an edit to the file on ``main``
silently changed notebook behaviour.
"""

from __future__ import annotations

from lorenz63.backend import xp

# --- Physical system -------------------------------------------------------
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
PARAMS = (SIGMA, RHO, BETA)

#: Textbook Lyapunov spectrum of Lorenz-63 (continuous time), used as the
#: analytic reference in tests.
TEXTBOOK_LYAPUNOV = (0.9056, 0.0, -14.5723)

#: The divergence of the flow is constant, so the Lyapunov exponents must sum
#: to it.  Used as an invariant check.
DIVERGENCE = -(SIGMA + 1.0 + BETA)


def lorenz_rhs(state):
    """Vector field as a plain function of state."""
    x, y, z = state[0], state[1], state[2]
    return xp.stack(
        [
            SIGMA * (y - x),
            x * (RHO - z) - y,
            x * y - BETA * z,
        ]
    )


def lorenz_vf(t, state, args=None):
    """``(t, state, args)`` adapter for the diffrax-style call signature."""
    return lorenz_rhs(state)


def jacobian(state):
    """Analytic Jacobian of :func:`lorenz_rhs` at ``state``."""
    x, y, z = state[0], state[1], state[2]
    z0 = xp.zeros((), dtype=state.dtype)
    row0 = xp.stack([-SIGMA + z0, SIGMA + z0, z0])
    row1 = xp.stack([RHO - z, -1.0 + z0, -x])
    row2 = xp.stack([y, x, -BETA + z0])
    return xp.stack([row0, row1, row2])


def rk4_step(state, dt):
    """One classical RK4 step.  Matches Tsit5 to ~1e-9 at dt=0.01."""
    k1 = lorenz_rhs(state)
    k2 = lorenz_rhs(state + 0.5 * dt * k1)
    k3 = lorenz_rhs(state + 0.5 * dt * k2)
    k4 = lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def flow_map(state, dt):
    """The one-step flow map ``x -> phi_dt(x)`` under RK4."""
    return rk4_step(state, dt)


def integrate(y0, steps, dt):
    """Integrate ``steps`` steps of size ``dt``.

    Returns an array of shape ``(steps, 3)`` holding the state *after* each step,
    i.e. ``out[k] = phi_{(k+1)dt}(y0)``.  This is the ``shift`` convention: the
    initial condition itself is not included, which is what the training windows
    and every diagnostic assume.
    """

    def body(carry, _):
        nxt = flow_map(carry, dt)
        return nxt, nxt

    if hasattr(xp, "lax"):  # JAX fast path
        import jax

        _, out = jax.lax.scan(body, y0, None, length=steps)
        return out
    ys = []
    carry = y0
    for _ in range(steps):
        carry = body(carry, None)[0]
        ys.append(carry)
    return xp.stack(ys)


def generate_trajectory(y0, steps, dt, burn_in=0):
    """Burn in, then return exactly ``steps`` states of shape ``(steps, 3)``.

    The previous implementation returned ``total_steps - burn_in_steps`` states,
    so two call sites disagreed about the length of a "trajectory" and one of
    them silently indexed past the end.  Here the contract is explicit.
    """
    total = burn_in + steps
    path = integrate(y0, total, dt)
    if burn_in == 0:
        return path
    # Discard the transient: index burn_in-1 so the first retained state is the
    # one produced by exactly `burn_in` steps.
    return path[burn_in - 1 : total - 1]


#: Convenience alias kept for readability in diagnostics.
def step(state, dt):
    return flow_map(state, dt)


__all__ = [
    "SIGMA",
    "RHO",
    "BETA",
    "PARAMS",
    "TEXTBOOK_LYAPUNOV",
    "DIVERGENCE",
    "lorenz_rhs",
    "lorenz_vf",
    "jacobian",
    "rk4_step",
    "flow_map",
    "integrate",
    "generate_trajectory",
    "step",
]
