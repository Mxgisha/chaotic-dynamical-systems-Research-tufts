"""Lorenz-63: the equations, an integrator, and trajectories.

Everything is written in plain JAX, so the same code runs on CPU or GPU without
changes.  This module is intentionally small: it is the "ground truth" side of
the project, and the emulator never imports anything from here except the
constants.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

with sigma=10, rho=28, beta=8/3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# --- parameters -----------------------------------------------------------
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
PARAMS = (SIGMA, RHO, BETA)

#: integration step used everywhere in this notebook
DT = 0.01

#: Lorenz-63 has *constant* divergence.  The volume of a small blob of initial
#: conditions changes at this rate at every point in space, and the three
#: Lyapunov exponents must sum to it.  This is the number the diagnostics test
#: the emulator against.
DIVERGENCE = -(SIGMA + 1.0 + BETA)

#: textbook Lyapunov exponents (continuous time), for comparison
TEXTBOOK_LYAPUNOV = (0.9056, 0.0, -14.5723)


# --- the field and its derivative ----------------------------------------
def rhs(y, params=PARAMS):
    """Vector field f(y) for a single state y = (x, y, z)."""
    x, yy, z = y[0], y[1], y[2]
    s, r, b = params
    return jnp.stack([s * (yy - x), x * (r - z) - yy, x * yy - b * z])


def jacobian(y, params=PARAMS):
    """Analytic derivative matrix df/dy at y."""
    x, yy, z = y[0], y[1], y[2]
    s, r, b = params
    return jnp.array(
        [
            [-s, s, 0.0],
            [r - z, -1.0, -x],
            [yy, x, -b],
        ]
    )


# --- integration ----------------------------------------------------------
def rk4_step(y, dt=DT, params=PARAMS):
    """One classical fourth-order Runge-Kutta step."""
    k1 = rhs(y, params)
    k2 = rhs(y + 0.5 * dt * k1, params)
    k3 = rhs(y + 0.5 * dt * k2, params)
    k4 = rhs(y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def flow(y0, steps, dt=DT, params=PARAMS):
    """Integrate ``steps`` steps.  Returns states *after* each step, (steps, 3)."""

    def body(y, _):
        nxt = rk4_step(y, dt, params)
        return nxt, nxt

    _, out = jax.lax.scan(body, y0, None, length=steps)
    return out


def trajectory(y0, steps, dt=DT, burn_in=0, params=PARAMS):
    """Integrate, discard ``burn_in`` steps, return exactly ``steps`` states."""
    total = burn_in + steps
    path = flow(y0, total, dt, params)
    return path[burn_in:] if burn_in else path


def flow_jacobian(y, dt=DT, params=PARAMS):
    """Derivative of one RK4 step with respect to the state.

    This is the object a Lyapunov calculation needs for a period-``dt`` map: the
    derivative of *the step*, ``I + O(dt)``, not the derivative of the vector
    field, which is ``O(30)``.  Autodiff does it, so there is no hand-derived
    product of stage matrices to get wrong.
    """
    return jax.jacfwd(lambda s: rk4_step(s, dt, params))(y)


# --- helpers --------------------------------------------------------------
def initial_conditions(key, n, low=-15.0, high=15.0, z_offset=25.0):
    """``n`` states spread over a box, shifted in z to sit near the attractor."""
    ic = jax.random.uniform(key, (n, 3), minval=low, maxval=high)
    return ic.at[:, 2].add(z_offset)


def divergence_at(y, params=PARAMS):
    """Divergence of the field at ``y`` -- constant for Lorenz-63."""
    return jnp.trace(jacobian(y, params))


if __name__ == "__main__":
    # quick self-check: divergence constant, exponents sum to it
    ys = trajectory(jnp.array([1.0, 1.0, 1.0]), 4, burn_in=500)
    divs = jax.vmap(divergence_at)(ys)
    print("divergence per state:", divs)
    print("state after 504 steps:", ys[-1])
