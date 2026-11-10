# Chaotic dynamical systems — Research@Tufts

Can a neural network learn a chaotic system's *dynamics*, or only its next state?

A small network is trained to imitate the **Lorenz-63** equations. Eight measurements then
test whether it reproduces the system's dynamics — its Lyapunov spectrum, its attractor
dimension, its lobe statistics, and the rate at which it destroys volume — rather than just
its one-step error.

The answer, in this study, is that the two come apart: an emulator with low one-step error
can get the dynamics badly wrong, and the failure is concentrated in a minority of states
rather than spread evenly.

## Start here

[`Lorenz63_emulator.ipynb`](Lorenz63_emulator.ipynb) — one notebook, executed end to end,
that builds the system from the equations, trains the emulator, measures all eight
properties, finds the failure, and fixes it. No physics background assumed.

## Files

| file | what it is |
|---|---|
| [`Lorenz63_emulator.ipynb`](Lorenz63_emulator.ipynb) | the walkthrough; runs in a few minutes on a CPU |
| [`lorenz.py`](lorenz.py) | the equations, RK4 integration, trajectories, exact Jacobians |
| [`train.py`](train.py) | equinox MLP, losses, optax training loop, precision handling |
| [`diagnostics.py`](diagnostics.py) | the eight measurements, all in JAX |
| [`PIPELINE.md`](PIPELINE.md) · [`PIPELINE.pdf`](PIPELINE.pdf) | how the code fits together: four diagrams of the modules, the call graph and the notebook |
| [`experiments/`](experiments) | the two follow-up experiments: timestep scaling, and estimating the divergence from data |
| [`PAPER.md`](PAPER.md) · [`PAPER.pdf`](PAPER.pdf) | the written account, with the full results |
| [`legacy/full_package/`](legacy/full_package) | the larger study this was cut down from: 103 tests, 5-seed statistics, comparison scripts |

About 700 lines of Python across the three modules, plus the notebook.

## Running it

```bash
pip install -r requirements.txt
jupyter lab Lorenz63_emulator.ipynb
```

**On a GPU.** The code is plain JAX and selects a GPU automatically when one is visible. On
Colab, set the runtime to GPU and install the CUDA build:

```python
!pip install -U "jax[cuda12]" diffrax equinox optax
```

The second cell prints the backend and devices it found. Training runs in float32; the
diagnostics run in float64, and section 7 of the notebook measures the difference.

## The idea in one paragraph

A chaotic system destroys volume at a rate (the divergence) that is *constant* for
Lorenz-63, and the three Lyapunov exponents must sum to it. That makes the volume change a
known target at every state, so an emulator's departure from it is unambiguous. It turns out
that the training signal cannot see this quantity at all: the divergence enters a one-step
prediction only at second order in the timestep, a factor of 100 down at `dt = 0.01`. An
emulator that learns the one-step map therefore has no incentive to get it right, and no
local quantity that could be penalised. Learning the *velocity field* instead exposes the
divergence as a cheap derivative, and penalising it recovers the correct contraction at a
modest cost in one-step accuracy.

## Where this came from

The larger study — five seeds per configuration, significance screening across 198
comparisons, and the parameterisation experiment in full — is in
[`legacy/full_package/`](legacy/full_package) and documented in [`PAPER.md`](PAPER.md). The
notebook is a reduction of it to the parts that carry the argument.
