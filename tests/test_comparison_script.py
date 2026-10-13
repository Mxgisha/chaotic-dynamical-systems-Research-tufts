"""The parameterisation comparison driver.

These guard the plumbing rather than the science: the first version of the
comparison passed the *arm name* where a backend was expected, so every arm
silently trained the map and the results were identical for arms that should
have differed.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_parameterisations import ARMS  # noqa: E402


def test_every_arm_declares_a_known_backend():
    for name, spec in ARMS.items():
        assert spec["backend"] in ("map", "field"), (name, spec)


def test_arms_are_distinct_in_backend_or_loss():
    """Two arms that agree on both would be the same experiment twice."""
    seen = []
    for name, spec in ARMS.items():
        seen.append((spec["backend"], tuple(sorted(spec.get("loss", {}).items()))))
    assert len(set(seen)) == len(seen), seen


def test_the_divergence_arm_is_the_only_one_with_a_penalty():
    penalised = [n for n, s in ARMS.items() if s.get("loss", {}).get("divergence_weight")]
    assert penalised == ["field_div"], penalised
    assert ARMS["field_div"]["backend"] == "field"
    assert ARMS["field_div"]["loss"]["divergence_target"] == pytest.approx(-13.666666666, abs=1e-6)


def test_map_and_field_arms_use_their_own_backend():
    assert ARMS["map"]["backend"] == "map"
    assert ARMS["field"]["backend"] == "field"
    # no loss overrides on the un-penalised arms
    assert ARMS["map"]["loss"] == {}
    assert ARMS["field"]["loss"] == {}


def test_loss_overrides_are_valid_loss_configs():
    """A target with zero weight is rejected, so the arm definition must be sane."""
    from lorenz63.losses import LossConfig

    for name, spec in ARMS.items():
        LossConfig(l1_weight=0.0, **spec.get("loss", {}))
