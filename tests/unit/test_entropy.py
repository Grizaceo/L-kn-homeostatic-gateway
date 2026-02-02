import math

import pytest

from homeostatic import HomeostaticConfig, HomeostaticSystem


def _logprobs_from_probs(probs):
    return [(f"t{i}", math.log(p)) for i, p in enumerate(probs)]


def _system():
    return HomeostaticSystem(HomeostaticConfig(), engine_client=None)


def test_entropy_k1_zero():
    system = _system()
    entropy_norm = system.calculate_entropy([("only", 0.0)])
    assert 0.0 <= entropy_norm <= 1.0
    assert entropy_norm == pytest.approx(0.0, abs=1e-6)


def test_entropy_uniform_k10_near_one():
    system = _system()
    probs = [1.0 / 10.0] * 10
    entropy_norm = system.calculate_entropy(_logprobs_from_probs(probs))
    assert 0.0 <= entropy_norm <= 1.0
    assert entropy_norm == pytest.approx(1.0, abs=1e-3)


def test_entropy_skewed_between_zero_and_one():
    system = _system()
    probs = [0.9] + [0.1 / 9.0] * 9
    entropy_norm = system.calculate_entropy(_logprobs_from_probs(probs))
    assert 0.0 <= entropy_norm <= 1.0
    assert 0.0 < entropy_norm < 1.0
