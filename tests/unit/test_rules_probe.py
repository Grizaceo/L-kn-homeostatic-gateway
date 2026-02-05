import pytest

from homeostatic import HomeostaticConfig, HomeostaticMode, HomeostaticSystem


def _rules_system(max_tokens_fluido=150):
    config = HomeostaticConfig(
        decision_strategy="rules",
        max_tokens_fluido=max_tokens_fluido,
    )
    return HomeostaticSystem(config, engine_client=None)


@pytest.mark.asyncio
async def test_rules_strategy_fluid_for_simple_prompt():
    system = _rules_system()
    decision = await system.decide_mode([{"role": "user", "content": "hola"}])
    assert decision.mode == HomeostaticMode.FLUIDO
    assert decision.decision_strategy == "rules"
    assert decision.probe_signals is not None


@pytest.mark.asyncio
async def test_rules_strategy_analitico_for_code_prompt():
    system = _rules_system()
    decision = await system.decide_mode(
        [{"role": "user", "content": "Write a python function: def add(a, b): return a + b"}]
    )
    assert decision.mode == HomeostaticMode.ANALITICO
    assert decision.probe_signals is not None
    assert decision.probe_signals.has_code is True


@pytest.mark.asyncio
async def test_rules_strategy_analitico_for_long_prompt():
    system = _rules_system(max_tokens_fluido=20)
    prompt = "token " * 25
    decision = await system.decide_mode([{"role": "user", "content": prompt}])
    assert decision.mode == HomeostaticMode.ANALITICO
    assert decision.probe_signals is not None
    assert decision.probe_signals.prompt_tokens_est > 20


@pytest.mark.asyncio
async def test_entropy_fallback_uses_rules(monkeypatch):
    config = HomeostaticConfig(
        decision_strategy="entropy",
        max_tokens_fluido=10,
    )
    system = HomeostaticSystem(config, engine_client=None)

    async def fake_probe(_messages):
        return None

    monkeypatch.setattr(system, "probe_entropy", fake_probe)
    prompt = "token " * 20
    decision = await system.decide_mode([{"role": "user", "content": prompt}])
    assert decision.mode == HomeostaticMode.ANALITICO
    assert decision.decision_strategy == "entropy_fallback_rules"
