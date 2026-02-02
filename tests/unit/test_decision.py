import pytest

from homeostatic import HomeostaticConfig, HomeostaticMode, HomeostaticSystem


@pytest.mark.asyncio
async def test_decision_fluid_below_threshold(monkeypatch):
    system = HomeostaticSystem(HomeostaticConfig(), engine_client=None)

    async def fake_probe(_messages):
        return system.config.entropy_threshold - 0.01

    monkeypatch.setattr(system, "probe_entropy", fake_probe)
    decision = await system.decide_mode([{"role": "user", "content": "hi"}])
    assert decision.mode == HomeostaticMode.FLUIDO


@pytest.mark.asyncio
async def test_decision_analitico_at_or_above_threshold(monkeypatch):
    system = HomeostaticSystem(HomeostaticConfig(), engine_client=None)

    async def fake_probe(_messages):
        return system.config.entropy_threshold

    monkeypatch.setattr(system, "probe_entropy", fake_probe)
    decision = await system.decide_mode([{"role": "user", "content": "hi"}])
    assert decision.mode == HomeostaticMode.ANALITICO


@pytest.mark.asyncio
async def test_decision_fallback_to_fluid_when_probe_returns_none(monkeypatch):
    system = HomeostaticSystem(HomeostaticConfig(), engine_client=None)

    async def fake_probe(_messages):
        return None

    monkeypatch.setattr(system, "probe_entropy", fake_probe)
    decision = await system.decide_mode([{"role": "user", "content": "hi"}])
    assert decision.mode == HomeostaticMode.FLUIDO
    assert decision.entropy_norm is None


@pytest.mark.asyncio
async def test_decision_fallback_when_probe_raises_in_generate():
    class ExplodingEngineClient:
        async def generate(self, _prompt, _sampling_params=None):
            raise RuntimeError("boom")

    system = HomeostaticSystem(HomeostaticConfig(), engine_client=ExplodingEngineClient())
    decision = await system.decide_mode([{"role": "user", "content": "hi"}])
    assert decision.mode == HomeostaticMode.FLUIDO
    assert decision.entropy_norm is None
