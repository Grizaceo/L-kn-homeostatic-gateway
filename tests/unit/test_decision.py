import pytest

from homeostatic import HomeostaticConfig, HomeostaticMode, HomeostaticSystem


@pytest.mark.asyncio
async def test_decision_fluid_below_threshold(monkeypatch):
    system = HomeostaticSystem(HomeostaticConfig(decision_strategy="entropy"), engine_client=None)

    async def fake_probe(_messages):
        return system.config.entropy_threshold - 0.01

    monkeypatch.setattr(system, "probe_entropy", fake_probe)
    decision = await system.decide_mode([{"role": "user", "content": "hi"}])
    assert decision.mode == HomeostaticMode.FLUIDO


@pytest.mark.asyncio
async def test_decision_analitico_at_or_above_threshold(monkeypatch):
    system = HomeostaticSystem(HomeostaticConfig(decision_strategy="entropy"), engine_client=None)

    async def fake_probe(_messages):
        return system.config.entropy_threshold

    monkeypatch.setattr(system, "probe_entropy", fake_probe)
    decision = await system.decide_mode([{"role": "user", "content": "hi"}])
    assert decision.mode == HomeostaticMode.ANALITICO


@pytest.mark.asyncio
async def test_decision_fallback_to_fluid_when_probe_returns_none(monkeypatch):
    system = HomeostaticSystem(HomeostaticConfig(decision_strategy="entropy"), engine_client=None)

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

    system = HomeostaticSystem(HomeostaticConfig(decision_strategy="entropy"), engine_client=ExplodingEngineClient())
    decision = await system.decide_mode([{"role": "user", "content": "hi"}])
    assert decision.mode == HomeostaticMode.FLUIDO
    assert decision.entropy_norm is None


@pytest.mark.asyncio
async def test_probe_entropy_falls_back_to_legacy_when_chat_logprobs_missing():
    class HybridEngineClient:
        def __init__(self):
            self.chat_calls = 0
            self.generate_calls = 0

        async def chat_completions(self, _messages, stream=False, **_kwargs):
            self.chat_calls += 1
            # No logprobs block -> force fallback to legacy.
            return {"choices": [{"message": {"role": "assistant", "content": "x"}}]}

        async def generate(self, _prompt, _sampling_params=None):
            self.generate_calls += 1
            return {
                "meta_info": {
                    "output_top_logprobs": [
                        {"A": -0.05, "B": -3.0, "C": -3.2}
                    ]
                }
            }

    engine = HybridEngineClient()
    system = HomeostaticSystem(HomeostaticConfig(decision_strategy="entropy"), engine_client=engine)

    entropy_1 = await system.probe_entropy([{"role": "user", "content": "hi"}])
    entropy_2 = await system.probe_entropy([{"role": "user", "content": "hi again"}])

    assert entropy_1 is not None
    assert entropy_2 is not None
    assert engine.chat_calls == 1
    assert engine.generate_calls == 2
