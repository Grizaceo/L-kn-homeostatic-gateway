import pytest

import l_kn_gateway
from homeostatic import HomeostaticConfig, HomeostaticSystem


class FakeEngineClient:
    async def chat_completions(self, messages, stream=False, **_kwargs):
        self.last_messages = messages
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"completion_tokens": 1},
        }


@pytest.mark.asyncio
async def test_gateway_always_analitico_forces_intervention(monkeypatch):
    fake_engine = FakeEngineClient()
    homeostat = HomeostaticSystem(HomeostaticConfig(decision_strategy="rules"), engine_client=None)

    monkeypatch.setattr(l_kn_gateway, "engine_client", fake_engine)
    monkeypatch.setattr(l_kn_gateway, "homeostatic_system", homeostat)
    monkeypatch.setattr(l_kn_gateway.settings, "lkn_mode", "always_analitico", raising=False)

    request = l_kn_gateway.ChatCompletionRequest(
        messages=[{"role": "user", "content": "hola"}],
        stream=False,
    )

    class DummyRequest:
        def __init__(self):
            self.scope = {"request_id": "always-analitico-test"}

    await l_kn_gateway.chat_completions(request, DummyRequest())

    assert fake_engine.last_messages[0]["role"] == "system"
    assert "step-by-step" in fake_engine.last_messages[0]["content"]
