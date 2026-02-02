import pytest

import l_kn_gateway


class FakeCircuitBreaker:
    def __init__(self):
        self.record_failure_called = 0

    def record_failure(self):
        self.record_failure_called += 1

    def can_attempt(self):
        return True


class FakeEngineClient:
    def __init__(self, circuit_breaker):
        self.circuit_breaker = circuit_breaker

    async def chat_completions(self, _messages, stream=False, **_kwargs):
        async def gen():
            yield "data: test\n\n"
            raise RuntimeError("stream boom")

        return gen()


class FakeLogger:
    def __init__(self):
        self.error_calls = []

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_stream_failure_records_circuit_breaker(monkeypatch):
    circuit_breaker = FakeCircuitBreaker()
    engine_client = FakeEngineClient(circuit_breaker)
    fake_logger = FakeLogger()

    monkeypatch.setattr(l_kn_gateway, "engine_client", engine_client)
    monkeypatch.setattr(l_kn_gateway.settings, "lkn_mode", "passthrough", raising=False)
    monkeypatch.setattr(l_kn_gateway, "logger", fake_logger)

    request = l_kn_gateway.ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        stream=True
    )

    class DummyRequest:
        def __init__(self):
            self.scope = {"request_id": "test-request"}

    response = await l_kn_gateway.chat_completions(request, DummyRequest())
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert circuit_breaker.record_failure_called == 1
    assert len(fake_logger.error_calls) == 1
