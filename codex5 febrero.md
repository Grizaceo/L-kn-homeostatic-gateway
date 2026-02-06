# Parte 1: Instrucciones para Codex — Fixes del código

## Fix 1: `time.time()` → `time.perf_counter()` en LatencyTracker

**Archivo:** `src/utils.py`

**Prompt para Codex:**

```
En src/utils.py, en la clase LatencyTracker, reemplaza time.time() por time.perf_counter() 
tanto en el método start() como en elapsed_ms(). Razón: perf_counter es monotónico y más 
preciso para medir latencia (no se ve afectado por ajustes de reloj del sistema).
```

**Cambio exacto:**

```python
# ANTES (líneas 79-87)
class LatencyTracker:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000

# DESPUÉS
class LatencyTracker:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000
```

---

## Fix 2: Probe de entropía — usar chat template compatible con RadixAttention

**Archivo:** `src/homeostatic.py`

**Prompt para Codex:**

```
En src/homeostatic.py, refactoriza el método probe_entropy para que use el endpoint 
/v1/chat/completions con max_tokens=1 y logprobs=true, en lugar de /generate con un 
template manual. 

Razón: el template manual ("System: ...\nUser: ...\nAssistant:") no coincide con el 
chat template de Qwen (<|im_start|>system...<|im_end|>), así que los tokens del probe 
y los de la generación real son diferentes y RadixAttention no puede reusar el KV cache.

El nuevo método debe:
1. Llamar a self.engine_client.chat_completions() con stream=False, max_tokens=1, 
   temperature=1.0, logprobs=True, top_logprobs=self.config.probe_top_k
2. Extraer logprobs del formato OpenAI (response["choices"][0]["logprobs"]["content"][0]["top_logprobs"])
3. Convertirlos a lista de (token, logprob) para pasar a self.calculate_entropy()
4. Si el endpoint no soporta logprobs (KeyError), loggear warning y retornar None
5. Mantener el método original renombrado como probe_entropy_legacy() por si SGLang 
   no soporta logprobs en chat/completions y hay que hacer fallback
```

**Cambio exacto:**

```python
# ANTES: probe_entropy (líneas 147-201)
async def probe_entropy(self, messages: List[Dict[str, str]]) -> Optional[float]:
    # ... template manual con /generate ...

# DESPUÉS: reemplazar con estos dos métodos

async def probe_entropy(self, messages: List[Dict[str, str]]) -> Optional[float]:
    """
    Probe engine entropy via /v1/chat/completions with logprobs.
    
    Uses the same endpoint and chat template as real generation,
    so the tokenized prefix is identical → RadixAttention can reuse KV cache.
    """
    if not messages:
        logger.warning("probe_empty_messages")
        return None

    if self.engine_client is None:
        logger.warning("probe_missing_engine_client")
        return None

    try:
        response = await self.engine_client.chat_completions(
            messages=messages,
            stream=False,
            max_tokens=1,
            temperature=1.0,
            logprobs=True,
            top_logprobs=self.config.probe_top_k,
        )

        # OpenAI format: choices[0].logprobs.content[0].top_logprobs
        choices = response.get("choices", [])
        if not choices:
            logger.warning("probe_no_choices")
            return None

        logprobs_data = choices[0].get("logprobs", {})
        content_logprobs = logprobs_data.get("content", [])
        if not content_logprobs:
            logger.warning("probe_no_content_logprobs", response_keys=list(response.keys()))
            return None

        top_logprobs = content_logprobs[0].get("top_logprobs", [])
        logprobs_list = [
            (entry["token"], entry["logprob"])
            for entry in top_logprobs
        ]

        if not logprobs_list:
            logger.warning("probe_empty_top_logprobs")
            return None

        entropy_norm = self.calculate_entropy(logprobs_list)
        logger.info("probe_completed", entropy_norm=round(entropy_norm, 3), method="chat_completions")
        return entropy_norm

    except KeyError as e:
        logger.warning("probe_logprobs_not_supported", missing_key=str(e))
        logger.info("probe_falling_back_to_legacy")
        return await self.probe_entropy_legacy(messages)

    except Exception as e:
        from utils import classify_engine_error
        category, reason = classify_engine_error(e)
        logger.error("probe_failed", failure_reason=reason, category=category, error=str(e))
        return None

async def probe_entropy_legacy(self, messages: List[Dict[str, str]]) -> Optional[float]:
    """
    Legacy probe via /generate with manual template.
    
    WARNING: This uses a different tokenization than chat/completions,
    so RadixAttention cannot reuse the KV cache from this probe.
    Only use as fallback if /v1/chat/completions doesn't support logprobs.
    """
    if not messages:
        return None
    if self.engine_client is None:
        return None

    try:
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        sampling_params = {
            "max_new_tokens": 1,
            "temperature": 1.0,
            "return_logprob": True,
            "top_logprobs_num": self.config.probe_top_k,
        }

        response = await self.engine_client.generate(prompt, sampling_params)

        if "meta_info" in response and "output_top_logprobs" in response["meta_info"]:
            top_logprobs = response["meta_info"]["output_top_logprobs"]
            if top_logprobs:
                first_token_logprobs = top_logprobs[0]
                logprobs_list = [(token, lp) for token, lp in first_token_logprobs.items()]
                entropy_norm = self.calculate_entropy(logprobs_list)
                logger.info("probe_completed", entropy_norm=round(entropy_norm, 3), method="legacy_generate")
                return entropy_norm

        logger.warning("probe_legacy_schema_mismatch", response_keys=list(response.keys()))
        return None

    except Exception as e:
        from utils import classify_engine_error
        category, reason = classify_engine_error(e)
        logger.error("probe_legacy_failed", failure_reason=reason, category=category, error=str(e))
        return None
```

---

## Fix 3: Agregar modo `always_analitico` para baseline de evaluación

**Archivo:** `src/l_kn_gateway.py`

**Prompt para Codex:**

```
En src/l_kn_gateway.py, en el endpoint chat_completions, agregar soporte para 
LKN_MODE=always_analitico. Cuando este modo está activo, se salta el probe y siempre 
aplica la intervención ANALÍTICO (inyectar system prompt de verificación). 
Esto sirve como baseline de "máxima calidad" para comparar contra el routing homeostático.

También agregar el valor "always_analitico" como opción válida en Settings.lkn_mode.
```

**Cambio exacto en l_kn_gateway.py:**

```python
# ANTES (línea 39):
lkn_mode: str = "homeostatic"  # "homeostatic" or "passthrough"

# DESPUÉS:
lkn_mode: str = "homeostatic"  # "homeostatic", "passthrough", or "always_analitico"
```

```python
# ANTES (líneas 252-259 en chat_completions):
        # Homeostatic decision
        if settings.lkn_mode == "homeostatic":
            decision = await homeostatic_system.decide_mode(messages)
            mode = decision.mode
            entropy_norm = decision.entropy_norm

            # Apply intervention if needed
            if decision.intervention_applied:
                messages = homeostatic_system.apply_intervention(messages, decision)

# DESPUÉS:
        # Homeostatic decision
        if settings.lkn_mode == "homeostatic":
            decision = await homeostatic_system.decide_mode(messages)
            mode = decision.mode
            entropy_norm = decision.entropy_norm

            if decision.intervention_applied:
                messages = homeostatic_system.apply_intervention(messages, decision)

        elif settings.lkn_mode == "always_analitico":
            mode = "ANALITICO"
            decision = HomeostaticDecision(
                mode="ANALITICO",
                entropy_norm=None,
                intervention_applied=True,
                rationale="always_analitico mode (baseline)",
                decision_strategy="forced",
            )
            messages = homeostatic_system.apply_intervention(messages, decision)
```

Agregar import en la cabecera si no existe:

```python
from homeostatic import HomeostaticSystem, HomeostaticConfig, HomeostaticDecision
```

---

## Fix 4: Relajar pinning de numpy

**Archivo:** `requirements.txt`

**Prompt para Codex:**

```
En requirements.txt, cambiar numpy>=1.26.0,<2.0.0 por numpy>=1.26.0 para evitar 
conflictos con sglang y otras dependencias que ya requieren numpy 2.x.
```

---

## Fix 5: Agregar `__init__.py` en `src/`

**Prompt para Codex:**

```
Crear un archivo src/__init__.py vacío para permitir imports como paquete Python. 
Contenido: solo un comentario.
```

```python
# src/__init__.py
# L-kn Gateway source package
```

---

## Test para verificar los fixes

**Prompt para Codex:**

```
En tests/unit/test_rules_probe.py, agregar un test que verifique que el modo 
always_analitico siempre retorna ANALITICO sin importar el contenido del mensaje.
```

```python
# Agregar a tests/unit/test_decision.py

@pytest.mark.asyncio
async def test_always_analitico_mode_forces_analitico():
    """Verify always_analitico baseline forces ANALITICO for any input."""
    system = HomeostaticSystem(HomeostaticConfig(decision_strategy="rules"), engine_client=None)
    
    # Even a trivial message should be ANALITICO in this mode
    decision = HomeostaticDecision(
        mode="ANALITICO",
        entropy_norm=None,
        intervention_applied=True,
        rationale="always_analitico mode (baseline)",
        decision_strategy="forced",
    )
    
    messages = [{"role": "user", "content": "hola"}]
    modified = system.apply_intervention(messages, decision)
    
    # Should have injected system prompt
    assert modified[0]["role"] == "system"
    assert "step-by-step" in modified[0]["content"]
```

---
---