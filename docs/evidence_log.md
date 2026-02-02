# L-kn Gateway - Evidence Log

Fecha: 2026-02-02
Alcance: L-kn v1 (gateway + homeostatica + runtime local)
Regla: todo claim sin evidencia externa se marca como SIN FUENTE y se acompana de "Como verificar en el PC".

## A) Evidencia verificada en el repo (codigo)
1) `src/homeostatic.py` define modos FLUIDO/ANALITICO y umbral por defecto 0.6.
2) `src/homeostatic.py` hace probe via `engine_client.generate` con `return_logprob` y `top_logprobs_num`.
3) `src/engine_client.py` implementa circuit breaker con estados CLOSED/OPEN/HALF_OPEN y `record_success`/`record_failure`.
4) `src/l_kn_gateway.py` maneja SSE; en excepcion registra failure del circuit breaker y emite evento de error.

## B) Hipotesis por verificar en el PC (SIN FUENTE)
1) Flags reales del launcher de SGLang (por ejemplo `--mem-fraction-static`, `--model-path`, `--host`, `--port`).
   Como verificar en el PC: ejecutar `python3 -m sglang.launch_server --help` y comparar con scripts/config.
2) Soporte real de logprobs en `/v1/chat/completions`.
   Como verificar en el PC: enviar request con `logprobs` y `top_logprobs` y revisar respuesta/errores.
3) Esquema de logprobs en `/generate` (`return_logprob`, `top_logprobs_num`, `meta_info.output_top_logprobs`).
   Como verificar en el PC: llamar a `/generate` con esos campos y validar la respuesta.

## C) Decisiones pendientes de validacion (SIN FUENTE)
- Umbral de entropia 0.6: requiere experimento controlado.
  Como verificar en el PC: correr un set de prompts y correlacionar `entropy_norm` con calidad.
