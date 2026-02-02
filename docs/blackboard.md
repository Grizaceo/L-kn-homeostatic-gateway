# L-kn Blackboard

Estado compartido del proyecto.

## Estado actual del MVP
- ✅ Unit tests offline existen y pasan (pytest -q).
- ✅ Streaming: exceptions registran record_failure() en circuit breaker.
- ⏳ Validacion local en PC (RTX 4060) pendiente para contrato real con SGLang (/generate schema, logprobs, VRAM/mem_fraction).

## Riesgos abiertos
- Pendiente validar contrato real con SGLang en PC RTX 4060 (/generate schema, logprobs, VRAM/mem_fraction).

## Proximos pasos (max 5)
1. Validar `/generate` schema real en PC RTX 4060 con SGLang.
2. Verificar `logprobs` end-to-end en PC RTX 4060.
3. Probar VRAM/mem_fraction bajo carga real en PC RTX 4060.

Ultima actualizacion: 2026-02-02
