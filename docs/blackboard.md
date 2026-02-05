# L-kn Blackboard

Estado compartido del proyecto.

## Estado actual del MVP
- Se inicio la aplicacion del plan MVP (2026-02-05).
- Gateway ahora soporta estrategia `rules` (default) y `entropy` (opcional).
- Se agrego telemetria JSONL por request en `logs/telemetry_YYYY-MM-DD.jsonl`.
- Validacion local en PC RTX 4060 sigue pendiente para contrato real con SGLang.

## Riesgos abiertos
- Falta validar contrato real de `/generate` con SGLang en PC RTX 4060.
- Falta medir latencia y hit-rate de RadixAttention con prompts compartidos.

## Proximos pasos (max 5)
1. Correr benchmark de setup (VRAM, TTFT, throughput) en PC RTX 4060.
2. Ejecutar prueba de hit-rate de RadixAttention en trafico real.
3. Evaluar reglas `FLUIDO/ANALITICO` sobre un lote inicial etiquetado.
4. Ajustar umbrales (`LKN_MAX_TOKENS_FLUIDO`, patrones) con telemetria real.
5. Definir dataset minimo de evaluacion para probe v1.

Ultima actualizacion: 2026-02-05
