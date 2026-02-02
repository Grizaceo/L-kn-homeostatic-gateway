# L-kn Blackboard

Estado compartido del proyecto.

## Estado actual del MVP
- El repositorio incluye gateway, logica homeostatica y scripts de runtime.
- La validacion local en PC (RTX 4060) sigue pendiente.
- La evidencia tecnica esta en docs/evidence_log.md (con items verificados y pendientes).

## Riesgos abiertos
- Dependencias del engine (flags y parametros) sin verificacion local.
- Falta de tests unitarios offline (solo smoke tests E2E).
- Comportamiento de streaming SSE ante fallos requiere validacion en PC.

## Proximos pasos (max 5)
1. Verificar flags reales del engine con `python3 -m sglang.launch_server --help`.
2. Ejecutar `tests/smoke_test.sh` en el PC.
3. Definir y correr tests unitarios offline (pytest) para homeostatica y utilidades.
4. Calibrar el umbral de entropia con un conjunto de prompts controlado.
5. Actualizar docs/evidence_log.md con resultados verificables.

Ultima actualizacion: 2026-02-02
