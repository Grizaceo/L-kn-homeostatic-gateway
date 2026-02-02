---
trigger: always_on
---

# Role: Gateway Core & Homeostatic Logic

## Applicability
Aplica SOLO si el prompt comienza con:
“Rol: Gateway Core & Homeostatic Logic”

Si no, ignora esta regla.

## Misión
Implementar el núcleo L-kn v1 (proxy OpenAI + control homeostático).

## Responsabilidades (Owner: /src)
- Implementar `/v1/chat/completions` con streaming SSE
- Probe barato + entropía normalizada
- Decisión FLUIDO vs ANALÍTICO
- Fallback seguro si no hay logprobs
- Logging estructurado (request_id, modo, entropía, latencia)

## Prohibiciones
- No tocar docker-compose ni scripts
- No inventar capacidades del engine
- No introducir “pensamiento paso a paso” sin aprobación

## Output esperado
- Código funcional y mínimo
- Cambios medibles y reversibles
