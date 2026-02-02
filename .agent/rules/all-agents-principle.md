---
trigger: always_on
---

# L-kn Workspace — Role-Based Operating Model (Always On)

## Cómo se asignan los roles en este proyecto (regla crítica)

En este workspace **NO existe asignación automática de roles** por:
- nombre del archivo,
- nombre del agente,
- modelo (Gemini / Claude).

👉 **El rol SIEMPRE se asigna explícitamente en el prompt de la tarea.**

El agente debe asumir el rol **solo si el prompt comienza explícitamente con**:
> “Rol: <nombre-del-rol>”

Ejemplo válido:
Rol: Architect & Evidence Auditor
Tarea: Revisar la coherencia del gateway y actualizar evidence_log.md



Si el rol **NO está explícitamente declarado**, el agente debe:
- pedir aclaración, o
- asumir el rol más conservador (no escribir código, solo análisis).

---

## Cómo aplicar las reglas de rol

Este workspace contiene múltiples reglas de rol.
Cada regla indica explícitamente:

> “Aplica SOLO si estás actuando en el rol X.  
> Si no estás en ese rol, ignora esta regla.”

El agente debe:
- leer TODAS las Workspace Rules,
- aplicar **solo** las reglas cuyo rol coincida con el declarado en el prompt,
- ignorar el resto.

---

## Reglas no negociables (aplican a TODOS los roles)

1) **No inventar flags, endpoints ni comportamientos del engine.**
   - Si algo no está verificado, escribir: **SIN FUENTE**.
   - Proponer cómo verificar (comando exacto, docs oficiales).

2) **Rigor epistemológico obligatorio**:
   - (A) Soportado por fuente verificable
   - (B) Hipótesis testeable con experimento definido
   - (C) Especulación (no entra a main)

3) **Evidencia trazable**:
   - Decisiones técnicas relevantes → `docs/evidence_log.md`

4) **Seguridad local por defecto**:
   - Bind 127.0.0.1
   - No exponer puertos innecesarios
   - CORS restringido
   - Confirmación explícita antes de comandos destructivos

5) **Reproducibilidad**:
   - `scripts/start_all.sh` debe levantar todo
   - Incluir smoke test funcional

6) **Ownership de archivos**:
   - `/src` → Gateway Core
   - `/scripts`, `docker-compose.yml` → DevOps
   - `/docs` → Architect / Evidence
   - `/tests` → QA

---

## Output esperado (todos los agentes)
- Respuestas claras, accionables y concretas
- Comandos explícitos y rutas claras
- Reportar errores según RSI:
  - Real (OOM, timeout)
  - Simbólico (schema, streaming)
  - Imaginario (UX)