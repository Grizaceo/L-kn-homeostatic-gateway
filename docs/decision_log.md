# Decision Log

## DEC-001: Start MVP with rules-first routing
- Date: 2026-02-05
- Context: `analisis_critico.md` and `MVP_plan_implementacion.md` recommend a cheap request-level baseline before model-internal probes.
- Decision: Make `rules` the default `LKN_DECISION_STRATEGY` and keep `entropy` as optional strategy.
- Why: The rules probe runs before model invocation and avoids extra prefill/probe cost.
- Impact:
  - Added request signal extraction and deterministic FLUIDO/ANALITICO routing.
  - Entropy mode now falls back to rules when probe fails.
  - Gateway now writes structured JSONL telemetry for each request.
