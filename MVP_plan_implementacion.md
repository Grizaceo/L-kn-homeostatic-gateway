# MVP: L-kn Gateway Homeostático — Plan de Implementación

**Versión:** 1.0  
**Fecha:** 2026-02-05  
**Hardware:** RTX 4060 8GB · WSL2/Linux · VS Code  
**Principio rector:** Baseline funcional primero, complejidad solo si los datos lo justifican.

---

## 0. Decisiones de diseño ya tomadas (no renegociables)

| Decisión | Justificación | Etiqueta |
|----------|---------------|----------|
| Motor de inferencia: SGLang + RadixAttention | Throughput demostrado 2-5× vs vLLM/Guidance | A |
| Modelo base: Qwen 2.5 3B cuantizado (4-5 bit) | Cabe en 8GB con margen para KV cache | A |
| Gateway: FastAPI proxy OpenAI-compatible | Estándar de industria, streaming, tool-calls | A |
| Señal de incertidumbre v0: entropía de logits | Gratis con la generación, sin overhead extra | A→B |
| Probe v0: clasificador lineal sobre features baratas | Validar antes de añadir complejidad tropical | B |
| Verificación: herramientas externas > probe estadístico | Errores confiados escapan a señales de confianza | A |
| Telemetría: JSON estructurado desde día 1 | Sin datos no hay ciencia | A |

---

## 1. Estructura del proyecto en VS Code

```
lkn-gateway/
├── .vscode/
│   ├── settings.json          # Python path, formatter, etc.
│   ├── launch.json            # Debug configs para gateway y scripts
│   └── tasks.json             # Tareas: levantar SGLang, correr benchmarks
├── pyproject.toml             # uv/pip, dependencias
├── README.md
│
├── infra/                     # Fase 1: levantar el motor
│   ├── sglang_launcher.sh     # Script para levantar SGLang server
│   ├── verify_setup.py        # Smoke test: VRAM, throughput, latencia
│   └── radix_hit_test.py      # Medir hit-rate de RadixAttention
│
├── gateway/                   # Fase 2: proxy OpenAI-compatible
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   ├── router.py              # Lógica FLUIDO vs ANALÍTICO
│   ├── probe.py               # Probe v0 (reglas) → v1 (clasificador lineal)
│   ├── telemetry.py           # Logger JSON estructurado
│   ├── schemas.py             # Pydantic models (request/response/telemetría)
│   └── config.py              # Umbrales, model_id, rutas
│
├── eval/                      # Fase 3: evaluación rigurosa
│   ├── datasets/
│   │   ├── README.md          # Documentación de cada dataset
│   │   ├── prepare_triviaqa.py
│   │   └── prepare_hotpotqa.py
│   ├── run_baseline.py        # Baseline: siempre ANALÍTICO
│   ├── run_probe_v0.py        # Evaluar probe con reglas
│   ├── run_probe_v1.py        # Evaluar probe clasificador lineal
│   ├── run_checkpoint.py      # Pipeline multi-paso con checkpoints
│   ├── metrics.py             # AUC, accuracy, latencia, error amplification
│   └── analyze_results.py     # Notebooks/scripts de análisis
│
├── experiments/               # Resultados versionados
│   ├── 001_setup_verification/
│   ├── 002_radix_hitrate/
│   ├── 003_probe_v0_rules/
│   ├── 004_probe_v1_linear/
│   └── 005_multi_step_checkpoint/
│
├── logs/                      # Telemetría JSON (gitignored)
│   └── .gitkeep
│
└── docs/
    ├── analisis_critico.md    # Tu auditoría (este archivo anterior)
    ├── decision_log.md        # Registro de decisiones técnicas
    └── experiment_reports/    # Un .md por experimento completado
```

---

## 2. Configuración del entorno (Día 1)

### 2.1 Dependencias

```toml
# pyproject.toml
[project]
name = "lkn-gateway"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "httpx>=0.28",           # Cliente async para SGLang
    "pydantic>=2.10",
    "numpy>=2.0",
    "scikit-learn>=1.6",     # Para probe v1 (clasificador lineal)
    "pandas>=2.2",           # Para análisis de telemetría
]

[project.optional-dependencies]
eval = [
    "datasets>=3.0",        # HuggingFace datasets
    "matplotlib>=3.9",
    "seaborn>=0.13",
]
```

### 2.2 VS Code settings recomendados

```jsonc
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true
    },
    "python.testing.pytestEnabled": true,
    "files.exclude": {
        "logs/*.jsonl": false  // Visibles para inspección rápida
    }
}
```

### 2.3 VS Code tasks para operación diaria

```jsonc
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "SGLang: Start Server",
            "type": "shell",
            "command": "bash infra/sglang_launcher.sh",
            "isBackground": true,
            "problemMatcher": []
        },
        {
            "label": "Gateway: Run Dev",
            "type": "shell",
            "command": "uvicorn gateway.main:app --reload --port 8000",
            "problemMatcher": []
        },
        {
            "label": "Eval: Run Baseline",
            "type": "shell",
            "command": "python eval/run_baseline.py --output experiments/001_setup_verification/",
            "problemMatcher": []
        }
    ]
}
```

### 2.4 Debug config

```jsonc
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Gateway",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": ["gateway.main:app", "--port", "8000"],
            "env": {"SGLANG_URL": "http://localhost:30000"}
        },
        {
            "name": "Debug Current Eval Script",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "args": ["--output", "experiments/debug/"],
            "console": "integratedTerminal"
        }
    ]
}
```

---

## 3. Fases de implementación

### FASE 1 — Motor de inferencia (Semanas 1–2)

**Objetivo:** SGLang corriendo, medido, y documentado.

#### Experimento 001: Verificación de setup

**Hipótesis:** Qwen 2.5 3B cuantizado (4-bit GPTQ) corre en RTX 4060 con al menos 2K tokens de contexto, throughput ≥ 30 tokens/s, y VRAM total < 7.5GB (dejando margen).

```bash
# infra/sglang_launcher.sh
#!/bin/bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
    --port 30000 \
    --mem-fraction-static 0.80 \
    --dtype float16 \
    --context-length 4096 \
    --log-level info
```

```python
# infra/verify_setup.py
"""
Experimento 001: Verificación de setup
Mide: VRAM, throughput, TTFT, latencia p50/p95
"""
import httpx, time, json, subprocess, statistics

SGLANG_URL = "http://localhost:30000"

def measure_vram() -> dict:
    """nvidia-smi → VRAM usado en MB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    used, total = result.stdout.strip().split(", ")
    return {"vram_used_mb": int(used), "vram_total_mb": int(total)}

def measure_generation(prompt: str, max_tokens: int = 200) -> dict:
    """Una generación, mide TTFT y total."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True
    }
    t0 = time.perf_counter()
    ttft = None
    tokens = 0

    with httpx.Client(timeout=60) as client:
        with client.stream("POST", f"{SGLANG_URL}/v1/chat/completions",
                           json=payload) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    tokens += 1

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "ttft_ms": round(ttft, 1) if ttft else None,
        "total_ms": round(total_ms, 1),
        "tokens_generated": tokens,
        "tokens_per_sec": round(tokens / (total_ms / 1000), 1) if total_ms > 0 else 0,
        "prompt_length": len(prompt.split())
    }

def run_experiment():
    prompts = [
        "¿Cuál es la capital de Francia?",                    # Trivial
        "Explica la fotosíntesis en 3 pasos.",                # Medio
        "Escribe un análisis de 200 palabras sobre el impacto "
        "de la IA en la educación superior.",                  # Largo
    ]

    print("=== Experimento 001: Verificación de Setup ===\n")

    vram = measure_vram()
    print(f"VRAM: {vram['vram_used_mb']}MB / {vram['vram_total_mb']}MB\n")

    results = []
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt[:50]}...")
        # Warmup
        measure_generation(prompt, max_tokens=10)
        # Medición real (3 repeticiones)
        reps = [measure_generation(prompt) for _ in range(3)]
        avg = {
            "prompt": prompt[:50],
            "ttft_ms_p50": statistics.median([r["ttft_ms"] for r in reps]),
            "total_ms_p50": statistics.median([r["total_ms"] for r in reps]),
            "tokens_per_sec_avg": statistics.mean([r["tokens_per_sec"] for r in reps]),
        }
        results.append(avg)
        print(f"  TTFT p50: {avg['ttft_ms_p50']}ms")
        print(f"  Throughput: {avg['tokens_per_sec_avg']:.1f} tok/s\n")

    # Guardar resultados
    output = {"vram": vram, "results": results}
    with open("experiments/001_setup_verification/results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Resultados guardados.")

if __name__ == "__main__":
    run_experiment()
```

**Criterio GO/NO-GO:**

| Métrica | Éxito (GO) | Fallo (replantear) |
|---------|------------|-------------------|
| VRAM total | < 7.5 GB | ≥ 7.5 GB → probar 2-bit o modelo más pequeño |
| Throughput | ≥ 30 tok/s | < 20 tok/s → evaluar llama.cpp como alternativa |
| TTFT p50 | < 500 ms | > 1000 ms → investigar chunked-prefill |

#### Experimento 002: RadixAttention hit-rate

**Hipótesis:** Con un system prompt compartido de 500+ tokens, RadixAttention logra ≥ 80% cache hit-rate en requests consecutivos, reduciendo TTFT p50 en ≥ 30%.

```python
# infra/radix_hit_test.py
"""
Experimento 002: RadixAttention hit-rate
Compara TTFT con prefijo compartido vs prefijo variable.
"""
import httpx, time, json, statistics

SGLANG_URL = "http://localhost:30000"
SYSTEM_PROMPT = (
    "Eres un asistente experto en análisis de datos. "
    "Tu tarea es responder de forma precisa y concisa. "
    "Siempre verifica la información antes de responder. "
    # ... extender hasta ~500 tokens para maximizar el beneficio de cache
    "Responde en español." + " contexto adicional." * 80
)

QUERIES_SHARED_PREFIX = [
    "¿Qué es la desviación estándar?",
    "¿Cómo interpreto un p-value de 0.03?",
    "¿Cuándo uso regresión logística vs lineal?",
    "Explica qué es un intervalo de confianza.",
    "¿Qué significa overfitting?",
] * 4  # 20 queries total, mismo system prompt

QUERIES_VARIABLE_PREFIX = [
    (f"Eres un experto en {domain}. Responde brevemente.", query)
    for domain, query in [
        ("estadística", "¿Qué es la media?"),
        ("física", "¿Qué es la gravedad?"),
        ("historia", "¿Quién fue Julio César?"),
        ("biología", "¿Qué es el ADN?"),
        ("economía", "¿Qué es la inflación?"),
    ] * 4
]

def measure_ttft(system: str, user: str) -> float:
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 50,
        "stream": True
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=30) as client:
        with client.stream("POST", f"{SGLANG_URL}/v1/chat/completions",
                           json=payload) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    return (time.perf_counter() - t0) * 1000
    return -1

def run_experiment():
    print("=== Experimento 002: RadixAttention Hit-Rate ===\n")

    # Grupo A: prefijo compartido
    print("Grupo A: Prefijo compartido (20 queries)...")
    ttfts_shared = []
    for q in QUERIES_SHARED_PREFIX:
        ttft = measure_ttft(SYSTEM_PROMPT, q)
        ttfts_shared.append(ttft)

    # Grupo B: prefijo variable
    print("Grupo B: Prefijo variable (20 queries)...")
    ttfts_variable = []
    for sys, q in QUERIES_VARIABLE_PREFIX:
        ttft = measure_ttft(sys, q)
        ttfts_variable.append(ttft)

    # Análisis
    # Descartar primera query de cada grupo (cold start)
    shared = ttfts_shared[1:]
    variable = ttfts_variable[1:]

    results = {
        "shared_prefix": {
            "p50_ms": round(statistics.median(shared), 1),
            "p95_ms": round(sorted(shared)[int(len(shared)*0.95)], 1),
            "mean_ms": round(statistics.mean(shared), 1),
        },
        "variable_prefix": {
            "p50_ms": round(statistics.median(variable), 1),
            "p95_ms": round(sorted(variable)[int(len(variable)*0.95)], 1),
            "mean_ms": round(statistics.mean(variable), 1),
        },
        "ttft_reduction_pct": round(
            (1 - statistics.median(shared) / statistics.median(variable)) * 100, 1
        ),
        "raw_shared": [round(t, 1) for t in ttfts_shared],
        "raw_variable": [round(t, 1) for t in ttfts_variable],
    }

    print(f"\nPrefijo compartido TTFT p50: {results['shared_prefix']['p50_ms']}ms")
    print(f"Prefijo variable   TTFT p50: {results['variable_prefix']['p50_ms']}ms")
    print(f"Reducción: {results['ttft_reduction_pct']}%")

    with open("experiments/002_radix_hitrate/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiment()
```

**Criterio:**

| Métrica | Éxito | Fallo |
|---------|-------|-------|
| Reducción TTFT p50 | ≥ 30% | < 15% → el prefijo es demasiado corto o SGLang no cachea bien |

---

### FASE 2 — Gateway + Probe v0 (Semanas 3–4)

**Objetivo:** Gateway funcional con routing por reglas simples, telemetría completa.

#### Gateway mínimo

```python
# gateway/schemas.py
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
import uuid

class Mode(str, Enum):
    FLUIDO = "FLUIDO"
    ANALITICO = "ANALITICO"

class ProbeSignals(BaseModel):
    prompt_tokens_est: int
    has_code: bool
    has_math: bool
    has_multi_step: bool
    session_turn_count: int
    # v1: añadir logit_entropy, gap_top1_top2

class TelemetryRecord(BaseModel):
    request_id: str = ""
    timestamp: str = ""
    mode: Mode = Mode.FLUIDO
    model_id: str = ""
    probe: ProbeSignals | None = None
    latency_ms: dict = {}  # {"ttft": 0, "total": 0}
    tokens_generated: int = 0
    error: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
```

```python
# gateway/probe.py
"""
Probe v0: Reglas simples. Sin ML, sin tropical.
Objetivo: tener un baseline medible antes de sofisticar.
"""
import re
from gateway.schemas import ProbeSignals, Mode
from gateway.config import THRESHOLDS

def extract_signals(messages: list[dict]) -> ProbeSignals:
    """Extrae features baratas del request (sin invocar modelo)."""
    last_user = ""
    turn_count = 0
    for msg in messages:
        if msg["role"] == "user":
            last_user = msg["content"]
            turn_count += 1

    tokens_est = len(last_user.split()) * 1.3  # Heurística rough
    has_code = bool(re.search(r"```|def |class |import |function ", last_user))
    has_math = bool(re.search(r"[=+\-*/^]|\d{3,}|ecuaci|integr|deriv|probab", last_user, re.I))
    has_multi_step = bool(re.search(
        r"paso a paso|step by step|primero.*luego|analiza.*compara|multi.?hop",
        last_user, re.I
    ))

    return ProbeSignals(
        prompt_tokens_est=int(tokens_est),
        has_code=has_code,
        has_math=has_math,
        has_multi_step=has_multi_step,
        session_turn_count=turn_count,
    )

def decide_mode(signals: ProbeSignals) -> Mode:
    """
    v0: Reglas determinísticas.
    Cada regla tiene justificación y es desactivable para ablation.
    """
    # R1: Prompts largos probablemente necesitan más cuidado
    if signals.prompt_tokens_est > THRESHOLDS["max_tokens_fluido"]:
        return Mode.ANALITICO

    # R2: Código y matemáticas tienen mayor riesgo de error sutil
    if signals.has_code or signals.has_math:
        return Mode.ANALITICO

    # R3: Tareas multi-paso requieren verificación
    if signals.has_multi_step:
        return Mode.ANALITICO

    return Mode.FLUIDO
```

```python
# gateway/config.py
"""Configuración centralizada. Cambiar umbrales aquí para ablations."""

SGLANG_URL = "http://localhost:30000"

THRESHOLDS = {
    "max_tokens_fluido": 150,  # Tokens estimados: >150 → ANALÍTICO
}

# Modos de modelo (si tienes múltiples modelos o configs)
MODEL_CONFIGS = {
    "FLUIDO": {
        "max_tokens": 256,
        "temperature": 0.7,
    },
    "ANALITICO": {
        "max_tokens": 1024,
        "temperature": 0.3,
        # Aquí irían: system prompt extendido, tool-calls, etc.
    },
}
```

```python
# gateway/telemetry.py
"""Logger de telemetría: un JSONL por día."""
import json
from pathlib import Path
from datetime import date
from gateway.schemas import TelemetryRecord

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def log_record(record: TelemetryRecord):
    filepath = LOG_DIR / f"{date.today().isoformat()}.jsonl"
    with open(filepath, "a") as f:
        f.write(record.model_dump_json() + "\n")
```

```python
# gateway/main.py
"""
Gateway L-kn v0: FastAPI proxy OpenAI-compatible.
"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx, time, json

from gateway.probe import extract_signals, decide_mode
from gateway.telemetry import log_record
from gateway.schemas import TelemetryRecord, Mode
from gateway.config import SGLANG_URL, MODEL_CONFIGS

app = FastAPI(title="L-kn Gateway", version="0.1.0")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # 1. Probe
    signals = extract_signals(messages)
    mode = decide_mode(signals)

    # 2. Configurar request según modo
    config = MODEL_CONFIGS[mode.value]
    body["max_tokens"] = body.get("max_tokens") or config["max_tokens"]
    body["temperature"] = body.get("temperature") or config["temperature"]

    # 3. Forward a SGLang
    t0 = time.perf_counter()

    async def stream_and_log():
        ttft = None
        tokens = 0
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", f"{SGLANG_URL}/v1/chat/completions",
                json=body
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        if ttft is None and line != "data: [DONE]":
                            ttft = (time.perf_counter() - t0) * 1000
                        tokens += 1
                    yield line + "\n"

        # 4. Telemetría
        total_ms = (time.perf_counter() - t0) * 1000
        record = TelemetryRecord(
            mode=mode,
            model_id="qwen2.5-3b-q4",
            probe=signals,
            latency_ms={"ttft": round(ttft or 0, 1), "total": round(total_ms, 1)},
            tokens_generated=tokens,
        )
        log_record(record)

    if body.get("stream", False):
        return StreamingResponse(stream_and_log(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{SGLANG_URL}/v1/chat/completions", json=body)
            total_ms = (time.perf_counter() - t0) * 1000
            record = TelemetryRecord(
                mode=mode,
                model_id="qwen2.5-3b-q4",
                probe=signals,
                latency_ms={"ttft": 0, "total": round(total_ms, 1)},
            )
            log_record(record)
            return resp.json()
```

---

### FASE 3 — Evaluación del Probe (Semanas 5–6)

**Objetivo:** ¿El probe v0 predice fallos? ¿Vale la pena el routing?

#### Experimento 003: Probe v0 con reglas

**Hipótesis:** El routing por reglas simples reduce latencia media ≥ 20% vs "siempre ANALÍTICO" con degradación de accuracy < 3% absoluto.

```python
# eval/run_baseline.py
"""
Baseline: todas las queries con configuración ANALÍTICO.
Produce: accuracy, latencia media, distribución de errores.
"""
import json, httpx, time
from pathlib import Path

GATEWAY_URL = "http://localhost:8000"

def load_dataset(path: str = "eval/datasets/triviaqa_500.jsonl") -> list[dict]:
    """Cada línea: {"question": "...", "answer": "...", "difficulty": "easy|hard"}"""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def evaluate_single(question: str, ground_truth: str) -> dict:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 256,
        "temperature": 0.1,  # Determinístico para reproducibilidad
    }
    t0 = time.perf_counter()
    resp = httpx.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, timeout=30)
    latency_ms = (time.perf_counter() - t0) * 1000
    answer = resp.json()["choices"][0]["message"]["content"]

    # Evaluación simple: ¿la respuesta contiene el ground truth?
    correct = ground_truth.lower() in answer.lower()

    return {
        "question": question[:80],
        "ground_truth": ground_truth,
        "model_answer": answer[:200],
        "correct": correct,
        "latency_ms": round(latency_ms, 1),
    }

def run(output_dir: str = "experiments/003_probe_v0_rules/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    results = [evaluate_single(d["question"], d["answer"]) for d in dataset]

    accuracy = sum(r["correct"] for r in results) / len(results)
    latency_p50 = sorted([r["latency_ms"] for r in results])[len(results)//2]

    summary = {
        "n": len(results),
        "accuracy": round(accuracy, 4),
        "latency_p50_ms": latency_p50,
        "latency_mean_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
    }

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Latencia p50: {latency_p50:.0f}ms")

if __name__ == "__main__":
    run()
```

#### Experimento 004: Probe v1 — clasificador lineal post-hoc

**Hipótesis:** Un clasificador lineal (logistic regression) entrenado sobre la telemetría del experimento 003 logra AUC ≥ 0.70 para predecir fallo del modelo.

```python
# eval/run_probe_v1.py
"""
Entrena un probe clasificador lineal sobre telemetría pasada.
Features: las de ProbeSignals.
Target: 1 si el modelo falló, 0 si acertó.
"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from pathlib import Path

def load_telemetry_and_results(
    telemetry_path: str = "logs/",
    results_path: str = "experiments/003_probe_v0_rules/results.json"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combina telemetría del probe con resultados de evaluación.
    Requiere que ambos estén alineados por request_id o por índice.
    """
    with open(results_path) as f:
        data = json.load(f)

    X, y = [], []
    for r in data["details"]:
        # Features del probe (extraídas de la query)
        tokens_est = len(r["question"].split()) * 1.3
        has_code = 1 if "```" in r["question"] or "def " in r["question"] else 0
        has_math = 1 if any(c in r["question"] for c in "=+*/^") else 0

        X.append([tokens_est, has_code, has_math])
        y.append(0 if r["correct"] else 1)  # 1 = fallo

    return np.array(X), np.array(y)

def run(output_dir: str = "experiments/004_probe_v1_linear/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X, y = load_telemetry_and_results()
    print(f"Dataset: {len(X)} queries, {sum(y)} fallos ({sum(y)/len(y):.1%})")

    # Cross-validated AUC
    clf = LogisticRegression(max_iter=1000)
    aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"AUC (5-fold CV): {aucs.mean():.3f} ± {aucs.std():.3f}")

    # Entrenar final y guardar coeficientes
    clf.fit(X, y)
    coefs = {
        "features": ["tokens_est", "has_code", "has_math"],
        "coefficients": clf.coef_[0].tolist(),
        "intercept": clf.intercept_[0],
        "auc_mean": round(aucs.mean(), 4),
        "auc_std": round(aucs.std(), 4),
    }

    with open(f"{output_dir}/probe_v1_model.json", "w") as f:
        json.dump(coefs, f, indent=2)

    print(f"\nCoeficientes: {coefs}")
    if aucs.mean() >= 0.70:
        print("✅ GO: El probe lineal predice fallos con AUC aceptable.")
    else:
        print("⚠️  AUC < 0.70: las features baratas no predicen fallo.")
        print("   → Considerar: añadir logit_entropy como feature (requiere overhead).")

if __name__ == "__main__":
    run()
```

---

### FASE 4 — Multi-paso con checkpoints (Semanas 7–8)

#### Experimento 005: Pipeline con checkpoints de entropía

**Hipótesis:** En una tarea de QA multi-hop (2 pasos: decompose + answer), insertar un checkpoint de entropía de logits entre pasos reduce errores finales ≥ 10% absoluto con overhead ≤ 50%.

```python
# eval/run_checkpoint.py (esqueleto)
"""
Pipeline:
  Paso 1: Descomponer pregunta multi-hop en sub-preguntas
  Checkpoint: Si entropía alta en paso 1 → re-generar con temperatura más baja
  Paso 2: Responder sub-preguntas → sintetizar respuesta final
"""
# Implementar después de que Fase 3 confirme que el setup funciona.
# Requiere: acceso a logprobs de SGLang (verificar --logprob-start-len).
```

---

## 4. Dataset de evaluación (preparar en Semana 3)

### Especificación

| Dataset | Queries | Ground truth | Propósito |
|---------|---------|-------------|-----------|
| TriviaQA (subset) | 300 | Respuesta corta exacta | QA factual, fácil de evaluar automáticamente |
| HotpotQA (subset) | 100 | Respuesta + supporting facts | Multi-hop, evalúa pipeline multi-paso |
| Custom "hard-easy" | 100 | Manual (50 fácil, 50 difícil) | Calibrar umbrales del probe |

```python
# eval/datasets/prepare_triviaqa.py
"""Descarga y prepara 300 queries de TriviaQA con ground truth."""
from datasets import load_dataset

def prepare():
    ds = load_dataset("trivia_qa", "rc", split="validation[:300]")
    records = []
    for item in ds:
        records.append({
            "question": item["question"],
            "answer": item["answer"]["value"],
            "aliases": item["answer"]["aliases"],
        })
    # Guardar como JSONL
    import json
    with open("eval/datasets/triviaqa_300.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Preparadas {len(records)} queries.")

if __name__ == "__main__":
    prepare()
```

---

## 5. Decision Log (plantilla)

Mantener en `docs/decision_log.md`:

```markdown
# Decision Log

## DEC-001: Motor de inferencia → SGLang
- **Fecha:** 2026-02-XX
- **Contexto:** Necesitamos inferencia local en RTX 4060 8GB
- **Opciones:** SGLang, vLLM, llama.cpp, TGI
- **Decisión:** SGLang por RadixAttention (reuso KV de prefijos)
- **Resultado del experimento 001:** [pendiente]

## DEC-002: Probe v0 → reglas simples
- **Fecha:** 2026-02-XX
- **Contexto:** Necesitamos routing FLUIDO/ANALÍTICO sin overhead
- **Decisión:** Reglas if/then sobre features baratas (longitud, keywords)
- **Alternativa descartada:** Probe tropical (no validado, overhead incierto)
- **Resultado del experimento 003:** [pendiente]
- **Condición para upgrade a v1:** AUC > 0.70 del clasificador lineal
```

---

## 6. Criterios de transición entre fases

```
FASE 1 → FASE 2:
  ✓ SGLang corriendo con throughput ≥ 30 tok/s
  ✓ VRAM < 7.5 GB medido
  ✓ RadixAttention verificado (hit-rate documentado)

FASE 2 → FASE 3:
  ✓ Gateway responde a /v1/chat/completions
  ✓ Telemetría JSON registrando cada request
  ✓ Dataset de evaluación preparado (≥ 300 queries con ground truth)

FASE 3 → FASE 4:
  ✓ Accuracy de baseline medida
  ✓ Probe v0 evaluado (ratio de escalado documentado)
  ✓ Probe v1 AUC medido (decide si seguir o pivotar)

FASE 4 → Decisión:
  ✓ Pipeline multi-paso evaluado
  ✓ Comparativa completa: sin routing vs v0 vs v1 vs siempre-ANALÍTICO
  ✓ Publicar experiment_report.md con conclusiones
  ✓ DECIDIR: ¿añadir features del modelo (logprobs)? ¿tropical? ¿otro?
```

---

## 7. Lo que NO está en este MVP (y por qué)

| Excluido | Razón | Cuándo reconsiderar |
|----------|-------|-------------------|
| Geometría tropical (probe, KV, métricas) | Sin baseline no se puede medir mejora | Después de Fase 4 si probe v1 tiene AUC < 0.70 |
| Topología Möbius / RSI / Sinthome | Desproporción complejidad/beneficio para un gateway | Proyecto separado de investigación teórica |
| Signed-tropical (s+/s-) | Sin dominio con separación natural pro/contra | Si aparece un caso de verificación de claims |
| Honeycombs / JL / compressed sensing | Especulación sin operacionalización | Publicación teórica futura |
| Entropía semántica | Overhead alto (M muestras) | Si entropía de logits falla en Fase 4 |
| Librería `tropica` | No construir infra sin caso de uso validado | Después de TropiUMAP MVP (proyecto paralelo) |
| Multi-agente complejo | Un solo modelo primero | Después de validar pipeline 2-3 pasos |

---

## 8. Checklist de arranque (Día 1 literal)

```
[ ] Crear repo lkn-gateway con la estructura de arriba
[ ] Instalar SGLang: pip install sglang[all]
[ ] Descargar modelo: Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
[ ] Copiar sglang_launcher.sh, ejecutar, verificar que responde
[ ] Correr verify_setup.py, documentar resultados en experiments/001/
[ ] Commit: "001: setup verification results"
[ ] Si GO → continuar con radix_hit_test.py
[ ] Si NO-GO → documentar por qué y evaluar alternativas
```
