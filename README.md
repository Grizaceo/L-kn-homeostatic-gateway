# L-kn Gateway Homeostático

**Homeostatic Inference Gateway** for local LLM deployment on RTX 4060 (8GB VRAM).

## 🛸 Antigravity Operating Model
Este proyecto utiliza el marco **Antigravity** para su desarrollo, bajo un modelo de **Roles Explícitos**.

### 1. ¿Qué es Antigravity?
Es el sistema operativo de agentes que rige este workspace.
- **Workspace Rules**: Reglas de oro (archivos `.md`) que dictan cómo se escribe código, se gestiona la seguridad y se documenta la evidencia.
- **"Always On"**: Estas reglas están siempre activas para garantizar rigor epistemológico y seguridad local (bind 127.0.0.1).
- **Asignación de Roles**: Para evitar ambigüedad, el agente solo activa sus capacidades de escritura si se le asigna un rol en el prompt.

### 2. Cómo iniciar una tarea correctamente
Para que el agente actúe con autoridad, el prompt **DEBE** comenzar con la declaración del rol.

**Ejemplo de Prompt:**
> "Rol: Gateway Core & Homeostatic Logic"
> Tarea: Ajustar el umbral de entropía en el sensor homeostático.

*Si no se declara un rol, el agente se mantendrá en modo observador (análisis sin cambios en archivos).*

### Regla Práctica
> **Nuevo rol o nueva tarea ⇒ Nuevo chat**
>
> Para mantener contexto limpio y evitar conflictos de ownership, se recomienda
> iniciar un chat nuevo cuando:
> - Cambias de rol (ej: de DevOps a Gateway Core)
> - Inicias una tarea no relacionada con la anterior

### 3. Roles Disponibles
- **Architect & Evidence Auditor**: Dueño de `/docs`. Vela por la coherencia y la trazabilidad.
- **Gateway Core & Homeostatic Logic**: Dueño de `/src`. Implementa la lógica de proxy y streaming.
- **DevOps & Runtime Orchestration**: Dueño de `/scripts` y `docker-compose.yml`. Garantiza la reproducibilidad.

### 4. Flujo de Trabajo
1. **Declarar Rol** al inicio del prompt.
2. **Ejecutar Tarea** (Implementación / Debug).
3. **Actualizar Evidencia** en `docs/evidence_log.md` (si hay decisiones técnicas).
4. **Verificar** con `bash tests/smoke_test.sh`.

---

## Architecture

L-kn implements a **Gateway Pattern** for adaptive inference:

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│  Open WebUI │ ───> │  L-kn Gateway│ ───> │  SGLang Engine │
│ (localhost  │      │  (FastAPI    │      │  (Qwen 2.5 3B) │
│    :3000)   │ <─── │   :8000)     │ <─── │     :30000     │
└─────────────┘      └──────────────┘      └────────────────┘
                             │
                             ├─ Entropy Probe
                             ├─ Mode Decision
                             └─ Intervention (if needed)
```

### Components

1. **Engine (SGLang)**: High-performance inference server with RadixAttention caching
2. **Gateway (FastAPI)**: OpenAI-compatible proxy with homeostatic logic
3. **UI (Open WebUI)**: Standard chat interface

### Homeostatic Operation

The gateway intercepts requests and makes adaptive decisions:

1. **Rules Probe (default)**: Cheap request-level signals (length, code/math patterns, multi-step cues)
2. **Optional Entropy Probe**: Engine logprob probe (`max_new_tokens=1`, `top_logprobs_num=K`) when `LKN_DECISION_STRATEGY=entropy`
3. **Mode Selection**:
   - **FLUIDO**: Low-risk request, pass-through
   - **ANALITICO**: High-risk request, inject verification prompt

## Requirements

### System
- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or equivalent
- **OS**: WSL2 (Ubuntu) or native Linux
- **Driver**: NVIDIA driver 525+ with CUDA 11.8+

### Software
- Python 3.10+
- Docker & Docker Compose
- curl (for tests)

## Installation

### 1. Install SGLang

```bash
# In WSL2/Linux
pip install "sglang[all]"

# Verify installation
python3 -c "import sglang; print(sglang.__version__)"
```

### 2. Install Python Dependencies

It is highly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp config/.env.example config/.env
# Edit config/.env to customize ports or model
```

### 4. Make Scripts Executable

```bash
chmod +x scripts/*.sh
chmod +x tests/*.sh
```

## Usage

### Quick Start

```bash
# Start the entire stack from the project root
bash scripts/start_all.sh
```

This will:
1. Validate GPU, Docker, and utilities (like curl)
2. Load configuration from `config/.env`
3. Start SGLang engine (downloads model on first run)
4. Start L-kn gateway (using PYTHONPATH for reliability)
5. Start Open WebUI via Docker Compose
6. Run smoke tests

**Access**: Open http://localhost:3000 in your browser

### Individual Components

All components should be started from the **root directory**:

```bash
# Start only engine
bash scripts/start_engine.sh

# Start only gateway (requires engine running)
PYTHONPATH=. python3 src/l_kn_gateway.py

# Start only UI (requires gateway running)
docker compose up -d
```

### Monitoring

```bash
# Check health of all components
bash scripts/healthcheck.sh

# View logs
tail -f logs/engine.log
tail -f logs/gateway.log
```

### Shutdown

```bash
# Stop everything
bash scripts/stop_all.sh
```

## Configuration

All settings are in [`config/.env`](config/.env.example):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE_MODEL` | `Qwen/Qwen2.5-3B-Instruct-AWQ` | HuggingFace model ID |
| `ENGINE_MEM_FRACTION` | `0.85` | GPU memory allocation (0.0-1.0) |
| `LKN_ENTROPY_THRESHOLD` | `0.6` | Decision threshold for mode switch |
| `LKN_PROBE_TOP_K` | `10` | Number of top logprobs for entropy |
| `LKN_DECISION_STRATEGY` | `rules` | `rules` (default) or `entropy` |
| `LKN_MAX_TOKENS_FLUIDO` | `150` | Long prompts above this value route to `ANALITICO` in rules mode |
| `LKN_MODE` | `homeostatic` | `homeostatic` or `passthrough` |

## Testing

```bash
# Run all smoke tests
bash tests/smoke_test.sh
```

### Running unit tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

Tests validate:
- ✓ Engine reachability
- ✓ Gateway OpenAI API compatibility
- ✓ Streaming support
- ✓ Error handling

## Troubleshooting

### Engine won't start

```bash
# Check GPU
nvidia-smi

# Check SGLang installation
python3 -m sglang.launch_server --help

# View engine logs
tail -n 50 logs/engine.log
```

### Gateway errors

```bash
# Check if engine is running
curl http://127.0.0.1:30000/health

# View gateway logs
tail -n 50 logs/gateway.log
```

### Out of Memory

Reduce memory fraction in `config/.env`:
```bash
ENGINE_MEM_FRACTION=0.75  # Down from 0.85
```

Or use smaller model:
```bash
ENGINE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

### Circuit Breaker Open

The gateway opens circuit breaker after 5 consecutive engine failures:

```bash
# Wait 30s for automatic recovery, or restart engine
bash scripts/stop_engine.sh
bash scripts/start_engine.sh
```


### Clean Shutdown / PID Lock

The scripts use `logs/*.pid` to track processes. If the engine fails to start due to a "stale PID file":

1. Try running stop script to clean up (it handles stale locks):
   ```bash
   bash scripts/stop_engine.sh
   ```
2. If that fails or loop persists, manually remove the lock:
   ```bash
   rm logs/engine.pid
   ```

## Architecture Details

### Verified SGLang Flags

Based on research, we use:
- `--mem-fraction-static`: Controls KV cache memory allocation
- `--host` / `--port`: Bind address
- `--model-path`: HuggingFace model ID
- RadixAttention **enabled by default** (no flag needed)

### Entropy Calculation

```python
# 1. Get top-K logprobs from engine
logprobs = [(token, logprob), ...]

# 2. Convert to probabilities
probs = exp(logprobs) / sum(exp(logprobs))

# 3. Shannon entropy
H = -sum(p * log2(p))

# 4. Normalize by max entropy
H_norm = H / log2(K)  # Range: [0, 1]
```

### Intervention Strategy (MVP)

For ANALÍTICO mode, we inject a system prompt:

> *"Think carefully, verify your assumptions, and reason step-by-step before answering."*

## Development

```bash
# Project structure
L-kn Gateway homeostatico/
├── src/
│   ├── l_kn_gateway.py      # Main FastAPI app
│   ├── homeostatic.py       # Decision logic
│   ├── engine_client.py     # SGLang client
│   └── utils.py             # Logging, errors
├── scripts/
│   ├── start_all.sh         # Orchestration
│   ├── start_engine.sh      # Engine launcher
│   ├── stop_all.sh          # Shutdown
│   └── healthcheck.sh       # Health checks
├── tests/
│   └── smoke_test.sh        # E2E tests
├── config/
│   └── .env.example         # Configuration template
├── docker-compose.yml       # Open WebUI
└── requirements.txt         # Python deps
```

## License

MIT

## Research Notes

See [`docs/evidence_log.md`](docs/evidence_log.md) for detailed research findings on SGLang capabilities and limitations.
