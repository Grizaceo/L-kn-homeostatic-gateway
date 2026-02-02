# L-kn Gateway Homeostático

**Homeostatic Inference Gateway** for local LLM deployment on RTX 4060 (8GB VRAM).

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

1. **Probe**: Query engine with `max_tokens=1` + `top_logprobs_num=10`
2. **Entropy Analysis**: Calculate normalized Shannon entropy: `H = -Σ p·log(p)`
3. **Mode Selection**:
   - **FLUIDO** (H < 0.6): Low uncertainty → pass-through
   - **ANALÍTICO** (H ≥ 0.6): High uncertainty → inject verification prompt

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

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp config/.env.example config/.env
# Edit config/.env if needed (optional)
```

### 4. Make Scripts Executable

```bash
chmod +x scripts/*.sh
chmod +x tests/*.sh
```

## Usage

### Quick Start

```bash
# Start the entire stack
bash scripts/start_all.sh
```

This will:
1. Validate GPU and environment
2. Start SGLang engine (downloads model on first run)
3. Start L-kn gateway
4. Start Open WebUI
5. Run smoke tests

**Access**: Open http://localhost:3000 in your browser

### Individual Components

```bash
# Start only engine
bash scripts/start_engine.sh

# Start only gateway (requires engine running)
cd src && python3 l_kn_gateway.py

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
| `LKN_MODE` | `homeostatic` | `homeostatic` or `passthrough` |

## Testing

```bash
# Run all smoke tests
bash tests/smoke_test.sh
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
