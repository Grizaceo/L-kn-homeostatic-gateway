#!/bin/bash
# L-kn Engine Startup Script
# Launches SGLang server with verified flags for RTX 4060 (8GB VRAM)

set -e

# Configuration
ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
ENGINE_MODEL="${ENGINE_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
ENGINE_MEM_FRACTION="${ENGINE_MEM_FRACTION:-0.85}"
LOG_FILE="logs/engine.log"

echo "=== L-kn Engine Startup ==="
echo "Model: $ENGINE_MODEL"
echo "Host: $ENGINE_HOST:$ENGINE_PORT"
echo "Memory Fraction: $ENGINE_MEM_FRACTION"

# 1. Verify GPU
echo -n "Checking GPU... "
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

if [ "$GPU_MEM" -lt 7000 ]; then
    echo "WARNING: GPU memory is less than 8GB. Expected RTX 4060."
fi

# 2. Verify Python environment
echo -n "Checking Python environment... "
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi
echo "OK"

# 3. Verify SGLang installation
echo -n "Checking SGLang installation... "
if ! python3 -c "import sglang" 2>/dev/null; then
    echo "ERROR: SGLang not installed."
    echo "Install with: pip install 'sglang[all]'"
    exit 1
fi

SGLANG_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
echo "OK (version: $SGLANG_VERSION)"

# 4. Check if port is already in use
if lsof -Pi :$ENGINE_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "ERROR: Port $ENGINE_PORT is already in use"
    echo "Stop the existing process or change ENGINE_PORT"
    exit 1
fi

# 5. Create log directory
mkdir -p logs

# 6. Launch SGLang server
echo "Starting SGLang server..."
echo "Logs: $LOG_FILE"

# Note: Radix cache is ENABLED by default in SGLang
# To disable, add: --disable-radix-cache

python3 -m sglang.launch_server \
    --model-path "$ENGINE_MODEL" \
    --host "$ENGINE_HOST" \
    --port "$ENGINE_PORT" \
    --mem-fraction-static "$ENGINE_MEM_FRACTION" \
    > "$LOG_FILE" 2>&1 &

ENGINE_PID=$!
echo $ENGINE_PID > logs/engine.pid
echo "Engine PID: $ENGINE_PID"

# 7. Health check with retry
echo -n "Waiting for engine to be ready..."
MAX_WAIT=60
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s "http://$ENGINE_HOST:$ENGINE_PORT/health" >/dev/null 2>&1; then
        echo " READY!"
        echo "Engine is running at http://$ENGINE_HOST:$ENGINE_PORT"
        exit 0
    fi
    
    # Check if process is still alive
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        echo " FAILED!"
        echo "Engine process died. Check logs:"
        tail -n 20 "$LOG_FILE"
        exit 1
    fi
    
    echo -n "."
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

echo " TIMEOUT!"
echo "Engine did not become ready within ${MAX_WAIT}s"
echo "Check logs: $LOG_FILE"
exit 1
