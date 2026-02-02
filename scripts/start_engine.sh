#!/bin/bash
# Start logic
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
enable_strict_mode
load_env

# Defaults (can be overridden by .env loaded above)
ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
ENGINE_MODEL="${ENGINE_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
ENGINE_MEM_FRACTION="${ENGINE_MEM_FRACTION:-0.85}"

log_info "=== L-kn Engine Startup ==="
log_info "Model: $ENGINE_MODEL"
log_info "Host: $ENGINE_HOST:$ENGINE_PORT"
log_info "Memory Fraction: $ENGINE_MEM_FRACTION"

# 0. Check if already running
if [ -f "$PID_FILE_ENGINE" ]; then
    OLD_PID=$(cat "$PID_FILE_ENGINE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        log_error "Engine appears to be already running (PID: $OLD_PID)."
        log_error "Run 'bash scripts/stop_engine.sh' first."
        exit 1
    else
        log_warn "Found stale PID file. Removing."
        rm -f "$PID_FILE_ENGINE"
    fi
fi

# 1. Verify GPU
echo -n "Checking GPU... "
if ! command -v nvidia-smi &>/dev/null; then
    log_warn "nvidia-smi not found. Assuming CPU-only or driver issue."
    # We exit if strict requirements, but user might want to try?
    # Spec says "Expected RTX 4060", so we should fail if no GPU? 
    # But let's fail fast as requested by user rules for "Runtime Orchestration"
    # User rule: "Reproducibilidad: start_all.sh debe levantar todo"
    # If no GPU, it won't work well.
    log_error "NVIDIA driver is required for L-kn Engine."
    exit 1
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 || echo "0")
if [ "$GPU_MEM" -lt 7000 ]; then
    log_warn "GPU memory ($GPU_MEM MiB) is less than 8GB. Expected RTX 4060."
fi
echo "OK ($GPU_MEM MiB)"

# 2. Verify Python environment
echo -n "Checking Python environment... "
check_cmd python3
echo "OK"

# 3. Verify SGLang installation
echo -n "Checking SGLang installation... "
if ! python3 -c "import sglang" 2>/dev/null; then
    log_error "SGLang not installed."
    log_error "Install with: pip install 'sglang[all]'"
    exit 1
fi

SGLANG_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
echo "OK (version: $SGLANG_VERSION)"

# 4. Check if port is already in use
if command -v lsof &>/dev/null; then
    if lsof -Pi :"$ENGINE_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_error "Port $ENGINE_PORT is already in use"
        log_error "Stop the existing process or change ENGINE_PORT"
        exit 1
    fi
else
    log_warn "'lsof' not found. Skipping port check."
fi

# 5. Create log directory
mkdir -p "$LOGS_DIR"

# 6. Launch SGLang server
log_info "Starting SGLang server..."
log_info "Logs: $LOG_FILE_ENGINE"

# Note: Radix cache is ENABLED by default in SGLang
# To disable, add: --disable-radix-cache

python3 -m sglang.launch_server \
    --model-path "$ENGINE_MODEL" \
    --host "$ENGINE_HOST" \
    --port "$ENGINE_PORT" \
    --mem-fraction-static "$ENGINE_MEM_FRACTION" \
    > "$LOG_FILE_ENGINE" 2>&1 &

ENGINE_PID=$!
echo "$ENGINE_PID" > "$PID_FILE_ENGINE"
log_info "Engine PID: $ENGINE_PID"

# 7. Health check with retry
echo -n "Waiting for engine to be ready..."
MAX_WAIT=60
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s "http://$ENGINE_HOST:$ENGINE_PORT/health" >/dev/null 2>&1; then
        echo " READY!"
        log_info "Engine is running at http://$ENGINE_HOST:$ENGINE_PORT"
        exit 0
    fi
    
    # Check if process is still alive
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo " FAILED!"
        log_error "Engine process died. Check logs:"
        tail -n 20 "$LOG_FILE_ENGINE"
        rm -f "$PID_FILE_ENGINE"
        exit 1
    fi
    
    echo -n "."
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

echo " TIMEOUT!"
log_error "Engine did not become ready within ${MAX_WAIT}s"
log_error "Check logs: $LOG_FILE_ENGINE"
# Kill the dangling process if any (though it might be stuck)
kill "$ENGINE_PID" 2>/dev/null
rm -f "$PID_FILE_ENGINE"
exit 1
