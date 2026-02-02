#!/bin/bash
# L-kn Gateway Homeostático - Full System Startup
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
enable_strict_mode
load_env

echo "==================================="
echo "  L-kn Gateway Homeostático"
echo "  Full System Startup"
echo "==================================="
echo

# 1. Environment validation
log_info "=== Step 1: Environment Validation ==="

check_cmd curl
check_cmd python3
check_cmd docker

# GPU Check (Optional for script flow, but strictly checked in start_engine)
if command -v nvidia-smi &>/dev/null; then
    echo "✓ GPU driver detected"
else
    log_warn "No NVIDIA driver detected. Engine start might fail if not CPU-only."
fi

# Check Python requirements
# We do a quick check for a key package
if ! python3 -c "import fastapi" 2>/dev/null; then
    log_warn "FastAPI not installed. Installing requirements..."
    pip install -r requirements.txt
    echo "✓ Python dependencies installed"
else
    echo "✓ Python dependencies available"
fi

echo

# 2. Start Engine
log_info "=== Step 2: Starting SGLang Engine ==="
# We allow start_engine.sh to handle its own logic. 
# Because of 'set -e', if it fails, we exit immediately.
bash "$SCRIPT_DIR/start_engine.sh"
echo "✓ Engine started"
echo

# 3. Start Gateway
log_info "=== Step 3: Starting L-kn Gateway ==="

# Check/Kill existing gateway
if [ -f "$PID_FILE_GATEWAY" ]; then
    OLD_PID=$(cat "$PID_FILE_GATEWAY")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        log_info "Stopping existing gateway (PID: $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    else
        # Stale PID file
        rm -f "$PID_FILE_GATEWAY"
    fi
fi

# Run gateway
# We use nohup or just background to keep it alive? Original used `&`.
# We strictly log to file.
log_info "Logs: $LOG_FILE_GATEWAY"
PYTHONPATH=. python3 src/l_kn_gateway.py > "$LOG_FILE_GATEWAY" 2>&1 &
GATEWAY_PID=$!

echo "$GATEWAY_PID" > "$PID_FILE_GATEWAY"
log_info "Gateway PID: $GATEWAY_PID"

# Wait for gateway to be ready
echo -n "Waiting for gateway..."
MAX_WAIT=30
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        echo " READY!"
        break
    fi
    
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
        echo " FAILED!"
        log_error "Gateway process died. Check logs:"
        tail -n 20 "$LOG_FILE_GATEWAY"
        # Cleanup
        rm -f "$PID_FILE_GATEWAY"
        exit 1
    fi
    
    echo -n "."
    sleep 1
    WAIT_TIME=$((WAIT_TIME + 1))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo " TIMEOUT!"
    log_error "Gateway did not start within ${MAX_WAIT}s"
    # Kill if stuck
    kill "$GATEWAY_PID" 2>/dev/null
    rm -f "$PID_FILE_GATEWAY"
    exit 1
fi

echo "✓ Gateway started"
echo

# 4. Start Open WebUI
log_info "=== Step 4: Starting Open WebUI ==="
docker compose up -d

# Docker exit code check happens via set -e? 
# docker compose up -d usually returns 0 if containers started or are running.
echo "✓ Open WebUI started"
echo

# 5. Run smoke test
log_info "=== Step 5: Running Smoke Test ==="
sleep 3  # Give UI time to fully initialize

# We temporarily simple disable strict mode for smoke test if we want to handle failure gracefully
set +e
bash tests/smoke_test.sh
SMOKE_EXIT=$?
set -e

echo

# 6. Display summary
echo "==================================="
echo "  L-kn Gateway - System Ready"
echo "==================================="
echo
echo "Services:"
echo "  • Engine:     http://127.0.0.1:30000"
echo "  • Gateway:    http://127.0.0.1:8000"
echo "  • Open WebUI: http://localhost:3000"
echo
echo "Logs:"
echo "  • Engine:  $LOG_FILE_ENGINE"
echo "  • Gateway: $LOG_FILE_GATEWAY"
echo
echo "Management:"
echo "  • Health:  bash scripts/healthcheck.sh"
echo "  • Shutdown: bash scripts/stop_all.sh"
echo

if [ $SMOKE_EXIT -eq 0 ]; then
    log_info "✓ All smoke tests passed"
    echo
    echo "🚀 System is ready! Open http://localhost:3000 in your browser"
else
    log_error "⚠ Some smoke tests failed. Check logs for details."
    exit 1
fi
