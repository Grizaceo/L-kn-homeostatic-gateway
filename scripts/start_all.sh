# Start of logic
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env

echo "==================================="
echo "  L-kn Gateway Homeostático"
echo "  Full System Startup"
echo "==================================="
echo

# 1. Environment validation
echo "=== Step 1: Environment Validation ==="

# Check curl
if ! check_cmd curl; then
    exit 1
fi
echo "✓ curl available"

# Check GPU
if ! check_cmd nvidia-smi; then
    echo "  (Non-GPU environment detected or driver missing)"
    # Don't exit here if we want to allow testing on CPU-only notebooks
    # but the user specified they have an RTX 4060, so normally we'd fail.
    # Logic in start_engine will handle specific engine failures.
fi
echo "✓ GPU driver check complete"

# Check Python
if ! check_cmd python3; then
    exit 1
fi
echo "✓ Python available"

# Check Docker
if ! check_cmd docker; then
    exit 1
fi
echo "✓ Docker available"

 # Check if requirements are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "WARNING: FastAPI not installed. Installing requirements..."
    pip install -r requirements.txt
fi
echo "✓ Python dependencies available"

echo

# 2. Start Engine
echo "=== Step 2: Starting SGLang Engine ==="
bash "$SCRIPT_DIR/start_engine.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start engine"
    exit 1
fi
echo "✓ Engine started"
echo

# 3. Start Gateway
echo "=== Step 3: Starting L-kn Gateway ==="

# Kill existing gateway if running
if [ -f "logs/gateway.pid" ]; then
    OLD_PID=$(cat logs/gateway.pid)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "Stopping existing gateway..."
        kill $OLD_PID
        sleep 2
    fi
fi

# Run gateway from root to ensure relative config paths work or use absolute
PYTHONPATH=. python3 src/l_kn_gateway.py > logs/gateway.log 2>&1 &
GATEWAY_PID=$!

echo $GATEWAY_PID > logs/gateway.pid
echo "Gateway PID: $GATEWAY_PID"

# Wait for gateway to be ready
echo -n "Waiting for gateway..."
MAX_WAIT=30
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        echo " READY!"
        break
    fi
    
    if ! kill -0 $GATEWAY_PID 2>/dev/null; then
        echo " FAILED!"
        echo "Gateway process died. Check logs:"
        tail -n 20 logs/gateway.log
        exit 1
    fi
    
    echo -n "."
    sleep 1
    WAIT_TIME=$((WAIT_TIME + 1))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo " TIMEOUT!"
    echo "Gateway did not start within ${MAX_WAIT}s"
    exit 1
fi

echo "✓ Gateway started"
echo

# 4. Start Open WebUI
echo "=== Step 4: Starting Open WebUI ==="
docker compose up -d

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start Open WebUI"
    exit 1
fi

echo "✓ Open WebUI started"
echo

# 5. Run smoke test
echo "=== Step 5: Running Smoke Test ==="
sleep 3  # Give UI time to fully initialize

bash tests/smoke_test.sh
SMOKE_EXIT=$?

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
echo "  • Engine:  logs/engine.log"
echo "  • Gateway: logs/gateway.log"
echo
echo "Management:"
echo "  • Health:  bash scripts/healthcheck.sh"
echo "  • Shutdown: bash scripts/stop_all.sh"
echo

if [ $SMOKE_EXIT -eq 0 ]; then
    echo "✓ All smoke tests passed"
    echo
    echo "🚀 System is ready! Open http://localhost:3000 in your browser"
else
    echo "⚠ Some smoke tests failed. Check logs for details."
    exit 1
fi
