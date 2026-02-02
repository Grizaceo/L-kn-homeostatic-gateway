#!/bin/bash
# L-kn Master Startup Script
# Orchestrates the complete stack: Engine → Gateway → UI

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load environment if exists
if [ -f "config/.env" ]; then
    echo "Loading configuration from config/.env"
    set -a
    source config/.env
    set +a
else
    echo "WARNING: config/.env not found."
    echo "  → Using default values."
    echo "  → Tip: Run 'cp config/.env.example config/.env' to configure environment."
fi

echo "==================================="
echo "  L-kn Gateway Homeostático"
echo "  Full System Startup"
echo "==================================="
echo

# 1. Environment validation
echo "=== Step 1: Environment Validation ==="

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi
echo "✓ GPU driver available"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi
echo "✓ Python available"

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found"
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
bash scripts/start_engine.sh
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

cd src
python3 l_kn_gateway.py > ../logs/gateway.log 2>&1 &
GATEWAY_PID=$!
cd ..

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
