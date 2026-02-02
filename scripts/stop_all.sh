#!/bin/bash
# L-kn System Shutdown Script

set -e

echo "=== L-kn System Shutdown ==="

# 1. Stop Open WebUI
echo "Stopping Open WebUI..."
docker compose down
echo "✓ Open WebUI stopped"

# 2. Stop Gateway
if [ -f "logs/gateway.pid" ]; then
    GATEWAY_PID=$(cat logs/gateway.pid)
    if kill -0 $GATEWAY_PID 2>/dev/null; then
        echo "Stopping gateway (PID: $GATEWAY_PID)..."
        kill $GATEWAY_PID
        sleep 2
        
        if kill -0 $GATEWAY_PID 2>/dev/null; then
            kill -9 $GATEWAY_PID
        fi
        echo "✓ Gateway stopped"
    fi
    rm -f logs/gateway.pid
fi

# 3. Stop Engine
echo "Stopping engine..."
bash scripts/stop_engine.sh
echo "✓ Engine stopped"

echo
echo "All services stopped."
