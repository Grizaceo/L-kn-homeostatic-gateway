#!/bin/bash
# Stop all L-kn services
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
enable_strict_mode

log_info "=== L-kn System Shutdown ==="

# 1. Stop Open WebUI
echo "Stopping Open WebUI..."
# We don't fail if docker compose isn't running or not installed (common.sh check handles it but we might be stopping on a machine without it running)
# However, if docker command exists, we run it.
if command -v docker &>/dev/null; then
    docker compose down || true
    echo "✓ Open WebUI stopped"
else
    log_warn "Docker not found, skipping Open WebUI shutdown"
fi

# 2. Stop Gateway
if [ -f "$PID_FILE_GATEWAY" ]; then
    GATEWAY_PID=$(cat "$PID_FILE_GATEWAY")
    
    if [ -n "$GATEWAY_PID" ] && kill -0 "$GATEWAY_PID" 2>/dev/null; then
        log_info "Stopping gateway (PID: $GATEWAY_PID)..."
        kill "$GATEWAY_PID"
        
        # Simple wait loop
        WAIT_TIME=0
        while kill -0 "$GATEWAY_PID" 2>/dev/null && [ $WAIT_TIME -lt 5 ]; do
            sleep 1
            WAIT_TIME=$((WAIT_TIME + 1))
        done
        
        if kill -0 "$GATEWAY_PID" 2>/dev/null; then
            log_warn "Force killing gateway..."
            kill -9 "$GATEWAY_PID"
        fi
        echo "✓ Gateway stopped"
    else
        echo "Gateway not running or stale PID."
    fi
    rm -f "$PID_FILE_GATEWAY"
else
     echo "Gateway PID file not found."
fi

# 3. Stop Engine
echo "Stopping engine..."
bash "$SCRIPT_DIR/stop_engine.sh"
echo "✓ Engine stopped"

echo
log_info "All services stopped."
