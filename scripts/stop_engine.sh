#!/bin/bash
# Stop the L-kn Engine reliably using PID file
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
enable_strict_mode

echo "=== L-kn Engine Shutdown ==="

if [ ! -f "$PID_FILE_ENGINE" ]; then
    log_warn "PID file '$PID_FILE_ENGINE' not found. Engine may not be running."
    exit 0
fi

ENGINE_PID=$(cat "$PID_FILE_ENGINE")

# Check if PID is empty
if [ -z "$ENGINE_PID" ]; then
    log_warn "PID file is empty. Removing."
    rm -f "$PID_FILE_ENGINE"
    exit 0
fi

# Check if process is running
if kill -0 "$ENGINE_PID" 2>/dev/null; then
    log_info "Stopping engine (PID: $ENGINE_PID)..."
    kill "$ENGINE_PID"
    
    # Wait for graceful shutdown (max 20s)
    WAIT_TIME=0
    MAX_WAIT=20
    while kill -0 "$ENGINE_PID" 2>/dev/null && [ $WAIT_TIME -lt $MAX_WAIT ]; do
        sleep 1
        WAIT_TIME=$((WAIT_TIME + 1))
        echo -n "."
    done
    echo ""
    
    # Force kill if still running
    if kill -0 "$ENGINE_PID" 2>/dev/null; then
        log_warn "Engine did not stop gracefully. Force killing..."
        kill -9 "$ENGINE_PID"
    fi
    
    log_info "Engine stopped."
else
    log_warn "Engine process $ENGINE_PID not running (stale PID file)."
fi

# Clean up
rm -f "$PID_FILE_ENGINE"
