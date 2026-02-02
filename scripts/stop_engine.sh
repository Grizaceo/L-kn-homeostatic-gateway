# Start of logic
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "=== L-kn Engine Shutdown ==="

if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found. Engine may not be running."
    exit 0
fi

ENGINE_PID=$(cat "$PID_FILE")

if kill -0 $ENGINE_PID 2>/dev/null; then
    echo "Stopping engine (PID: $ENGINE_PID)..."
    kill $ENGINE_PID
    
    # Wait for graceful shutdown
    WAIT_TIME=0
    while kill -0 $ENGINE_PID 2>/dev/null && [ $WAIT_TIME -lt 10 ]; do
        sleep 1
        WAIT_TIME=$((WAIT_TIME + 1))
    done
    
    # Force kill if still running
    if kill -0 $ENGINE_PID 2>/dev/null; then
        echo "Force stopping engine..."
        kill -9 $ENGINE_PID
    fi
    
    echo "Engine stopped."
else
    echo "Engine process not running (stale PID file)."
fi

rm -f "$PID_FILE"
