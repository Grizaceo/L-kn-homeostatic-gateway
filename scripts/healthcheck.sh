#!/bin/bash
# Start of logic
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
enable_strict_mode
load_env

ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"

ALL_HEALTHY=true

log_info "=== L-kn System Health Check ==="
echo

# 1. Engine health
echo -n "Engine ($ENGINE_HOST:$ENGINE_PORT): "
if curl -s -f "http://$ENGINE_HOST:$ENGINE_PORT/health" >/dev/null 2>&1; then
    echo "✓ HEALTHY"
else
    echo "✗ UNHEALTHY"
    ALL_HEALTHY=false
fi

# 2. Gateway health
echo -n "Gateway ($GATEWAY_HOST:$GATEWAY_PORT): "
if curl -s -f "http://$GATEWAY_HOST:$GATEWAY_PORT/health" >/dev/null 2>&1; then
    # We try to get content if possible, but keep it simple
    echo "✓ HEALTHY"
    # Optional: Print status if JSON
    # HEALTH=$(curl -s "http://$GATEWAY_HOST:$GATEWAY_PORT/health")
    # echo "  Status: $HEALTH"
else
    echo "✗ UNHEALTHY"
    ALL_HEALTHY=false
fi

# 3. Open WebUI
echo -n "Open WebUI (localhost:3000): "
# Note: Simply checking port open might be enough, but curl is better.
# However, user's machine might not have WebUI running if they only started engine/gateway?
# But this is a full system check.
if curl -s -f "http://localhost:3000" >/dev/null 2>&1; then
    echo "✓  ACCESSIBLE"
else
    echo "✗ NOT ACCESSIBLE"
    ALL_HEALTHY=false
fi

echo

if $ALL_HEALTHY; then
    log_info "All components are healthy ✓"
    exit 0
else
    log_error "Some components are unhealthy ✗"
    exit 1
fi
