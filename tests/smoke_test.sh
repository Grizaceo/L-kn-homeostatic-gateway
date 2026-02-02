#!/bin/bash
# L-kn Smoke Tests
# End-to-end validation of the complete system

ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"

PASS_COUNT=0
FAIL_COUNT=0

echo "=== L-kn Smoke Tests ==="
echo

# Helper functions
pass() {
    echo "✓ PASS: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo "✗ FAIL: $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

# Test 1: Engine reachability
echo "Test 1: Engine reachability"
if curl -s -f "http://$ENGINE_HOST:$ENGINE_PORT/health" >/dev/null 2>&1; then
    pass "Engine is reachable"
else
    fail "Engine is not reachable at http://$ENGINE_HOST:$ENGINE_PORT/health"
fi

# Test 2: Gateway reachability
echo "Test 2: Gateway reachability"
HEALTH=$(curl -s "http://$GATEWAY_HOST:$GATEWAY_PORT/health")
if echo "$HEALTH" | grep -q "engine"; then
    pass "Gateway is reachable and returns health status"
else
    fail "Gateway health check failed"
fi

# Test 3: Gateway OpenAI compatibility (non-streaming)
echo "Test 3: Gateway OpenAI API compatibility (non-streaming)"
RESPONSE=$(curl -s -X POST "http://$GATEWAY_HOST:$GATEWAY_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": false
    }' 2>&1)

if echo "$RESPONSE" | grep -q "choices\|content\|message"; then
    pass "Gateway returned valid chat completion response"
else
    fail "Gateway did not return valid response: $RESPONSE"
fi

# Test 4: Gateway streaming support
echo "Test 4: Gateway streaming support"
STREAM_RESPONSE=$(curl -s -X POST "http://$GATEWAY_HOST:$GATEWAY_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "stream": true
    }' 2>&1 | head -n 5)

if echo "$STREAM_RESPONSE" | grep -q "data:"; then
    pass "Gateway supports streaming responses"
else
    fail "Gateway streaming not working properly"
fi

# Test 5: Gateway error handling
echo "Test 5: Gateway error handling (invalid request)"
ERROR_RESPONSE=$(curl -s -X POST "http://$GATEWAY_HOST:$GATEWAY_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "invalid_role", "content": "test"}]
    }' 2>&1)

if echo "$ERROR_RESPONSE" | grep -q "error\|validation"; then
    pass "Gateway properly validates requests"
else
    fail "Gateway did not return error for invalid request"
fi

# Summary
echo
echo "==================================="
echo "  Test Results"
echo "==================================="
echo "PASSED: $PASS_COUNT"
echo "FAILED: $FAIL_COUNT"
echo

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓ All smoke tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
