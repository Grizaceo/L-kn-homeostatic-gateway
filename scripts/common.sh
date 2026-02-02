#!/bin/bash
# L-kn Common Script Utilities
# Centralizes path resolution and environment loading

# 0. Fail Fast Settings (Individual scripts should also set this, but good practice here)
# We don't force it here to avoid breaking sourcing scripts that aren't ready, 
# but we provide a function to enable it.
enable_strict_mode() {
    set -euo pipefail
}

# 1. Path Resolution
# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is one level up from scripts/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 2. Shared Constants (Consistency)
LOGS_DIR="logs"
PID_FILE_ENGINE="$LOGS_DIR/engine.pid"
LOG_FILE_ENGINE="$LOGS_DIR/engine.log"
PID_FILE_GATEWAY="$LOGS_DIR/gateway.pid"
LOG_FILE_GATEWAY="$LOGS_DIR/gateway.log"

# 3. Environment Loading
load_env() {
    local env_file="$PROJECT_ROOT/config/.env"
    if [ -f "$env_file" ]; then
        # Load environment variables, ignoring comments and empty lines
        set -a
        source "$env_file"
        set +a
    else
        # We don't error out because defaults are handled in scripts, 
        # but we warn.
        echo "WARNING: $env_file not found. Using defaults."
    fi
}

# 4. Logging Utility
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

log_warn() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $*"
}

# 5. Check Command
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        log_error "Command '$1' is required but not found."
        return 1
    fi
    return 0
}

# Change to project root by default so relative paths (logs/) work
cd "$PROJECT_ROOT"
