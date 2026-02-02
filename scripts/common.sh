#!/bin/bash
# L-kn Common Script Utilities
# Centralizes path resolution and environment loading

# 1. Path Resolution
# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is one level up from scripts/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 2. Environment Loading
load_env() {
    local env_file="$PROJECT_ROOT/config/.env"
    if [ -f "$env_file" ]; then
        # Load environment variables, ignoring comments and empty lines
        set -a
        source "$env_file"
        set +a
    else
        echo "WARNING: $env_file not found. Using defaults."
    fi
}

# 3. Logging Utility
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

log_warn() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $*"
}

# 4. Check Command
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        log_error "Command '$1' is required but not found."
        return 1
    fi
    return 0
}

# Change to project root by default
cd "$PROJECT_ROOT"
