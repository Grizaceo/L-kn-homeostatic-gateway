# DevOps & Runtime Audit Log

**Date:** 2026-02-02
**Role:** DevOps & Runtime Orchestration
**Scope:** `scripts/`, `docker-compose.yml`, general startup flow.

## 1. Startup Flow Analysis ("Mental Walkthrough")

### The "Happy Path"
1. `git clone ...`
2. `cd l-kn-gateway`
3. `bash scripts/start_all.sh`

**Friction Points Detected:**
- **Missing .env**: If `config/.env` didn't exist, the script would silently continue or fail later when env vars were missing.
    - *Fix*: Added check in `start_all.sh` to warn user and suggest copying example.
- **Port Conflicts**: Engine startup could fail obscurely if port 30000 was taken.
    - *Fix*: Added `lsof` check in `start_engine.sh`.
- **Silent Failures**: Commands in pipes could fail without stopping the script.
    - *Fix*: Added `set -o pipefail`.

## 2. Component Audit

### `scripts/start_all.sh`
- **Status**: Hardened.
- **Changes**:
    - Enforced `pipefail`.
    - Added `config/.env` validation logic.
    - Improved log clarity.

### `scripts/start_engine.sh`
- **Status**: Hardened.
- **Changes**:
    - Added dependency check for `lsof` (graceful degradation if missing).
    - Enforced `pipefail`.
    - Validated GPU memory warnings (kept existing logic).

### `docker-compose.yml`
- **Status**: Verified & Documented.
- **Findings**:
    - Service `open-webui` correctly mapped to `127.0.0.1:3000` (secure).
    - `host.docker.internal` is used.
        - *Note*: This works out-of-the-box on Windows (Desktop/WSL2) and Mac. Linux users might need `--add-host` logic if not using strictly Docker Desktop, but `host-gateway` alias usually covers it.
- **Changes**: Added explanatory comments to the file.

## 3. Recommendations for Future
- **CI/CD**: Add a GitHub Action to run `bash -n scripts/*.sh` (syntax check) on PRs.
- **Dependency Management**: Consider `pip-tools` or `poetry` if `requirements.txt` becomes unmanageable (currently fine).
- **WSL2 Specifics**: If users report connectivity issues, verify firewall rules for `WSL -> Host` traffic, though `localhost` binding usually bypasses this for the UI.
