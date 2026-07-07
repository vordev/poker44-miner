#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

PROCESS_NAME="${PROCESS_NAME:-poker44_validator}"
WALLET_NAME="${WALLET_NAME:-}"
WALLET_HOTKEY="${WALLET_HOTKEY:-}"
SUBTENSOR_PARAM="${SUBTENSOR_PARAM:---subtensor.network finney}"
VALIDATOR_ENV_DIR="${VALIDATOR_ENV_DIR:-validator_env}"
VALIDATOR_EXTRA_ARGS="${VALIDATOR_EXTRA_ARGS:-}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
STATE_FILE="${STATE_FILE:-}"
SLEEP_INTERVAL="${SLEEP_INTERVAL:-600}"

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -z "$REPO_ROOT" ]; then
  echo "Error: not inside a Git repository" >&2
  exit 1
fi

if [ -z "$STATE_FILE" ]; then
  STATE_FILE="${HOME:-$REPO_ROOT}/.poker44_auto_update_state"
elif [[ "$STATE_FILE" != /* ]]; then
  STATE_FILE="$REPO_ROOT/$STATE_FILE"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="$SCRIPT_DIR/update_validator.sh"
VERSION_GATES_FILE="poker44/__init__.py"
VERSION_KEY="VALIDATOR_DEPLOY_VERSION"

extract_named_version() {
  local key="$1"
  local file="$2"
  grep "^${key}[[:space:]]*=" "$file" 2>/dev/null | \
    head -n1 | \
    sed -E "s/^${key}[[:space:]]*=[[:space:]]*[\"']?([^\"']+)[\"']?.*/\1/"
}

get_local_version() {
  extract_named_version "$VERSION_KEY" "$REPO_ROOT/$VERSION_GATES_FILE" || echo ""
}

get_local_commit() {
  git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo ""
}

get_remote_version() {
  git -C "$REPO_ROOT" fetch origin "$TARGET_BRANCH" --quiet || return 1
  git -C "$REPO_ROOT" show "origin/$TARGET_BRANCH:$VERSION_GATES_FILE" 2>/dev/null | \
    extract_named_version "$VERSION_KEY" /dev/stdin || echo ""
}

get_remote_commit() {
  git -C "$REPO_ROOT" rev-parse --short "origin/$TARGET_BRANCH" 2>/dev/null || echo ""
}

upsert_state_value() {
  local key="$1"
  local value="$2"
  local tmp_file

  mkdir -p "$(dirname "$STATE_FILE")"
  tmp_file="$(mktemp)"
  if [ -f "$STATE_FILE" ]; then
    grep -v "^${key}=" "$STATE_FILE" > "$tmp_file" || true
  fi
  printf '%s=%s\n' "$key" "$value" >> "$tmp_file"
  mv "$tmp_file" "$STATE_FILE"
}

is_remote_newer() {
  [ -z "$1" ] && return 1
  [ -z "$2" ] && return 1
  [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ] && [ "$1" != "$2" ]
}

[ -f "$UPDATE_SCRIPT" ] || {
  echo "Error: update script not found at $UPDATE_SCRIPT" >&2
  exit 1
}

[ -f "$REPO_ROOT/$VERSION_GATES_FILE" ] || {
  echo "Error: version gate file not found at $REPO_ROOT/$VERSION_GATES_FILE" >&2
  exit 1
}

chmod +x "$UPDATE_SCRIPT" || echo "[WARN] Could not chmod +x $UPDATE_SCRIPT"
git -C "$REPO_ROOT" config --local core.fileMode false || true

echo "[INFO] Poker44 auto-update watcher starting in $REPO_ROOT"
echo "[INFO] Process=$PROCESS_NAME branch=$TARGET_BRANCH env_dir=$VALIDATOR_ENV_DIR"
echo "[INFO] State file: $STATE_FILE"
echo "[INFO] Poll interval: ${SLEEP_INTERVAL}s"

while true; do
  LOCAL_VERSION="$(get_local_version)"
  REMOTE_VERSION="$(get_remote_version)"
  LOCAL_COMMIT="$(get_local_commit)"
  REMOTE_COMMIT="$(get_remote_commit)"
  echo "[INFO] $VERSION_KEY local=$LOCAL_VERSION remote=$REMOTE_VERSION"
  echo "[INFO] Git commit local=$LOCAL_COMMIT remote=$REMOTE_COMMIT"

  if is_remote_newer "$LOCAL_VERSION" "$REMOTE_VERSION"; then
    echo "[INFO] New Poker44 deploy version detected, updating validator"
    PROCESS_NAME="$PROCESS_NAME" \
    WALLET_NAME="$WALLET_NAME" \
    WALLET_HOTKEY="$WALLET_HOTKEY" \
    SUBTENSOR_PARAM="$SUBTENSOR_PARAM" \
    VALIDATOR_ENV_DIR="$VALIDATOR_ENV_DIR" \
    VALIDATOR_EXTRA_ARGS="$VALIDATOR_EXTRA_ARGS" \
    TARGET_BRANCH="$TARGET_BRANCH" \
    bash -x "$UPDATE_SCRIPT"
    upsert_state_value "LAST_APPLIED_VALIDATOR_DEPLOY_VERSION" "$REMOTE_VERSION"
    echo "[INFO] Persisted LAST_APPLIED_VALIDATOR_DEPLOY_VERSION=$REMOTE_VERSION"
  else
    echo "[INFO] No Poker44 validator update needed"
  fi

  echo "[INFO] Sleeping ${SLEEP_INTERVAL}s..."
  sleep "$SLEEP_INTERVAL"
done
