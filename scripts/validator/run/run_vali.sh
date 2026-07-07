#!/bin/bash

set -euo pipefail

# Poker44 Validator Startup Script

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-poker44-test-ck}"
HOTKEY="${HOTKEY:-poker44-hk}"
NETWORK="${NETWORK:-finney}"
SUBTENSOR_PARAM="${SUBTENSOR_PARAM:-}"
VALIDATOR_SCRIPT="${VALIDATOR_SCRIPT:-./neurons/validator.py}"
PM2_NAME="${PM2_NAME:-poker44_validator}"  ##  name of validator, as you wish
VALIDATOR_ENV_DIR="${VALIDATOR_ENV_DIR:-validator_env}"
WALLET_PATH="${WALLET_PATH:-}"
VALIDATOR_EXTRA_ARGS="${VALIDATOR_EXTRA_ARGS:-}"
POKER44_RUNTIME_MODE="${POKER44_RUNTIME_MODE:-provider_runtime}"
POKER44_POLL_INTERVAL_SECONDS="${POKER44_POLL_INTERVAL_SECONDS:-300}"
POKER44_MINERS_PER_CYCLE="${POKER44_MINERS_PER_CYCLE:-16}"
NEURON_TIMEOUT="${NEURON_TIMEOUT:-180}"
POKER44_MINER_QUERY_TIMEOUT_SECONDS="${POKER44_MINER_QUERY_TIMEOUT_SECONDS:-$NEURON_TIMEOUT}"
POKER44_EVAL_API_BASE_URL="${POKER44_EVAL_API_BASE_URL:-https://api.poker44.net}"
POKER44_PROVIDER_API_BASE_URL="${POKER44_PROVIDER_API_BASE_URL:-$POKER44_EVAL_API_BASE_URL}"
POKER44_PROVIDER_INTERNAL_SECRET="${POKER44_PROVIDER_INTERNAL_SECRET:-}"
POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS="${POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS:-180}"
POKER44_PROVIDER_MIN_EVAL_HANDS="${POKER44_PROVIDER_MIN_EVAL_HANDS:-120}"
POKER44_PROVIDER_MAX_EVAL_HANDS="${POKER44_PROVIDER_MAX_EVAL_HANDS:-100}"
POKER44_MIN_HANDS_PER_CHUNK="${POKER44_MIN_HANDS_PER_CHUNK:-100}"
POKER44_MAX_HANDS_PER_CHUNK="${POKER44_MAX_HANDS_PER_CHUNK:-100}"
POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT="${POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT:-true}"
POKER44_PROVIDER_VALIDATOR_ID="${POKER44_PROVIDER_VALIDATOR_ID:-}"

if [ -x "$VALIDATOR_ENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VALIDATOR_ENV_DIR/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Error: No Python interpreter found"
    exit 1
fi

if [ ! -f "$VALIDATOR_SCRIPT" ]; then
    echo "Error: Validator script not found at $VALIDATOR_SCRIPT"
    exit 1
fi

if [ "$POKER44_RUNTIME_MODE" != "provider_runtime" ]; then
    echo "Error: Only POKER44_RUNTIME_MODE=provider_runtime is supported."
    exit 1
fi

if [ "$POKER44_RUNTIME_MODE" = "provider_runtime" ] && [ "$POKER44_PROVIDER_INTERNAL_SECRET" = "force-start-secret" ]; then
    echo "Error: POKER44_PROVIDER_INTERNAL_SECRET cannot be set to the placeholder force-start-secret."
    exit 1
fi

if [ "$POKER44_RUNTIME_MODE" = "provider_runtime" ] && [ -z "$POKER44_PROVIDER_INTERNAL_SECRET" ]; then
    echo "Warning: POKER44_PROVIDER_INTERNAL_SECRET is unset; validator-facing eval calls will use signed hotkey auth only."
    echo "Warning: admin eval actions such as publish-current will be skipped unless the backend auto-publishes the active chunk."
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import bittensor, dotenv, numpy, pandas, sklearn" >/dev/null 2>&1; then
    echo "Error: Python environment is missing required packages for validator startup."
    echo "Checked interpreter: $PYTHON_BIN"
    echo "Run ./scripts/validator/main/setup.sh or fix the virtualenv before starting PM2."
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"
export POKER44_RUNTIME_MODE="$POKER44_RUNTIME_MODE"
export POKER44_POLL_INTERVAL_SECONDS="$POKER44_POLL_INTERVAL_SECONDS"
export POKER44_MINERS_PER_CYCLE="$POKER44_MINERS_PER_CYCLE"
export POKER44_MINER_QUERY_TIMEOUT_SECONDS="$POKER44_MINER_QUERY_TIMEOUT_SECONDS"
export POKER44_EVAL_API_BASE_URL="$POKER44_EVAL_API_BASE_URL"
export POKER44_PROVIDER_API_BASE_URL="$POKER44_PROVIDER_API_BASE_URL"
export POKER44_PROVIDER_INTERNAL_SECRET="$POKER44_PROVIDER_INTERNAL_SECRET"
export POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS="$POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS"
export POKER44_PROVIDER_MIN_EVAL_HANDS="$POKER44_PROVIDER_MIN_EVAL_HANDS"
export POKER44_PROVIDER_MAX_EVAL_HANDS="$POKER44_PROVIDER_MAX_EVAL_HANDS"
export POKER44_MIN_HANDS_PER_CHUNK="$POKER44_MIN_HANDS_PER_CHUNK"
export POKER44_MAX_HANDS_PER_CHUNK="$POKER44_MAX_HANDS_PER_CHUNK"
export POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT="$POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT"
export POKER44_PROVIDER_VALIDATOR_ID="$POKER44_PROVIDER_VALIDATOR_ID"
export PM2_NAME="$PM2_NAME"
export VALIDATOR_ENV_DIR="$VALIDATOR_ENV_DIR"

if [ -n "$SUBTENSOR_PARAM" ]; then
  read -r -a SUBTENSOR_ARG_ARRAY <<< "$SUBTENSOR_PARAM"
else
  SUBTENSOR_ARG_ARRAY=(--subtensor.network "$NETWORK")
fi

VALIDATOR_ARGS=(
  "$VALIDATOR_SCRIPT"
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --neuron.timeout "$NEURON_TIMEOUT"
  --logging.debug
)

VALIDATOR_ARGS+=("${SUBTENSOR_ARG_ARRAY[@]}")

if [ -n "$WALLET_PATH" ]; then
  VALIDATOR_ARGS+=(--wallet.path "$WALLET_PATH")
fi

if [ -n "$VALIDATOR_EXTRA_ARGS" ]; then
  read -r -a EXTRA_ARG_ARRAY <<< "$VALIDATOR_EXTRA_ARGS"
  VALIDATOR_ARGS+=("${EXTRA_ARG_ARRAY[@]}")
fi

pm2 start "$PYTHON_BIN" \
  --name $PM2_NAME -- \
  "${VALIDATOR_ARGS[@]}"

pm2 save

echo "Validator started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY python=$PYTHON_BIN"
echo "Subtensor args: ${SUBTENSOR_PARAM:---subtensor.network $NETWORK}"
echo "Runtime extras: wallet_path=${WALLET_PATH:-<default>} extra_args=${VALIDATOR_EXTRA_ARGS:-<none>}"
echo "Profile: runtime_mode=$POKER44_RUNTIME_MODE chunks=backend_controlled reward_window=backend_controlled poll_interval_s=$POKER44_POLL_INTERVAL_SECONDS miners_per_cycle=$POKER44_MINERS_PER_CYCLE timeout_s=$NEURON_TIMEOUT miner_query_timeout_s=$POKER44_MINER_QUERY_TIMEOUT_SECONDS"
if [ "$POKER44_RUNTIME_MODE" = "provider_runtime" ]; then
  echo "Provider runtime: eval_api=$POKER44_EVAL_API_BASE_URL request_timeout_s=$POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS min_eval_hands=$POKER44_PROVIDER_MIN_EVAL_HANDS max_eval_hands=$POKER44_PROVIDER_MAX_EVAL_HANDS min_hands_per_chunk=$POKER44_MIN_HANDS_PER_CHUNK max_hands_per_chunk=$POKER44_MAX_HANDS_PER_CHUNK attempt_publish_current=$POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT"
  echo "Provider source: central platform backend validator_id=${POKER44_PROVIDER_VALIDATOR_ID:-<wallet hotkey>}"
fi
