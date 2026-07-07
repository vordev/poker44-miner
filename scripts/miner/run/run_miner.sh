#!/bin/bash

# Poker44 Miner Startup Script

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-poker44-miner-ck}"
HOTKEY="${HOTKEY:-poker44-miner-hk}"
NETWORK="${NETWORK:-finney}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
PM2_NAME="${PM2_NAME:-poker44_miner}"  ##  name of Miner, as you wish
AXON_PORT="${AXON_PORT:-8091}"
# Public endpoint announced to validators. On a VDS set EXTERNAL_IP to the server's
# public IP so the on-chain axon record is correct even if auto-detection differs.
EXTERNAL_IP="${EXTERNAL_IP:-}"
EXTERNAL_PORT="${EXTERNAL_PORT:-$AXON_PORT}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"

# bittensor 10.x makes CLI arg parsing opt-in. Without this, bt.Config ignores every
# --flag (wallet/netuid/neuron/axon), leaving config.neuron=None and crashing
# check_config at startup. Must be exported so the pm2-spawned process inherits it.
export BT_NO_PARSE_CLI_ARGS=false

# Optional: capture unlabeled live queries for drift diagnosis (no-op unless =1).
# Exported here so the pm2-spawned miner inherits it.
export POKER44_CAPTURE="${POKER44_CAPTURE:-}"

# --- Model manifest identity (published to validators for transparency/compliance) ---
# Auto-derived from git; override by exporting these before running the script.
export POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL:-$(git config --get remote.origin.url 2>/dev/null || echo '')}"
export POKER44_MODEL_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo '')}"
export POKER44_MODEL_VERSION="${POKER44_MODEL_VERSION:-2}"
# Optional: publish the trained artifact if you host it (helps a manual audit).
# export POKER44_MODEL_ARTIFACT_URL="https://.../model.pkl"
# export POKER44_MODEL_ARTIFACT_SHA256="$(sha256sum miner_training/model.pkl 2>/dev/null | awk '{print $1}')"

case "$POKER44_MODEL_REPO_URL" in
  ""|*"Poker44/Poker44-subnet"*)
    echo "WARNING: POKER44_MODEL_REPO_URL is empty or points to the reference repo;"
    echo "         the manifest will be 'opaque'. Set it to YOUR public model repo:"
    echo "         export POKER44_MODEL_REPO_URL=https://github.com/<you>/<your-repo>"
    ;;
esac
echo "Manifest identity: repo=${POKER44_MODEL_REPO_URL:-<unset>} commit=${POKER44_MODEL_REPO_COMMIT:0:12}"

MINER_ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

# Announce a reachable public endpoint (recommended on a VDS/behind NAT).
if [ -n "$EXTERNAL_IP" ]; then
  MINER_ARGS+=(--axon.external_ip "$EXTERNAL_IP" --axon.external_port "$EXTERNAL_PORT")
  echo "Announcing external endpoint: $EXTERNAL_IP:$EXTERNAL_PORT (bind port $AXON_PORT)"
fi

pm2 start $MINER_SCRIPT \
  --name $PM2_NAME -- \
  "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY axon_port=$AXON_PORT"
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
else
    echo "Access mode: validator_permit fallback"
fi
