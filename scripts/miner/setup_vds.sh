#!/bin/bash
# setup_vds.sh - Provision a fresh Ubuntu VDS/VPS to serve the Poker44 miner.
#
# Run from the repo root on the VDS:
#     bash scripts/miner/setup_vds.sh
#
# Installs system deps + LightGBM (required to load the trained model.pkl) + pm2,
# then builds the miner venv via scripts/miner/setup.sh. Firewall is opt-in
# (SETUP_FIREWALL=1) and always allows SSH first to avoid locking yourself out.
set -e

info(){ echo -e "\e[34m[INFO]\e[0m $1"; }
ok(){ echo -e "\e[32m[OK]\e[0m $1"; }
err(){ echo -e "\e[31m[ERROR]\e[0m $1" >&2; exit 1; }

[ -f "requirements.txt" ] || err "Run from the repo root (requirements.txt not found)."

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

info "Installing system packages..."
$SUDO apt-get update -y
# libgomp1 = OpenMP runtime LightGBM needs at import time.
$SUDO apt-get install -y python3-venv python3-dev build-essential git curl libgomp1

if ! command -v pm2 >/dev/null 2>&1; then
  info "Installing Node.js + pm2 (process manager used by run_miner.sh)..."
  if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO -E bash -
    $SUDO apt-get install -y nodejs
  fi
  $SUDO npm install -g pm2
fi
ok "pm2 $(pm2 --version 2>/dev/null || echo 'installed')"

info "Building miner venv + Python deps (scripts/miner/setup.sh)..."
bash scripts/miner/setup.sh

info "Installing LightGBM into miner_env (required to load model.pkl)..."
# shellcheck disable=SC1091
source miner_env/bin/activate
pip install lightgbm
python -c "import lightgbm; print('lightgbm', lightgbm.__version__)"
ok "LightGBM ready"

# The pinned bittensor-cli pulls in `scalecodec`, which collides with the `cyscale`
# used by the bittensor SDK's async-substrate-interface and breaks `import bittensor`.
# Keep cyscale (the SDK needs it); the miner host does not need bittensor-cli.
info "Resolving scale-codec namespace conflict..."
pip uninstall -y scalecodec cyscale >/dev/null 2>&1 || true
pip install --force-reinstall cyscale
python -c "import bittensor; print('bittensor import OK:', bittensor.__version__)" \
  || err "bittensor still fails to import after conflict fix."
ok "bittensor import verified"

if [ "${SETUP_FIREWALL:-0}" = "1" ]; then
  PORT="${AXON_PORT:-8091}"
  info "Configuring ufw (SSH first, then axon $PORT)..."
  $SUDO ufw allow OpenSSH 2>/dev/null || $SUDO ufw allow 22/tcp
  $SUDO ufw allow "${PORT}/tcp"
  $SUDO ufw --force enable
  $SUDO ufw status
else
  info "Firewall unchanged. If ufw/cloud-firewall is active, open SSH + the axon port:"
  echo "    sudo ufw allow OpenSSH && sudo ufw allow ${AXON_PORT:-8091}/tcp && sudo ufw --force enable"
fi

PUBIP="$(curl -s --max-time 8 https://api.ipify.org 2>/dev/null || echo '<VDS_PUBLIC_IP>')"

cat <<NEXT

$(ok "VDS provisioned. This server's public IP looks like: ${PUBIP}")

NEXT STEPS
  1) Get the trained model on this box. Recommended: build it here (no version skew):
        source miner_env/bin/activate
        python -m miner_training.train --all          # downloads benchmark, writes miner_training/model.pkl
     OR copy it from your local machine (ensure sklearn/lightgbm versions match):
        # on LOCAL:  scp miner_training/model.pkl USER@${PUBIP}:~/Poker44-subnet/miner_training/

  2) Put ONLY your hotkey here (coldkey stays on your local machine):
        # on LOCAL:
        scp ~/.bittensor/wallets/<wallet>/coldkeypub.txt USER@${PUBIP}:~/.bittensor/wallets/<wallet>/coldkeypub.txt
        scp ~/.bittensor/wallets/<wallet>/hotkeys/<hotkey> USER@${PUBIP}:~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>

  3) Register the hotkey on netuid 126 FROM LOCAL (needs the coldkey + TAO):
        btcli subnet register --netuid 126 --wallet.name <wallet> --wallet.hotkey <hotkey> --subtensor.network finney

  4) Launch here (announces THIS server's public IP to validators):
        source miner_env/bin/activate
        EXTERNAL_IP=${PUBIP} AXON_PORT=8091 \\
        WALLET_NAME=<wallet> HOTKEY=<hotkey> \\
        POKER44_MODEL_REPO_URL=https://github.com/<you>/<your-repo> \\
        ALLOWED_VALIDATOR_HOTKEYS="<vali_hk_1> <vali_hk_2>" \\
        ./scripts/miner/run/run_miner.sh

  5) Confirm reachability from a DIFFERENT machine:
        nc -vz ${PUBIP} 8091
     Logs:  pm2 logs poker44_miner    (look for "Loaded trained model")
NEXT
