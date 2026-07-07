#!/bin/bash
# setup.sh - Setup Poker44 miner environment
set -e

SAFE_BITTENSOR_CLI_VERSION="9.20.0"
SAFE_BITTENSOR_WALLET_VERSION="4.0.1"
BLOCKED_BITTENSOR_CLI_VERSION="9.18.2"
BLOCKED_BITTENSOR_WALLET_VERSION="4.0.2"

handle_error() {
  echo -e "\e[31m[ERROR]\e[0m $1" >&2
  exit 1
}

success_msg() {
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}

info_msg() {
  echo -e "\e[34m[INFO]\e[0m $1"
}

resolve_python() {
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done

  handle_error "Python 3.10+ is required."
}

check_python() {
  resolve_python
  info_msg "Using Python interpreter: $PYTHON_BIN"
  "$PYTHON_BIN" --version || handle_error "Failed to execute $PYTHON_BIN."
}

create_activate_venv() {
  VENV_DIR="miner_env"
  info_msg "Creating virtualenv in $VENV_DIR..."
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR" || handle_error "Failed to create virtualenv"
    success_msg "Virtualenv created."
  else
    info_msg "Virtualenv already exists. Skipping creation."
  fi

  info_msg "Activating virtualenv..."
  source "$VENV_DIR/bin/activate" || handle_error "Failed to activate virtualenv"
}

upgrade_pip() {
  info_msg "Upgrading pip and setuptools..."
  python -m pip install --upgrade pip setuptools || handle_error "Failed to upgrade pip/setuptools"
  success_msg "pip and setuptools upgraded."
}

install_python_reqs() {
  info_msg "Installing Python dependencies from requirements.txt..."
  [ -f "requirements.txt" ] || handle_error "requirements.txt not found"

  pip install -r requirements.txt || handle_error "Failed to install Python dependencies"
  success_msg "Packages installed"
}

install_modules() {
  info_msg "Installing current package in editable mode..."
  pip install -e . || handle_error "Failed to install current package"
  success_msg "Main package installed."
}

install_bittensor_cli() {
  info_msg "Installing pinned Bittensor CLI and wallet versions..."
  pip install "bittensor-cli==${SAFE_BITTENSOR_CLI_VERSION}" "bittensor-wallet==${SAFE_BITTENSOR_WALLET_VERSION}" \
    || handle_error "Failed to install pinned Bittensor packages"
  success_msg "Pinned Bittensor packages installed."
}

guard_bittensor_versions() {
  info_msg "Checking installed Bittensor package versions..."
  python - <<'PY' || handle_error "Blocked or unexpected Bittensor package versions detected"
from importlib import metadata
from sys import exit

safe_cli = "9.20.0"
safe_wallet = "4.0.1"
blocked = {
    "bittensor-cli": "9.18.2",
    "bittensor-wallet": "4.0.2",
}

packages = {}
for name in ("bittensor-cli", "bittensor-wallet"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

for name, blocked_version in blocked.items():
    if packages[name] == blocked_version:
        print(f"Blocked version installed: {name}=={blocked_version}")
        exit(1)

if packages["bittensor-cli"] != safe_cli or packages["bittensor-wallet"] != safe_wallet:
    print(
        "Unexpected Bittensor package versions:",
        f"bittensor-cli=={packages['bittensor-cli']}",
        f"bittensor-wallet=={packages['bittensor-wallet']}",
    )
    exit(1)

print(
    "Verified pinned Bittensor package versions:",
    f"bittensor-cli=={packages['bittensor-cli']}",
    f"bittensor-wallet=={packages['bittensor-wallet']}",
)
PY
  success_msg "Pinned Bittensor package versions verified."
}

verify_installation() {
  info_msg "Verifying participant environment setup..."
  python -c "import bittensor; print(f'✓ Bittensor: {bittensor.__version__}')" || info_msg "Warning: Bittensor import failed"
  success_msg "Installation verification completed."
}

show_completion_info() {
  echo
  success_msg "Poker44 miner environment configured!"
  echo
  echo -e "\e[33m[INFO]\e[0m Virtual environment: $(pwd)/miner_env"
  echo -e "\e[33m[INFO]\e[0m To activate: source miner_env/bin/activate"
  echo
  echo -e "\e[34m[NEXT STEPS]\e[0m"
  echo "1. Review scripts/miner/run/run_miner.sh and set wallet, hotkey, axon port, and allowlisted validators."
  echo "   source miner_env/bin/activate"
  echo "   ./scripts/miner/run/run_miner.sh"
}

main() {
  check_python
  create_activate_venv
  upgrade_pip
  install_python_reqs
  install_modules
  install_bittensor_cli
  guard_bittensor_versions
  verify_installation
  show_completion_info
}

main "$@"
