#!/bin/bash
# redeploy.sh - retrain -> commit -> push -> relaunch, keeping the manifest consistent.
#
# Usage (on the miner host, from anywhere):
#     bash scripts/miner/redeploy.sh [--min-reward 0.68] [--force] [--no-push]
#
#   --min-reward X : abort (keep the current miner) if the retrained held-out reward < X
#   --force        : deploy even if below --min-reward
#   --no-push      : skip the git push (leaves the manifest commit unreachable - avoid)
#
# Config comes from scripts/miner/miner.env (copy miner.env.example first) or exported env
# vars. `git push` must authenticate non-interactively (SSH key, or
# `git config --global credential.helper store`). The commit is pushed BEFORE relaunch so the
# manifest never points at an unreachable commit; if the push fails, the old miner keeps running.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Activate the miner venv so both training and the pm2-spawned miner use the same interpreter.
# shellcheck disable=SC1091
[ -f miner_env/bin/activate ] && source miner_env/bin/activate
PYBIN="${PYBIN:-python}"

# shellcheck disable=SC1091
[ -f scripts/miner/miner.env ] && source scripts/miner/miner.env

MIN_REWARD=""; FORCE=0; DO_PUSH=1; ROBUST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --min-reward) MIN_REWARD="${2:?}"; shift 2;;
    --force) FORCE=1; shift;;
    --no-push) DO_PUSH=0; shift;;
    --robust) ROBUST="--robust"; shift;;  # drift-select features from live captures
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

: "${WALLET_NAME:?set WALLET_NAME in scripts/miner/miner.env or env}"
: "${HOTKEY:?set HOTKEY}"
: "${POKER44_MODEL_REPO_URL:?set POKER44_MODEL_REPO_URL}"
export WALLET_NAME HOTKEY POKER44_MODEL_REPO_URL
export EXTERNAL_IP="${EXTERNAL_IP:-}"
export AXON_PORT="${AXON_PORT:-8091}"
export ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

echo "== [1/5] retrain on the latest benchmark =="
LOG="$(mktemp)"
$PYBIN -m miner_training.train --all --force-download $ROBUST 2>&1 | tee "$LOG"
REWARD="$(grep 'val (held-out)' "$LOG" | grep -oE 'subnet_reward=[0-9.]+' | grep -oE '[0-9.]+' | tail -1 || true)"
echo "held-out reward: ${REWARD:-unknown}"

if [ -n "$MIN_REWARD" ] && [ -n "$REWARD" ] && awk "BEGIN{exit !($REWARD < $MIN_REWARD)}"; then
  if [ "$FORCE" -ne 1 ]; then
    echo "ABORT: held-out reward $REWARD < --min-reward $MIN_REWARD. Old miner left running (nothing changed)." >&2
    exit 2
  fi
  echo "WARN: reward below min, but --force set; continuing."
fi

echo "== [2/5] commit the retrained model + code =="
git add -A
git add -f miner_training/model.pkl
git reset -q -- scripts/miner/miner.env 2>/dev/null || true   # never publish local deploy config
if git diff --cached --quiet; then
  echo "No changes vs last deploy (model identical). Done."; exit 0
fi
VER="$(date +%Y.%m.%d)"
git commit -q -m "retrain $VER (held-out reward=${REWARD:-na})"
COMMIT="$(git rev-parse HEAD)"
echo "commit=$COMMIT  version=$VER"

if [ "$DO_PUSH" -eq 1 ]; then
  echo "== [3/5] push (must succeed before relaunch) =="
  if ! git push origin HEAD:main; then
    echo "ABORT: git push failed. Old miner still running; manifest NOT updated. Fix auth and re-run." >&2
    exit 3
  fi
else
  echo "== [3/5] push SKIPPED (--no-push): manifest commit will be INACCESSIBLE; don't leave it this way. =="
fi

echo "== [4/5] relaunch with the new commit + version =="
pm2 delete poker44_miner 2>/dev/null || true
POKER44_MODEL_REPO_COMMIT="$COMMIT" POKER44_MODEL_VERSION="$VER" \
  bash scripts/miner/run/run_miner.sh

echo "== [5/5] verify =="
sleep 6
tail -n 40 "$HOME"/.pm2/logs/poker44*out.log 2>/dev/null \
  | grep -iE "transparency|Loaded trained|Miner UID|served on" \
  || echo "(no lines yet - check: pm2 logs poker44_miner)"
echo
echo "Redeployed. Confirm 'transparent' above and that the commit is live:"
echo "  $POKER44_MODEL_REPO_URL/commit/$COMMIT"
