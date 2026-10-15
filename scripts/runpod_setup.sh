#!/usr/bin/env bash
set -Eeuo pipefail

# Run once inside GPU pod or local Linux environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
QWEN_REPO_DIR="${QVC_QWEN_REPO_DIR:-/workspace/Qwen3-TTS}"
QWEN_GIT_REF="${QVC_QWEN_GIT_REF:-main}"
LOG_DIR="${PROJECT_DIR}/runs/setup-$(date -u +%Y%m%dT%H%M%SZ)"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/.tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$LOG_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
exec > >(tee -a "$LOG_DIR/setup.log") 2>&1

log() { printf '[setup %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
trap 'log "FAILED at line $LINENO"' ERR

log "Checking GPU and environment"
command -v nvidia-smi || true
nvidia-smi || true
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"; print("torch=", torch.__version__, "gpu=", torch.cuda.get_device_name(0))' || true
if ! command -v sox >/dev/null 2>&1; then
  log "Installing system SoX package required by audio stack"
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sox libsox-fmt-all || true
fi
sox --version || true

export PIP_PROGRESS_BAR=on
if [[ ! -d "$QWEN_REPO_DIR/.git" ]]; then
  log "Cloning official Qwen3-TTS repository"
  git clone --progress https://github.com/QwenLM/Qwen3-TTS.git "$QWEN_REPO_DIR"
fi
git -C "$QWEN_REPO_DIR" fetch --tags --progress origin
git -C "$QWEN_REPO_DIR" checkout "$QWEN_GIT_REF"
git -C "$QWEN_REPO_DIR" rev-parse HEAD | tee "$LOG_DIR/qwen_git_commit.txt"

log "Installing project and official Qwen dependencies"
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
  python3 -m venv --system-site-packages "$PROJECT_DIR/.venv"
fi
source "$PROJECT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$QWEN_REPO_DIR"
python -m pip install -e "$PROJECT_DIR[train,eval,dev]"
if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
  python -m pip install -U hf_transfer
fi
log "Checking FlashAttention"
python -m pip install -U flash-attn --no-build-isolation || true

log "Prefetching Qwen model weights"
qwen-voiceclone models prefetch || true
qwen-voiceclone budget show
log "Setup complete. Inspect $LOG_DIR/setup.log, then run scripts/runpod_train_and_eval.sh"
