#!/usr/bin/env bash
# Imagine Cloud Worker Boot
#
# 외부 GPU 인스턴스(Elice Cloud G-NAHP-80 등)에서 1회 실행.
# git clone → venv → pip → worker_daemon.
#
# 필수 환경변수:
#   IMAGINE_SERVER_URL       메인 서버의 외부 노출 URL (Firebase 경유)
#   IMAGINE_WORKER_EMAIL     워커 전용 user 계정 이메일
#   IMAGINE_WORKER_PASSWORD  위 계정 비밀번호
#
# 선택 환경변수:
#   IMAGINE_REPO_URL    기본: https://github.com/<your-account>/Imagine.git
#   IMAGINE_BRANCH      기본: main
#   IMAGINE_WORK_DIR    기본: $HOME/imagine
#   CUDA_WHEEL_INDEX    기본: https://download.pytorch.org/whl/cu121
#   HF_HOME             기본: $IMAGINE_WORK_DIR/.hf-cache (가중치 캐시 위치)
#   HF_TOKEN            private HF 모델 사용 시
#
# 사용 예:
#   export IMAGINE_SERVER_URL=https://imagine.example.com
#   export IMAGINE_WORKER_EMAIL=cloud-worker@imagine.local
#   export IMAGINE_WORKER_PASSWORD='...'
#   curl -O https://raw.githubusercontent.com/<repo>/main/scripts/cloud_worker_boot.sh
#   chmod +x cloud_worker_boot.sh
#   ./cloud_worker_boot.sh

set -euo pipefail

# ── 0. 환경변수 검증 ───────────────────────────────────────
: "${IMAGINE_SERVER_URL:?Set IMAGINE_SERVER_URL (메인 서버 외부 URL)}"
: "${IMAGINE_WORKER_EMAIL:?Set IMAGINE_WORKER_EMAIL}"
: "${IMAGINE_WORKER_PASSWORD:?Set IMAGINE_WORKER_PASSWORD}"

REPO_URL="${IMAGINE_REPO_URL:-https://github.com/sungchulJE/Imagine.git}"
BRANCH="${IMAGINE_BRANCH:-main}"
WORK_DIR="${IMAGINE_WORK_DIR:-$HOME/imagine}"
CUDA_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu121}"
export HF_HOME="${HF_HOME:-$WORK_DIR/.hf-cache}"

echo "[boot] server=$IMAGINE_SERVER_URL  worker=$IMAGINE_WORKER_EMAIL"
echo "[boot] repo=$REPO_URL@$BRANCH  work_dir=$WORK_DIR"
echo "[boot] cuda_index=$CUDA_INDEX  hf_home=$HF_HOME"

# ── 1. GPU 가시성 확인 ─────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "[boot] WARN: nvidia-smi not found — GPU 가시성 미확인"
fi

# ── 2. 시스템 의존성 (Ubuntu 가정) ─────────────────────────
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-pip git
fi

# ── 3. 코드 준비 ───────────────────────────────────────────
if [ ! -d "$WORK_DIR/.git" ]; then
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORK_DIR"
else
    echo "[boot] $WORK_DIR already cloned — pulling latest"
    git -C "$WORK_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$WORK_DIR" reset --hard "origin/$BRANCH"
fi
cd "$WORK_DIR"

# ── 4. venv ────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# ── 5. torch (CUDA wheel) → 나머지 의존성 ──────────────────
pip install torch>=2.6.0 --index-url "$CUDA_INDEX"
pip install -r requirements-worker-cuda.txt

# ── 6. CUDA + vLLM 동작 확인 ───────────────────────────────
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — boot 중단"
p = torch.cuda.get_device_properties(0)
print(f"[boot] CUDA OK: {p.name}, VRAM={p.total_memory/1024**3:.1f}GB, "
      f"capability={p.major}.{p.minor}")
try:
    import vllm
    print(f"[boot] vLLM OK: {vllm.__version__} (continuous batching enabled)")
except ImportError as e:
    print(f"[boot] WARN: vLLM 미설치 → transformers fallback 으로 동작: {e}")
PY

# ── 7. 워커 데몬 실행 (PYTHONPATH=프로젝트 루트) ──────────
mkdir -p "$HF_HOME"
export PYTHONPATH="$WORK_DIR"
echo "[boot] starting worker_daemon..."
exec python -m backend.worker.worker_daemon
