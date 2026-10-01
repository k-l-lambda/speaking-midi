#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$PWD/.local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HTTP_PROXY="${HTTP_PROXY:-http://localhost:1091}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://localhost:1091}"
export HF_HUB_DISABLE_XET=1
export PATH="$PWD/.local/usr/bin:$PATH"
export OMP_NUM_THREADS=4
exec .venv/bin/python "$@"
