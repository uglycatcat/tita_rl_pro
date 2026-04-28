#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  python "${PROJECT_ROOT}/sim2sim/mujoco_onnx_runner.py" --help
  exit 0
fi

ONNX_PATH="${PROJECT_ROOT}/sim2sim/stairs_test.onnx"
if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[sim2sim] missing ONNX file: ${ONNX_PATH}"
  echo "[sim2sim] place your model at sim2sim/stairs_test.onnx"
  exit 1
fi

python "${PROJECT_ROOT}/sim2sim/mujoco_onnx_runner.py" \
  --sim-time "${SIM_TIME:-30}" \
  "${@}"
