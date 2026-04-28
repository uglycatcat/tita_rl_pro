#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Isaac Gym often needs conda lib path for libpython.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

python "sim2sim/experiment_suite.py" \
  --scenarios "sim2sim/scenarios_default.json" \
  --repeats "${REPEATS:-3}" \
  --seed "${SEED:-0}" \
  --output-dir "${OUTPUT_DIR:-sim2sim/results/mujoco_vs_phyX}" \
  --physx-model "${PHYSX_MODEL:-tita_example_10000.pt}" \
  --metrics-warmup-s "${METRICS_WARMUP_S:-1.0}" \
  "$@"
