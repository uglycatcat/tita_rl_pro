#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_benchmark.sh
# Optional env overrides:
#   DEVICE=cuda:0 BATCH_SIZE=1 WARMUP=100 ITERS=1000 REPEATS=3 NVSMI_MS=100

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-100}"
ITERS="${ITERS:-1000}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-42}"
NVSMI_MS="${NVSMI_MS:-100}"

CHECKPOINT="${CHECKPOINT:-model_stairs_10000.pt}"
TORCHSCRIPT="${TORCHSCRIPT:-model.pt}"
ONNX_MODEL="${ONNX_MODEL:-test.onnx}"
ENGINE_FP32="${ENGINE_FP32:-test_fp32.engine}"
ENGINE_FP16="${ENGINE_FP16:-test_fp16.engine}"

OUT_JSON="${OUT_JSON:-benchmark_results.json}"
OUT_MD="${OUT_MD:-benchmark_results.md}"
OUT_CSV="${OUT_CSV:-benchmark_samples.csv}"

echo "[benchmark] device=${DEVICE}, batch=${BATCH_SIZE}, warmup=${WARMUP}, iters=${ITERS}, repeats=${REPEATS}"
echo "[benchmark] models: ckpt=${CHECKPOINT}, jit=${TORCHSCRIPT}, onnx=${ONNX_MODEL}"
echo "[benchmark] engines: fp32=${ENGINE_FP32}, fp16=${ENGINE_FP16}"

python "benchmark_inference.py" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --repeats "${REPEATS}" \
  --seed "${SEED}" \
  --nvidia-smi-interval-ms "${NVSMI_MS}" \
  --checkpoint "${CHECKPOINT}" \
  --torchscript "${TORCHSCRIPT}" \
  --onnx "${ONNX_MODEL}" \
  --engine-fp32 "${ENGINE_FP32}" \
  --engine-fp16 "${ENGINE_FP16}" \
  --output-json "${OUT_JSON}" \
  --output-md "${OUT_MD}" \
  --output-csv "${OUT_CSV}"

echo "[benchmark] done -> ${OUT_JSON}, ${OUT_MD}, ${OUT_CSV}"
