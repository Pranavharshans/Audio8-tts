#!/usr/bin/env bash
set -Eeuo pipefail

# Single-GPU, non-Slurm entrypoint for a full Praha-Labs/TTS-Ml run on Vast.ai.
# Keep DATA_ROOT and VENV on persistent storage so preparation and checkpoints
# survive instance restarts.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"

MODEL="${MODEL:-Audio8/Audio8-TTS-Preview-0.6b}"
DATASET="${DATASET:-Praha-Labs/TTS-Ml}"
# Dataset main at the time this full-run recipe was created (2026-08-10).
DATASET_REVISION="${DATASET_REVISION:-33cef946925f89ee48511951da3049f5281cfd2e}"
DATA_ROOT="${DATA_ROOT:-/workspace/audio8_ml}"
VENV="${VENV:-/workspace/venvs/audio8tts}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/full-original-tokenizer}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# "all" means the complete dataset. Set a positive integer for a pilot.
MAX_SAMPLES="${MAX_SAMPLES:-all}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
IMPORT_SEED="${IMPORT_SEED:-1337}"
FORCE_IMPORT="${FORCE_IMPORT:-false}"
PREP_BATCH_SIZE="${PREP_BATCH_SIZE:-4}"
FULL_DATASET_MIN_ROWS="${FULL_DATASET_MIN_ROWS:-70000}"

RAW_DIR="${DATA_ROOT}/raw"
PREPARED_DIR="${DATA_ROOT}/prepared"
TRAIN_JSONL="${PREPARED_DIR}/train.jsonl"
EVAL_JSONL="${PREPARED_DIR}/eval.jsonl"
RUN_CONFIG="${OUTPUT_DIR}/vast_run_config.env"

usage() {
  cat <<'EOF'
Usage: bash scripts/vast_audio8_ml_full.sh [setup|prepare|train|all]

Stages:
  setup    Create the virtual environment and install training dependencies.
  prepare  Download Praha-Labs/TTS-Ml and precompute codec indices.
  train    Train from prepared manifests, resuming the latest checkpoint.
  all      Run setup, prepare, and train (default).

Important environment overrides:
  DATA_ROOT=/workspace/audio8_ml   Persistent data/checkpoint directory
  VENV=/workspace/venvs/audio8tts Python virtual environment
  MAX_SAMPLES=all                 Complete dataset; use an integer for a pilot
  OUTPUT_DIR=...                  Training checkpoints and final export
  CUDA_VISIBLE_DEVICES=0          The one GPU to use

Training defaults can also be overridden with BATCH_SIZE,
GRADIENT_ACCUMULATION_STEPS, NUM_TRAIN_EPOCHS, LEARNING_RATE,
FREEZE_FAST_AR, SAVE_STEPS, and EVAL_STEPS.
EOF
}

fail() {
  echo "[audio8_tts.vast] error: $*" >&2
  exit 2
}

case "${STAGE}" in
  setup|prepare|train|all) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    fail "unknown stage: ${STAGE}"
    ;;
esac

[[ "${MAX_SAMPLES}" == "all" || "${MAX_SAMPLES}" == "0" || "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] \
  || fail "MAX_SAMPLES must be 'all', 0, or a positive integer"
[[ "${EVAL_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || fail "EVAL_SAMPLES must be a positive integer"
[[ "${PREP_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "PREP_BATCH_SIZE must be positive"
[[ "${IMPORT_SEED}" =~ ^[0-9]+$ ]] || fail "IMPORT_SEED must be a non-negative integer"
[[ "${FULL_DATASET_MIN_ROWS}" =~ ^[1-9][0-9]*$ ]] \
  || fail "FULL_DATASET_MIN_ROWS must be positive"
[[ "${FORCE_IMPORT}" == "true" || "${FORCE_IMPORT}" == "false" ]] \
  || fail "FORCE_IMPORT must be true or false"
if [[ "${MAX_SAMPLES}" != "all" && "${MAX_SAMPLES}" != "0" ]]; then
  (( 10#${MAX_SAMPLES} >= 2 )) || fail "MAX_SAMPLES must be at least 2"
  (( 10#${EVAL_SAMPLES} < 10#${MAX_SAMPLES} )) \
    || fail "EVAL_SAMPLES must be smaller than MAX_SAMPLES"
fi

# A multi-GPU Vast offer may expose every device. This launcher deliberately
# masks all but one unless the caller already selected a single device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]] \
  || fail "select exactly one GPU, for example CUDA_VISIBLE_DEVICES=0"
export HF_HOME HF_DATASETS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_ROOT}/pip_cache}"

mkdir -p "${DATA_ROOT}" "${HF_HOME}" "${PIP_CACHE_DIR}"

setup_environment() {
  command -v python3 >/dev/null 2>&1 || fail "python3 is not installed"
  mkdir -p "$(dirname "${VENV}")"
  if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[audio8_tts.vast] creating venv=${VENV}"
    python3 -m venv "${VENV}"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  python -m pip install --upgrade pip wheel
  python -m pip install -r "${PROJECT_ROOT}/requirements-train.txt"
  if python -m pip show deepspeed >/dev/null 2>&1; then
    echo "[audio8_tts.vast] removing DeepSpeed; it provides no benefit on one GPU"
    python -m pip uninstall -y deepspeed
  fi
}

activate_environment() {
  [[ -x "${VENV}/bin/python" ]] \
    || fail "venv not found at ${VENV}; run the setup stage first"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
}

check_single_gpu() {
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("this run uses BF16; select an Ampere-or-newer GPU with BF16 support")
print(f"[audio8_tts.vast] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
PY
}

manifest_rows() {
  local train_manifest="${RAW_DIR}/train.jsonl"
  local eval_manifest="${RAW_DIR}/eval.jsonl"
  local train_rows=0
  local eval_rows=0
  [[ -f "${train_manifest}" ]] && train_rows="$(wc -l < "${train_manifest}")"
  [[ -f "${eval_manifest}" ]] && eval_rows="$(wc -l < "${eval_manifest}")"
  echo $((train_rows + eval_rows))
}

raw_import_is_complete() {
  local total_rows
  local eval_rows
  [[ -s "${RAW_DIR}/train.jsonl" && -s "${RAW_DIR}/eval.jsonl" ]] || return 1
  total_rows="$(manifest_rows)"
  eval_rows="$(wc -l < "${RAW_DIR}/eval.jsonl")"
  (( eval_rows == 10#${EVAL_SAMPLES} )) || return 1
  if [[ "${MAX_SAMPLES}" == "all" || "${MAX_SAMPLES}" == "0" ]]; then
    (( total_rows >= FULL_DATASET_MIN_ROWS ))
  else
    (( total_rows == 10#${MAX_SAMPLES} ))
  fi
}

prepare_dataset() {
  local -a import_args
  mkdir -p "${RAW_DIR}" "${PREPARED_DIR}"
  if [[ "${FORCE_IMPORT}" == "false" ]] && raw_import_is_complete; then
    echo "[audio8_tts.vast] reusing raw manifests ($(manifest_rows) rows)"
  else
    import_args=(
      --dataset "${DATASET}"
      --revision "${DATASET_REVISION}"
      --output-dir "${RAW_DIR}"
      --eval-samples "${EVAL_SAMPLES}"
      --seed "${IMPORT_SEED}"
    )
    if [[ "${MAX_SAMPLES}" != "all" && "${MAX_SAMPLES}" != "0" ]]; then
      import_args+=(--max-samples "${MAX_SAMPLES}")
    fi
    python "${PROJECT_ROOT}/audio8_tts_import_hf.py" "${import_args[@]}"
  fi

  # The original tokenizer is intentional. No token mining or vocabulary file
  # is produced by this workflow.
  python "${PROJECT_ROOT}/audio8_tts_prepare.py" \
    --input-jsonl "${RAW_DIR}/train.jsonl" \
    --output-jsonl "${TRAIN_JSONL}" \
    --model "${MODEL}" \
    --device cuda \
    --dtype bfloat16 \
    --batch-size "${PREP_BATCH_SIZE}"
  python "${PROJECT_ROOT}/audio8_tts_prepare.py" \
    --input-jsonl "${RAW_DIR}/eval.jsonl" \
    --output-jsonl "${EVAL_JSONL}" \
    --model "${MODEL}" \
    --device cuda \
    --dtype bfloat16 \
    --batch-size "${PREP_BATCH_SIZE}"
}

write_run_config() {
  mkdir -p "${OUTPUT_DIR}"
  local git_commit="unknown"
  if command -v git >/dev/null 2>&1 && git -C "${PROJECT_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
    git_commit="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  fi
  {
    printf 'created_utc=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%q\n' "${git_commit}"
    printf 'dataset=%q\n' "${DATASET}"
    printf 'dataset_revision=%q\n' "${DATASET_REVISION}"
    printf 'model=%q\n' "${MODEL}"
    printf 'max_samples=%q\n' "${MAX_SAMPLES}"
    printf 'eval_samples=%q\n' "${EVAL_SAMPLES}"
    printf 'import_seed=%q\n' "${IMPORT_SEED}"
    printf 'train_rows=%q\n' "$(wc -l < "${TRAIN_JSONL}")"
    printf 'eval_rows=%q\n' "$(wc -l < "${EVAL_JSONL}")"
    printf 'tokenizer=%q\n' original
    printf 'batch_size=%q\n' "${BATCH_SIZE:-1}"
    printf 'gradient_accumulation_steps=%q\n' "${GRADIENT_ACCUMULATION_STEPS:-16}"
    printf 'num_train_epochs=%q\n' "${NUM_TRAIN_EPOCHS:-3}"
    printf 'learning_rate=%q\n' "${LEARNING_RATE:-5e-6}"
    printf 'freeze_slow_ar=%q\n' "${FREEZE_SLOW_AR:-false}"
    printf 'freeze_fast_ar=%q\n' "${FREEZE_FAST_AR:-true}"
    printf 'save_steps=%q\n' "${SAVE_STEPS:-250}"
    printf 'eval_steps=%q\n' "${EVAL_STEPS:-250}"
  } > "${RUN_CONFIG}"
}

train_model() {
  [[ -s "${TRAIN_JSONL}" ]] || fail "missing prepared train manifest: ${TRAIN_JSONL}"
  [[ -s "${EVAL_JSONL}" ]] || fail "missing prepared eval manifest: ${EVAL_JSONL}"
  write_run_config

  export PYTHON="${VENV}/bin/python"
  export MODEL TRAIN_JSONL EVAL_JSONL OUTPUT_DIR
  export EXPORT_DIR="${EXPORT_DIR:-${OUTPUT_DIR}/export}"
  export ADDITIONAL_TOKENS_JSON=""
  export NPROC_PER_NODE=1
  export DEEPSPEED_CONFIG=none
  export BATCH_SIZE="${BATCH_SIZE:-1}"
  export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
  export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
  export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
  export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
  export FREEZE_SLOW_AR="${FREEZE_SLOW_AR:-false}"
  export FREEZE_FAST_AR="${FREEZE_FAST_AR:-true}"
  export BF16=true
  export GRADIENT_CHECKPOINTING=false
  export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
  export EVAL_STEPS="${EVAL_STEPS:-250}"
  export SAVE_STEPS="${SAVE_STEPS:-250}"
  export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
  export RESUME_MODE=auto
  export REPORT_TO="${REPORT_TO:-tensorboard}"

  echo "[audio8_tts.vast] train=${TRAIN_JSONL} eval=${EVAL_JSONL}"
  echo "[audio8_tts.vast] tokenizer=original output=${OUTPUT_DIR} resume=auto"
  echo "[audio8_tts.vast] config=${RUN_CONFIG}"
  bash "${PROJECT_ROOT}/audio8_tts_sft.sh"
}

on_interrupted() {
  echo "[audio8_tts.vast] interrupted; rerun the same command to reuse codec files and resume the latest checkpoint" >&2
  exit 130
}
trap on_interrupted INT TERM

if [[ "${STAGE}" == "setup" || "${STAGE}" == "all" ]]; then
  setup_environment
else
  activate_environment
fi

check_single_gpu

if [[ "${STAGE}" == "prepare" || "${STAGE}" == "all" ]]; then
  prepare_dataset
fi
if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then
  train_model
fi

echo "[audio8_tts.vast] stage=${STAGE} complete"
