#!/bin/bash -l
#SBATCH --job-name=audio8-ml-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=audio8-ml-sft-%j.out
#SBATCH --error=audio8-ml-sft-%j.err

set -euo pipefail
unset SLURM_EXPORT_ENV

PROJECT_ROOT="$(realpath "${1:?usage: $0 PROJECT_ROOT TRAIN_JSONL TOKENS_JSON [MODEL] [VENV] [OUTPUT_DIR] [EVAL_JSONL] [HF_HOME]}")"
TRAIN_JSONL="$(realpath "${2:?TRAIN_JSONL is required}")"
ADDITIONAL_TOKENS_JSON="$(realpath "${3:?TOKENS_JSON is required}")"
MODEL="${4:-Audio8/Audio8-TTS-Preview-0.6b}"
VENV="${5:-${PROJECT_ROOT}/.venv}"
JOB_OUTPUT_DIR="${6:-${PROJECT_ROOT}/outputs/audio8_tts_ml}"
EVAL_JSONL="${7:-}"
HF_HOME="${8:-${PROJECT_ROOT}/.cache/huggingface}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Python environment does not exist: ${VENV}" >&2
  echo "Create it and install requirements-train.txt before submitting." >&2
  exit 2
fi

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#visible_gpus[@]} -lt 1 || -z "${visible_gpus[0]}" ]]; then
  echo "No Slurm-assigned GPUs are visible; submit with --gres=gpu:<type>:<count>." >&2
  exit 2
fi

module purge
module load python

cd "${PROJECT_ROOT}"
source "${VENV}/bin/activate"
mkdir -p "${HF_HOME}" "${JOB_OUTPUT_DIR}"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME
export TRITON_CACHE_DIR="${TMPDIR:-${JOB_OUTPUT_DIR}/.triton}"
mkdir -p "${TRITON_CACHE_DIR}"
export PYTHON="${VENV}/bin/python"
export MODEL
export TRAIN_JSONL
export ADDITIONAL_TOKENS_JSON
if [[ -n "${EVAL_JSONL}" ]]; then
  export EVAL_JSONL="$(realpath "${EVAL_JSONL}")"
fi
export OUTPUT_DIR="${JOB_OUTPUT_DIR}"
export EXPORT_DIR="${OUTPUT_DIR}/export"
export NPROC_PER_NODE="${#visible_gpus[@]}"
if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  # ZeRO provides no sharding benefit on one GPU, while importing DeepSpeed
  # requires an Alex CUDA compiler module solely for optional-op detection.
  export DEEPSPEED_CONFIG=none
fi
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export FREEZE_SLOW_AR=false
export FREEZE_FAST_AR=true
export BF16=true
# ArkttsModel does not advertise Transformers gradient-checkpointing support.
# Enabling this makes Trainer fail before the first optimization step.
export GRADIENT_CHECKPOINTING=false
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export RESUME_MODE=auto
export REPORT_TO=tensorboard

echo "[audio8_tts.slurm] job=${SLURM_JOB_ID} host=$(hostname) gpus=${NPROC_PER_NODE}"
echo "[audio8_tts.slurm] train=${TRAIN_JSONL} tokens=${ADDITIONAL_TOKENS_JSON}"
echo "[audio8_tts.slurm] output=${OUTPUT_DIR}"

srun \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  bash "${PROJECT_ROOT}/audio8_tts_sft.sh"
