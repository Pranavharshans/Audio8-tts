#!/bin/bash -l
#SBATCH --job-name=audio8-ml-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=audio8-ml-prep-%j.out
#SBATCH --error=audio8-ml-prep-%j.err

set -euo pipefail
unset SLURM_EXPORT_ENV

PROJECT_ROOT="$(realpath "${1:?usage: $0 PROJECT_ROOT DATA_ROOT VENV [MAX_SAMPLES] [MODEL] [EVAL_SAMPLES]}")"
DATA_ROOT="$(realpath -m "${2:?DATA_ROOT is required and should normally be under \$WORK}")"
VENV="${3:?VENV is required}"
MAX_SAMPLES="${4:-2000}"
MODEL="${5:-Audio8/Audio8-TTS-Preview-0.6b}"
if [[ "${MAX_SAMPLES}" == "0" || "${MAX_SAMPLES}" == "all" ]]; then
  DEFAULT_EVAL_SAMPLES=500
else
  DEFAULT_EVAL_SAMPLES=100
fi
EVAL_SAMPLES="${6:-${DEFAULT_EVAL_SAMPLES}}"

module purge
module load python
source "${VENV}/bin/activate"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false

RAW_DIR="${DATA_ROOT}/raw"
PREPARED_DIR="${DATA_ROOT}/prepared"
mkdir -p "${RAW_DIR}" "${PREPARED_DIR}" "${HF_HOME}"

import_args=(
  --dataset Praha-Labs/TTS-Ml
  --output-dir "${RAW_DIR}"
  --eval-samples "${EVAL_SAMPLES}"
)
if [[ "${MAX_SAMPLES}" != "0" && "${MAX_SAMPLES}" != "all" ]]; then
  import_args+=(--max-samples "${MAX_SAMPLES}")
fi

python "${PROJECT_ROOT}/audio8_tts_import_hf.py" "${import_args[@]}"
python "${PROJECT_ROOT}/audio8_tts_mine_tokens.py" \
  --input-jsonl "${RAW_DIR}/train.jsonl" \
  --output-json "${PREPARED_DIR}/malayalam_tokens.json" \
  --model "${MODEL}" \
  --max-tokens "${MAX_TOKENS:-2048}"
python "${PROJECT_ROOT}/audio8_tts_prepare.py" \
  --input-jsonl "${RAW_DIR}/train.jsonl" \
  --output-jsonl "${PREPARED_DIR}/train.jsonl" \
  --model "${MODEL}" \
  --batch-size "${PREP_BATCH_SIZE:-4}"
python "${PROJECT_ROOT}/audio8_tts_prepare.py" \
  --input-jsonl "${RAW_DIR}/eval.jsonl" \
  --output-jsonl "${PREPARED_DIR}/eval.jsonl" \
  --model "${MODEL}" \
  --batch-size "${PREP_BATCH_SIZE:-4}"

echo "[audio8_tts.ml_prepare] prepared=${PREPARED_DIR}"
