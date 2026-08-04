#!/usr/bin/env bash
#SBATCH --job-name=audio8-infer
#SBATCH --output=audio8-infer-%j.out
#SBATCH --error=audio8-infer-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 PROJECT_ROOT MODEL_DIR VENV OUTPUT_WAV [TEXT]" >&2
  exit 2
fi

PROJECT_ROOT="$(realpath "$1")"
MODEL_DIR="$(realpath "$2")"
VENV="$(realpath "$3")"
OUTPUT_WAV="$4"
TEXT="${5:-നമസ്കാരം. മലയാളം ശബ്ദസംശ്ലേഷണത്തിന്റെ ഒരു പരീക്ഷണമാണിത്.}"

mkdir -p "$(dirname "$OUTPUT_WAV")"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "[audio8_tts.infer] job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[audio8_tts.infer] model=${MODEL_DIR}"
echo "[audio8_tts.infer] output=${OUTPUT_WAV}"

cd "$PROJECT_ROOT"
"${VENV}/bin/python" "${PROJECT_ROOT}/audio8_tts_infer.py" \
  --model "$MODEL_DIR" \
  --device cuda \
  --dtype bfloat16 \
  --text "$TEXT" \
  --output "$OUTPUT_WAV" \
  --seed 42 \
  --overwrite

ls -lh "$OUTPUT_WAV"
