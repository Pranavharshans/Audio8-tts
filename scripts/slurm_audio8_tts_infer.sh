#!/usr/bin/env bash
#SBATCH --job-name=audio8-infer
#SBATCH --output=audio8-infer-%j.out
#SBATCH --error=audio8-infer-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 || $# == 6 ]]; then
  echo "Usage: $0 PROJECT_ROOT MODEL_DIR VENV OUTPUT_WAV [TEXT [REFERENCE_AUDIO REFERENCE_TEXT]]" >&2
  exit 2
fi

PROJECT_ROOT="$(realpath "$1")"
MODEL_INPUT="$2"
if [[ -e "$MODEL_INPUT" ]]; then
  MODEL_DIR="$(realpath "$MODEL_INPUT")"
else
  # Allow a Hugging Face model ID such as Audio8/Audio8-TTS-Preview-0.6b.
  MODEL_DIR="$MODEL_INPUT"
fi
VENV="$(realpath "$3")"
OUTPUT_WAV="$4"
TEXT="${5:-നമസ്കാരം. മലയാളം ശബ്ദസംശ്ലേഷണത്തിന്റെ ഒരു പരീക്ഷണമാണിത്.}"
REFERENCE_AUDIO="${6:-}"
REFERENCE_TEXT="${7:-}"

mkdir -p "$(dirname "$OUTPUT_WAV")"
export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "[audio8_tts.infer] job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[audio8_tts.infer] model=${MODEL_DIR}"
echo "[audio8_tts.infer] output=${OUTPUT_WAV}"

infer_args=(
  --model "$MODEL_DIR"
  --device cuda
  --dtype bfloat16
  --text "$TEXT"
  --output "$OUTPUT_WAV"
  --seed 42
  --overwrite
)
if [[ -n "$REFERENCE_AUDIO" ]]; then
  infer_args+=(--reference-audio "$REFERENCE_AUDIO" --reference-text "$REFERENCE_TEXT")
fi

cd "$PROJECT_ROOT"
"${VENV}/bin/python" "${PROJECT_ROOT}/audio8_tts_infer.py" "${infer_args[@]}"

ls -lh "$OUTPUT_WAV"
