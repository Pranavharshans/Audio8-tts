# Malayalam training

This guide records the Malayalam fine-tuning workflow that was added to Audio8 TTS and used successfully for the pilot training run. The implementation was introduced in Git commit `e8b2cc9` and refined by the subsequent checkpoint, single-GPU, and Alex-cluster commits.

## Result

The Malayalam training run completed without training errors. The workflow adapts Audio8 TTS to Malayalam while preserving the pretrained tokenizer IDs, including the reserved semantic-audio IDs.

The pilot workflow is:

1. Materialize Malayalam audio and transcripts from `Praha-Labs/TTS-Ml`.
2. Create a deterministic train/eval split.
3. Mine frequent Malayalam grapheme and subword candidates.
4. Extend the tokenizer by appending new tokens.
5. Precompute the target codec indices.
6. Fine-tune the slow semantic/text branch while keeping the fast acoustic branch frozen.
7. Evaluate generated Malayalam audio and resume from checkpoints when a Slurm job ends.

## What was trained

The pilot used the following settings:

| Setting | Pilot value |
| --- | --- |
| Base model | `Audio8/Audio8-TTS-Preview-0.6b` |
| Dataset | `Praha-Labs/TTS-Ml` |
| Pilot size | 2,000 utterances |
| Split | 1,900 train / 100 eval |
| Learning rate | `5e-6` |
| Fast AR branch | Frozen |
| Slow AR branch | Trainable |
| Precision | BF16 |
| Per-GPU batch size | 1 |
| Gradient accumulation | 16 |
| Epochs | 3 |
| Evaluation/checkpoint interval | Every 100 steps |
| Resume behavior | Latest checkpoint automatically |

The fast branch is frozen initially because the Malayalam adaptation primarily requires new text and semantic representations. The slow branch must remain trainable when additional Malayalam tokens are used, otherwise the new embeddings cannot learn.

## FAU Alex workflow

Do all compute-heavy work inside Slurm allocations. Keep the repository, virtual environment, Hugging Face cache, prepared data, and outputs under `$WORK`; do not install packages or train on the login node.

### 1. Set up the training environment

```bash
PROJECT_ROOT="$WORK/audio8tts"
VENV="$WORK/venvs/audio8tts"
DATA_ROOT="$WORK/audio8_ml"

cd "$PROJECT_ROOT"
sbatch \
  --partition=a40 \
  --gres=gpu:a40:1 \
  --cpus-per-task=16 \
  scripts/slurm_audio8_setup.sh "$PROJECT_ROOT" "$VENV"
```

### 2. Prepare the 2,000-example pilot

This downloads the pilot rows, writes raw manifests, mines Malayalam tokens, and encodes the train and eval audio into codec arrays.

```bash
sbatch \
  --partition=a100 \
  --gres=gpu:a100:1 \
  --cpus-per-task=16 \
  scripts/slurm_audio8_ml_prepare.sh \
  "$PROJECT_ROOT" "$DATA_ROOT" "$VENV" 2000
```

The generated files are:

```text
$DATA_ROOT/raw/train.jsonl
$DATA_ROOT/raw/eval.jsonl
$DATA_ROOT/prepared/malayalam_tokens.json
$DATA_ROOT/prepared/train.jsonl
$DATA_ROOT/prepared/eval.jsonl
```

### 3. Submit Malayalam SFT

```bash
sbatch \
  --partition=a100 \
  --gres=gpu:a100:1 \
  --cpus-per-task=16 \
  scripts/slurm_audio8_tts_sft.sh \
  "$PROJECT_ROOT" \
  "$DATA_ROOT/prepared/train.jsonl" \
  "$DATA_ROOT/prepared/malayalam_tokens.json" \
  Audio8/Audio8-TTS-Preview-0.6b \
  "$VENV" \
  "$DATA_ROOT/outputs/pilot" \
  "$DATA_ROOT/prepared/eval.jsonl" \
  "$DATA_ROOT/hf_cache"
```

The wrapper derives the number of distributed processes from the GPUs assigned by Slurm, saves checkpoints every 100 steps, and resumes the latest checkpoint when the job is resubmitted after a time limit.

Useful job commands:

```bash
squeue --me
scontrol show job JOB_ID
scancel JOB_ID
```

Inspect the `audio8-*.out` and `audio8-*.err` files in the submission directory for training progress and failures.

## Tokenizer safety

Malayalam candidates are mined from the training transcripts:

```bash
python audio8_tts_mine_tokens.py \
  --input-jsonl "$DATA_ROOT/raw/train.jsonl" \
  --output-json "$DATA_ROOT/prepared/malayalam_tokens.json" \
  --model Audio8/Audio8-TTS-Preview-0.6b \
  --max-tokens 2048
```

New tokens are appended; existing token IDs are never reindexed. Each new embedding is initialized from the mean of the original byte-token embeddings used to represent that token. Do not replace the pretrained tokenizer with a newly trained tokenizer, because that would invalidate the checkpoint's fixed semantic-audio token IDs.

For a controlled comparison, pass `none` as the token JSON argument to the Slurm SFT wrapper. This keeps the original tokenizer and tests whether vocabulary expansion is providing a measurable benefit.

## Monitoring and validation

Training loss and slow/fast token accuracy are useful for detecting a broken run, but they are not sufficient to choose the best-sounding checkpoint. For each pilot, keep a fixed Malayalam test set and generate the same sentences from:

- the base model;
- the first usable fine-tuning checkpoint;
- the best evaluation checkpoint;
- the final exported model.

Compare intelligibility, pronunciation, speaker similarity, and audio artifacts. If available, record Malayalam ASR CER/WER as an objective companion to listening tests.

Before a full run, also check:

- duplicate or near-duplicate transcripts and audio;
- duration and transcript-length outliers;
- Unicode normalization and unexpected non-Malayalam text;
- whether train and eval contain the same speaker or recording session;
- available `$WORK` space for extracted audio and codec arrays.

## Next run recommendations

The pilot worked, so the next improvements should be measured experiments rather than changes to the training architecture:

1. Compare the original tokenizer with 256–512 and 2,048 mined tokens.
2. Use a speaker-disjoint eval split when speaker metadata is available.
3. Pin and record the dataset and model revisions for reproducibility.
4. Retain the best checkpoint based on held-out audio quality, not only loss.
5. Run the full 71,608-utterance dataset only after the pilot samples pass the listening and ASR checks.

The full dataset is approximately 20.8 GB of Parquet data, and extracted audio plus codec arrays require substantially more storage. Check the quota before submitting full preparation:

```bash
shownicerquota.pl
```

To prepare the full dataset, replace `2000` with `all` in the preparation command. Full preparation and SFT may require multiple 24-hour submissions; the preparation scripts reuse valid outputs and the training wrapper resumes checkpoints.

## Single-GPU Vast.ai full run

Vast.ai does not need Slurm, FAU modules, `srun`, or the Alex proxy settings. The
standalone launcher performs environment setup, downloads the full
`Praha-Labs/TTS-Ml` dataset, precomputes codec indices, and starts single-GPU
SFT using the original pretrained tokenizer:

```bash
cd /workspace/audio8tts
DATA_ROOT=/workspace/audio8_ml \
VENV=/workspace/venvs/audio8tts \
bash scripts/vast_audio8_ml_full.sh all
```

Keep `/workspace/audio8_ml` on persistent storage. The full dataset is the
default (`MAX_SAMPLES=all`), the eval split contains 500 examples, the fast
acoustic branch is frozen, and the slow semantic/text branch is trained for
three epochs with effective batch size 16. No Malayalam tokens are mined or
added. The dataset is pinned to the Hugging Face commit current when this
launcher was added; set `DATASET_REVISION` explicitly to select another
revision. Checkpoints are saved every 250 optimizer steps and a repeated `train`
or `all` invocation resumes the latest checkpoint automatically.

The Vast workflow also preserves complete resumable checkpoints at the end of
epochs 1, 2, and 3. They are stored outside normal checkpoint rotation:

```text
/workspace/audio8_ml/outputs/full-original-tokenizer/epoch_checkpoints/epoch-1/
/workspace/audio8_ml/outputs/full-original-tokenizer/epoch_checkpoints/epoch-2/
/workspace/audio8_ml/outputs/full-original-tokenizer/epoch_checkpoints/epoch-3/
```

### Reference-conditioned samples

The reference audio is included in the repository at
`assets/training/Maya.wav`, so a normal clone contains everything needed for
qualitative sampling. Its matching transcript is:

```text
i went to the store to buy some fresh fruits and snacks for the evening
```

Every 1,000 optimizer steps, training pauses briefly and generates five
reference-conditioned comparison files: the four supplied Malayalam prompts
and a second stochastic variant of the first prompt. Results and failure logs
are written without overwriting earlier steps:

```text
/workspace/audio8_ml/outputs/full-original-tokenizer/samples/step-00001000/
/workspace/audio8_ml/outputs/full-original-tokenizer/samples/step-00002000/
...
```

Sampling failures are non-fatal. On a 24 GB RTX 3090, the callback temporarily
moves optimizer state to CPU, generates one file at a time, moves the codec back
to CPU, and then restores optimizer state before training continues. This makes
64 GB of system RAM preferable even though 32 GB may work.

The stages can also be run independently, which is useful when changing Vast
instances between preparation and training:

```bash
bash scripts/vast_audio8_ml_full.sh setup
bash scripts/vast_audio8_ml_full.sh prepare
bash scripts/vast_audio8_ml_full.sh train
```

For a smoke test before paying for the full materialization, use a separate
data directory so the pilot cannot be mistaken for the full manifests:

```bash
DATA_ROOT=/workspace/audio8_ml_pilot \
MAX_SAMPLES=2000 \
EVAL_SAMPLES=100 \
bash scripts/vast_audio8_ml_full.sh all
```

Codec preparation defaults to `PREP_BATCH_SIZE=1` for the RTX 3090. Training
defaults can be overridden through the existing
environment controls, for example `GRADIENT_ACCUMULATION_STEPS`,
`NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `SAVE_STEPS`, and `EVAL_STEPS`.

For this full run, allocate 220 GB of persistent disk. A 150 GB disk may fit,
but leaves little safety margin for the Hugging Face cache, extracted audio,
codec arrays, virtual environment, rotating recovery checkpoints, three
permanent epoch checkpoints, and final export. Vast disk allocations cannot be
resized after instance creation.
