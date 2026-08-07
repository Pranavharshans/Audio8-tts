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
