# Audio8 TTS Inference Optimization Results

## Acceptance rules

1. No quality regression against the reference path.
2. One material change or tightly bounded experiment per entry.
3. Results must be reproducible from a recorded commit, environment, command, and input set.
4. The deployment path must support the change in a clean environment.
5. A speed improvement must be end-to-end or materially reduce a measured resource bottleneck; kernel-only gains are not sufficient by themselves.

Exact greedy code equality is the primary gate. A valid WAV alone is not sufficient. For sampling paths, distribution and seeded behavior must also be checked before accepting an optimization.

## Environment

See `profiling.md`. All short baseline measurements below use the 0.6B checkpoint, BF16, CUDA, batch size 1, greedy decoding, and `max_new_tokens=128`. Unless noted, measurements use two warm-up requests and six measured requests. The long test uses three measured requests and `max_new_tokens=256`.

## Baseline

### No-reference synthesis

Command workload: `Welcome to Audio8 TTS.`

| Metric | Result |
|---|---:|
| Request p50 | 2,385.42 ms |
| Request p95 | 2,394.78 ms |
| Request p99 | 2,395.72 ms |
| Generation p50 | 2,348.57 ms |
| Decode p50 | 35.92 ms |
| Preprocess + H2D p50 | 0.75 ms |
| Peak allocated VRAM | 2,060,130,304 bytes (~1.92 GiB) |
| Peak reserved VRAM | 2,157,969,408 bytes (~2.01 GiB) |
| Output | 106,496 samples at 44.1 kHz |
| Codes SHA-256 | `0aeba5fbffe80f006eb366e5190a05d0d2c726b7791cc99c654736e551766dc4` |
| Audio SHA-256 | `4d10a142d2a8af7afb9398b7fcb7242b2920e15ec7f8bc56ecab481add1185a8` |

### Reference-voice synthesis

Workload: `assets/training/Maya.wav`, matching transcript from the repository guide, target text `The evening market was quiet after the rain.`

| Metric | Result |
|---|---:|
| Request p50 | 2,549.82 ms |
| Request p95 | 2,557.93 ms |
| Request p99 | 2,558.89 ms |
| Generation p50 | 2,507.14 ms |
| Decode p50 | 37.57 ms |
| Preprocess + H2D p50 | 5.07 ms |
| Peak allocated VRAM | 2,072,694,784 bytes (~1.93 GiB) |
| Peak reserved VRAM | 2,225,078,272 bytes (~2.07 GiB) |
| Output | 112,640 samples at 44.1 kHz |
| Codes SHA-256 | `131e04fec4fe06e7fcee0cf0281b9f41312373936d0e7bebfa7de6c679f3b5da` |
| Audio SHA-256 | `03450042bc746e5228508ebfd50b85748c533324b0fe9ed0bd5aef7dc1a14d39` |

## Experiment log

| ID | Commit / patch | Change | Quality gate | p50 | p95 | Throughput | VRAM peak | Decision |
|---|---|---|---|---:|---:|---:|---:|---|
| E000 | `3698607` | Eager BF16 reference path | pass: valid WAV and deterministic greedy artifacts | 2,385.42 ms | 2,394.78 ms | 0.419 req/s | 1.92 GiB | baseline |
| E001 | `3698607` + CLI wrapper | Wrap generation and decode in `torch.inference_mode()` | pass: exact codes/audio; model already does this internally | 2,387.05 ms | 2,422.02 ms | 0.419 req/s | 1.92 GiB | reject: redundant, +0.07% p50 |
| E002 | `3698607` + model patch | Restrict semantic projection to EOS + semantic-token rows when processors/criteria are absent | pass: exact codes/audio on short no-reference, reference voice, and long text | 2,380.73 ms | 2,484.73 ms | 0.420 req/s | ~1.92 GiB | candidate: ~0.50% long-request gain; needs deployable integration |
| E003 | `3698607` + compile | `torch.compile(mode=reduce-overhead)` on `_slow_step`/`_fast_step` | fail during warm-up: CUDA-graph output was overwritten | n/a | n/a | n/a | n/a | reject |
| E003b | `3698607` + compile | `torch.compile(mode=default)` on `_slow_step`/`_fast_step` | fail: generated code changed shape and values | 1,104.16 ms | 1,105.14 ms | 0.906 req/s | ~1.92 GiB | reject despite ~53.7% p50 gain |
| E004 | `3698607` + model patch | Allow PyTorch’s default SDPA backend instead of forced math SDPA in decode | fail: generated code changed shape and values | 2,199.70 ms | 2,228.08 ms | 0.455 req/s | ~1.91 GiB | reject despite ~7.8% p50 gain |

## Detailed experiments

### E000 — eager BF16 reference

- The repository path loads the checkpoint with BF16 and runs greedy generation.
- The deterministic artifact hashes above are the correctness reference.
- Full profile findings are in `profiling.md`.

### E001 — explicit inference-mode wrapper

- The CLI was temporarily wrapped around `model.generate()` and `model.decode_audio()`.
- Codes and audio were byte-for-byte identical to E000.
- The model’s downloaded `generate()` already uses `@torch.inference_mode()`, so the wrapper added no meaningful work reduction and was reverted.

### E002 — reduced semantic projection

- Profile evidence showed a repeated full `[896, 155776]` semantic projection.
- The candidate computes only the EOS row and the 4,096 semantic-token rows when no custom logits processor or stopping criterion is present, then maps compact candidate IDs back to model IDs.
- Short no-reference: 2,380.73 ms p50 versus 2,385.42 ms baseline; exact codes/audio.
- Reference voice: 2,548.35 ms p50 versus 2,549.82 ms baseline; exact codes/audio.
- Long text (`I will keep you, Susy, busy, make your head with heat grow dizzy; tear in eye, your dress you will tear; queer, fair seer, hear my prayer.`, `max_new_tokens=256`): 8,991.38 ms p50 versus 9,036.15 ms baseline, a 0.50% improvement; exact codes/audio.
- This is promising but is not yet accepted as a repository change: the tested patch was applied to the downloaded model custom code. It must be integrated into the repository’s clean deployment path and validated for seeded sampling before acceptance.

### E003 — compiler/CUDA-graph mode

- `reduce-overhead` failed during warm-up with a CUDA-graph tensor-overwrite error.
- `default` completed and was substantially faster, but produced a different code tensor shape (`116,736` versus `106,496` samples after decode) and different code/audio hashes. It is rejected under the quality gate.

### E004 — default SDPA backend

- The candidate removed the forced `SDPBackend.MATH` context around autoregressive decode.
- It measured 2,199.70 ms p50, but the output code tensor shape and values differed from E000. It is rejected.

## Final comparison

No optimization is accepted yet. E002 is the leading candidate because it preserves exact greedy artifacts across three workloads and improves the long workload by ~0.50%, but it requires a tracked, deployment-safe integration and seeded-sampling validation before it can be merged.
