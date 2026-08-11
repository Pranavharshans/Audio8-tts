# Audio8 TTS Inference Optimization Results

## Acceptance rules

1. No quality regression against the reference path.
2. One material change or tightly bounded experiment per entry.
3. Results must be reproducible from a recorded commit, environment, command, and input set.
4. The deployment path must support the change in a clean environment.
5. A speed improvement must be end-to-end or materially reduce a measured resource bottleneck; kernel-only gains are not sufficient by themselves.

Exact greedy code equality is the primary gate. A valid WAV alone is not sufficient. For sampling paths, distribution and seeded behavior must also be checked before accepting an optimization.

The active performance target is end-to-end RTF <= 0.5 and the lowest practical time to first playable audio (TTFA). For a non-streaming path, TTFA is effectively the full request latency.

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

| ID | Commit / patch | Change | Quality gate | Total p50 | p95 | TTFA p50 | RTF | Decision |
|---|---|---|---|---:|---:|---:|---:|---|
| E000 | `3698607` | Eager BF16 reference path | pass: valid WAV and deterministic greedy artifacts | 2,385.42 ms | 2,394.78 ms | ~2,385 ms | 0.988 | baseline |
| E001 | `3698607` + CLI wrapper | Wrap generation and decode in `torch.inference_mode()` | pass: exact codes/audio; model already does this internally | 2,387.05 ms | 2,422.02 ms | ~2,387 ms | 0.988 | reject: redundant, +0.07% p50 |
| E002 | `3698607` + model patch | Restrict semantic projection to EOS + semantic-token rows when processors/criteria are absent | pass: exact codes/audio on short no-reference, reference voice, and long text | 2,380.73 ms | 2,484.73 ms | ~2,381 ms | 0.986 | candidate: ~0.50% long-request gain; needs deployable integration |
| E003 | `3698607` + compile | `torch.compile(mode=reduce-overhead)` on `_slow_step`/`_fast_step` | fail during warm-up: CUDA-graph output was overwritten | n/a | n/a | n/a | n/a | reject |
| E003b | `3698607` + compile | `torch.compile(mode=default)` on `_slow_step`/`_fast_step` | fail: generated code changed shape and values | 1,104.16 ms | 1,105.14 ms | ~1,104 ms | n/a | reject despite ~53.7% p50 gain |
| E004 | `3698607` + model patch | Allow PyTorch’s default SDPA backend instead of forced math SDPA in decode | fail: generated code changed shape and values | 2,199.70 ms | 2,228.08 ms | ~2,200 ms | n/a | reject despite ~7.8% p50 gain |
| E005 | `86c9e5f` + pinned serving runtime | SGLang service, CUDA Graph, FlashInfer slow AR, SDPA fast head, 12-frame streaming chunks, compile off | deterministic and same length; fail exact waveform equivalence, objective quality suite pending | **860.97 ms** | **866.71 ms** | **219.28 ms** | **0.357** | performance target met; quality acceptance pending |
| E006 | `c09b0c2` + runtime config | Reduce streaming chunk from 12 to 4 codec frames, retain one-frame guard | deterministic and same length; versus SGLang full decode: cosine 0.999981, SNR 44.31 dB | 1,001.29 ms | 1,005.80 ms | **132.01 ms** | **0.415** | TTFA candidate: -39.8% TTFA, +16.3% total latency |
| E007 | `7dc7cda` + runtime config | Reduce streaming chunk from 4 to 2 codec frames, retain one-frame guard | deterministic and same length; versus SGLang full decode: cosine 0.999982, SNR 44.40 dB | 1,201.19 ms | 1,206.58 ms | **105.34 ms** | **0.497** | aggressive TTFA profile; RTF target met with little margin |
| E008 | `621bbb1` + runtime config | Enable SGLang TorchInductor compilation with CUDA Graph and 4-frame chunks | fail: output changed from 106,496 to 116,736 samples | 991.37 ms | 997.22 ms | 120.38 ms | 0.375 reported | reject: duration/output changed for ~1% latency gain |
| E009 | `72dc8c0` + runtime config | Enable greedy-only sampling fast path, compile off, 4-frame chunks | exact PCM hash match to E006 | **971.00 ms** | **977.53 ms** | **125.22 ms** | **0.402** | accept for greedy requests: -3.0% total, -5.1% TTFA |
| E010 | `2ba6a08` + runtime config | Combine greedy-only fast path with 2-frame chunks | exact PCM hash match to E007 | **1,169.01 ms** | **1,214.66 ms** | **102.21 ms** | **0.484** p50 / 0.503 p95 | aggressive: p50 target met, p95 narrowly misses |
| E011 | `aee19c2` + runtime config | Combine greedy-only fast path with 3-frame chunks | deterministic and same length; versus SGLang full decode: cosine 0.999980, SNR 44.06 dB | **1,029.56 ms** | **1,031.75 ms** | **113.95 ms** | **0.426** p50 / 0.427 p95 | recommended low-TTFA profile with robust RTF margin |
| E012 | `c2a2940` + safety validation | Reject sampled requests on a greedy-fast-path server | greedy PCM hash unchanged; sampled request rejected | 1,043.08 ms single check | n/a | 129.82 ms single check | 0.432 single check | accept: prevents silent sampling-quality changes |
| E013 | `f7fd5b9` + backend resolver fix | Auto-select portable attention on non-Hopper GPUs; no backend override | exact PCM hash match to E011 | **1,030.07 ms** | **1,035.51 ms** | **113.90 ms** | **0.427** p50 / 0.429 p95 | accept: RTX 4060 Ti deployment works without manual backend config |

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

### E005 — SGLang streaming baseline on RTX 4060 Ti

- Runtime: pinned SGLang Omni `68a5723` (`0.1.0`), SGLang `0.5.8`, PyTorch `2.9.1+cu128`, Transformers `4.57.1`, BF16.
- Configuration: CUDA Graph batch sizes 1/2/4, FlashInfer slow-AR attention, portable SDPA fast-head cache, Torch compilation disabled, 12 codec frames per streaming chunk.
- Workload: `Welcome to Audio8 TTS.`, greedy decoding, `max_new_tokens=128`, two warm-ups and six measured requests.
- Total p50 was 860.97 ms and p95 was 866.71 ms for 2.41488 seconds of audio: RTF 0.357.
- TTFA p50 was 219.28 ms and p95 was 238.80 ms. The first chunk carried 0.51084 seconds of audio.
- Every run produced 106,496 samples and the same PCM SHA-256 (`e3a548aa86c6b922cc63a38878b66d292a6294335f53c51c54c890e1e6cc2fa0`).
- The waveform was not the eager baseline artifact: cosine similarity 0.00886 and SNR -5.10 dB despite identical length. This does not prove perceptual degradation, but it fails the exact-artifact gate. E005 remains a performance-qualified candidate until multi-prompt ASR, speaker-similarity, and listening checks establish equivalent quality.

### E006 — four-frame streaming chunks

- Only `AUDIO8_TTS_STREAM_CHUNK_FRAMES` changed, from 12 to 4; context remained 128 frames and the one-frame boundary guard remained enabled.
- TTFA p50 improved from 219.28 ms to 132.01 ms (39.8% lower); p95 was 141.30 ms.
- The first response contained 0.13932 seconds of playable audio instead of 0.51084 seconds.
- More frequent codec work increased total p50 from 860.97 ms to 1,001.29 ms and RTF from 0.357 to 0.415. The RTF target is still met.
- All six runs produced the same 106,496-sample output and PCM hash (`608e655df117e9e8e321f441aaba7bb53521d589410d2280b6f4feb062db4cd3`).
- Against a non-streaming full codec decode from the same SGLang request, the four-frame stream measured cosine 0.999981 and SNR 44.31 dB. The 12-frame stream measured cosine 0.999978 and SNR 43.65 dB, so reducing the chunk did not worsen this waveform-equivalence check.

### E007 — two-frame streaming chunks

- Only `AUDIO8_TTS_STREAM_CHUNK_FRAMES` changed, from 4 to 2. Context and guard settings were unchanged.
- TTFA p50 improved from 132.01 ms to 105.34 ms; p95 was 107.53 ms. The first response carried 0.04644 seconds of playable audio.
- Total p50 increased to 1,201.19 ms because the vocoder produced 27 chunks. RTF p50 was 0.4974 and p95 was 0.4996, both below the 0.5 target but with little margin for deployment noise.
- All six requests produced the same 106,496-sample output and PCM hash (`5ae0bafc0fe3550bd7b77962f23ae4a29502cb31f5829a9317ab6fbcdb35dec3`).
- Against the same-path full codec decode, cosine similarity was 0.999982 and SNR was 44.40 dB. Smaller chunks did not degrade this waveform-equivalence check.

### E008 — SGLang TorchInductor compilation

- The E006 four-frame configuration was held constant while `AUDIO8_TTS_ENABLE_TORCH_COMPILE=1` compiled and captured batch sizes 1, 2, and 4. First compilation took 208 seconds and used a persistent TorchInductor cache.
- Total p50 was 991.37 ms versus 1,001.29 ms without compilation; TTFA was 120.38 ms versus 132.01 ms.
- The generated result changed from 106,496 samples (2.41488 seconds) to 116,736 samples (2.64707 seconds), with a different deterministic PCM hash. This reproduces the output-length divergence seen in E003b.
- The reported RTF of 0.375 uses the longer candidate audio and therefore cannot be treated as a quality-preserving improvement. E008 is rejected.

### E009 — greedy-only sampling fast path

- The E006 four-frame streaming configuration was held constant while `AUDIO8_TTS_GREEDY_FASTPATH=1` bypassed sorting, top-k/top-p filtering, probability construction, and random sampling for greedy requests.
- Total p50 improved from 1,001.29 ms to 971.00 ms (-3.0%); TTFA improved from 132.01 ms to 125.22 ms (-5.1%); RTF improved from 0.415 to 0.402.
- All six requests produced exactly the same 106,496-sample output and PCM hash as E006 (`608e655df117e9e8e321f441aaba7bb53521d589410d2280b6f4feb062db4cd3`). The optimization is accepted for greedy inference only; sampled requests must continue to use the general sampling path.

### E010 — greedy fast path with two-frame streaming

- The exact-output greedy fast path from E009 was combined with E007's two-frame streaming chunks.
- Total p50 improved from 1,201.19 ms to 1,169.01 ms (-2.7%); TTFA improved from 105.34 ms to 102.21 ms (-3.0%); p50 RTF improved from 0.497 to 0.484.
- All requests retained E007's 106,496-sample output and exact PCM hash (`5ae0bafc0fe3550bd7b77962f23ae4a29502cb31f5829a9317ab6fbcdb35dec3`). However, p95 RTF was 0.503 and the maximum was 0.507, so this profile does not provide robust tail-latency margin under the 0.5 target.

### E011 — greedy fast path with three-frame streaming

- Three-frame chunks were tested as the midpoint between E009's four-frame profile and E010's aggressive two-frame profile.
- Total p50 was 1,029.56 ms, TTFA p50 was 113.95 ms, and RTF was 0.426 at p50 and 0.427 at p95. The narrow p50/p95 spread makes this materially more stable than E010 while reducing TTFA by 11.27 ms versus E009.
- All six requests produced the same 106,496-sample output and PCM hash (`dd6aefc83ab74a1f378f74c5eae60ac2ec7a3cb468ce4f1afb0a6a97525da71e`). Against the same-path full codec decode, cosine similarity was 0.999980 and SNR was 44.06 dB. E011 is the recommended low-TTFA profile.

### E012 — safe greedy-only deployment gate

- The server now rejects nonzero-temperature requests when `AUDIO8_TTS_GREEDY_FASTPATH=1`, rather than silently applying argmax to a request that asked for sampling.
- A `temperature=0.8` request returned HTTP 500 with an explicit configuration error. A subsequent greedy request completed normally and retained E011's exact 106,496-sample PCM hash (`dd6aefc83ab74a1f378f74c5eae60ac2ec7a3cb468ce4f1afb0a6a97525da71e`).
- This is a deployment-safety result, not a new performance measurement; the single-request timing is included only as a smoke check.

### E013 — automatic non-Hopper attention backend

- Backend resolution was changed from a consumer-Blackwell exception to an allowlist for validated Hopper capability `(9, 0)`. Ampere, Ada, consumer Blackwell, and unknown future capabilities now default to the portable FlashInfer/SDPA path.
- With `AUDIO8_TTS_ATTENTION_BACKEND` unset, the RTX 4060 Ti reported capability `(8, 9)`, selected `flashinfer`, and started successfully. Direct resolver assertions cover Hopper, Ampere, Ada, Blackwell, missing-CUDA, and explicit-override cases.
- All six requests retained E011's exact 106,496-sample PCM hash. Total p50 was 1,030.07 ms, TTFA p50 was 113.90 ms, and RTF was 0.427 at p50 and 0.429 at p95, confirming no material performance regression.

## Final comparison

E009 is accepted as an exact-output optimization for greedy SGLang requests. E011 is the recommended low-TTFA profile at 114 ms with stable RTF headroom, while E010 reaches 102 ms TTFA but meets the RTF target only at p50. E008 confirms that TorchInductor is not quality-safe for this path. The SGLang generation path still requires a multi-prompt objective quality suite before production acceptance because it differs numerically from eager. E002 remains the leading exact-artifact eager optimization, although its speedup is too small to meet the RTF target alone.
