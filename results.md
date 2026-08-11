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
| E014 | `fec7a20` + quality harness and reference-encoder warm-up | Six-prompt eager/stream/full quality suite; 3 no-reference + 3 reference-voice prompts | pass: stream/full cosine >0.999; macro WER equal; mean speaker cosine delta -0.0028 | n/a | n/a | n/a | n/a | accept quality gate; fixes first-reference cold-start divergence |
| E015 | `152569e` + Nsight Systems | Profile accepted 3-frame serving path to select kernel-fusion targets | observational; no output change | n/a | n/a | n/a | n/a | weight norm 7.3% GPU time; layout transforms 11.4%; target kernel elimination first |
| E016 | `8d5b252` + codec weight baking | Materialize 80 inference-invariant weight-normalization hooks per codec after load | exact byte match for all 12 E014 WAV artifacts | **1,012.17 ms** | **1,017.39 ms** | **112.62 ms** | **0.419** p50 / 0.421 p95 | accept and enable by default: -1.7% total latency |
| E017 | `6d2f54b` + codec-only compile | Compile dynamic-shape codec decode with TorchInductor `reduce-overhead` | no output produced; startup compiler failure | n/a | n/a | n/a | n/a | reject: backend storage-lifetime error in quantizer attention graph |
| E018 | `d7a9e91` + streaming final-decode elimination | Reuse incrementally emitted PCM instead of decoding the complete codec sequence again at stream completion | exact byte match for all 12 E016 WAV artifacts | **975.49 ms** | **981.43 ms** | **99.57 ms** | **0.404** p50 / 0.406 p95 | accept and enable by default: -3.6% total latency versus E016 |
| E019 | `5adc7ba` + Nsight Systems | Re-profile the accepted E018 path after weight baking and final-decode elimination | observational; no output change | n/a | n/a | n/a | n/a | weight-normalization kernels gone; layout transforms 10.8%; fused Snake 9.8%; convolutions remain dominant |
| E020 | `6afdf48` + hybrid Triton Snake | Custom single-pass Snake kernel for large BF16 decoder activations; TorchScript fusion retained below 1,048,576 elements | pass: macro WER unchanged; speaker delta -0.00349; direct E018 cosine >=0.9999937 | **968.95 ms** | **976.06 ms** | **112.13 ms** | **0.401** p50 / 0.404 p95 | accept with automatic fallback: -0.7% p50 versus E018 |
| E021 | `f102397` + 30-run threshold sweep | Compare 524,288, 1,048,576, and 2,097,152-element hybrid Snake cutoffs | pass: macro WER unchanged; speaker delta -0.00365; stream/full cosine >0.99994 | **968.16 ms** | **973.73 ms** | **89.40 ms** | **0.401** p50 / 0.403 p95 | accept 524,288 default: best mean and p95; p50 tied within 0.23 ms |
| E022 | `3cc7872` + cuDNN benchmark mode | Autotune convolution algorithms for dynamic codec shapes | fail: repeated prompt produced varying PCM hashes | **2,513.69 ms** | **4,017.47 ms** | **89.01 ms** p50 / 626.91 p95 | **1.041** p50 / 1.664 p95 | reject and remove: repeated searches cause severe latency and nondeterminism |
| E023 | `789216a` + zero-extra-copy CPU handoff | Let NumPy retain decoded CPU tensor storage instead of copying each full context window | exact byte match for all 12 E021 WAV artifacts | **967.47 ms** | **973.34 ms** | **88.35 ms** | **0.401** p50 / 0.403 p95 | accept: -0.69 ms p50 and -0.82 ms mean versus E021 |

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

### E014 — multi-prompt objective quality gate

- A reproducible six-prompt manifest and evaluator now compare streamed output with same-path full decode and the eager repository path. The suite contains three no-reference and three Maya-reference voice prompts.
- The first fresh-server run exposed a cold-start defect: the first reference-conditioned stream had cosine -0.083 versus its full decode, while an immediate repeat reached 0.999945. Warming the reference codec encoder with one zero frame after loading removed the divergence; a fresh-server rerun passed all six waveform checks with cosine above 0.999 and 39.49-44.89 dB SNR.
- Whisper Small English produced identical eager and optimized macro WER (0.0833). Five prompts had zero WER; both paths received the same word-level penalty on `Audio8`, while the optimized path had lower character error for that prompt.
- WavLM speaker verification across the three reference prompts measured mean reference cosine 0.9388 for optimized streaming versus 0.9415 for eager, a -0.0028 delta inside the predefined 0.01 tolerance. The six-prompt quality gate passes; this remains a bounded evaluation rather than a claim of universal perceptual equivalence.

### E015 — accepted-path kernel profile

- Nsight Systems captured the accepted RTX 4060 Ti configuration under live HTTP streaming requests. The timed capture ended before the client completed every planned iteration, so it is used for kernel attribution rather than latency reporting.
- Runtime weight-normalization kernels consumed 7.3% of GPU kernel time across 547 launches. NCHW-to-NHWC and NHWC-to-NCHW transforms consumed 11.4% combined, while codec convolutions occupied most of the remaining top kernels.
- CUDA API time was dominated by `cudaStreamSynchronize` (58.9%), followed by `cudaMemcpyAsync` (20.7%) and kernel launch calls. The first custom-kernel phase therefore prioritizes eliminating inference-invariant weight-normalization work, followed by codec layout/elementwise fusion; the already-fused RMSNorm kernels account for only about 0.1% and are not worth replacing.

### E016 — bake codec weight normalization

- Legacy and parametrized weight-normalization hooks are now materialized once after loading each codec. The runtime logged 80 baked modules for the vocoder codec, eliminating repeated normalization kernels from each streaming decode.
- Total p50 improved from E013's 1,030.07 ms to 1,012.17 ms (-1.7%); TTFA improved from 113.90 ms to 112.62 ms; p50 RTF improved from 0.427 to 0.419.
- The primary benchmark retained the exact E011/E013 PCM hash. More importantly, every streamed and full-decode WAV across the six-prompt E014 suite was byte-for-byte identical before and after weight baking, including all three reference-conditioned prompts. E016 is accepted and enabled by default.

### E017 — codec-only TorchInductor fusion

- Only `codec.decode` was wrapped with dynamic-shape `torch.compile(mode="reduce-overhead")`; autoregressive generation remained eager/CUDA Graph.
- The first warm-up failed during startup with a TorchInductor backend storage-lifetime error while compiling the quantizer post-module attention projection. The server never accepted a request, so there was no quality exposure.
- The experimental switch was removed rather than shipping a dormant, unvalidated path. E017 is rejected; subsequent kernel work uses narrower, directly testable targets.

### E018 — eliminate redundant final streaming decode

- Streaming already decodes and emits the complete waveform chunk by chunk. The terminal pipeline payload now concatenates those emitted PCM chunks instead of invoking the codec once more over the full frame sequence. Non-streaming requests retain their original full single-pass decode.
- Total p50 improved from E016's 1,012.17 ms to 975.49 ms (-3.6%); p50 RTF improved from 0.419 to 0.404, with p95 RTF at 0.406. TTFA measured 99.57 ms, but this optimization runs only at stream completion, so the apparent TTFA change is treated as benchmark variance rather than a causal gain.
- The primary PCM hash remained `dd6aefc83ab74a1f378f74c5eae60ac2ec7a3cb468ce4f1afb0a6a97525da71e`. All 12 streamed and full-decode WAV artifacts in the six-prompt quality suite were byte-for-byte identical to E016. E018 is accepted and enabled by default.

### E019 — post-optimization kernel profile

- Nsight Systems captured 12 warmed streaming requests on the accepted E018 configuration. The trace is stored on the benchmark instance at `/workspace/E019_audio8_e018_serving.nsys-rep`. Instrumented latency is excluded from performance comparisons because profiler overhead is material.
- The weight-normalization kernels identified in E015 are absent, confirming that E016 removed the intended runtime work. NCHW-to-NHWC transforms now account for 9.2% of GPU kernel time and NHWC-to-NCHW for 1.6%. The codec's existing fused Snake activation accounts for 9.8%, while several cuDNN convolution kernels occupy most of the remaining top positions.
- CUDA API time remains dominated by `cudaStreamSynchronize` (57.0%) and `cudaMemcpyAsync` (26.6%). Device-to-host transfers are only 11.6% of GPU memory-operation time; most GPU memory-operation time is device-to-device traffic. The next experiments therefore target codec layout/convolution selection and avoid replacing already-efficient SGLang RMSNorm kernels.

### E020 — hybrid custom Triton Snake kernel

- A custom Triton kernel evaluates the complete Snake expression in one pass for contiguous BF16 decoder tensors with at least 1,048,576 elements. The existing TorchScript-fused kernel remains faster for smaller tensors and is retained there. The microbenchmark measured the custom kernel up to 2.16x faster on the largest tested shape, while confirming that it is inappropriate below the threshold.
- End-to-end p50 improved from E018's 975.49 ms to 968.95 ms (-0.7%); p95 improved to 976.06 ms and p50/p95 RTF measured 0.401/0.404. TTFA measured 112.13 ms and is treated as normal run variance because the kernel primarily affects later, larger decoder shapes.
- The custom sine path is not byte-identical. Direct comparison with E018 across six prompts measured waveform cosine 0.9999937-0.9999983 and 48.97-54.67 dB SNR. The full objective gate passed: stream/full cosine exceeded 0.99994, macro WER remained 0.0833 for both optimized and eager audio, and mean reference-speaker cosine was 0.93806 versus eager's 0.94155 (delta -0.00349, inside the 0.01 tolerance). The default `auto` mode enables the hybrid kernel when CUDA and Triton are available and otherwise retains TorchScript.

### E021 — tune the hybrid Snake cutoff

- Three cutoffs were compared, followed by 30-request runs for the two plausible boundaries. At 524,288 elements, mean/p50/p95 were 968.38/968.16/973.73 ms. At 1,048,576 elements, they were 969.10/967.93/976.14 ms. At 2,097,152 elements, they regressed to 973.83/970.97/983.37 ms.
- The 524,288 cutoff improves mean latency by 0.72 ms and p95 by 2.41 ms versus 1,048,576, while its p50 is 0.23 ms slower and therefore effectively tied. Its p50/p95 RTF is 0.401/0.403; TTFA p50/p95 is 89.40/113.10 ms. The lower cutoff is selected for better tail behavior.
- Because the selected cutoff applies the non-byte-exact sine kernel to additional shapes, the complete six-prompt gate was rerun. Stream/full cosine remained above 0.99994, macro WER stayed equal to eager at 0.0833, and mean reference-speaker cosine was 0.93789 versus eager's 0.94155 (delta -0.00365, inside tolerance). The default threshold is changed to 524,288 elements.

### E022 — cuDNN convolution autotuning

- `torch.backends.cudnn.benchmark` was enabled before codec load and warm-up to test per-shape convolution algorithm selection. Streaming context growth produces many changing convolution shapes, so three warm-up requests did not stabilize the search cost.
- Across 30 measured requests, total p50/p95 regressed to 2,513.69/4,017.47 ms and RTF to 1.041/1.664. TTFA p50 remained 89.01 ms, but p95 rose to 626.91 ms. This misses both latency and RTF goals by a wide margin.
- The same deterministic prompt produced different PCM hashes across requests, so the experiment also fails the correctness gate. The switch and implementation were removed; fixed cuDNN heuristics remain the supported path.

### E023 — remove redundant CPU full-window copy

- After each codec decode, `.cpu().numpy().copy()` duplicated the complete context window even though only the newly stable suffix is serialized. NumPy already retains the CPU tensor's storage through its base object, so the final `.copy()` was removed without changing ownership or payload lifetime.
- In a 30-request comparison against E021, mean/p50/p95 improved from 968.38/968.16/973.73 ms to 967.56/967.47/973.34 ms. P50/p95 RTF remained 0.401/0.403, and TTFA p50 measured 88.35 ms.
- The primary PCM hash was unchanged, and every streamed/full WAV in the six-prompt suite was byte-for-byte identical to E021. E023 is accepted.

## Final comparison

E023 is the recommended performance profile at 0.401 p50 / 0.403 p95 RTF with byte-identical output versus E021, a passing objective quality gate, and an automatic TorchScript fallback. E018 remains the pre-custom-kernel exact-output profile at 0.404 p50 / 0.406 p95 RTF and about 100 ms measured TTFA. E010 reaches similarly low TTFA but meets the RTF target only at p50. E008 confirms that broad TorchInductor compilation is not quality-safe for this path, while E017 shows codec-only dynamic compilation is not currently viable. E002 remains the leading exact-artifact eager optimization, although its speedup is too small to meet the RTF target alone.
