# ROCm G11 performance reproduction

Use eight MI300X GPUs, the updated [companion source patches](../../examples/vime_rocm_attention_ablation/companion_patches/README.md),
and rebuild RL-Kernel with `PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace`.
Configure the framework, model and dataset paths in `.rlk-profile.json` once.
Omit `paths.rl_kernel_root` to use the current checkout; a stale explicit path
can silently select a different revision.

```bash
./rlk run --tp 4 --cp 2 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 --max-response-len 6912 \
  --grpo-std-normalization disabled --steps 200
```

Add `--mode native` for the native comparison. Both commands independently
recompute training logprobs. Neither passes `--use-rollout-logprobs`.
Sampling, learning rate, KL coefficient, response limit and topology remain
configurable. Use `plan` instead of `run` to inspect the resolved arguments.
The eight-GPU profile supplies eight samples, global batch size eight and a
4096 token budget per training GPU; preserve those when comparing this workload.

The consistency route selects Triton chunked Attention and the deterministic
MFMA GEMM. For top-p below one it scores the complete retained support with a
compiled HIP sparse kernel and fuses monitoring entropy in the same forward.
It reuses materialized LM-head logits, avoiding a second projection and an
extra full-vocabulary log-softmax in the rollout sampler. It still computes
the full local LM-head logits. Entropy with gradients uses the existing full
entropy path; the fused value is the entropy of the retained distribution.
Top-p equal to one uses the existing full-vocabulary strict scorer.

Complete support transport is required. The scorer accepts support sets larger
than 128 tokens; no 64-token sampling truncation is introduced. Larger supports
can cost more. The launcher saves source fingerprints, runtime backend
readbacks and raw train/rollout logprob comparisons and rejects nonzero mismatch.

## Historical performance reference

The September 16/17 runs used temperature 0.7, top-p 0.95, LR 5e-7, KL 0.01,
response cap 6912, training TP4/CP2 and rollout TP4/CP1. Both enabled rollout
logprob reuse. The native run saved checkpoints; the strict run did not.
They used different source snapshots and generated different token counts.

| 200-step measurement | Native G10 | Strict G11 |
|---|---:|---:|
| Total response tokens | 9,919,832 | 9,459,539 |
| Mean rollout seconds | 81.4696 | 68.0234 |
| Mean training seconds | 14.7771 | 27.6626 |
| Mean step seconds | 99.2430 | 99.0377 |
| Pooled end-to-end tok/GPU/s | 62.4719 | 59.6966 |

Pooled throughput is total response tokens divided by total step seconds and
eight GPUs. G11 was 4.4424% lower by this measure. The 0.2069% lower mean step
time does not establish the cost at equal generated work. Neither percentage
is a guarantee for a new source revision, no-reuse run, sampling configuration
or topology. New performance claims require an unprofiled native/strict pair;
record token counts and warmup policy alongside timing. Profiler runs must be
kept separate from throughput runs.

## No-reuse smoke validation, September 20

The command above was run with `--steps 3` for each arm on eight MI300X GPUs.
The strict run passed the complete validator: 130,890 distinct selected logprobs
matched bit for bit, with maximum absolute difference zero. A separate raw audit
also found zero mismatches in all 523,560 stored values including TP replicas.
Backend readbacks and before/after source fingerprints passed. The reuse flag
was zero in both launches; temperature and top-p were 0.7 and 0.95.

| Three-step measurement | Native | Strict |
|---|---:|---:|
| Response tokens | 136,086 | 130,890 |
| Total step seconds | 426.2328 | 424.3251 |
| Mean rollout seconds | 108.0953 | 103.7735 |
| Mean training seconds | 30.7289 | 33.8964 |
| End-to-end tok/GPU/ms | 0.039910 | 0.038558 |
| End-to-end tok/ms, eight GPUs | 0.319276 | 0.308466 |

The pooled strict/native difference was **-3.39%** including every step and
**-5.47%** after excluding each arm's first step. Model startup is outside the
step timer; lazy compilation remains inside it. These are different generated
trajectories and only three steps, so neither establishes 200-step performance.

The native training job completed, but its artifact validator reported a missing
vLLM rollout FFN execution record (zero recorded calls). Its timing is therefore
provisional: native provenance validation did **not** pass. The strict run's
provenance and bitwise checks did pass. This change does not establish a new
all-topology or all-sampling-parameter matrix, and does not include a new trace
or an ablation assigning speedup to individual kernels.

Targeted tests: **54 passed**, including GPU support widths 3/65/129/513,
forward/backward numerical checks, batching/padding bit equality, and the
applied VIME adapter's temperature and native-mask contracts. All three companion
patches reproduce the recorded Git trees from their bases. The training adapter
keeps sparse logits unscaled until the shared HIP scorer; pre-scaling them in
the framework caused FP32 rounding differences and was corrected before this run.

See the [machine-readable measurements](../validation/rocm-readme-20260920/readme-pair-audit.json)
and [tested source mapping](../validation/rocm-readme-20260920/tested-source-equivalence.json)
for revision `a2b9fd6`. Runtime Python differences in that mapping were formatting
only (equal ASTs). The subsequent launcher affinity change is validated separately
below; the earlier timings are not measurements of that final launcher.

## Absolute timing diagnosis and CPU allocation

The historical G11 and current run had the same model/data fingerprints and
effective workload parameters. Their first-step generated token multisets were
identical, yet rollout took 57.354 versus 86.260 seconds. The first-three-step
rollout increase accounted for 94.06% of the total step-time increase. This
particular gap cannot be attributed to generating longer responses.

On the installed PyTorch `2.12.0+rocm7.14.0a20260608` / HIP `7.14.60850`
runtime, `torch.cuda.is_available()` narrowed the calling thread's allowed CPU
mask from 160 CPUs to CPUs 0-7. The full run's vLLM processes shared that mask.
An existing one-step no-reuse control reproduced the slow rollout. A second
one-step run widened only that experiment's vLLM actor/worker threads from
CPUs 0-7 to the controller's 160 allowed CPUs:

| One-step strict measurement | Original affinity | Widened vLLM affinity |
|---|---:|---:|
| Response tokens | 35,878 | 35,878 |
| Rollout seconds | 86.2402 | 50.4340 |
| Training seconds | 35.6649 | 50.4299 |
| Step seconds | 125.1104 | 103.0554 |
| Distinct train/rollout bit mismatches | 0 / 35,878 | 0 / 35,878 |

Rollout time fell 41.52% and step time fell 17.63%. Token sequences and selected
logprobs matched across runs after sorting by token sequence. Each arm passed
the strict validator and an independent raw-bit audit of all 143,512 stored
values including TP replicas. This is a strict/strict diagnostic comparison,
not a native/strict performance comparison.

Training time increased in this experiment. Sample order changed and the saved
rank-0 recomputation batches increased from six to seven despite identical token
multisets. Thus the training comparison also changed batching; it does not
isolate an affinity cost or explain the entire training increase. The historical
first step was 95.110 seconds, so this experiment does not establish the
historical 0.2069% mean step-time advantage.

The launcher now leaves `AMD_CPU_AFFINITY` empty by default before HIP
initialization and forwards an explicit operator override unchanged to Ray
workers in both native and consistency modes. This preserves the inherited
CPU allocation instead of applying the old eight-core affinity bottleneck.
The effective value is recorded in `launch.json`; an operator can still set a
specific mask explicitly. Initialization probes confirmed caller masks remain
unchanged through device discovery and `torch.cuda.init()`. The actual shell
exports and Ray runtime-env construction were checked with default, `0`, `1`,
and empty overrides. CLI regression tests: 32 passed, one Windows symlink
test skipped.

The final environment default applies before initialization to training as well
as rollout. It is not identical to the diagnostic's live vLLM-only intervention;
the one-step diagnostic did not measure this final environment default. The
subsequent three-step pair is reported below. The Quick start defaults to independent training
logprobs and Triton chunked Attention with sparse logp/monitoring-entropy scoring.

Another measured cost remains: complete top-p support transport reads a GPU
scalar via `finite_counts.max().item()` on each decode step. In short dual-TP4
replays, fixed-size transport with and without that synchronization took 4.2738
and 3.8455 seconds (10.02% less). **Correction:** those standalone replays did
not explicitly select the vLLM 0.26 Attention backend and executed native paged
Attention. That percentage cannot be applied directly to the README Triton
route. The corrected Triton causal comparison is linked below. This is diagnostic evidence, not a production
64-token cap or a percentage to add to the full-run affinity gain. Arbitrary
top-p and complete support remain supported; this change does not remove that
synchronization. Native FFN provenance and the historical performance target
remain open, so the PR retains draft status.

See the [one-step raw audit](../validation/rocm-readme-20260920/e2e-affinity-audit.json)
and [launcher/initialization checks](../validation/rocm-readme-20260920/affinity-fix-verification.json),
plus the [short decode replay measurements](../validation/rocm-readme-20260920/decode-sync-audit.json).

## Final launcher three-step pair, revision b7fd0ba

The exact no-reuse reproduction command above was subsequently run with
`--steps 3`, first consistency and then native, in separate PID/network
namespaces. Both used the launcher's default `AMD_CPU_AFFINITY=0`; sampled
training and rollout actors retained all 160 allowed CPUs throughout. All 31
files changed by the PR matched the published revision after LF normalization.
The kernels and companion sources were unchanged during both arms.

| Three-step measurement | Native | Strict |
|---|---:|---:|
| Response tokens | 134,504 | 132,044 |
| Step 0 seconds | 63.7306 | 70.2361 |
| Step 1 seconds | 51.8381 | 72.1041 |
| Step 2 seconds | 60.2969 | 84.1282 |
| Mean rollout seconds | 42.7283 | 57.3279 |
| Mean training seconds | 14.3292 | 16.5488 |
| Mean step seconds | 58.6219 | 75.4895 |
| Pooled end-to-end tok/GPU/ms | 0.095601 | 0.072882 |

Strict mean step time was **28.77% higher** and throughput **23.76% lower**.
Excluding step 0, the respective differences were **+39.33%** and **-27.96%**.
Rollout accounted for 14.600 of the 16.868-second mean step difference (86.55%).
Both arms generated different trajectories, including different first-step
tokens. Both last steps reached the same response-length cap for all eight
samples; strict/native step times there were 84.1282/60.2969 seconds. These
short runs do not establish steady-state or 200-step performance. The affinity
fix improves absolute times but does **not** reproduce the historical 0.2069%
step-time advantage; the measured relative gap remains substantial.

Strict passed the full validator, backend readbacks and source sealing.
All **132,044 distinct train/rollout logprobs** matched bitwise, maximum difference
zero. An independent raw-bit audit checked **528,176 values** including TP
replicas with zero mismatches. Reuse was disabled in both launch manifests;
strict selected Triton Attention and sparse top-p scoring. Two strict and six
native inference-time JIT warnings were logged, so excluding only step 0 does
not guarantee all compilation is removed.

Native completed all three training steps, but artifact validation still failed
with `missing vllm/rollout ffn execution record` and `vllm/rollout ffn had zero
calls`. The native timing remains provisional. No kernel speedup is inferred
from this incomplete provenance, and the validator was not relaxed. The PR
remains draft; both the native execution evidence and performance target are
unresolved.

See the [full final-pair audit](../validation/rocm-readme-20260920/final-affinity-pair-audit.json)
for per-step metrics, source hashes, CPU-mask observations and validation errors.

## Direct comparison with the historical G11 implementation

The saved G11 RL-Kernel, VIME and Megatron revision/diff seals match the
preserved source trees. Attention, deterministic GEMM and collective HIP
sources are unchanged. Materialized LM-head logits reuse and sparse HIP
logp/monitoring-entropy fusion are also present in both implementations.
The historical VIME directory was subsequently modified for G11, so it no
longer matches G10's seal; vLLM was not sealed in the historical manifest.
This is therefore not a complete reconstruction of the G10 baseline.

| Difference from historical G11 | Current handling |
|---|---|
| Fixed 128-column, unrolled sparse HIP forward became runtime-width loops | Restore compile-time specialization and unrolling for 128 columns; retain the dynamic fallback for larger support |
| Capacity-64 top-p transport with asynchronous overflow failure became complete dynamic support | Retain complete support; its per-token scalar synchronization remains unresolved |
| Raw-logit `.contiguous()` became a protective clone | Retain the clone because a contiguous one-row view can alias logits modified by the native sampler |
| CP token/gradient ordering and gradient-replica corrections | Retain the correctness fixes |
| GPU sparse-ID assembly became CPU assembly followed by one copy | Retain the batched copy |
| Rollout-logprob reuse was enabled | Keep it disabled, as required by the Quick start |

Historical G11 also enabled mismatch metrics, so it already executed independent
training logprob recomputation. Disabling reuse does not introduce an entire
new recomputation forward relative to that run. At TP4/CP2 with rollout TP4/CP1,
canonical TP is also four: newer multi-chunk projection and rollout PCP branches
are inactive. None of these findings attributes historical timings to CPU
contention; historical CPU-affinity telemetry is unavailable.

The restored sparse forward passed **35 tests**, covering FP32/BF16/FP16,
temperature 0.7/1.3, widths 3/65/128/129/513, gradients, exact batching/padding
invariance for logp and monitoring entropy, and raw-logit preservation.
Isolated 128-column forward time fell 17.7–19.1% for 1/4/8/4096 rows; all
eight before/after forward-output digests matched.

An unprofiled dual-TP4 replay then switched only the sparse forward binary in
the same workers. Each replica generated four fixed 3072-token prefixes plus
512 new tokens at temperature 0.7/top-p 0.95. Both variants retained complete
support and the same CPU allocation. In baseline/aligned/aligned/baseline
order, with warmups and six measurements per variant, mean paired maximum
replica time was **4.2913 / 4.3022 seconds (+0.25%)**. All 24 replay outputs
had matching token/selected-logprob digests within their replica. **Correction:**
this standalone replay also used native paged Attention because explicit backend
selection was missing. It is not a timing measurement of the README Triton route.
There is no demonstrated rollout speedup from this kernel change in that workload.

No new end-to-end training result or 200-step performance advantage is claimed.
Historical G11/G10 first-three-step means were 98.3222/90.6144 seconds (+8.506%),
even though the 200-step means differed by -0.2069%; that percentage is not
a constant per-step advantage. The historical performance target remains open.
See the [source comparison, numerical checks and raw replay measurements](../validation/rocm-readme-20260920/historical-alignment-audit.json).

### Follow-up limited to historical execution-path alignment

The ROCm sampler again skips CUDA-only greedy-temperature preprocessing.
ROCm still passes the command's positive scalar temperature into the shared
HIP scorer; CUDA retains its existing greedy handling. The earlier restoration
of the 128-column HIP specialization remains. This follow-up introduces no
sparse-score graph cache, speculative gather, or deferred overflow mechanism.

Training and rollout topology remain independent command arguments, sampling
parameters remain configurable, and rollout-logprob reuse remains disabled by
default. Complete support, raw-logit preservation, and canonical TP/CP arithmetic
corrections remain: restoring the historical capacity-64 overflow failure or
removing correctness fixes would violate those requirements.

Validation: 43 targeted ROCm tests passed; 44 local sampler/CLI tests passed
with one Windows symlink skip. All three companion patches still reconstruct
their recorded source trees. This is source and correctness validation, not
evidence that the historical 0.2% step-time result has been reproduced. See the
[scoped audit](../validation/rocm-readme-20260920/historical-only-alignment-audit.json).

### Corrected causal rollout comparison

With actual Triton Attention verified on all 36 layers, changing only the
historical worker's top-p transport from capacity 64 to complete dynamic support
increased fixed replay time **4.4561 to 4.7870 seconds (+7.43%)**. Current
RL-Kernel with matching transport measured 4.7314 seconds on the other TP4 GPU
group. Tokens and selected logprobs matched exactly. The intervention changes
both per-token synchronization and payload width; it does not isolate `.item()`
alone. Current native also uses dynamic support. The historical sampler cap
cannot be restored without restricting valid sampling inputs.

Restoring just the historical Python scorer wrapper did not reduce time.
Historical strict rollout itself measured 4.5291 seconds versus 3.4026 for current
native in another six-repeat fixed-length replay. This is neither a full old
G10 environment reconstruction nor an E2E benchmark. The native FFN provenance
gap remains. It demonstrates that restoring the old RL-Kernel source alone did
not recover the historical 0.2% comparison on this workload.

No training rerun or new production optimization was added. Default reuse remains
off, and sampling/topology remain configurable. See the [causal source analysis](../validation/rocm-readme-20260920/historical-active-path-diff.md)
and [raw timings, sources and runtime evidence](../validation/rocm-readme-20260920/causal-rollout-audit.json).
