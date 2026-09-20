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

The launcher now defaults `AMD_CPU_AFFINITY=0` before HIP initialization and
forwards it to Ray workers in both native and consistency modes. Explicit
operator overrides are preserved and the effective value is recorded in
`launch.json`. This disables the runtime's affinity reset without choosing a
CPU count or overriding an inherited `taskset` allocation. Initialization probes
confirmed 160 CPUs stay 160; caller masks 0-7 and 80-87 also remain unchanged
through device discovery and `torch.cuda.init()`. The actual shell exports and
Ray runtime-env construction were checked with default, `0`, `1`, and empty
overrides. CLI regression tests: 32 passed, one Windows symlink test skipped.

The final environment default applies before initialization to training as well
as rollout. It is not identical to the diagnostic's live vLLM-only intervention;
the one-step diagnostic did not measure this final environment default. The
subsequent three-step pair is reported below. The Quick start defaults to independent training
logprobs and Triton chunked Attention with sparse logp/monitoring-entropy scoring.

Another measured cost remains: complete top-p support transport reads a GPU
scalar via `finite_counts.max().item()` on each decode step. In short dual-TP4
replays, fixed-size transport with and without that synchronization took 4.2738
and 3.8455 seconds (10.02% less). This is diagnostic evidence, not a production
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
