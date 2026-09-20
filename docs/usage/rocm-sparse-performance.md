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
and [tested source mapping](../validation/rocm-readme-20260920/tested-source-equivalence.json).
Runtime Python differences in that mapping are formatting only (equal ASTs).
