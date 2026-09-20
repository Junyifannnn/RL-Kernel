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
