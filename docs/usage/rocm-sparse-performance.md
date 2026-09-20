# ROCm G11 performance path

Use the updated [ROCm companion patches](../../examples/vime_rocm_attention_ablation/companion_patches/README.md),
rebuild RL-Kernel for `gfx942`, and configure the local paths in the profile.
The README Quick start command is the canonical entry point:

```bash
./rlk run --backend rocm --mode consistency \
  --tp 4 --cp 2 --rollout-tp 4 --rollout-cp 1 \
  --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 --max-response-len 6912 \
  --grpo-std-normalization disabled --steps 200
```

The consistency arm uses Triton chunked Attention, deterministic MFMA GEMM,
the sparse HIP top-p logprob/entropy path, and independent training logprob
recomputation. `--use-rollout-logprobs` is not passed. Complete retained-token
support is transported, so arbitrary top-p values remain supported. The
launcher leaves `AMD_CPU_AFFINITY` empty by default and preserves an explicit
operator override, avoiding the old eight-core CPU bottleneck.

The launcher records source fingerprints, backend readbacks and raw train /
rollout logprob comparisons. A consistency run fails if any compared value is
not bitwise equal. Temperature, top-p, learning rate, KL coefficient, response
length, training TP/CP and rollout TP/CP are command-line parameters.

## Historical reference

The original 200-step G10/G11 runs used the same TP4/CP2 training and TP4/CP1
rollout topology, temperature 0.7, top-p 0.95, LR 5e-7, KL 0.01 and response
cap 6912. Both runs enabled rollout-logprob reuse and generated different token
counts, so these numbers are a reference rather than a guarantee for the
current no-reuse path.

| 200-step measurement | Native G10 | Strict G11 |
|---|---:|---:|
| Mean rollout seconds | 81.4696 | 68.0234 |
| Mean training seconds | 14.7771 | 27.6626 |
| Mean step seconds | 99.2430 | 99.0377 |
| Pooled end-to-end tok/GPU/s | 62.4719 | 59.6966 |

The mean step difference was 0.2069% in favor of G11 because its rollout
savings offset the slower strict training path. A short run cannot establish
that 200-step number; compare native and consistency with the same command,
token workload and warmup policy.

## Validated strict path

The no-reuse TP4/CP2 / TP4/CP1 run used the exact command above with
`temperature=0.7`, `top-p=0.95` and strict Triton routing. The one-step CPU
allocation check measured 52.93 s rollout, 18.70 s training and 72.80 s step
time, with 35,878 compared tokens and zero bitwise mismatches. The focused
CLI/platform test set passed 26 tests.

The largest measured regression before this fix was the forced
`AMD_CPU_AFFINITY=0` mask, which constrained vLLM/Ray workers to CPUs 0–7.
The launcher now matches the historical contract by inheriting the host mask;
the diagnosis is recorded in [rollout-gap-cause.md](../validation/rocm-readme-20260920/rollout-gap-cause.md).
