# ROCm G11 performance path

Use the updated [ROCm companion patches](../../examples/vime_rocm_attention_ablation/companion_patches/README.md),
rebuild RL-Kernel for `gfx942`, and configure the local paths in the profile.
The README Quick start command is the canonical entry point:

```bash
./rlk run --backend rocm --mode consistency \
  --tp 4 --cp 2 --rollout-tp 4 --rollout-cp 1 \
  --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 --max-response-len 6912 \
  --max-tokens-per-gpu 4096 --grpo-std-normalization enabled --steps 200
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

## Current measured path

The latest controlled comparison used the command above with `--steps 3`,
on eight MI300X GPUs, PyTorch 2.12 / ROCm 7.14 and Qwen3-8B. Both arms used
the same prompts, seeds, sampling and training settings, inherited CPU affinity,
and independent training logprob recomputation. The native arm changes only
`--mode native`. Generated lengths differ between implementations.

| Three-step measurement | Native | Strict |
|---|---:|---:|
| Mean rollout seconds | 39.8674 | 44.1356 |
| Mean training seconds | 14.1679 | 16.7198 |
| Mean step seconds | 55.5846 | 62.7527 |
| Response tokens | 129,391 | 129,868 |
| End-to-end tok/GPU/ms | 0.096993 | 0.086230 |
| Mean step seconds, excluding step 0 | 54.2528 | 63.9601 |

Strict is 12.90% slower in mean step time (17.89% excluding step 0).
This does **not** reproduce the historical 0.2% difference. Native completed
all three steps, but its validation reports a missing compiled FFN execution
record; its timings remain provisional until that evidence gap is resolved.
The strict run passed validation and a raw FP32-bit audit: 129,868 distinct
logprobs, 519,472 including TP replicas, and zero mismatches.

The retained decode changes fuse eager RMSNorm/residual addition, combine
Q/K normalization with RoPE, fuse GEMM chunk reduction with SwiGLU, and use
a smaller split Attention query tile. Small-batch GEMMs use narrower tiles and
unroll complete QKV/down-projection chunks without reordering accumulation.
Single-token decode also avoids materializing an identity sequence-index tensor
for every Attention layer. Other batches and supplied mappings retain their path.
They preserve the intermediate BF16 rounding and reduction order. The
normalization shortcuts are restricted to the verified PyTorch/gfx942
contract and fall back on unsupported inputs. Sampling parameters and
training/rollout parallelism remain configurable.

The preceding simplified strict checkout measured 70.3220 seconds/step and
52.4108 seconds/rollout under the same settings. The current run is 10.76%
faster per step and 15.79% faster in rollout. The immediately preceding fusion
baseline measured 65.1766 seconds/step and 47.1067 seconds/rollout: this increment
improves step time by 3.72% and rollout by 6.31%. Step 0's generated token multiset
matches that baseline; later trajectories differ (129,868 vs 129,575 total response
tokens), so these are end-to-end observations rather than fixed-workload timings.
A separate fixed-workload rollout comparison (four 3072-token prefixes,
256 generated tokens each, TP4, temperature 0.7, top-p 0.95) measured
2.0413 -> 1.9044 seconds for the immediately preceding fusion baseline and the
current path, a 6.70% reduction, with identical output token IDs and 5,120 raw
logprob-bit comparisons. These are rollout timings, not training step times.
The earlier simplified baseline took 2.2479 seconds, and native took 1.6668 seconds
on that fixed workload in earlier runs.

The SwiGLU fusion initially measured slower in rollout. Full-model tracing
confirmed 2,304 fewer launches per rank; same-process alternating FFN/communication
replay improved 42.0290 -> 39.8478 microseconds, and a reversed-order rollout
recheck improved 2.0537 -> 2.0079 seconds with identical token and logprob bits.
IPC completion fusion remained slower after implementation review and is excluded.
IPC collectives remain unchanged.

The expanded GPU/integration suite reports 130 passed, one skipped and one
existing integer-square-bin kernel-loading failure (HIP status 500), reproduced
on the unchanged baseline. Five separate SwiGLU graph tests pass, including
changed inputs and weights. Raw traces and experiment scripts stay outside the
repository. These checks do not establish a new full parallelism matrix or
a 200-step performance guarantee.

The incremental GEMM/Attention/SwiGLU suite reports 46 passed, including default
GEMM dispatch across batch sizes and Attention graph replay with changed queries
and sequence lengths. This suite overlaps the earlier tests; counts are not additive.
