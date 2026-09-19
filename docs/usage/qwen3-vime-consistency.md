# Qwen3-8B VIME train–rollout consistency

This guide is the supported path for a new user who wants to compare native
VIME operators with RL-Kernel operators on Qwen3-8B. The standard experiment
does **not** require edits to `run_arm.py`, VIME model scripts, or shell
launchers. Machine-specific paths belong in CLI options or `RLK_REPRO_*`
environment variables; stable experiment changes belong in a copied profile.

## CUDA quick path

On one 8×H100 node, a new user only needs to provide the local Qwen3-8B
checkpoint and a workspace. No VIME script or RL-Kernel example file needs to
be edited:

```bash
python3 -m pip install -e .

export RLK_REPRO_WORKSPACE=/data/rlk-repro
export RLK_REPRO_MODEL_ROOT=/models/Qwen3-8B

rlk-repro prepare \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --download-data \
  --convert-checkpoint

ray start --head \
  --include-dashboard=true \
  --dashboard-host=127.0.0.1 \
  --num-gpus=8 \
  --object-store-memory=200000000000

rlk-repro doctor --workspace "$RLK_REPRO_WORKSPACE"
rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode native \
  --rollouts 8 \
  --wait
rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --rollouts 8 \
  --wait
```

The default CUDA command uses the reference training topology TP4/CP2 and two
TP4/CP1 rollout engines. Use `rlk-repro plan` before every non-default
topology. Formal evidence runs must use clean source checkouts and must not pass
`--allow-dirty`.

## Supported reference configuration

The bundled CUDA profile targets one Linux node with eight 80 GB NVIDIA H100 GPUs.
Megatron runs TP4/CP2 and two colocated vLLM engines run TP4. The frozen runtime
contract is Python 3.11.15, PyTorch 2.9.1, vLLM 0.16.0, Ray 2.57.0, and
Transformer Engine 2.18. Other hardware or topology is development work, not a
drop-in path change.

Before starting, provide:

- a VIME-compatible Python environment containing the frozen dependencies;
- a local Hugging Face Qwen3-8B checkpoint;
- enough disk for source checkouts, the converted Megatron checkpoint, run
  artifacts, and per-step debug tensors;
- a large `/dev/shm` or a deliberately sized Ray object store and spill path.

The launcher uses the active Python environment by default. Set
`RLK_REPRO_RUNTIME_ROOT` only when the experiment must run in a different
environment.

## Topology support and evidence

Topology validity and end-to-end evidence are different things. The launcher
accepts any topology that passes its GPU-count and Qwen3-8B sharding checks,
while the table below records how much runtime evidence each layout has.

| Backend | Training topology | Rollout topology | Status |
|---|---|---|---|
| CUDA H100 | TP4/CP2 | TP4/CP1, two engines | Supported reference topology |
| CUDA H100 | TP1/CP8 | TP4/CP1, two engines | One-round strict smoke passed on September 19, 2026 with zero mean/max LogP difference and zero mismatches |
| CUDA H100 | TP2/CP4 | TP4/CP1, two engines | One-round strict smoke passed on September 19, 2026 with zero mean/max LogP difference and zero mismatches |
| CUDA H100 | TP8/CP1 | TP4/CP1, two engines | One-round strict smoke passed on September 19, 2026 with zero mean/max LogP difference and zero mismatches |
| CUDA H100 | Any valid training TP×CP | Rollout CP greater than 1 | Experimental; requires vLLM prefill context-parallel integration and separate evidence |
| CUDA H100 | TP4/CP2 | TP2/CP1 or TP8/CP1 | One-round strict smoke passed on September 19, 2026 with zero mean/max LogP difference and zero mismatches |
| CUDA H100 | TP4/CP2 | TP1/CP1 | Canonical-shard code path is present; end-to-end validation pending |
| ROCm MI300X | TP4/CP2 | TP4, two engines | Backend/topology has 200-step evidence; rerun the new no-reuse pair |
| ROCm gfx942 | Other valid TP×CP | Rollout TP dividing available GPUs | Experimental; configuration checks only |

The validated TP2/CP4 CUDA smoke uses the same TP4 rollout topology:

```bash
rlk-repro plan \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --tp-size 2 \
  --cp-size 4 \
  --rollout-tp-size 4 \
  --rollout-cp-size 1

rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --tp-size 2 \
  --cp-size 4 \
  --rollout-tp-size 4 \
  --rollout-cp-size 1 \
  --rollouts 8 \
  --wait
```

The same command form covers every validated eight-GPU training factorization.
For example, TP8/CP1 is:

```bash
rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --tp-size 8 \
  --cp-size 1 \
  --rollout-tp-size 8 \
  --rollout-cp-size 1 \
  --rollouts 8 \
  --wait
```

Passing argument validation proves only that GPU counts, Qwen3-8B sharding,
and manifest relationships are coherent. A new training topology is supported
only after both `native` and `consistency` pass the runtime validator on the
target hardware.

TP1/CP8 keeps the complete actor weights on every GPU. The CUDA launcher
therefore defaults vLLM memory utilization to `0.2` for training TP1 and `0.4`
otherwise. Users normally do not need to pass
`--vllm-gpu-memory-utilization`; an explicit value still overrides the
topology-aware default.

### One-round CUDA smoke performance

These numbers are single-run, 256-response-token smoke measurements from one
8×H100 node on September 19, 2026. They verify that the strict paths remain
usable; they are not a statistically controlled benchmark or a long-run
throughput claim.

| Training topology | Rollout topology | Train tokens/s | Rollout tokens/GPU/s | Step time |
|---|---|---:|---:|---:|
| TP4/CP2 | TP4/CP1 | 1292 | 71.7 | 28.0 s |
| TP2/CP4 | TP4/CP1 | 1195 | 71.6 | 29.3 s |
| TP1/CP8 | TP4/CP1 | 882 | 71.9 | 32.2 s |
| TP8/CP1 | TP4/CP1 | 1328 | 52.7 | 29.4 s |
| TP4/CP2 | TP2/CP1 | 1292 | 45.1 | 33.3 s |
| TP4/CP2 | TP8/CP1 | — | 51.8 | — |

The TP4/CP2 reference retains its original single-shard launches. Coarser
physical TP uses additional canonical-shard GEMM launches, so equal bitwise
results do not imply equal throughput. Different rollout TP values also change
the number of engines and request concurrency.

## Canonical topology strategy

The non-default CUDA work reuses the TP4/CP2 implementation rather than
maintaining a second operator stack. The launcher records a shared
`canonical_tp`, currently the finer of training TP and rollout TP. A physical
rank that owns more than one canonical shard executes those shards separately
and combines them with the same fixed tree used by the finer topology.

For the validated TP2/CP4 training and TP4/CP1 rollout pair:

- one TP2 rank owns two adjacent canonical TP4 shards;
- Attention output projection computes two TP4-width partial projections
  before the existing deterministic TP reduction;
- FFN gate, up, and down projections execute at TP4 shard widths, and the down
  partials are combined in the TP4 fixed-tree order;
- selected-token logp exposes TP4-sized virtual vocabulary summaries and
  merges them in ascending global-vocabulary order;
- CP changes token ownership only. Strict Attention reconstructs global
  positions before the attention kernel, and CP is not a logp reduction axis.

This keeps the reference TP4/CP2 hot path unchanged: `canonical_tp=4` gives
exactly one shard per TP4 rank, so it does not add projection launches or a new
reduction. Coarser TP layouts add launches but retain the existing
cuBLASLt no-split-K kernels and fixed-tree collectives. Their performance must
be measured; bitwise success alone is not a throughput claim.

TP1/CP8 uses four TP4 virtual shards per physical training rank. TP8/CP1 or
rollout TP8 selects canonical TP8; the vLLM QKV projection, output projection,
packed FFN, and padded-vocabulary logp statistics use the corresponding TP8
virtual shards while retaining CUDA Graph capture. Rollout CP greater than 1
is a separate vLLM PCP integration task, not just another value for
`canonical_tp`.

## Prepare once

From a clean RL-Kernel checkout, activate the experiment environment and run:

```bash
python3 -m pip install -e .

export RLK_REPRO_WORKSPACE=/data/rlk-repro
export RLK_REPRO_MODEL_ROOT=/models/Qwen3-8B

rlk-repro prepare \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --download-data \
  --convert-checkpoint
```

`prepare` pins VIME and Megatron-LM to the profile revisions, downloads and
converts DAPO-Math-17k, and creates the Megatron torch-dist checkpoint from the
local Qwen3-8B model. It does not download model weights or create a Python
environment. Existing source directories must be clean because selecting a
profile revision changes their checked-out commit.

If Transformer Engine or CUDA Python lives outside the active environment,
set these before `doctor` and `run`:

```bash
export RLK_REPRO_TE_ROOT=/opt/transformer-engine/site-packages
export RLK_REPRO_CUDA_PYTHON_SITE=/opt/cuda-python/site-packages
export RLK_REPRO_CUDA_RUNTIME_ROOT=/opt/python/site-packages/nvidia/cuda_runtime
```

Use path flags with the same names (`--te-root`, `--cuda-python-site`, and
`--cuda-runtime-root`) when per-command configuration is clearer.

## Start Ray and check the host

Start one Ray head node. Size the object store for the host; the reference run
used 200 GB:

```bash
ray start --head \
  --include-dashboard=true \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=8265 \
  --num-gpus=8 \
  --object-store-memory=200000000000 \
  --temp-dir=/tmp/rlk-repro

rlk-repro doctor --workspace "$RLK_REPRO_WORKSPACE" --mode native
rlk-repro plan --workspace "$RLK_REPRO_WORKSPACE" --mode consistency
```

Do not continue past a failed `doctor`. Its path report is also the fastest way
to see which model, checkpoint, data, runtime, or source location the launcher
resolved.

## Run the paired comparison

Use an eight-step pair first. `--wait` streams the Ray job and saves `run.log`
and `ray-status.txt` in the append-only run directory:

```bash
rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode native \
  --rollouts 8 \
  --wait

rlk-repro run \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --rollouts 8 \
  --wait
```

`native` uses VIME's native Attention, FFN, and logp operators.
`consistency` uses RL-Kernel for all three. Neither mode reuses rollout
log-probabilities. After the short pair passes, repeat both commands with
`--rollouts 200` for the primary evidence run.

Each invocation prints its `run_dir`. Validate and seal successful runs:

```bash
rlk-repro validate --run-dir /data/rlk-repro/data/runs/convergence/<run-id> --seal
rlk-repro report --workspace "$RLK_REPRO_WORKSPACE"
```

The validator checks the topology, operator readbacks, CUDA Graph settings,
fallback markers, step count, and train–rollout log-probability evidence. A
passing sealed run contains `COMPLETE`.

## ROCm MI300X and gfx942

ROCm uses a separate launcher because its strict path requires AITER/CK, RCCL,
HIP Graph settings, and ROCm-specific runtime evidence. The user-facing
`rlk-repro` command and topology flags are nevertheless shared with CUDA.
Activate a ROCm VIME environment, install this checkout with
`pip install -e .`, and stop any existing Ray cluster; the ROCm launcher owns a
fresh cluster for each arm.

Run the no-rollout-logprob-reuse pair with unique append-only directories:

```bash
rlk-repro plan \
  --backend rocm \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode native \
  --rollouts 8

rlk-repro run \
  --backend rocm \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode native \
  --rollouts 8 \
  --run-id rocm-native-8 \
  --wait

rlk-repro run \
  --backend rocm \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --rollouts 8 \
  --run-id rocm-consistency-8 \
  --wait
```

Both commands recompute training logp. Neither sets
`RLK_ABLATION_USE_ROLLOUT_LOGPROBS` or passes `--use-rollout-logprobs`.
The runner validates readbacks and mismatch artifacts before returning success.

ROCm accepts the same `--rollouts`, `--tp-size`, `--cp-size`,
`--rollout-tp-size`, and `--rollout-cp-size` flags as CUDA. It additionally
accepts `--num-gpus` and `--visible-gpus` for hosts whose visible device set is
not the default `0,1,2,3,4,5,6,7`. The compatibility spelling
`--num-rollout` remains accepted by the direct Python launcher, but new user
commands should use `--rollouts`.

The topology validator requires:

- training `TP × CP == num_gpus` for colocated execution;
- both training TP and rollout TP to divide Qwen3-8B's 32 attention heads,
  eight query groups, and padded vocabulary;
- rollout `TP × CP` to divide `num_gpus`;
- enough generated requests to feed every rollout engine.

For example, the following configurations can be planned without editing any
Python or shell file:

```bash
# TP2/CP4 training, two rollout engines using TP2/CP2 each.
rlk-repro plan \
  --backend rocm \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --rollouts 8 \
  --tp-size 2 \
  --cp-size 4 \
  --rollout-tp-size 2 \
  --rollout-cp-size 2

# TP8/CP1 training and one TP8/CP1 rollout engine.
rlk-repro plan \
  --backend rocm \
  --workspace "$RLK_REPRO_WORKSPACE" \
  --mode consistency \
  --rollouts 8 \
  --tp-size 8 \
  --cp-size 1 \
  --rollout-tp-size 8 \
  --rollout-cp-size 1
```

The launcher records the finer of training TP and rollout TP as
`canonical_tp`, matching CUDA's topology strategy, and forwards rollout CP to
vLLM prefill context parallelism. The TP4/CP2 training with TP4/CP1 rollout
layout remains the ROCm reference configuration. Every other topology is
experimental until an eight-step native/consistency pair passes on the target
ROCm system; argument validation alone is not a bitwise or performance claim.

The direct module remains available for automation and older scripts:

```bash
python -m examples.vime_rocm_attention_ablation.run_qwen3_8b \
  --mode consistency \
  --rollouts 8 \
  --run-dir "$RLK_REPRO_WORKSPACE/data/runs/rocm-consistency-8" \
  --tp-size 4 \
  --cp-size 2 \
  --rollout-tp-size 4 \
  --rollout-cp-size 1
```

## Configuration without script edits

Use the following order of precedence:

1. CLI path options for one-off runs.
2. `RLK_REPRO_*` environment variables for one host.
3. A copied JSON profile for a shared, reviewed experiment definition.

To create a custom profile, copy
`examples/vime_qwen3_8b_tp4_cp2_200/profiles/qwen3-8b-tp4-cp2.json`, update
repository revisions, paths, or `runner_args`, and pass `--profile FILE` to
every command. Run `rlk-repro plan` and review the expanded command before
submission.

Only change Python or shell runners when changing experiment semantics that the
profile cannot express, such as GPU topology, model architecture, operator
routing rules, validation gates, or Ray runtime construction. Supported
topology sizes are now CLI options and do not require script edits. Other such
changes must update the profile, manifest schema or validator when applicable,
and the launcher tests. Host paths, model locations, output directories, and
Ray API addresses are not reasons to edit a script.

## Common failure modes

- **`doctor` reports a missing Python or package path:** activate the intended
  environment, or set `RLK_REPRO_RUNTIME_ROOT` and the specific path override.
- **The model or reference checkpoint is missing:** set
  `RLK_REPRO_MODEL_ROOT`, then rerun `prepare --convert-checkpoint`.
- **Ray submission cannot connect:** start Ray and, for a non-default dashboard,
  pass `--ray-address http://host:port` to `plan` and `run`.
- **A source checkout is dirty:** commit or clean it. Use `--allow-dirty` only
  for disposable development runs; such runs are not publishable evidence.
- **A run directory already exists:** choose a new `--run-id`. Run directories
  are append-only and are never overwritten.

The long-form [reproduction runbook](../../examples/vime_qwen3_8b_tp4_cp2_200/REPRODUCTION.md)
remains the audit reference for historical experiments and manual recovery.
