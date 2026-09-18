# Qwen3-8B VIME train–rollout consistency

This guide is the supported path for a new user who wants to compare native
VIME operators with RL-Kernel operators on Qwen3-8B. The standard experiment
does **not** require edits to `run_arm.py`, VIME model scripts, or shell
launchers. Machine-specific paths belong in CLI options or `RLK_REPRO_*`
environment variables; stable experiment changes belong in a copied profile.

## Supported reference configuration

The bundled profile targets one Linux node with eight 80 GB NVIDIA H100 GPUs.
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
routing rules, validation gates, or Ray runtime construction. Such changes
must update the profile, manifest schema or validator when applicable, and the
launcher tests. Host paths, model locations, output directories, and Ray API
addresses are not reasons to edit a script.

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
