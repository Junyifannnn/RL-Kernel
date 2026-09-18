# CLI Reference

RL-Kernel exposes `rlk-repro` for the supported Qwen3-8B VIME consistency
workflow:

```bash
rlk-repro prepare --workspace /data/rlk-repro --download-data --convert-checkpoint
rlk-repro doctor --workspace /data/rlk-repro
rlk-repro plan --workspace /data/rlk-repro --mode consistency
rlk-repro run --workspace /data/rlk-repro --mode consistency --rollouts 8 --wait
rlk-repro validate --run-dir /data/rlk-repro/data/runs/convergence/<run-id> --seal
rlk-repro report --workspace /data/rlk-repro
```

Use `COMMAND --help` for path overrides and execution options. `native` runs
VIME's operators and `consistency` runs RL-Kernel's operators; neither mode
reuses rollout log-probabilities. See the
[Qwen3-8B consistency guide](../usage/qwen3-vime-consistency.md) for required
hardware, runtime setup, Ray configuration, and profile customization.

## Developer commands

The repository also contains lower-level developer commands:

```bash
python scripts/run_perf.py
python benchmarks/benchmark_sampling.py
python benchmarks/benchmark_grpo_op.py
```
