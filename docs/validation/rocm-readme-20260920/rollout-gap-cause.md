# Rollout gap diagnosis

The largest measured regression is the default ROCm CPU affinity, not the
Triton Attention arithmetic. The launcher exported `AMD_CPU_AFFINITY=0`, and
the vLLM/Ray worker threads then ran on CPU 0–7 while only two process threads
retained the full 0–159 mask. This throttles the host side of decode, sampler
support preparation, and Ray/vLLM scheduling.

The controlled one-step strict run kept the model, prompts, TP4/CP2 training,
TP4/CP1 rollout, temperature 0.7, top-p 0.95, and `use_rollout_logprobs=0`
fixed. It produced 35,878 tokens and matched all 143,512 audited log-probability
bits. Narrow affinity measured 86.2402 s rollout / 125.1104 s step; widening
only the vLLM descendants' CPU mask measured 50.4340 s rollout / 103.0554 s
step. The exact token and log-probability digests were unchanged.

The historical G10/G11 launch snapshots do not set `AMD_CPU_AFFINITY`. Their
200-step rollout means were 81.4696 s and 68.0234 s. This makes the forced
eight-core mask the first confirmed current-path cause to remove when comparing
against the historical runs. It does not by itself prove that every second of
the historical 13.4462 s G10/G11 rollout difference came from affinity.

The remaining measured strict-path cost is complete dynamic top-p support:
the same historical worker changed from the old fixed-capacity transport to the
current unrestricted transport at 4.4561 -> 4.7870 s (+7.43%). That change
cannot be reverted to a hard cap because arbitrary top-p must remain supported.

The launcher now preserves an explicit `AMD_CPU_AFFINITY` supplied by the user
and leaves it unset by default, matching the historical launch contract.
