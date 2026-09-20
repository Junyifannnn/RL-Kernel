# Historical G11 versus the current active execution path

Compared the sealed historical G11 RL-Kernel, VIME and Megatron sources with
PR #437 at `1ab26ec`, including newly added files. All three historical revision
and dirty-diff seals match. The historical vLLM checkout is corroborating evidence
only: the run did not seal it. The preserved G10 VIME checkout no longer matches
its recorded seal, so an exact historical native baseline is not reconstructed.

The requested repeated E2E experiment was interrupted. Only focused operator and
serializer checks were run for this follow-up. No performance target is marked met.

## What the reported 29.33% slowdown measures

The screenshot's 71.8858 versus 55.5841 seconds came from `sparse-graph-e2e`,
a three-step diagnostic whose additional graph optimization was subsequently
withdrawn. These timings must not be attributed to the current published source.
They still locate the time gap in that experiment:

| Mean seconds per step | Native | Strict | Strict minus native |
|---|---:|---:|---:|
| Rollout | 40.1440 | 53.5601 | 13.4161 |
| Training | 13.7920 | 16.6319 | 2.8399 |
| Step | 55.5841 | 71.8858 | 16.3017 |

Rollout accounts for 82.30% of that measured gap. Even at step 2, where both arms
generated eight responses of 6912 tokens, rollout was 47.0372 versus 61.4790 seconds.
Actor training was 6.0653 versus 9.8787 seconds; the independent old-logprob forward
was 2.5344 versus 2.5405 seconds. Thus simply disabling rollout-logprob reuse does
not explain the strict/native gap. Equal lengths do not imply identical token
values or identical complete trajectories. Native FFN execution provenance also
remains incomplete, so the baseline is provisional.

Historical G10/G11 both enabled reuse and mismatch metrics; the independent
training logprob forward already existed. The historical 200-step means were
99.2430/99.0377 seconds (-0.2069%); their first three means were
90.6144/98.3222 seconds (+8.51%). A three-step comparison cannot establish the
historical 200-step advantage.

## Two missed implementation differences corrected

1. `csrc/hip/hip_deterministic_logp_kernel.hip`: the forward fixed-128 specialization
   had been restored, but backward still used the later dynamic strided loop.
   Restore the historical one-element-per-lane backward at width 128. Other widths
   retain the existing dynamic path. The temperature and nucleus membership are
   still runtime inputs.
2. `rl_engine/integrations/vllm_runtime.py::_patch_tokens_api_top_logprobs`: the pinned
   vLLM 0.26 token-in/token-out endpoint already emits `token_id:<id>` entries.
   The compatibility wrapper rebuilt those objects after the native implementation
   had created them. Keep the native serializer on that API; retain the repair for
   the older disaggregated API. This restores the historical native serving path.

The compiled extension passed 45 focused ROCm tests. A separate 45-case comparison
with the preceding extension found identical backward bytes for FP32/BF16/FP16,
widths 3/65/128/129/513 and temperatures 0.3/0.7/1.6. Local serializer/worker tests:
14 passed. The actual installed native serializer remained unwrapped; its complete
output matched the previous wrapper on a 6912-token synthetic response.

At width 128, isolated backward medians changed from 4.3069 to 4.2186 microseconds
for 4096 rows and 6.8432 to 6.7367 microseconds for 8192 rows. These tiny savings
cannot explain seconds of E2E latency. CPU serialization of the synthetic response
changed from 50.17 to 33.95 milliseconds, with cyclic GC controlled identically and
output destruction excluded. Neither result is an E2E speedup claim. Raw samples,
source hashes and correctness results are in `backward-serializer-alignment-audit.json`.

## Differences that still execute

| Area | Historical implementation | Current implementation and consequence |
|---|---|---|
| Top-p support transport | Fixed capacity 64 plus asynchronous overflow assertion in the corroborating vLLM snapshot | Actual worker sampler uses `finite_counts.max().item()` to choose a complete payload. This adds a host/device synchronization versus that snapshot. The **current native sampler also does this**, so it is not an exclusively strict-path cost or a complete attribution of the strict/native gap. |
| CP parameter gradients | QKV/output/LM-head parameter sums used local tokens | `canonical_cp.weight_gradient` gathers and orders complete CP tokens before GEMMs. This is active at training TP4/CP2 and preserves partition-independent gradient arithmetic. |
| Norm and embedding gradients | Framework local backward | Canonical CP norm backward computes local activation gradients and full-token parameter gradients; shared Q/K gamma also gathers TP heads. Embedding backward and the exact integer-bin gradient norm add work absent historically. |
| FFN backward | Already gathered five CP payloads in one collective | Current code additionally orders logical tokens and fixes gradient ownership. The five-tensor gather itself is **not newly introduced**. |
| Raw sampler logits | `.contiguous()` on a narrowed shard | Protective `.clone()` avoids aliasing when a single-row view is already contiguous and native sampling mutates its input. Reverting it would reintroduce a correctness error. |
| Compiled arithmetic | No explicit precision-emulation configuration | `emulate_precision_casts=True` preserves BF16 intermediate rounding. Its independent runtime cost has not been measured; it must not be blamed from source diff alone or disabled without bitwise validation. |

The canonical training changes are numerical correctness changes, not a necessary
cost of parsing configurable CLI arguments. Restoring the historical local
reductions wholesale would remove the added cross-configuration gradient contract.
Likewise, reverting to capacity 64 with overflow failure would restrict valid
top-p/temperature combinations. This follow-up does neither.

## Differences excluded from this configuration

- The Triton chunked Attention core, deterministic GEMM and deterministic collective
  GPU kernel sources match the sealed G11 sources.
- With canonical TP=physical TP=4, the extra multi-chunk QKV/FFN/output-projection
  branches do not execute. With rollout CP=1, the PCP transport does not execute.
- The active GPU worker sampler reads top-p state from CPU NumPy metadata. The
  classic sampler's GPU boolean branch is not its hot path.
- Sparse scoring still consumes materialized LM-head logits and the shared HIP
  logp/monitor-entropy operation. No duplicate LM-head projection was found there.
- Complete-support transport, configurable sampling and independent training/rollout
  topology remain enabled. Default rollout-logprob reuse remains off.

There is no remaining evidence that switching one Attention flag, restoring a
few lines, or re-enabling reuse explains the entire gap. The two verified missed
alignments above are small. The residual E2E gap is not fully attributed; this
audit does not establish that the historical advantage is impossible, nor promise
that it has been recovered. No graph cache, speculative gather, deferred transport
or communication-elision optimization was introduced.
