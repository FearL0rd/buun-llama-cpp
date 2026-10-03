# Round 22 — decode: 2-GPU vs 3-GPU split (2026-10-03)

Question: round 21's nsys profile showed high launch counts and ~7% per-kernel GPU
utilization per decode cycle; does cutting one pipeline stage (3 backends -> 2)
speed up decode?

Setup: Coder IQ1_M + MTP sidecar (n=3), short prompt (~32 tok), 384-token decode,
3 measured runs per scenario, production config copy.
- 3gpu-baseline: CUDA_VISIBLE_DEVICES=1,3,0 — production split (3090 main + 2x V100)
- 2v100: CUDA_VISIBLE_DEVICES=0,3 — both V100s, ctx reduced to 131072 for KV headroom

| scenario | tg t/s (runs) |
|---|---|
| 3gpu-baseline | 61.5 / 61.3 / 62.0 |
| 2v100 | 58.5 / 58.5 / 58.6 |

Result: the 2-GPU split loses ~5%. The 3-GPU production split stays.

Interpretation: decode wall time is carried by per-GPU layer compute (kernel
latency), not by dispatch/launch overhead. Dropping the 3090 (the fastest dp4a
card in the mix) forces each V100 to carry ~14 GB of layers instead of ~10 GB, and
the extra per-GPU work outweighs the one removed pipeline boundary + backend.

Consequences for the decode queue:
- launch-count trimming / further fusion for decode: deprioritized
- draft depth (verify forward is batch-insensitive): next — round 23
- MTP head GEMV (20.1% of decode kernel time): draft-vocab restriction candidate

Raw data: round22-results.md (also ~/.strata-bench/phase2-round22-results.md on AISERVER).
