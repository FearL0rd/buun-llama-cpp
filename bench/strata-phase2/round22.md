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

Interpretation (corrected after cycle-time math): acceptance is identical across
splits (same model, prompt, and sampling), so the implied spec-cycle time is
46.0 ms (3 GPU) vs 48.4 ms (2x V100) — only +5% despite each V100 carrying ~40%
more layer bytes in the 2-GPU split. Decode wall time is therefore NOT carried by
per-GPU weight-read bandwidth or compute; it matches round 21's profile (GPU busy
only ~8% of the cycle): the dominant cost is per-op host dispatch + sync structure,
plus a small bandwidth-sensitive remainder (which is what the 2-GPU split lost).
The "fewer pipeline boundaries" lever is dead — removing one of three backends
does not pay.

Consequences for the decode queue:
- draft depth (verify forward is batch-insensitive): next — round 23
- decompose the ~42 ms/cycle of non-kernel time: nsys cuda_api_sum + gpu-trace
  gaps on the existing round-21 capture (decode-r21.sqlite)
- fusion / op-count reduction for decode: back on the table pending that profile
- MTP head GEMV: 20.1% of kernel time, but kernel time is only ~8% of the cycle
  (~1.5% of wall) — draft-vocab restriction parked

Raw data: round22-results.md (also ~/.strata-bench/phase2-round22-results.md on AISERVER).
