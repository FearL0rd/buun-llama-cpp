# Round 23 — MTP draft-depth sweep n={2,3,4,5} (2026-10-03)

Question: round 21 showed verify (~37 ms) is 80% of the cycle; is the verify forward
batch-insensitive so that drafting deeper raises tokens/cycle faster than cycle time?

Setup: Coder IQ1_M + MTP sidecar, production 3-GPU split, production config copy with
spec-draft-n-max set per scenario, short prompt, 384-token decode, 3 runs each.

| n | tg t/s (median) | draft | verify | cycle | tokens/cycle |
|---|---|---|---|---|---|
| 2 | 62.97 | 4.2 ms | 31.2 ms | 38.3 ms | 2.41 |
| 3 (prod) | 61.64 | 5.8 ms | 36.2 ms | 44.7 ms | 2.76 |
| 4 | 53.01 | 7.5 ms | 41.7 ms | 52.5 ms | 2.78 |
| 5 | 54.84 | 9.2 ms | 47.4 ms | 60.2 ms | 3.30 |

Result: **n=2 is fastest; production n=3 is second; deeper drafting loses badly.**

Interpretation: verify is NOT batch-insensitive — it costs ~5.3 ms per additional
token (31.2 -> 36.2 -> 41.7 -> 47.4, linear), and each draft step costs ~1.7 ms.
Marginal cost of one more drafted token is ~7 ms against ~0.2-0.5 marginal accepted
tokens — a losing trade at every depth. The whole curve fits T(n) = 24.6 + 7.0*n
(R^2 ~ 1), and the same model reproduces the no-MTP baseline (43 t/s = 23.3 ms
single-token cycle). Curiosity: the implied chain-acceptance increments are
non-monotonic (+0.35, +0.02, +0.49 tokens/cycle at n=3,4,5) — some drafting-mode
change appears to kick in at n>=5; does not affect the recommendation.

Cycle structure (n=3): verify = ~15.3 ms fixed + 5.3 ms/token; draft = 0.7 + 1.7*n;
accept+other ~3 ms. The 5.3 ms/token marginal and most of the fixed cost are
host-side, not GPU (total kernel time is 3-4 ms/cycle). The round 21 api-sum
decomposition confirms: ~77% of capture wall time sits in cudaStreamSynchronize
(7.05 s) + cudaEventSynchronize (1.43 s) vs 0.47 s in cudaLaunchKernel; ~136 stream
syncs/cycle, median 1 us, with ~11 long (~3.5 ms) waits per cycle at backend-segment
drains — waiting on segments whose GPU work is well under 1 ms.

Recommendation: set production spec-draft-n-max = 2 (+2% decode, one-line config
change; speculative decoding is lossless, n only trades speed). The bigger lever is
the sync-wait structure: round 24 sweeps --threads (decode needs almost no CPU),
followed by spin-wait device flags / fewer segment drains if it pays.

Raw: round23-results.md; ~/.strata-bench/phase2-round23-results.md on AISERVER.
