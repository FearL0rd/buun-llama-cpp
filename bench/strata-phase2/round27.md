# Round 27 — graph-cache pointer-key fix: verified (2026-10-03)

Change under test (commit 0e100d9b7): fold nodes[0]'s data pointers into the
O(1) CUDA-graph cache key, so the scheduler's input-copy ring
(GGML_SCHED_MAX_COPIES=4) lands each slot in its own cache entry — captured once,
replayed thereafter — instead of recapturing the verify tail graph every cycle.
Cache cap raised 64 -> 256 for the extra variants. Node-props memcmp still gates
every replay.

| scenario | tg t/s (median) | warmup resets | node_props recaptures |
|---|---|---|---|
| n=2 pre-fix (round 23) | 62.97 | ~623 / 589 cycles | 1,177 |
| **n=2 with fix** | **64.29** | **29 / 625 cycles** | **32** |
| n=3 pre-fix | 61.64 | ~623 | 1,177 |
| n=3 with fix | 62.39 | 38 / 589 cycles | 42 |

Results:
- The recapture storm is eliminated exactly as designed. The residual ~30
  resets are load-time captures plus rare variants; the remaining uid reasons
  (~1,200) are benign log lines — uid differs on rebuild but properties match,
  so nothing recaptures.
- Decode gain: +2.1% at n=2 (62.97 -> 64.29), +1.2% at n=3. n=2 remains the
  optimum with the fix. Cumulative decode from session start: 59.8 -> 64.3 t/s
  (+7.5%) once n=2 is applied to production.
- Honest accounting: the per-cycle recapture cost was only ~1 ms, not the
  ~15 ms projected from the warmup+capture passes. Verify wall time at n=2 is
  still ~31 ms with 3-4 ms of GPU kernel time; the remaining ~27 ms is
  split-boundary orchestration — ~5 per-backend pipeline segments per forward,
  each paying serialized event-wait / peer-copy / replay handoffs. That is the
  next (deeper, diminishing-return) decode target: fewer splits per forward or
  cheaper handoffs, not more graph work.

Production rollout (pending user action):
1. config.ini, Coder section: spec-draft-n-max = 3 -> 2 (one line).
2. Restart the service (it is down since this bench; the binary already carries
   the fix). Expected steady state: ~64-65 t/s decode, prefill unchanged.

Raw: round27-results.md; ~/.strata-bench/phase2-round27-results.md on AISERVER.
