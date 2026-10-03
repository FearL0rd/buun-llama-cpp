# Round 26 — CUDA graph recapture diagnosis (2026-10-03)

Question: which trigger in ggml_cuda_graph_update_required (ggml-cuda.cu) forces the
per-cycle recapture of the verify CUDA graphs (623 warmup resets vs 589 cycles in
round 23)?

Method: env-gated instrumentation (GGML_CUDA_LOG_GRAPH_UPDATE, commit 1f8c7b236),
one production-config decode scenario (MTP n3), 589 cycles. t/s unchanged
(61.7/61.2/60.9) - logging is free.

Reason counts (a single recapture can log uid then node_props):
- reason=node_props: 1,243  (dominant)
- reason=uid:        1,248  (pairs with node_props)
- reason=fattn_epoch:    24  (load-time only)
- reason=node_count:     19  (first captures: 1985/2391/2162/150-node shapes)

Decode-phase detail:
- Verify graph uids increment +6 per cycle (prev=3900 new=3906 ...) - a fresh
  cgraph identity every cycle, never repeating. The sidecar's uids are stable
  (3757/3758/3759) - that is why only the sidecar replays.
- node_props always fails at i=0, op=RMS_NORM, name="norm-48" (1,177x) - the
  first node of the owning backend's subgraph. The uid mismatch is not itself
  fatal; the memcmp of the full ggml_tensor struct (including data pointers)
  is what forces the warmup + recapture every cycle.

Interpretation: the verify forward's compute graph is structurally rebuilt every
decode cycle (fresh cgraph, fresh uid, fresh tensor structs), and the rebuilt
tensors do not land on the previous cycle's addresses - unlike stock llama.cpp,
where the graph allocator is deterministic per shape and rebuilt graphs memcmp
clean, giving stable CUDA-graph replays.

Next: log old/new data + src pointers at the first mismatching node to identify
the rotating buffer, then fix the instability (graph-object reuse or address
stability for the verify path) so verify replays a stable graph. Projected:
verify 36 ms -> ~10 ms, decode 61 -> 100+ t/s.

Raw: ~/.strata-bench/phase2-round26-results.md on AISERVER; full log
~/.strata-bench/server-r26-diag.log.
