# Round 25 — expert layout A/B (default vs LLAMA_SPLIT_EXPERTS=slice) (2026-10-03)

Question: does the expert-parallel whole-expert spread (llama-model.cpp splits stacked
expert tensors into single-expert segments across device buffers) fragment decode into
cross-backend expert windows + reduce per MoE layer?

Setup: production 3-GPU split, MTP n3, threads 40, short prompt, 384-token decode.

| scenario | tg t/s (median) | cycle |
|---|---|---|
| default | 61.41 | ~44.9 ms |
| LLAMA_SPLIT_EXPERTS=slice | 61.60 | ~44.8 ms |

Result: **no effect — and the load buffers were byte-identical in both scenarios**
(CUDA1 10478.04 / CUDA2 10085.15 MiB), so slice never changed the placement. The fit
system plans this model's placement with whole tensors (the tensor-config split
machinery is bypassed), and the ot= pins in the config belong to the gemma-4 and
GLM-5.3 sections, not the Coder. Expert layout is decode-neutral as tested; the
fragmentation hypothesis is dead.

Where the decode cost actually is (established in rounds 21-25):
- ~77% of decode wall time is host-blocked in cudaStreamSynchronize (7.05 s) +
  cudaEventSynchronize (1.43 s) over the round 21 capture; GPU busy ~8%;
  cudaLaunchKernel only 0.47 s. Threads are neutral (round 24). Cycle time is
  split-invariant (round 22).
- The fork's CUDA-graph replay works — the dense MTP sidecar replays stable graphs
  every cycle (ids 3757/3758/3759 "reused"), including on the Volta — but the
  TARGET's verify graphs do not hit the stable-replay path: the round 23 log shows
  ~623 "CUDA graph warmup reset" events against ~589 spec cycles — a
  recapture/warmup essentially every cycle, paying the individual node-launch cost
  each time (~600 launches/cycle, the per-node orchestration tax that is the
  bottleneck).
- ggml_cuda_graph_update_required (ggml-cuda.cu:4064) recaptures when the
  fattn_scratch epoch changes OR any node's properties memcmp differs (the whole
  ggml_tensor struct, including node and src data pointers). Something in the
  verify graph changes every step; the sidecar proves the replay path is reachable
  on this rig.

Next: instrument ggml_cuda_graph_update_required (env-gated log of the failing
check: epoch vs uid vs node index/op) and fix the per-step instability so verify
replays a stable graph. Projected: verify 36 ms -> ~10 ms, decode 61 -> 100+ t/s.

Raw: round25-results.md; ~/.strata-bench/phase2-round25-results.md on AISERVER.
