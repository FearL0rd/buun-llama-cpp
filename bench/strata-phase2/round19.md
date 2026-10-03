# Round 19 (manual ot residency) — partial success, checkpoint (2026-10-03)

Setup: `ot = blk.3[2-9].=CUDA1,blk.4[0-7].=CUDA2` (whole layers 32-47) and an
experts-only variant, ctx 131072, n-gpu-layers still pinned, gate fix in the binary.

What happened:
- The ot overrides applied (option accepted; CUDA1 +4.5 GB) — but the bytes came from
  CUDA2 (-4.5 GB), NOT from CPU: `CPU_Mapped = 27465.95 MiB` unchanged. The host-mapped
  set is therefore not simply "layers 32-47"; the actual tensor->device map must be dumped
  before the next attempt (per-tensor "overriding" lines are not logged at this verbosity;
  use -lv 4 or parse the loader plan).
- The per-model CHILD crashed on the first long prefill (3.58 s error response), the
  front stayed up, refused briefly, then auto-reloaded and served decode (37.6 t/s —
  degraded vs the 47 t/s stock). Child crash signature not yet captured (front log has
  no error lines; the child's stderr needs capture).
- No host OOM (125 GB RAM, 121 available, no kernel events).

Conclusions at checkpoint:
1. Manual ot placement is mechanically viable and bypasses the fit's buggy margins.
2. Two open items before it can work: (a) dump the real placement map and pin the
   CPU_Mapped tensors specifically (they are expert tensors of specific layers — get
   names from a -lv 4 load or the loader plan); (b) capture and diagnose the child crash
   under mid-range pinning (could be cross-device layer splits breaking hybrid state
   handling — the pinned layer's recurrent state may need to stay with its experts).
3. Production is untouched and stable at the verified numbers; all of this is bench-only.

Queue for next session (in order): fix the pin ranges from the real map + child-crash
capture -> PR #540 port 1 (bf16-via-f16 on sm_70, plan in path-c-port-plan.md) ->
PR #540 port 2 (sm_70 attention kernel) -> 2-stage split test -> cross-card prefill
chunk pipelining.
