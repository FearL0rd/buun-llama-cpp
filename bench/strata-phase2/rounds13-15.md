# Phase 2 rounds 13-15 — the residency question (Coder on AISERVER)

Background: the round-12 nsys profile showed the Coder runs with 27.5 GB of weights
CPU_Mapped (only ~28 GB on the 3 GPUs) and streams 48.4 GB host->device per capture
window, while ~21 GB sat free per V100. Prefill is partially PCIe-bound at ~1,040 t/s.

## Round 13 — ctx-size does not move the placement

ctx-size 262144 / 131072 (KV pools did shrink: V100 usage 20.6 -> 17.9 -> 15.4 GB) but the
layer placement and `CPU_Mapped model buffer size = 27465.95 MiB` stayed byte-identical;
prefill stayed ~1,020-1,044 t/s. The placement is not free-VRAM driven and not actual-ctx
driven.

## Round 14 — neither is it fit or metadata-ctx driven

- `fitfree` (n-gpu-layers unpinned so the fit system can plan): identical placement.
- `meta262k` (metadata context_length halved): identical placement.
Both to the MiB: CUDA0 7341.07 / CUDA1 10478.04 / CUDA2 10085.15 / CPU_Mapped 27465.95.
Prefill ~1,040 t/s in all arms.

## The actual driver (from the logs)

```
sched_reserve: MoE cache requested=auto resolved=off
operator(): MoE cache fit kept stock placement because the complete model already meets the fit targets
common_init_: MoE cache: mode=auto budget=free-minus-reserve
```

The fork's auto-fit concludes "the complete model already meets the fit targets" — while
27.5 GB of experts sit CPU_Mapped — and resolves the moe cache to OFF. Free VRAM is left
for the vram-demand streaming pool instead ("plan on 0000:02:00.0: 22166.0 MiB total,
10896.4 already landed, 11269.6 remaining", "demand committed — waiting for donors"), so
expert pages re-stream over PCIe per chunk. The auto decision is wrong for this rig.

## Round 15 (running) — explicit moe-cache budget

`moe-cache = 20000` (20 GB/device page budget) and `moe-cache = on`, Coder section only,
no n-cpu-moe (round 11 showed that combination crashes during prefill). Success = cache
lines resolve to pools, H2D streaming drops across repeated prefills, prefill rises above
the 1,040 t/s baseline. Risk: the round-11 crash signature (illegal access on a V100).

If it pays: production sets an explicit moe-cache budget for the Qwen3.8 models, and the
auto-fit's "complete model meets fit targets" check deserves a fork fix (it ignores
CPU_Mapped expert bytes), which is the code-level follow-up.
