# Round 20: the placement map — the residency problem is ONE tensor (2026-10-03)

New tooling: env-gated placement dump (`LLAMA_DUMP_PLACEMENT=1`, implemented in
common/common.cpp after model load, using llama_internal_get_tensor_map; declaration added
to src/llama-ext.h). Captured 1,224 tensor->buffer assignments for the Coder:

| buffer | tensors | MiB | content |
|---|---:|---:|---|
| CUDA0 (3090) | 359 | 7,341 | layers 0-13 |
| CUDA1 (V100#1) | 455 | 10,478 | layers 14-27 |
| CUDA2 (V100#2) | 408 | 10,085 | layers 28-47 (incl. all of 32-47 — round 19's pins hit these) |
| CPU_Mapped | **1** | **27,466** | **per_layer_token_embd.weight** |
| CUDA_Host | 1 | 322 | pinned staging |

**Every MoE expert tensor is GPU-resident. The entire host-mapped 27.5 GB is the PLE
(per-layer token embedding) table** — a single bf16 tensor created with TENSOR_READ_LAZY
in src/models/qwen4exp.cpp:210, gathered row-by-row per layer per token during prefill.
That row gathering is the 48 GB/request H2D traffic from the round-12 profile. All
previous "expert residency" framing was wrong; round 19's whole-layer pins could never
have moved it (layers 32-47 were already fully on CUDA2).

The fork was built for the fix: qwen4exp.cpp already handles a quantized PLE — explicit
F8_E4M3 support with per_layer_token_embd_scale/bias tensors (line 207-217) — and
Strata ships exactly this optimization (its packer quantizes PLE to IQ4; "ple_iq4.cpp").

## Next work (crisp)

1. Produce a quantized-PLE Coder GGUF: `llama-quantize` has `--tensor-type
   per_layer_token_embd=IQ4_XS` (or fp8-e4m3 with a synthesized scale) + `--include-weights`
   + a `COPY` base type so everything else is copied unchanged. One-time ~40-90 min job.
2. Validate the TENSOR_READ_LAZY gather path handles the quantized type on CUDA
   (get_rows on IQ quants is standard; fp8-with-scale is the fork's own supported path).
3. Bench: expect the H2D traffic to collapse and prefill to move off the 1,040 t/s
   plateau by the streamed-gather share (PLE was the dominant PCIe cost).
4. Re-check decode: PLE lookups also occur at decode (fewer rows); decode may gain too.

Accuracy note: Strata ships IQ4 PLE in its own Coder builds; the fork's fp8+scale path
is the higher-fidelity option. Both beat bf16-from-host streaming on speed by design.
