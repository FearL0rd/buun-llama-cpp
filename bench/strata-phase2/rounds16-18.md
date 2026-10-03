# Rounds 16-18: the fit path, three bugs deep (Coder on AISERVER)

Working hypothesis from the profile: the Coder runs with 27.5 GB CPU_Mapped experts +
48 GB H2D streaming because the moe cache never arms. Tracing and fixing the chain:

1. **Stock-fit gate bug (fixed, `common/fit.cpp`)**: the gate declared "the complete
   model already meets the fit targets" because it only checked free-space margins —
   which under-placement trivially satisfies — and exited with the cache disarmed.
   Fixed: fall through when >256 MiB of model weights are host-resident (the measured
   host breakdown entry, dmds_full.back().mb.model — the no_alloc dry load cannot see
   placement via llama_model_has_host_moe_weights, it must come from the breakdown)
   and a MoE cache is requested. Verified: the new INF line fires with 27,788 MiB.
   Two commits on strata (first attempt used the no_alloc-blind API, second uses the
   measured breakdown).
2. **n-gpu-layers pin abort (documented)**: with `n-gpu-layers = 999` in the model
   section, the re-placement aborts ("already set by user to 999"). Note: rounds 14/18's
   "fitfree" attempts to remove it silently failed — an awk ordering bug (the header rule
   `next`ed before the flag rule ran), fixed with a python config editor that asserts
   both edits (`scripts/edit-coder-config.py`).
3. **Auto-placement over-commit (new, open)**: with the pin removed, the fit re-places
   and moves ~9 GB onto the 3090 (23.7/24.6 GB used) — then the first prefill crashes
   the server (no room for the compute buffer). The fit's margin model under-reserves
   for the prefill graph on the main GPU. Round 18b rows show the crash (11.25 s
   request, then connection refused).

## Current state and options

- Prefill remains ~1,040 t/s (Coder) / 585 t/s (Flash-Next) in production, unchanged
  and stable (the crashes above are bench-only configs).
- Path A (next): manual placement via `ot` tensor overrides — move the CPU_Mapped
  expert tensors onto the V100s' free VRAM deterministically, bypassing the fit's
  margin math entirely. Needs the exact qwen4exp expert tensor names.
- Path B: fix the fit's margin computation (code; the third fit-path bug).
- Path C (independent): the Strata PR #540 Volta ports (bf16-via-f16 GEMMs, sm_70
  attention kernel) — attack ~15% of measured prefill kernel time with a reference
  implementation, no placement changes needed.

The fit-path bugs are real fork findings worth fixing regardless (they affect every
multi-GPU rig that runs a model with host-mapped experts), but the residency goal may
be reachable faster via Path A.
