# Strata Phase 0 baseline — AISERVER (2026-10-01)

Hardware: 2x Tesla V100-PCIE-32GB + RTX 3090 24 GB, 48-core host, `CUDA_VISIBLE_DEVICES=1,3,0`
(3090 = GPU0/main), layer split. Models: `Qwen3.8-Flash-Next-Coder` (ISTA IQ1_M) and
`Qwen3.8-Flash-Next` (unsloth UD-IQ4_XS), MTP sidecar `mtp-...-shared-Q4_K_M.gguf`, VBR KV
(turbo8, 8.125 bpv), `draft-mtp` speculation. Server command mirrors the production one
(`scripts/bench-strata-phase0*.sh`, `--models-preset` variants of `/home/cesar/models/config.ini`).

Decode numbers are the server's own `predicted_per_second` (wall-clock agrees within ~2%).
Prefill numbers use a 21,208-token code prompt with `cache-prompt = off`.

## Decode (short context, 384 tokens out, greedy)

| scenario | Coder t/s | Flash-Next t/s | notes |
|---|---:|---:|---|
| no speculation (`md` sidecar removed) | 43.4 | 42.2 | true baseline |
| MTP n2 (production config) | 56.0 | 56.1 | +30% |
| MTP n3 (adaptive depth controller on) | ~57 | ~57–59 | +33%, acceptance 0.69, mean run 2.87 |
| MTP n4 (fixed window) | 53.3 | 53.4 | worse: acceptance drops to 0.53 |
| **MTP n3 + CUDA fusion ON** | **60.4–60.7** | **59.5–59.9** | **+40% vs no-spec; production pins `GGML_CUDA_DISABLE_FUSION=1`** |
| moe-cache off | 56.2 | 56.0 | no change — experts already fit VRAM |
| moe-cache-cpu-overlap 0 | 55.3 | 55.7 | ~1% — overlap not load-bearing here |
| MTP n3 @ 21K-token context | 45.4 | 45.4 | long-context cost ~-20%; acceptance rises to 0.83 |

Spec cycle anatomy (n3): draft 7.5 ms + verify 38.1 ms + accept 2.5 ms + other 1.8 ms = 50.3 ms
for ~2.87 tokens. Verify scales sub-linearly (4-token window = 2.2x single-token cost), so
acceptance, not window size, converts to t/s. The shared MTP head runs a full-vocab GEMM
(`output.weight` is in the sidecar GGUF) for every drafted position.

## Prefill (21,208-token code prompt, uncached)

| scenario | Coder t/s | Flash-Next t/s | notes |
|---|---:|---:|---|
| `-ub 512` (production CLI; overrides the config's `ubatch-size = 8192`) | 484 | 401 | |
| `-ub 8192` | 691–735 | FAILED | +45% for Coder; Flash-Next OOMs loading mmproj (862 MB) on the 3090 after the bigger compute buffers — needs `-ub 4096` or a non-GPU0 mmproj |
| Strata reference (same Coder model) | ~2,100–2,230 on one RTX 5070 | | remaining ~3x gap is kernel/pipeline-level |

## Conclusions driving the strata branch

1. Models fit the 88 GB VRAM pool -> Strata's expert-residency tiering is irrelevant on this
   rig, and buun already ships a moe-cache + cpu-overlap (both no-ops here).
2. The production command leaves measured wins behind: fusion disabled (-5–6% t/s), `-ub 512`
   (-45% prefill vs 8192), `spec-draft-n-max 2` (n3 adaptive is better).
3. Real Strata-derived code work: draft-head vocab restriction (targets the 7.5 ms draft
   cycle), combined MTP+ngram per-round drafter economics, and prefill pipeline work
   (chunked MoE prefill + inter-GPU overlap) to close the remaining 3x prompt-speed gap.

Files: `round1.md` (scenarios), `round2.md` (n4, long3), `round3.md` (true no-spec, prefill
@512), `round4.md` (fusion on, prefill @8192). Raw server logs stay in `~/.strata-bench/` on
AISERVER.
