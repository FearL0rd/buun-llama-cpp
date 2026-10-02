# Phase 1 round 5 (Volta MoE prefill MMQ A/B) - 2026-10-02 10:40

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |
|---|---|---|---|---|---|---|---|
| off-prefill | Qwen3.8-Flash-Next-Coder | req1 | 21208 | 256 | 555.7575968169409 | 42.93675264793479 | 44.159627 |
| off-prefill | Qwen3.8-Flash-Next-Coder | req2 | 21208 | 256 | 712.5243695563682 | 42.77563191350667 | 35.819577 |
| off-prefill | Qwen3.8-Flash-Next-Coder | req3 | 21208 | 256 | 696.0888948318583 | 44.538847394014574 | 36.282043 |
| off-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 108.2335273644798 | 45.2095254935688 | 8.866878 |
| off-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 103.63164133088935 | 45.07065325028251 | 8.905884 |
| off | Qwen3.8-Flash-Next | SERVER_FAILED | | | | | |
| on-prefill | Qwen3.8-Flash-Next-Coder | req1 | 21208 | 256 | 1025.707892243795 | 46.04122224097976 | 26.276016 |
| on-prefill | Qwen3.8-Flash-Next-Coder | req2 | 21208 | 256 | 1049.4971415207406 | 44.2401954618581 | 26.071305 |
| on-prefill | Qwen3.8-Flash-Next-Coder | req3 | 21208 | 256 | 1048.7481977731013 | 41.47621493592656 | 26.477023 |
| on-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 106.25087158918102 | 44.95240268244954 | 8.938092 |
| on-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 106.15183028976134 | 45.19684405660911 | 8.871712 |
| on | Qwen3.8-Flash-Next | SERVER_FAILED | | | | | |
