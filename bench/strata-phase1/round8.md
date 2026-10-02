# Phase 1 round 8 (Flash-Next MMQ A/B @ub512, no mmproj) - 2026-10-02 15:20

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |
|---|---|---|---|---|---|---|---|
| off-prefill | Qwen3.8-Flash-Next | req1 | 21208 | 256 | 356.2692288664422 | 41.48816586675856 | 65.732127 |
| off-prefill | Qwen3.8-Flash-Next | req2 | 21208 | 256 | 398.94383719901094 | 43.56795216614732 | 59.100858 |
| off-prefill | Qwen3.8-Flash-Next | req3 | 21208 | 256 | 394.7122773999471 | 45.165334348343045 | 59.461931 |
| off-decode | Qwen3.8-Flash-Next | req1 | 32 | 384 | 108.71189987634021 | 43.45469105303083 | 9.205237 |
| off-decode | Qwen3.8-Flash-Next | req2 | 32 | 384 | 107.41354887652142 | 43.47756001662818 | 9.194382 |
| on-prefill | Qwen3.8-Flash-Next | req1 | 21208 | 256 | 582.3050808012823 | 44.80896094872695 | 42.168663 |
| on-prefill | Qwen3.8-Flash-Next | req2 | 21208 | 256 | 588.2363872633508 | 44.40498108957286 | 41.887394 |
| on-prefill | Qwen3.8-Flash-Next | req3 | 21208 | 256 | 588.2281153512893 | 46.31571223303306 | 41.659025 |
| on-decode | Qwen3.8-Flash-Next | req1 | 32 | 384 | 108.63550411966202 | 43.71345204445989 | 9.152738 |
| on-decode | Qwen3.8-Flash-Next | req2 | 32 | 384 | 107.31736763911853 | 43.596901455521824 | 9.169660 |
