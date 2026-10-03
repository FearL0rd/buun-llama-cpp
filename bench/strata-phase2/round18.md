# Phase 2 round 18 (gate fix + ngl unpinned + reduced ctx) - 2026-10-02 23:30

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### fitfree262k load:
    [53245] 0.27.402.379 I load_tensors:        CUDA2 model buffer size =     0.00 MiB
    [53245] 0.28.271.026 I load_tensors:        CUDA0 model buffer size = 18746.60 MiB
    [53245] 0.28.271.027 I load_tensors:        CUDA1 model buffer size =  9157.66 MiB
    [53245] 0.28.271.028 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [53245] 0.28.271.029 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [53245] 0.32.345.319 I load_tensors:        CUDA1 model buffer size =  1805.24 MiB
    [53245] 0.01.945.524 I sched_reserve: MoE cache requested=auto resolved=off
    [53245] 0.02.490.062 I sched_reserve: MoE cache requested=auto resolved=off
    [53245] 0.02.970.939 I sched_reserve: MoE cache requested=auto resolved=off
    [53245] 0.03.113.538 I common_params_fit_impl: free-memory targets met, but 27788 MiB of model weights are host-resident with a MoE cache requested - continuing to cache fit
    [53245] 0.03.601.121 I sched_reserve: MoE cache requested=auto resolved=off
    [53245] 0.04.156.593 I sched_reserve: MoE cache requested=auto resolved=off
    [53245] 0.04.612.825 I sched_reserve: MoE cache requested=auto resolved=off
    [53245] 0.05.128.488 I sched_reserve: MoE cache requested=auto resolved=off
| fitfree262k-prefill | Qwen3.8-Flash-Next-Coder | req1 | 11.253467 | 0 | 0 | 0 | 0 | 0, 815 MiB, 32768 MiB|1, 23697 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 15421 MiB, 32768 MiB| |
| fitfree262k-prefill | Qwen3.8-Flash-Next-Coder | req2 | 0.001504 | 0 | 0 | 0 | 0 | 0, 813 MiB, 32768 MiB|1, 23697 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 15419 MiB, 32768 MiB| |
| fitfree262k-prefill | Qwen3.8-Flash-Next-Coder | req3 | 0.001504 | 0 | 0 | 0 | 0 | 0, 813 MiB, 32768 MiB|1, 23697 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 15419 MiB, 32768 MiB| |
| fitfree262k-decode | Qwen3.8-Flash-Next-Coder | req1 | 0.000636 | 0 | 0 | 0 | 0 | 0, 813 MiB, 32768 MiB|1, 23697 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 15419 MiB, 32768 MiB| |
| fitfree262k-decode | Qwen3.8-Flash-Next-Coder | req2 | 0.000799 | 0 | 0 | 0 | 0 | 0, 813 MiB, 32768 MiB|1, 23697 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 15419 MiB, 32768 MiB| |
### fitfree131k load:
    [45777] 0.25.640.655 I load_tensors:        CUDA2 model buffer size =     0.00 MiB
    [45777] 0.26.491.216 I load_tensors:        CUDA0 model buffer size = 19935.58 MiB
    [45777] 0.26.491.217 I load_tensors:        CUDA1 model buffer size =  7968.68 MiB
    [45777] 0.26.491.218 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [45777] 0.26.491.220 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [45777] 0.30.035.040 I load_tensors:        CUDA1 model buffer size =  1805.24 MiB
    [45777] 0.02.006.918 I sched_reserve: MoE cache requested=auto resolved=off
    [45777] 0.02.563.337 I sched_reserve: MoE cache requested=auto resolved=off
    [45777] 0.03.063.011 I sched_reserve: MoE cache requested=auto resolved=off
    [45777] 0.03.179.096 I common_params_fit_impl: free-memory targets met, but 27788 MiB of model weights are host-resident with a MoE cache requested - continuing to cache fit
    [45777] 0.03.645.658 I sched_reserve: MoE cache requested=auto resolved=off
    [45777] 0.04.212.078 I sched_reserve: MoE cache requested=auto resolved=off
    [45777] 0.04.683.565 I sched_reserve: MoE cache requested=auto resolved=off
    [45777] 0.05.243.872 I sched_reserve: MoE cache requested=auto resolved=off
| fitfree131k-prefill | Qwen3.8-Flash-Next-Coder | req1 | 11.752340 | 0 | 0 | 0 | 0 | 0, 815 MiB, 32768 MiB|1, 24119 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 12987 MiB, 32768 MiB| |
| fitfree131k-prefill | Qwen3.8-Flash-Next-Coder | req2 | 0.001572 | 0 | 0 | 0 | 0 | 0, 813 MiB, 32768 MiB|1, 24119 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 12985 MiB, 32768 MiB| |
| fitfree131k-prefill | Qwen3.8-Flash-Next-Coder | req3 | 0.001610 | 0 | 0 | 0 | 0 | 0, 813 MiB, 32768 MiB|1, 24119 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 12985 MiB, 32768 MiB| |
| fitfree131k-decode | Qwen3.8-Flash-Next-Coder | req1 | 0.000593 | 0 | 0 | 0 | 0 | 0, 467 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 77 MiB, 32768 MiB| |
| fitfree131k-decode | Qwen3.8-Flash-Next-Coder | req2 | 0.000730 | 0 | 0 | 0 | 0 | 0, 467 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 77 MiB, 32768 MiB| |
