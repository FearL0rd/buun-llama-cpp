# Phase 2 round 19 (manual ot residency, layers 32-47 to V100s) - 2026-10-03 00:39

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### ot131k load:
    [58439] 0.03.996.805 I load_tensors:        CUDA0 model buffer size =  7341.07 MiB
    [58439] 0.03.996.806 I load_tensors:        CUDA1 model buffer size = 15017.03 MiB
    [58439] 0.03.996.807 I load_tensors:        CUDA2 model buffer size =  5546.16 MiB
    [58439] 0.03.996.808 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [58439] 0.03.996.809 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [58439] 0.08.096.822 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
    [58439] 0.02.087.463 I sched_reserve: MoE cache requested=auto resolved=off
    [58439] 0.02.661.598 I sched_reserve: MoE cache requested=auto resolved=off
    [58439] 0.03.143.228 I sched_reserve: MoE cache requested=auto resolved=off
    [58439] 0.03.264.982 I cmn  common_init_: MoE cache: mode=auto budget=free-minus-reserve; use -lv 4 for resolved backend state, actual pools, and statistics
    [58439] 0.06.851.922 I sched_reserve: MoE cache requested=auto resolved=off
    [58439] 0.08.398.680 I sched_reserve: MoE cache requested=auto resolved=off
| ot131k-prefill | Qwen3.8-Flash-Next-Coder | req1 | 3.585773 | 0 | 0 | 0 | 0 | 0, 10585 MiB, 32768 MiB|1, 10577 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 17359 MiB, 32768 MiB| |
| ot131k-prefill | Qwen3.8-Flash-Next-Coder | req2 | 0.001847 | 0 | 0 | 0 | 0 | 0, 10583 MiB, 32768 MiB|1, 10577 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 17357 MiB, 32768 MiB| |
| ot131k-prefill | Qwen3.8-Flash-Next-Coder | req3 | 0.001653 | 0 | 0 | 0 | 0 | 0, 10583 MiB, 32768 MiB|1, 10577 MiB, 24576 MiB|2, 220 MiB, 3072 MiB|3, 17357 MiB, 32768 MiB| |
| ot131k-decode | Qwen3.8-Flash-Next-Coder | req1 | 0.000706 | 0 | 0 | 0 | 0 | 0, 467 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 77 MiB, 32768 MiB| |
| ot131k-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 34.08904484626906 | 37.600871947530784 | 20.161839 | 0, 10639 MiB, 32768 MiB|1, 10489 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 17281 MiB, 32768 MiB| |
### ot131kexp load:
    [36459] 0.04.087.344 I load_tensors:        CUDA0 model buffer size =  7341.07 MiB
    [36459] 0.04.087.345 I load_tensors:        CUDA1 model buffer size = 14453.04 MiB
    [36459] 0.04.087.346 I load_tensors:        CUDA2 model buffer size =  6110.15 MiB
    [36459] 0.04.087.346 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [36459] 0.04.087.347 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [36459] 0.08.221.788 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
    [36459] 0.02.068.988 I sched_reserve: MoE cache requested=auto resolved=off
    [36459] 0.02.659.760 I sched_reserve: MoE cache requested=auto resolved=off
    [36459] 0.03.186.601 I sched_reserve: MoE cache requested=auto resolved=off
    [36459] 0.03.312.996 I cmn  common_init_: MoE cache: mode=auto budget=free-minus-reserve; use -lv 4 for resolved backend state, actual pools, and statistics
    [36459] 0.06.987.738 I sched_reserve: MoE cache requested=auto resolved=off
    [36459] 0.08.524.143 I sched_reserve: MoE cache requested=auto resolved=off
| ot131kexp-prefill | Qwen3.8-Flash-Next-Coder | req1 | 3.602156 | 0 | 0 | 0 | 0 | 0, 11283 MiB, 32768 MiB|1, 10577 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 16795 MiB, 32768 MiB| |
| ot131kexp-prefill | Qwen3.8-Flash-Next-Coder | req2 | 0.001699 | 0 | 0 | 0 | 0 | 0, 11281 MiB, 32768 MiB|1, 10577 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 16793 MiB, 32768 MiB| |
| ot131kexp-prefill | Qwen3.8-Flash-Next-Coder | req3 | 0.001571 | 0 | 0 | 0 | 0 | 0, 11281 MiB, 32768 MiB|1, 10577 MiB, 24576 MiB|2, 220 MiB, 3072 MiB|3, 16793 MiB, 32768 MiB| |
| ot131kexp-decode | Qwen3.8-Flash-Next-Coder | req1 | 0.000638 | 0 | 0 | 0 | 0 | 0, 467 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 77 MiB, 32768 MiB| |
| ot131kexp-decode | Qwen3.8-Flash-Next-Coder | req2 | 0.000704 | 0 | 0 | 0 | 0 | 0, 467 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 204 MiB, 3072 MiB|3, 77 MiB, 32768 MiB| |
