# Phase 2 round 27 (graph-cache pointer-key fix, MTP n sweep 2/3) - 2026-10-03 16:22

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |
|---|---|---|---|---|---|---|---|
| n2-fix | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 130.50304846964784 | 64.854390041075 | 6.212996 |
| n2-fix | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 121.66652725710418 | 64.29360050373953 | 6.271788 |
| n2-fix | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 134.0673520860042 | 64.2911182407667 | 6.250351 |
### n=2 fix-check (cycles 625, resets 29, reused 2469):
reason counts:
     22 reason=fattn_epoch
     28 reason=node_count
     32 reason=node_props
   1286 reason=uid
spec cycles (last 6):
    [57345] 0.39.008.796 D srv    post_cycle: spec cycle (1 slots): draft=3.8ms verify=31.0ms accept=1.7ms other=0.9ms total=37.5ms
    [57345] 0.39.048.335 D srv    post_cycle: spec cycle (1 slots): draft=3.9ms verify=31.2ms accept=1.9ms other=2.5ms total=39.5ms
    [57345] 0.39.086.556 D srv    post_cycle: spec cycle (1 slots): draft=3.9ms verify=31.1ms accept=2.2ms other=1.0ms total=38.2ms
    [57345] 0.39.125.082 D srv    post_cycle: spec cycle (1 slots): draft=4.1ms verify=31.1ms accept=2.0ms other=1.3ms total=38.5ms
    [57345] 0.39.162.831 D srv    post_cycle: spec cycle (1 slots): draft=3.9ms verify=31.2ms accept=1.7ms other=1.0ms total=37.7ms
    [57345] 0.39.200.704 D srv    post_cycle: spec cycle (1 slots): draft=3.9ms verify=31.0ms accept=2.0ms other=0.9ms total=37.9ms
| n3-fix | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 130.3011177392756 | 61.74733345833272 | 6.511541 |
| n3-fix | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 131.15506627429443 | 62.39166981611562 | 6.434380 |
| n3-fix | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 127.28466020962192 | 62.885729241305846 | 6.402392 |
### n=3 fix-check (cycles 589, resets 38, reused 2517):
reason counts:
     78 reason=fattn_epoch
     92 reason=node_count
     42 reason=node_props
   1194 reason=uid
spec cycles (last 6):
    [35085] 0.38.733.074 D srv    post_cycle: spec cycle (1 slots): draft=5.6ms verify=35.9ms accept=2.1ms other=1.0ms total=44.6ms
    [35085] 0.38.777.170 D srv    post_cycle: spec cycle (1 slots): draft=5.5ms verify=36.4ms accept=1.1ms other=1.0ms total=44.1ms
    [35085] 0.38.821.201 D srv    post_cycle: spec cycle (1 slots): draft=5.5ms verify=36.2ms accept=1.4ms other=0.9ms total=44.0ms
    [35085] 0.38.865.189 D srv    post_cycle: spec cycle (1 slots): draft=5.5ms verify=36.1ms accept=1.4ms other=1.0ms total=44.0ms
    [35085] 0.38.909.830 D srv    post_cycle: spec cycle (1 slots): draft=5.5ms verify=36.2ms accept=2.0ms other=0.9ms total=44.6ms
    [35085] 0.38.947.484 D srv    post_cycle: spec cycle (1 slots): draft=2.2ms verify=32.4ms accept=1.7ms other=1.3ms total=37.6ms
