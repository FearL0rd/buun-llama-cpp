# Phase 2 round 26 (CUDA graph update-recapture diagnosis, MTP n3) - 2026-10-03 13:24

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |
|---|---|---|---|---|---|---|---|
| diag | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 132.11239461972272 | 61.66316007756357 | 6.513790 |
| diag | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 133.49241179071728 | 61.23633114753207 | 6.545691 |
| diag | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 132.45416878800296 | 60.89028270579246 | 6.585756 |
### diagnosis:
spec cycles: 589
warmup resets: 623
graph id reused: 2517
### reason counts:
     24 reason=fattn_epoch
     19 reason=node_count
   1243 reason=node_props
   1248 reason=uid
### first 30 reason lines:
    [54959] 0.09.972.100 D graph update: reason=fattn_epoch key=2842364363958198541
    [54959] 0.09.972.106 D graph update: reason=node_count 0 -> 1985
    [54959] 0.10.136.344 D graph update: reason=fattn_epoch key=9274789413517799951
    [54959] 0.10.136.350 D graph update: reason=node_count 0 -> 2391
    [54959] 0.10.179.133 D graph update: reason=fattn_epoch key=188533766057446084
    [54959] 0.10.179.137 D graph update: reason=node_count 0 -> 2164
    [54959] 0.12.511.527 D graph update: reason=fattn_epoch key=15420093704155139869
    [54959] 0.12.511.533 D graph update: reason=node_count 0 -> 1985
    [54959] 0.12.526.370 D graph update: reason=fattn_epoch key=15143804147030410239
    [54959] 0.12.526.376 D graph update: reason=node_count 0 -> 2391
    [54959] 0.12.533.187 D graph update: reason=fattn_epoch key=1753664511704222076
    [54959] 0.12.533.190 D graph update: reason=node_count 0 -> 2162
    [54959] 0.12.560.672 D graph update: reason=uid prev=0 new=200
    [54959] 0.12.560.675 D graph update: reason=node_count 0 -> 150
    [54959] 0.12.565.692 D graph update: reason=uid prev=0 new=203
    [54959] 0.12.565.694 D graph update: reason=node_count 0 -> 150
    [54959] 0.12.711.060 D graph update: reason=fattn_epoch key=12882219406553734611
    [54959] 0.12.711.067 D graph update: reason=node_count 0 -> 1985
    [54959] 0.12.722.198 D graph update: reason=fattn_epoch key=14321322349072906621
    [54959] 0.12.722.204 D graph update: reason=node_count 0 -> 2391
    [54959] 0.12.729.180 D graph update: reason=fattn_epoch key=12699964343202606001
    [54959] 0.12.729.184 D graph update: reason=node_count 0 -> 2162
    [54959] 0.12.753.290 D graph update: reason=uid prev=0 new=211
    [54959] 0.12.753.296 D graph update: reason=node_count 0 -> 150
    [54959] 0.12.760.299 D graph update: reason=fattn_epoch key=16242575502112643487
    [54959] 0.12.760.304 D graph update: reason=node_count 0 -> 1985
    [54959] 0.12.768.365 D graph update: reason=fattn_epoch key=10960966253513997745
    [54959] 0.12.768.371 D graph update: reason=node_count 0 -> 2391
    [54959] 0.12.775.042 D graph update: reason=fattn_epoch key=689741983878625689
    [54959] 0.12.775.045 D graph update: reason=node_count 0 -> 2162
