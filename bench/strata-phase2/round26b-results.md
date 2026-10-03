# Phase 2 round 26 (CUDA graph update-recapture diagnosis, MTP n3) - 2026-10-03 13:42

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |
|---|---|---|---|---|---|---|---|
| diag | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 135.6524245751324 | 61.579423790237655 | 6.513515 |
| diag | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 132.89202107999685 | 61.71033330669423 | 6.506011 |
| diag | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 134.46338604017933 | 61.391447578275695 | 6.525394 |
### diagnosis:
spec cycles: 589
warmup resets: 623
graph id reused: 2517
### reason counts:
     24 reason=fattn_epoch
     19 reason=node_count
   1243 reason=node_props
   1248 reason=uid
### last 40 reason lines:
    [55995] 0.38.995.567 D graph update: reason=uid prev=3864 new=3870
    [55995] 0.38.995.573 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0500000->0x7b60b0000000 src1=(nil)->(nil)
    [55995] 0.38.998.910 D graph update: reason=uid prev=3867 new=3873
    [55995] 0.38.998.914 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0780000->0x7b60b0280000 src1=(nil)->(nil)
    [55995] 0.39.041.096 D graph update: reason=uid prev=3870 new=3876
    [55995] 0.39.041.100 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0000000->0x7b60b0500000 src1=(nil)->(nil)
    [55995] 0.39.043.336 D graph update: reason=uid prev=3873 new=3879
    [55995] 0.39.043.340 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0280000->0x7b60b0780000 src1=(nil)->(nil)
    [55995] 0.39.085.645 D graph update: reason=uid prev=3876 new=3882
    [55995] 0.39.085.651 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0500000->0x7b60b0000000 src1=(nil)->(nil)
    [55995] 0.39.088.893 D graph update: reason=uid prev=3879 new=3885
    [55995] 0.39.088.898 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0780000->0x7b60b0280000 src1=(nil)->(nil)
    [55995] 0.39.131.206 D graph update: reason=uid prev=3882 new=3888
    [55995] 0.39.131.212 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0000000->0x7b60b0500000 src1=(nil)->(nil)
    [55995] 0.39.135.444 D graph update: reason=uid prev=3885 new=3891
    [55995] 0.39.135.448 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0280000->0x7b60b0780000 src1=(nil)->(nil)
    [55995] 0.39.178.047 D graph update: reason=uid prev=3888 new=3894
    [55995] 0.39.178.054 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0500000->0x7b60b0000000 src1=(nil)->(nil)
    [55995] 0.39.181.102 D graph update: reason=uid prev=3891 new=3897
    [55995] 0.39.181.106 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0780000->0x7b60b0280000 src1=(nil)->(nil)
    [55995] 0.39.223.455 D graph update: reason=uid prev=3894 new=3900
    [55995] 0.39.223.461 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0000000->0x7b60b0500000 src1=(nil)->(nil)
    [55995] 0.39.226.079 D graph update: reason=uid prev=3897 new=3903
    [55995] 0.39.226.083 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0280000->0x7b60b0780000 src1=(nil)->(nil)
    [55995] 0.39.268.262 D graph update: reason=uid prev=3900 new=3906
    [55995] 0.39.268.267 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0500000->0x7b60b0000000 src1=(nil)->(nil)
    [55995] 0.39.270.839 D graph update: reason=uid prev=3903 new=3909
    [55995] 0.39.270.843 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0780000->0x7b60b0280000 src1=(nil)->(nil)
    [55995] 0.39.313.367 D graph update: reason=uid prev=3906 new=3912
    [55995] 0.39.313.373 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0000000->0x7b60b0500000 src1=(nil)->(nil)
    [55995] 0.39.318.531 D graph update: reason=uid prev=3909 new=3915
    [55995] 0.39.318.535 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0280000->0x7b60b0780000 src1=(nil)->(nil)
    [55995] 0.39.326.000 D graph update: reason=uid prev=2992 new=3918
    [55995] 0.39.326.008 D graph update: reason=node_props i=0 op=REPEAT name=hc_init data=0x7b657a054a00->0x7b657a054a00 src0=0x7b657a00a000->0x7b657a000000 src1=(nil)->(nil)
    [55995] 0.39.330.654 D graph update: reason=uid prev=2993 new=3919
    [55995] 0.39.330.659 D graph update: reason=node_props i=0 op=MUL name=node_1989 data=0x7b6422144a00->0x7b6422144a00 src0=0x7b6422028000->0x7b6422000000 src1=0x7b6af6000000->0x7b6af6000000
    [55995] 0.39.336.220 D graph update: reason=uid prev=2994 new=3920
    [55995] 0.39.336.225 D graph update: reason=node_props i=0 op=MUL name=node_4380 data=0x7b62ca144c00->0x7b62ca144c00 src0=0x7b62ca028000->0x7b62ca000000 src1=0x7b689ddda800->0x7b689ddda800
    [55995] 0.39.353.656 D graph update: reason=uid prev=2997 new=3923
    [55995] 0.39.353.663 D graph update: reason=node_props i=0 op=RMS_NORM name=norm-48 data=0x7b60d3208200->0x7b60d3208200 src0=0x7b60b0780000->0x7b60b0000000 src1=(nil)->(nil)
