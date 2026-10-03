# Strata phase 2 round 12 — prefill profile (Coder, b/ub 2048, MMQ on, nsys)

One 420 s nsys capture: model load + ping + 2x 21K-token prefill + 1x decode. Kernel
ranking from `cuda_gpu_kern_sum` (share of total GPU kernel time):

| bucket | ~% | detail |
|---|---:|---|
| MoE expert GEMMs (mul_mat_q, 5 quant types + mm_ids_helper + weighted reduction) | ~40 | the MMQ patch's own path, now the dominant cost; type-20 (IQ2_XS-class) avg 3.8 ms/call, 1,166 calls |
| dense f16 GEMMs on V100 (volta_sgemm 128x64/32x128, cutlass 70 s884) | ~15 | dense projections of attention/DeltaNet layers on the V100s |
| DeltaNet recurrent scan (gated_delta_net_cuda, 1,620 calls, avg 3.0 ms) | ~12 | hybrid recurrent prefill |
| attention (flash sm70_d256_splitd_dense avg 7.6 ms x348 on V100; tile/ext small) | ~8 | the V100 FA path is the slow part (max 24.8 ms/call) |
| MoE routing (CUB DeviceSegmentedSort 2.9% + argsort 0.2%) | ~3 | |
| norms/elementwise/quantize/convert (rms_norm, dsv4_hc_pre/post, silu, converts) | ~10 | |
| turbo KV kernels | ~1 | |

Read: after the MMQ routing patch, the prefill is **V100-dominated** — the V100s host 2/3
of the layers and their kernels (dp4a MMQ, sm_70 FA, volta sgemm) are the slow variants.
The remaining ~2x gap to Strata lives in (in order): MoE MMQ kernel efficiency on sm_70,
the DeltaNet chunk-scan, and the sm_70 FA kernel. None are config-fixable; all are
kernel-level. The MoE routing sort (argsort bug area) is only ~3% of prefill time, so the
b-4096 argsort fix is not warranted by this data unless bigger ubatches become viable.

Raw: `~/.strata-bench/prefill-r12.nsys-rep` (AISERVER), summary `phase2-round12.txt`.
