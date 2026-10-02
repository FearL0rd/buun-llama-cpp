# Strata Phase 0 baseline - 2026-10-01 15:08
GPUs: 0, Tesla V100-PCIE-32GB;1, NVIDIA GeForce RTX 3090;2, NVIDIA GeForce GTX 1060 3GB;3, Tesla V100-PCIE-32GB;
Server: /home/cesar/buun-llama-cpp/build/bin/llama-server | CUDA_VISIBLE_DEVICES=1,3,0 | port 8091
| scenario | model | run | prompt_tok | out_tok | wall_s | out_t/s | vram(idx,used,total) |
|---|---|---|---|---|---|---|---|
| base | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 8.147183 | 47.1 | 0, 18071 MiB, 32768 MiB|1, 11656 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 14202 MiB, 32768 MiB| |
| base | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 6.851571 | 56.0 | 0, 18081 MiB, 32768 MiB|1, 11666 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 14214 MiB, 32768 MiB| |
### spec stats: base / Qwen3.8-Flash-Next-Coder
| base | Qwen3.8-Flash-Next | req1 | 32 | 384 | 8.253562 | 46.5 | 0, 28791 MiB, 32768 MiB|1, 22382 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 26058 MiB, 32768 MiB| |
| base | Qwen3.8-Flash-Next | req2 | 32 | 384 | 6.848407 | 56.1 | 0, 28803 MiB, 32768 MiB|1, 22392 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 26070 MiB, 32768 MiB| |
### spec stats: base / Qwen3.8-Flash-Next
| nospec | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 7.207423 | 53.3 | 0, 18071 MiB, 32768 MiB|1, 11656 MiB, 24576 MiB|2, 174 MiB, 3072 MiB|3, 14202 MiB, 32768 MiB| |
| nospec | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 6.846274 | 56.1 | 0, 18081 MiB, 32768 MiB|1, 11666 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 14214 MiB, 32768 MiB| |
| nospec | Qwen3.8-Flash-Next | req1 | 32 | 384 | 7.177408 | 53.5 | 0, 28791 MiB, 32768 MiB|1, 22382 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 26058 MiB, 32768 MiB| |
| nospec | Qwen3.8-Flash-Next | req2 | 32 | 384 | 6.815971 | 56.3 | 0, 28803 MiB, 32768 MiB|1, 22392 MiB, 24576 MiB|2, 174 MiB, 3072 MiB|3, 26070 MiB, 32768 MiB| |
| moeoff | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 7.198387 | 53.3 | 0, 18071 MiB, 32768 MiB|1, 11656 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 14202 MiB, 32768 MiB| |
| moeoff | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 6.832570 | 56.2 | 0, 18081 MiB, 32768 MiB|1, 11666 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 14214 MiB, 32768 MiB| |
### spec stats: moeoff / Qwen3.8-Flash-Next-Coder
| moeoff | Qwen3.8-Flash-Next | req1 | 32 | 384 | 7.199823 | 53.3 | 0, 28791 MiB, 32768 MiB|1, 22382 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 26058 MiB, 32768 MiB| |
| moeoff | Qwen3.8-Flash-Next | req2 | 32 | 384 | 6.857183 | 56.0 | 0, 28803 MiB, 32768 MiB|1, 22392 MiB, 24576 MiB|2, 158 MiB, 3072 MiB|3, 26070 MiB, 32768 MiB| |
### spec stats: moeoff / Qwen3.8-Flash-Next
| nooverlap | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 7.302057 | 52.6 | 0, 18071 MiB, 32768 MiB|1, 11656 MiB, 24576 MiB|2, 186 MiB, 3072 MiB|3, 14202 MiB, 32768 MiB| |
| nooverlap | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 6.946197 | 55.3 | 0, 18081 MiB, 32768 MiB|1, 11666 MiB, 24576 MiB|2, 198 MiB, 3072 MiB|3, 14214 MiB, 32768 MiB| |
### spec stats: nooverlap / Qwen3.8-Flash-Next-Coder
| nooverlap | Qwen3.8-Flash-Next | req1 | 32 | 384 | 7.267064 | 52.8 | 0, 28797 MiB, 32768 MiB|1, 22395 MiB, 24576 MiB|2, 209 MiB, 3072 MiB|3, 26065 MiB, 32768 MiB| |
| nooverlap | Qwen3.8-Flash-Next | req2 | 32 | 384 | 6.897096 | 55.7 | 0, 28809 MiB, 32768 MiB|1, 22405 MiB, 24576 MiB|2, 209 MiB, 3072 MiB|3, 26077 MiB, 32768 MiB| |
### spec stats: nooverlap / Qwen3.8-Flash-Next
| nmax3 | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 8.206002 | 46.8 | 0, 18161 MiB, 32768 MiB|1, 11749 MiB, 24576 MiB|2, 209 MiB, 3072 MiB|3, 14303 MiB, 32768 MiB| |
| nmax3 | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 6.842081 | 56.1 | 0, 18161 MiB, 32768 MiB|1, 11749 MiB, 24576 MiB|2, 209 MiB, 3072 MiB|3, 14303 MiB, 32768 MiB| |
### spec stats: nmax3 / Qwen3.8-Flash-Next-Coder
| nmax3 | Qwen3.8-Flash-Next | req1 | 32 | 384 | 7.424070 | 51.7 | 0, 28883 MiB, 32768 MiB|1, 22475 MiB, 24576 MiB|2, 209 MiB, 3072 MiB|3, 26161 MiB, 32768 MiB| |
| nmax3 | Qwen3.8-Flash-Next | req2 | 32 | 384 | 6.514138 | 58.9 | 0, 28883 MiB, 32768 MiB|1, 22475 MiB, 24576 MiB|2, 209 MiB, 3072 MiB|3, 26161 MiB, 32768 MiB| |
### spec stats: nmax3 / Qwen3.8-Flash-Next
