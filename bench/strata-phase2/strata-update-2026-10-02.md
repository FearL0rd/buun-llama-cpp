# Strata update (2026-10-02): what it changes for our plan

New docs: MULTI_GPU.md, SECOND_GPU.md, MODELS.md, HOW_IT_WORKS.md, COMMUNITY_BENCHMARKS.md.
New bench dirs: 2026-09-29-layer-split{,-limits}, 2026-09-30-{community-rtx-5090,tc-gemv-sm75}.

## Findings that redirect the "beat Strata" work

1. **Their split pipelines prefill chunks across cards** ("while a later card reads chunk c,
   the first card already reads chunk c+1"). buun/llama.cpp runs ubatches strictly serially
   through all stages. This is the biggest structural prefill gap; their own split gains
   +18-20% prompt speed from it on top of high residency.
2. **Residency is what makes their split fast**: 96-97% expert-cache hit rate on a
   2-card split vs 59-71% single-card (bigger combined caches). Our round-12 profile found
   the Coder running with 27.5 GB CPU_Mapped experts, 48 GB H2D per capture window, while
   ~21 GB sat free per V100 for a 512K-ctx KV reservation the workload never uses.
   Round 13 tests shrinking that reservation (ctx-size) to let experts go resident.
3. **A split loses to a single card when the model fits** (their split_skip_if_fits:
   4K prompts 1,776 vs 1,244 t/s; decode 60 vs 51). Every extra stage costs a handoff;
   they recommend dropping much-slower third cards (a 2080 Ti helper cut decode 62.5->43).
   For us: the V100s' dp4a MMQ is ~3.5x slower per FLOP than the 3090's MMA path, so once
   residency is high, a 2-stage split (3090 + one V100) or even 3090-heavy ot overrides
   may beat the 3-way split. Test after round 13.
4. **"Speed projection" is NOT an engine optimization**: it is a refusal-direction
   control vector (per-layer projection, layers 4-44) that makes the model decline/refuse
   less, so chats measure faster tokens/s while costing +0.2-0.4% per token on identical
   text and shifting outputs (perplexity +15% on code). Excluded from our plan.
5. Their prefill chunk is 2048 tokens by default (1.5 GB prompt-path buffers per card;
   `--prefill 1024` halves). Not 8192 as the older README implied.

Reference numbers to beat (their rig, Coder IQ1_M, 2-card split): 16K prompts 1,685-1,968
t/s, 28K prompts 2,039-2,357 t/s, decode 75-104 t/s. Single 5080: 1,487-1,837 prefill,
66-74 decode.
