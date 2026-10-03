#!/usr/bin/env bash
log=~/.strata-bench/server-r12.log
echo "=== bf16/f32 tensor lines (first 30):"
grep -ai 'bf16\|BF16' $log | grep -av 'chunk\|GET\|POST\|option\|kv cache\|cache_type\|f16 claude' | head -30
echo "=== tensor type summary lines:"
grep -a 'load_tensors:' $log | grep -ai 'type\|blk' | head -12
echo "=== print: model summary"
grep -a 'llm_load_print\|model params\|model size' $log | head -8
