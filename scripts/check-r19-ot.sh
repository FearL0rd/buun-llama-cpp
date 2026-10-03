#!/usr/bin/env bash
log=~/.strata-bench/server-r19-ot131k.log
echo "=== overriding lines (which tensors matched):"
grep -a 'overriding' $log | head -12
echo "=== matched-tensor count by prefix:"
grep -a 'overriding' $log | grep -ao 'blk\.[0-9]*\.[a-z_]*' | sort | uniq -c | sort -rn | head -12
echo "=== fatal error:"
grep -a 'CUDA error\|GGML_ASSERT\|Compute error\|abort\|illegal' $log | grep -av 'chunk\|GET\|POST' | tail -5
