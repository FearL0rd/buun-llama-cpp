#!/usr/bin/env bash
log=~/.strata-bench/server-r19-ot131k.log
echo "=== per-tensor load line format (sample):"
grep -a "load_tensors: tensor" $log | head -6 | cut -c1-160
echo "=== total per-tensor lines:"
grep -ac "load_tensors: tensor" $log
echo "=== big tensors (expert-sized) sample with context:"
grep -a "load_tensors: tensor" $log | grep -a "_exps" | head -8 | cut -c1-160
