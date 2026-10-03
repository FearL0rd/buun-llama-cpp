#!/usr/bin/env bash
log=~/.strata-bench/server-r19-ot131k.log
awk '/spawning server instance with args:/{f=1;next} f && /load:   /{sub(/.*load:   /,""); print; next} f{exit}' $log \
  | grep -v '^$' > ~/.strata-bench/child-args-r19.txt
wc -l ~/.strata-bench/child-args-r19.txt
echo "=== placement-relevant args (value on next line):"
grep -A1 -E '^--(n-gpu-layers|tensor-split|override-tensor|n-cpu-moe|fit|moe-cache)$' ~/.strata-bench/child-args-r19.txt | head -14
echo "=== full arg list (first 40):"
head -40 ~/.strata-bench/child-args-r19.txt
