#!/usr/bin/env bash
log=~/.strata-bench/server-r19-ot131k.log
echo "=== any 'load:' arg-dump lines:"
grep -ac 'load:' $log
grep -a 'load:' $log | head -8 | cut -c1-130
echo "=== n-gpu-layers anywhere (case-insensitive):"
grep -aic 'n-gpu-layers' $log
grep -ai 'n-gpu-layers' $log | head -4 | cut -c1-150
