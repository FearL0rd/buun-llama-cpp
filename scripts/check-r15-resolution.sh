#!/usr/bin/env bash
for f in cache20k cacheon; do
    echo "== $f"
    grep -a 'resolved=' ~/.strata-bench/server-r15-$f.log | sort | uniq -c | head -6
    grep -a 'MoE cache' ~/.strata-bench/server-r15-$f.log | grep -av 'accepted option\|option:' | head -6
done
