#!/usr/bin/env bash
f=~/.strata-bench/placement-map.txt
echo "=== per-buffer totals (correct parse):"
grep -a 'placement: ' $f | sed 's/.*placement: //' | awk '{s[$3]+=$4; n[$3]++} END {for (b in s) printf "%-14s %5d tensors %10.1f MiB\n", b, n[b], s[b]}' | sort -k3 -rn
echo "=== non-CUDA device totals for blk tensors:"
grep -a 'placement: blk' $f | sed 's/.*placement: //' | awk '{s[$3]+=$4; n[$3]++} END {for (b in s) printf "%-14s %5d tensors %10.1f MiB\n", b, n[b], s[b]}' | sort -k3 -rn
echo "=== per-layer layer-histogram per buffer:"
grep -a 'placement: blk' $f | sed 's/.*placement: //' | grep -o '^blk\.[0-9]*\|-> [A-Za-z0-9_]*' | paste - - | sort | uniq -c | head -20
