#!/usr/bin/env bash
# Parse phase0 server logs: per-request predicted_per_second, spec activity, moe cache init.
set -u
cd ~/.strata-bench
for f in base nospec moeoff nooverlap nmax3; do
    log="server-$f.log"
    [ -f "$log" ] || continue
    echo "== $f"
    echo "  spec_cycles: $(grep -ac 'spec cycle' "$log" || true)"
    echo "  t/s (predicted_per_second, unique):"
    grep -ao 'predicted_per_second":[0-9.]*' "$log" | grep -o '[0-9.]*$' | sort -rn | head -4 | sed 's/^/    /'
    grep -a 'draft acceptance' "$log" | tail -2 | sed 's/^/  /'
    echo "  moe-cache lines:"
    grep -ai 'moe.cache' "$log" | grep -av 'GET\|POST' | head -3 | sed 's/^/    /'
    echo "  vbr lines:"
    grep -ai 'vbr' "$log" | grep -av 'GET\|POST' | head -2 | sed 's/^/    /'
done
