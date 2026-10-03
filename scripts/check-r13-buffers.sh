#!/usr/bin/env bash
for f in ctx262k ctx131k; do
    echo "== $f"
    grep -a 'load_tensors:' ~/.strata-bench/server-r13-$f.log | grep -a 'model buffer size' | tail -6
done
