#!/usr/bin/env bash
echo "=== kernel oom-killer events:"
dmesg 2>/dev/null | grep -ai 'oom\|killed process' | tail -5
journalctl -k --since '-2 hours' --no-pager 2>/dev/null | grep -ai 'oom\|killed process' | tail -5
echo "=== ot option acceptance in r19 log:"
grep -a 'accepted option: ot\|accepted option: override' ~/.strata-bench/server-r19-ot131k.log | head -3
echo "=== last 12 lines of r19 server log:"
tail -12 ~/.strata-bench/server-r19-ot131k.log | cut -c1-140
echo "=== host memory now:"
free -g | head -2
