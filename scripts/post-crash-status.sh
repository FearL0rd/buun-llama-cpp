#!/usr/bin/env bash
# Post-crash status: is production listening, which processes run, crash signature.
echo "=== port 8080 ==="
(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep -E ':8080|:54739' | head -4
echo
echo "=== llama-server processes ==="
pgrep -af 'build/bin/llama-server' | head -3 | cut -c1-160
echo
echo "=== service state ==="
systemctl is-active llama-server; systemctl show llama-server -p Restart -p NRestarts 2>/dev/null
echo
echo "=== crash signature (journal, filtered) ==="
journalctl -u llama-server --since '-30 min' --no-pager 2>/dev/null \
  | grep -a -e 'CUDA error' -e 'GGML_ASSERT' -e 'illegal memory' -e 'cudaMalloc failed' -e 'alloc_tensor_range' \
  | tail -8
