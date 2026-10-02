#!/usr/bin/env bash
# Read-only check of the production llama-server rollout state.
echo "=== systemd unit (ExecStart/Environment) ==="
systemctl cat llama-server 2>/dev/null | grep -E 'ExecStart|Environment' | head -25
echo
echo "=== full ExecStart (continuation lines joined) ==="
systemctl cat llama-server 2>/dev/null | awk '/^ExecStart/{f=1} f{printf "%s ", $0} f && !/\\$/{print ""; f=0}' | head -5
echo
echo "=== running server processes ==="
pgrep -af 'build/bin/llama-server' | head -3
echo
for pid in $(pgrep -f 'build/bin/llama-server' | head -2); do
    echo "--- pid $pid cmdline:"
    tr '\0' ' ' < /proc/$pid/cmdline | head -c 700; echo
    echo "--- pid $pid env (GGML/LLAMA/TURBO):"
    tr '\0' '\n' < /proc/$pid/environ | grep -E 'GGML_CUDA_MMQ|DISABLE_FUSION|LLAMA_HC|TURBO' | head -6
done
echo
echo "=== models config: section -> keys ==="
awk '/^\[/{s=$0} /spec-draft-n-max|ubatch-size|cache-prompt/{print s ": " $0}' /home/cesar/models/config.ini
