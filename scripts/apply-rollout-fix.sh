#!/usr/bin/env bash
# Apply the validated config fix to /home/cesar/models/config.ini and bounce the front server.
# Edits:
#   [*]                                   batch-size = 2048 (default for all models)
#   Qwen3.8-Flash-Next-GSQ-RCO-Coder      batch-size 4096 -> 2048 (18GB alloc fail at 4096)
#   Qwen3.8-Flash-Next (and Instruct)     batch-size 4096 -> 2048, ubatch-size 4096 -> 512
#     (mmproj cannot load on the 3090 at ub > 512; prefill compute-errors at ub >= 2048)
# Leaves every other section (Nematron, GLM, gemma, ...) untouched.
set -u

CFG=/home/cesar/models/config.ini
BAK="$CFG.bak-strata-$(date +%Y%m%d-%H%M%S)"
cp "$CFG" "$BAK"
echo "backup: $BAK"

awk '
    /^\[/ { sec = $0 }
    # Coder: batch 2048, keep ubatch 4096 (validated b2048/ub4096)
    sec ~ /GSQ-RCO-Coder/ && /^batch-size *= *4096/ { print "batch-size = 2048"; next }
    # Flash-Next + Instruct: batch 2048, ubatch 512 (validated)
    sec ~ /Qwen3.8-Flash-Next-GGUF/ && sec !~ /Coder/ {
        if (/^batch-size *= *4096/)  { print "batch-size = 2048";  next }
        if (/^ubatch-size *= *4096/) { print "ubatch-size = 512"; next }
    }
    { print }
' "$CFG" > "$CFG.new" && mv "$CFG.new" "$CFG"

# [*] default batch-size, inserted right after the [*] line if absent
if ! sed -n '/^\[\*\]/,/^\[/p' "$CFG" | head -n -1 | grep -q '^batch-size'; then
    sed -i '/^\[\*\]/a batch-size = 2048' "$CFG"
fi

echo "=== resulting Qwen3.8 keys ==="
awk '/^\[/{s=$0} /batch-size|ubatch-size/{print s ": " $0}' "$CFG" | grep -E 'Qwen3\.8|\[\*\]'

echo
echo "=== bouncing front server (systemd Restart=always) ==="
pkill -f 'build/bin/llama-server --models-preset' || true
for _ in $(seq 1 30); do
    pgrep -f 'build/bin/llama-server --models-preset' >/dev/null || break
    sleep 1
done
sleep 2
systemctl is-active llama-server
pgrep -af 'build/bin/llama-server --models-preset' | head -1 | cut -c1-120
