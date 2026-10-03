#!/usr/bin/env bash
awk '/^\[/{s=$0} /ngl|gpu-layer|cpu-moe|ncmoe|^ot |tensor-split|^ts /{print s ": " $0}' /home/cesar/models/config.ini | grep -i coder
echo "=== all keys in the Coder section:"
sed -n '/^\[ISTA-DASLab\/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF:IQ1_M\]$/,/^\[/p' /home/cesar/models/config.ini | grep -v '^#\|^;'
