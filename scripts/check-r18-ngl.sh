#!/usr/bin/env bash
echo "=== n-gpu-layers lines in base config (with sections):"
awk '/^\[/{s=$0} /^n-gpu-layers/{print s ": " $0}' /home/cesar/models/config.ini | head -8
echo "=== n-gpu-layers lines in round-18 variant:"
awk '/^\[/{s=$0} /^n-gpu-layers/{print s ": " $0}' ~/.strata-bench/config-r18.ini | head -8
