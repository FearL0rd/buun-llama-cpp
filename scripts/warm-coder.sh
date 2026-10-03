#!/usr/bin/env bash
# Warm up the production Coder: trigger models-autoload with a 4-token request.
curl -sS --max-time 600 -H 'Content-Type: application/json' \
    -d '{"model":"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF:IQ1_M","messages":[{"role":"user","content":"Say OK."}],"max_tokens":4,"temperature":0}' \
    http://127.0.0.1:8080/v1/chat/completions > ~/.strata-bench/warm-coder.out 2>&1
echo WARM_COMPLETE >> ~/.strata-bench/warm-coder.out
