#!/usr/bin/env bash
echo "=== build log: was fit.cpp compiled?"
grep -a 'fit.cpp' ~/.strata-bench/build-r16.log | head -4
echo "=== binary contains the new log string?"
strings ~/buun-llama-cpp/build/bin/libllama-server-impl.so 2>/dev/null | grep -a 'host-resident with a MoE cache' | head -2
echo "=== r16 log: new INF line present?"
grep -ac 'host-resident with a MoE cache requested' ~/.strata-bench/server-r16-fixed.log
echo "=== llama_model_has_host_moe_weights impl:"
grep -n -A20 'llama_model_has_host_moe_weights' ~/buun-llama-cpp/src/llama-model.cpp | head -30
