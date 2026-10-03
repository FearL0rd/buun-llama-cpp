#!/usr/bin/env bash
# Phase 2 round 21: decode profile (nsys) of the Coder under steady speculative decoding.
# Capture: model load, short prompt (32 tok), ~512 decoded tokens with MTP n3 + fusion on.
# The trace isolates decode cycles; kern_sum ranks where the 50.3ms/cycle goes.
set -u

PORT=8092
ALIAS="Qwen3.8-Flash-Next-Coder"
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
NSYS=/usr/local/cuda-12.8/bin/nsys
DURATION=300
mkdir -p "$WORK"

cp "$BASE_CONFIG" "$WORK/config-r21.ini"
sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r21.ini"

# decode-heavy request: short prompt, many tokens
python3 - "$WORK" <<'PYEOF'
import json, sys
w = sys.argv[1]
json.dump({"model":"Qwen3.8-Flash-Next-Coder",
           "messages":[{"role":"user","content":"Write a long, detailed technical explanation of how speculative decoding works, including the draft and verify passes."}],
           "max_tokens":512,"temperature":0}, open(w+"/req-decode-512.json","w"))
PYEOF

pkill -f llama-server 2>/dev/null || true; sleep 3; pkill -9 -f llama-server 2>/dev/null || true; sleep 2

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0
export GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 LLAMA_HC_FUSED=1
export TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin"
export TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin"

nohup "$NSYS" profile -t cuda,nvtx -o "$WORK/decode-r21" --force-overwrite true \
    --duration "$DURATION" \
    "$BIN" \
        --models-preset "$WORK/config-r21.ini" \
        --threads 40 --threads-batch 40 --load-mode mlock \
        --host 0.0.0.0 --port "$PORT" --models-max 1 --main-gpu 0 --split-mode layer \
        --models-autoload --parallel 2 -b 2048 -ub 2048 -fa 1 --jinja \
        --spec-draft-threads 3 --cache-ram -1 --webui-mcp-proxy --verbose \
    > "$WORK/server-r21.log" 2>&1 &
NSYS_PID=$!

ready=0
for _ in $(seq 1 80); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4,\"temperature\":0}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { ready=1; break; }
    kill -0 "$NSYS_PID" 2>/dev/null || break
    sleep 6
done
echo "ready: $ready"

if [ "$ready" = "1" ]; then
    t0=$(date +%s.%N)
    curl -sS -o "$WORK/decode-r21-resp.json" --max-time 600 -H 'Content-Type: application/json' \
        -d @"$WORK/req-decode-512.json" "http://127.0.0.1:$PORT/v1/chat/completions" || true
    t1=$(date +%s.%N)
    echo "decode request wall: $(echo "$t1 - $t0" | bc)"
    grep -o '"completion_tokens":[0-9]*\|"predicted_per_second":[0-9.]*' "$WORK/decode-r21-resp.json" | head -2
fi

wait "$NSYS_PID" 2>/dev/null || true
for _ in $(seq 1 60); do kill -0 "$NSYS_PID" 2>/dev/null || break; sleep 5; done
pkill -f llama-server 2>/dev/null || true; sleep 2

OUT="$WORK/phase2-round21.txt"
{
    echo "# Phase 2 round 21 decode profile (Coder, MTP n3, fusion on) - $(date)"
    echo
    echo "## GPU kernel time (top 40)"
    "$NSYS" stats --report cuda_gpu_kern_sum --format table "$WORK/decode-r21.nsys-rep" 2>/dev/null | head -55
    echo
    echo "## Spec cycle stats from server log"
    grep -a 'spec cycle' "$WORK/server-r21.log" | tail -4
} > "$OUT"
echo "round21 complete: $OUT"
