#!/usr/bin/env bash
# Phase 2 round 12: per-op prefill profile of the Coder under the current best config.
# nsys wraps the front server (child processes traced by default); one bounded capture
# window covering: model load, ping, 2x 21K-token prefill, 1x decode.
# Output: ~/.strata-bench/prefill-r12.nsys-rep + kern/mem summaries in phase2-round12.txt
set -u

PORT=8092
ALIAS="Qwen3.8-Flash-Next-Coder"
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
NSYS=/usr/local/cuda-12.8/bin/nsys
DURATION=420
mkdir -p "$WORK"

cp "$BASE_CONFIG" "$WORK/config-r12.ini"
sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r12.ini"

# build request payloads (Coder)
python3 - "$WORK" <<'PYEOF'
import json, sys
w = sys.argv[1]
q = ("Explain how flash-attention decoding works, then write a small CUDA kernel "
     "that implements one tile of it.")
json.dump({"model":"Qwen3.8-Flash-Next-Coder",
           "messages":[{"role":"user","content":q}],
           "max_tokens":384,"temperature":0}, open(w + "/req-std-coder.json","w"))
d = json.load(open(w + "/long-prompt.json"))
d["model"] = "Qwen3.8-Flash-Next-Coder"
d["max_tokens"] = 256
json.dump(d, open(w + "/req-long-coder.json","w"))
PYEOF

pkill -f llama-server 2>/dev/null || true
sleep 3
pkill -9 -f llama-server 2>/dev/null || true
sleep 2

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0
export GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 LLAMA_HC_FUSED=1
export TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin"
export TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin"

nohup "$NSYS" profile -t cuda,nvtx -o "$WORK/prefill-r12" --force-overwrite true \
    --duration "$DURATION" \
    "$BIN" \
        --models-preset "$WORK/config-r12.ini" \
        --threads 40 --threads-batch 40 \
        --load-mode mlock \
        --host 0.0.0.0 --port "$PORT" \
        --models-max 1 \
        --main-gpu 0 \
        --split-mode layer \
        --models-autoload \
        --parallel 2 \
        -b 2048 -ub 2048 -fa 1 \
        --jinja \
        --spec-draft-threads 3 \
        --cache-ram -1 \
        --webui-mcp-proxy \
        --verbose \
    > "$WORK/server-r12.log" 2>&1 &
NSYS_PID=$!
echo "nsys pid: $NSYS_PID"

# wait for the model to be ready (max ~6 min; nsys slows the load)
ready=0
for _ in $(seq 1 100); do
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
    for i in 1 2; do
        curl -sS -o /dev/null --max-time 600 -H 'Content-Type: application/json' \
            -d @"$WORK/req-long-coder.json" \
            "http://127.0.0.1:$PORT/v1/chat/completions" || true
    done
    curl -sS -o /dev/null --max-time 600 -H 'Content-Type: application/json' \
        -d @"$WORK/req-std-coder.json" \
        "http://127.0.0.1:$PORT/v1/chat/completions" || true
fi

# let the capture window close and nsys finalize
wait "$NSYS_PID" 2>/dev/null || true
for _ in $(seq 1 60); do
    kill -0 "$NSYS_PID" 2>/dev/null || break
    sleep 5
done
pkill -f llama-server 2>/dev/null || true
sleep 2

# summaries
OUT="$WORK/phase2-round12.txt"
{
    echo "# Phase 2 round 12 prefill profile (Coder, b/ub 2048, MMQ on) - $(date)"
    echo
    echo "## GPU kernel time (top 40)"
    "$NSYS" stats --report cuda_gpu_kern_sum --format table "$WORK/prefill-r12.nsys-rep" 2>/dev/null | head -60
    echo
    echo "## GPU memcpy time"
    "$NSYS" stats --report cuda_gpu_mem_time_sum --format table "$WORK/prefill-r12.nsys-rep" 2>/dev/null | head -20
    echo
    echo "## GPU mem size"
    "$NSYS" stats --report cuda_gpu_mem_size_sum --format table "$WORK/prefill-r12.nsys-rep" 2>/dev/null | head -20
} > "$OUT"
echo "round12 complete: $OUT"
