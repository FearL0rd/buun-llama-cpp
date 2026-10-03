#!/usr/bin/env bash
# Phase 2 round 22: decode on 2-GPU splits (fewer pipeline stages per cycle).
# Round 21 showed decode is dispatch/launch-bound (~7% GPU utilization, ~260 launches
# per cycle across 3 backends). Fewer stages = fewer boundaries and less dispatch.
#   2v100:    CUDA_VISIBLE_DEVICES=0,3 (both V100s, 64 GB; ctx 131072 for KV headroom)
#   3gpu-baseline: production split (3090+V100s) for a same-session comparison
# Output: ~/.strata-bench/phase2-round22-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase2-round22-results.md"
mkdir -p "$WORK"

PROMPT='Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it.'
MAXTOK=384

vram() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' '|'; }

kill_server() {
    pkill -f llama-server 2>/dev/null || true
    for _ in $(seq 1 60); do
        pgrep -f 'build/bin/llama-server' >/dev/null || break
        sleep 2
    done
    pkill -9 -f 'build/bin/llama-server' 2>/dev/null || true
    sleep 3
}

start_server() { # $1 = cuda devices, $2 = ctx line or empty
    local dev=$1 ctx=$2
    cp "$BASE_CONFIG" "$WORK/config-r22.ini"
    sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r22.ini"
    if [ -n "$ctx" ]; then
        python3 ~/buun-llama-cpp/scripts/edit-coder-config.py "$WORK/config-r22.ini" "$WORK/config-r22.tmp" "$ctx" none keep-ngl \
            && mv "$WORK/config-r22.tmp" "$WORK/config-r22.ini" || return 1
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$dev" \
    GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 \
    LLAMA_HC_FUSED=1 \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$WORK/config-r22.ini" \
        --threads 40 --threads-batch 40 --load-mode mlock \
        --host 0.0.0.0 --port "$PORT" --models-max 1 --main-gpu 0 --split-mode layer \
        --models-autoload --parallel 2 -b 2048 -ub 2048 -fa 1 --jinja \
        --spec-draft-threads 3 --cache-ram -1 --webui-mcp-proxy --verbose \
        > "$WORK/server-current.log" 2>&1 &
    SERVER_PID=$!
}

wait_ready() {
    local alias=$1
    for _ in $(seq 1 150); do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:" >&2; tail -8 "$WORK/server-current.log" >&2; return 1; }
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4,\"temperature\":0}" \
            "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || echo 000)
        [ "$code" = "200" ] && return 0
        sleep 6
    done
    return 1
}

g() { grep -o "\"$1\":[0-9.]*" "$RESP" | head -1 | grep -o '[0-9.]*$'; }

measure() {
    local json=$1 t
    RESP=$(mktemp)
    t=$(curl -sS -o "$RESP" -w '%{time_total}' --max-time 900 \
        -H 'Content-Type: application/json' \
        -d @- "http://127.0.0.1:$PORT/v1/chat/completions" < "$json")
    echo "$(g prompt_tokens) $(g completion_tokens) $(g prompt_per_second) $(g predicted_per_second) ${t:-0}"
}

std_json() { python3 -c 'import json,sys; json.dump({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":int(sys.argv[3]),"temperature":0}, sys.stdout)' "$1" "$PROMPT" "$MAXTOK"; }

run_scenario() { # $1 = scenario, $2 = alias
    local scenario=$1 alias=$2 dev ctx sjl sjs row i
    case "$scenario" in
        2v100)          dev="0,3"; ctx="131072" ;;
        3gpu-baseline)  dev="1,3,0"; ctx="" ;;
    esac
    sjs="$WORK/req-std.json"; std_json "$alias" > "$sjs"
    sjl="$WORK/req-std.json"  # decode-only scenario: short prompt both requests
    kill_server
    start_server "$dev" "$ctx" || { echo "| $scenario | $alias | CONFIG_EDIT_FAILED | | | | | | |" >> "$RESULTS"; return; }
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | | | $(vram) |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r22-$scenario-FAILED.log"
        kill_server; return
    fi
    { echo "### $scenario load:"; } >> "$RESULTS"
    grep -a 'load_tensors:' "$WORK/server-current.log" | grep -a 'model buffer size' | tail -5 | sed 's/^/    /' >> "$RESULTS"
    # warmup then 3 measured decode requests (short prompt, 384 tokens)
    curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$sjs" "http://127.0.0.1:$PORT/v1/chat/completions" || true
    for i in 1 2 3; do
        row=$(measure "$sjs")
        set -- $row
        echo "| $scenario-decode | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} | $(vram) |" >> "$RESULTS"
    done
    cp "$WORK/server-current.log" "$WORK/server-r22-$scenario.log"
    kill_server
}

{
    echo "# Phase 2 round 22 (decode on 2-GPU splits vs 3-GPU) - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |"
    echo "|---|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in 3gpu-baseline 2v100; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "round22 complete: $RESULTS"
