#!/usr/bin/env bash
# Phase 2 round 26: WHY do the verify CUDA graphs recapture every cycle?
# Rounds 21-25 established the decode bottleneck: verify pays individual node
# launches (~600/cycle) instead of stable graph replays (~623 warmup resets vs
# ~589 cycles). This round runs one decode scenario with the env-gated
# ggml_cuda_graph_update_required instrumentation (commit 1f8c7b236) and
# aggregates which check forces the recapture: fattn_epoch / uid / node_count /
# node_props (with the first differing node's op and name).
# Output: ~/.strata-bench/phase2-round26-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase2-round26-results.md"
mkdir -p "$WORK"

PROMPT='Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it.'
MAXTOK=384

kill_server() {
    pkill -f llama-server 2>/dev/null || true
    for _ in $(seq 1 60); do
        pgrep -f 'build/bin/llama-server' >/dev/null || break
        sleep 2
    done
    pkill -9 -f 'build/bin/llama-server' 2>/dev/null || true
    sleep 3
}

start_server() {
    cp "$BASE_CONFIG" "$WORK/config-r26.ini"
    sed -i 's/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r26.ini"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="1,3,0" \
    GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 \
    LLAMA_HC_FUSED=1 \
    GGML_CUDA_LOG_GRAPH_UPDATE=1 \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$WORK/config-r26.ini" \
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

{
    echo "# Phase 2 round 26 (CUDA graph update-recapture diagnosis, MTP n3) - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for alias in "${ALIASES[@]}"; do
    sjs="$WORK/req-std.json"; std_json "$alias" > "$sjs"
    kill_server
    start_server || { echo "| diag | $alias | START_FAILED | | | | | |" >> "$RESULTS"; exit 1; }
    if ! wait_ready "$alias"; then
        echo "| diag | $alias | SERVER_FAILED | | | | | |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r26-diag-FAILED.log"
        kill_server
        exit 1
    fi
    curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$sjs" "http://127.0.0.1:$PORT/v1/chat/completions" || true
    for i in 1 2 3; do
        row=$(measure "$sjs")
        set -- $row
        echo "| diag | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} |" >> "$RESULTS"
    done
    {
        echo "### diagnosis:"
        echo "spec cycles: $(grep -ac 'spec cycle' "$WORK/server-current.log")"
        echo "warmup resets: $(grep -ac 'warmup reset' "$WORK/server-current.log")"
        echo "graph id reused: $(grep -ac 'Graph id' "$WORK/server-current.log")"
        echo "### reason counts:"
        grep -a -o 'reason=[a-z_]*' "$WORK/server-current.log" | sort | uniq -c
        echo "### first 30 reason lines:"
        grep -a 'graph update: reason' "$WORK/server-current.log" | head -30 | sed 's/^/    /'
    } >> "$RESULTS"
    cp "$WORK/server-current.log" "$WORK/server-r26-diag.log"
    kill_server
done

kill_server
echo "round26 complete: $RESULTS"
