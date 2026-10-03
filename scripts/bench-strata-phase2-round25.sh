#!/usr/bin/env bash
# Phase 2 round 25: expert tensor layout A/B (default whole-expert spread vs
# LLAMA_SPLIT_EXPERTS=slice), decode focus.
# Round 21 api-sum: decode is sync-wait-bound (~77% of wall in stream+event
# syncs; ~226 event syncs/cycle). The expert-axis placement (llama-model.cpp
# get_split_segments: stacked expert tensors split into single-expert segments
# spread across ALL device buffers) forces per-MoE-layer cross-backend expert
# windows + reduce at decode. Hypothesis: that machinery is the sync storm.
#   default: whole experts spread across devices (per-device expert windows)
#   slice:   every expert row-sliced across all devices (old layout; every
#            active expert must touch every device -> expect WORSE if the
#            expert window/reduce path is on the critical path)
# All scenarios: production 3-GPU split, production config (MTP n3), threads 40.
# Output: ~/.strata-bench/phase2-round25-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase2-round25-results.md"
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

start_server() { # $1 = scenario
    local scen=$1
    cp "$BASE_CONFIG" "$WORK/config-r25.ini"
    sed -i 's/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r25.ini"
    local extra_env=""
    if [ "$scen" = "exp-slice" ]; then
        extra_env="LLAMA_SPLIT_EXPERTS=slice"
    fi
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="1,3,0" \
    GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 \
    LLAMA_HC_FUSED=1 \
    $extra_env \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$WORK/config-r25.ini" \
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
    local scen=$1 alias=$2 sjl sjs row i
    sjs="$WORK/req-std.json"; std_json "$alias" > "$sjs"
    sjl="$WORK/req-std.json"
    kill_server
    start_server "$scen" || { echo "| $scen | $alias | START_FAILED | | | | | | |" >> "$RESULTS"; return; }
    if ! wait_ready "$alias"; then
        echo "| $scen | $alias | SERVER_FAILED | | | | | | $(vram) |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r25-$scen-FAILED.log"
        kill_server; return
    fi
    { echo "### $scen load:"; } >> "$RESULTS"
    grep -a 'load_tensors:' "$WORK/server-current.log" | grep -a 'model buffer size' | tail -5 | sed 's/^/    /' >> "$RESULTS"
    # warmup then 3 measured decode requests (short prompt, 384 tokens)
    curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$sjs" "http://127.0.0.1:$PORT/v1/chat/completions" || true
    for i in 1 2 3; do
        row=$(measure "$sjs")
        set -- $row
        echo "| $scen-decode | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} | $(vram) |" >> "$RESULTS"
    done
    { echo "### $scen spec cycles (last 8):"; grep -a 'spec cycle' "$WORK/server-current.log" | tail -8 | sed 's/^/    /'; } >> "$RESULTS"
    cp "$WORK/server-current.log" "$WORK/server-r25-$scen.log"
    kill_server
}

{
    echo "# Phase 2 round 25 (expert layout: default spread vs LLAMA_SPLIT_EXPERTS=slice) - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |"
    echo "|---|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scen in exp-default exp-slice; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scen" "$alias"
    done
done

kill_server
echo "round25 complete: $RESULTS"
