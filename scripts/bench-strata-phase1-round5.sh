#!/usr/bin/env bash
# Phase 1 round 5: A/B the Volta MoE prefill MMQ route.
#   off - env unset: V100 MoE prefill takes the sorted per-expert cuBLAS fallback
#   on  - GGML_CUDA_MMQ_MOE_ALL_BATCHES=1: dp4a MMQ with device-side expert routing
# Both scenarios: cache-prompt off, -ub 4096, long 21K code prompt (prefill measurement)
# followed by a short decode request (regression check). Both Qwen3.8 models.
# Output: ~/.strata-bench/phase1-round5-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder" "Qwen3.8-Flash-Next")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase1-round5-results.md"
mkdir -p "$WORK"

PROMPT='Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it.'
MAXTOK=384

kill_server() {
    pkill -f llama-server 2>/dev/null || true
    for _ in $(seq 1 60); do
        pgrep -f llama-server >/dev/null || break
        sleep 2
    done
    pkill -9 -f llama-server 2>/dev/null || true
    sleep 3
}

start_server() { # $1 = 0|1 (MMQ MoE env)
    local mmqmoe=$1
    if [ "$mmqmoe" = "1" ]; then
        export GGML_CUDA_MMQ_MOE_ALL_BATCHES=1
    else
        unset GGML_CUDA_MMQ_MOE_ALL_BATCHES
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0 \
    LLAMA_HC_FUSED=1 \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$WORK/config-r5.ini" \
        --threads 40 --threads-batch 40 \
        --load-mode mlock \
        --host 0.0.0.0 --port "$PORT" \
        --models-max 1 \
        --main-gpu 0 \
        --split-mode layer \
        --models-autoload \
        --parallel 2 \
        -b 2048 -ub 4096 -fa 1 \
        --jinja \
        --spec-draft-threads 3 \
        --cache-ram -1 \
        --webui-mcp-proxy \
        --verbose \
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
    echo "timeout; last log:" >&2; tail -8 "$WORK/server-current.log" >&2
    return 1
}

g() { grep -o "\"$1\":[0-9.]*" "$RESP" | head -1 | grep -o '[0-9.]*$'; }

measure() { # $1 = request json file
    local json=$1 t
    RESP=$(mktemp)
    t=$(curl -sS -o "$RESP" -w '%{time_total}' --max-time 900 \
        -H 'Content-Type: application/json' \
        -d @- "http://127.0.0.1:$PORT/v1/chat/completions" < "$json")
    echo "$(g prompt_tokens) $(g completion_tokens) $(g prompt_per_second) $(g predicted_per_second) ${t:-0}"
}

std_json() { python3 -c 'import json,sys; json.dump({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":int(sys.argv[3]),"temperature":0}, sys.stdout)' "$1" "$PROMPT" "$MAXTOK"; }
long_json() { python3 -c 'import json,sys; d=json.load(open(sys.argv[2])); d["model"]=sys.argv[1]; json.dump(d, sys.stdout)' "$1" "$WORK/long-prompt.json"; }

run_scenario() { # $1=off|on $2=alias
    local scenario=$1 alias=$2 sjl sjs row i
    cp "$BASE_CONFIG" "$WORK/config-r5.ini"
    sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r5.ini"
    sjs="$WORK/req-std.json"; std_json "$alias" > "$sjs"
    sjl="$WORK/req-long.json"; long_json "$alias" > "$sjl"
    kill_server
    if [ "$scenario" = "on" ]; then start_server 1; else start_server 0; fi
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | | |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r5-$scenario-FAILED.log"
        kill_server; return
    fi
    # refuse to measure on a degraded load (repeated buffer alloc failures)
    if grep -aq 'cudaMalloc failed' "$WORK/server-current.log"; then
        echo "| $scenario | $alias | LOAD_DEGRADED_OOM | | | | | |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r5-$scenario-FAILED.log"
        kill_server; return
    fi
    # long prompt, 3 measured (uncached, cache-prompt off): prefill speed
    for i in 1 2 3; do
        row=$(measure "$sjl")
        set -- $row
        echo "| $scenario-prefill | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} |" >> "$RESULTS"
    done
    # decode regression check on the same server (short prompt, spec on)
    curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$sjs" "http://127.0.0.1:$PORT/v1/chat/completions" || true
    for i in 1 2; do
        row=$(measure "$sjs")
        set -- $row
        echo "| $scenario-decode | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} |" >> "$RESULTS"
    done
    cp "$WORK/server-current.log" "$WORK/server-r5-$scenario.log"
    kill_server
}

{
    echo "# Phase 1 round 5 (Volta MoE prefill MMQ A/B) - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in off on; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "round5 complete: $RESULTS"
