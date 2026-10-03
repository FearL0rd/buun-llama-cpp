#!/usr/bin/env bash
# Phase 1 round 11: Coder with CPU-resident MoE experts (Strata-style tiering via buun's
# own moe-cache machinery) so the TRUE 4096-token prefill graph fits.
#   expertsoff4096: n-cpu-moe = 99 (all MoE layers on CPU), moe-cache = 4096 MiB budget,
#                   moe-cache-cpu-overlap = auto (existing), -b 4096 -ub 4096
# Compare prefill/decode vs the current best: resident weights, effective ubatch 2048,
# 1,024 t/s prefill / ~59.8 t/s decode (verified in production).
# Output: ~/.strata-bench/phase1-round11-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase1-round11-results.md"
mkdir -p "$WORK"

PROMPT='Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it.'
MAXTOK=384

vram() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' '|'; }

kill_server() {
    pkill -f 'llama-server' 2>/dev/null || true
    for _ in $(seq 1 60); do
        pgrep -f 'build/bin/llama-server' >/dev/null || break
        sleep 2
    done
    pkill -9 -f 'build/bin/llama-server' 2>/dev/null || true
    sleep 3
}

start_server() { # $1 = ubatch (b == ub)
    local ub=$1
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0 \
    GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 \
    LLAMA_HC_FUSED=1 \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$WORK/config-r11.ini" \
        --threads 40 --threads-batch 40 \
        --load-mode mlock \
        --host 0.0.0.0 --port "$PORT" \
        --models-max 1 \
        --main-gpu 0 \
        --split-mode layer \
        --models-autoload \
        --parallel 2 \
        -b "$ub" -ub "$ub" -fa 1 \
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
    for _ in $(seq 1 200); do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:" >&2; tail -8 "$WORK/server-current.log" >&2; return 1; }
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
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

measure() {
    local json=$1 t
    RESP=$(mktemp)
    t=$(curl -sS -o "$RESP" -w '%{time_total}' --max-time 900 \
        -H 'Content-Type: application/json' \
        -d @- "http://127.0.0.1:$PORT/v1/chat/completions" < "$json")
    echo "$(g prompt_tokens) $(g completion_tokens) $(g prompt_per_second) $(g predicted_per_second) ${t:-0}"
}

std_json() { python3 -c 'import json,sys; json.dump({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":int(sys.argv[3]),"temperature":0}, sys.stdout)' "$1" "$PROMPT" "$MAXTOK"; }
long_json() { python3 -c 'import json,sys; d=json.load(open(sys.argv[2])); d["model"]=sys.argv[1]; json.dump(d, sys.stdout)' "$1" "$WORK/long-prompt.json"; }

run_scenario() { # $1 = scenario, $2 = alias
    local scenario=$1 alias=$2 ub sjl sjs row i
    case "$scenario" in
        expertsoff4096) ub=4096 ;;
    esac
    # build config variant: Coder section gets n-cpu-moe 99 + moe-cache 4096 MiB
    # (rewrite moe-cache in place — a later duplicate key would win over an inserted one)
    awk '
        /^\[ISTA-DASLab\/Qwen3\.8-Flash-Next-GSQ-RCO-Coder-GGUF:IQ1_M\]$/ { print; print "n-cpu-moe = 99"; incoder=1; next }
        /^\[/ { incoder=0 }
        incoder && /^moe-cache *=/ { print "moe-cache = 4096"; next }
        { print }
    ' "$BASE_CONFIG" > "$WORK/config-r11.ini"
    sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r11.ini"
    sjs="$WORK/req-std.json"; std_json "$alias" > "$sjs"
    sjl="$WORK/req-long.json"; long_json "$alias" > "$sjl"
    kill_server
    start_server "$ub"
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | | | $(vram) |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r11-$scenario-FAILED.log"
        kill_server; return
    fi
    { echo "### $scenario: accepted moe keys:"; } >> "$RESULTS"
    grep -a 'accepted option: \(n-cpu-moe\|moe-cache\)' "$WORK/server-current.log" | head -4 | sed 's/^/    /' >> "$RESULTS"
    for i in 1 2 3; do
        row=$(measure "$sjl")
        set -- $row
        echo "| $scenario-prefill | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} | $(vram) |" >> "$RESULTS"
    done
    curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$sjs" "http://127.0.0.1:$PORT/v1/chat/completions" || true
    for i in 1 2; do
        row=$(measure "$sjs")
        set -- $row
        echo "| $scenario-decode | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} | $(vram) |" >> "$RESULTS"
    done
    { echo "### $scenario: pool OOM lines: $(grep -ac 'CUDA pool allocation failed' "$WORK/server-current.log")"; echo; } >> "$RESULTS"
    cp "$WORK/server-current.log" "$WORK/server-r11-$scenario.log"
    kill_server
}

{
    echo "# Phase 1 round 11 (Coder CPU experts + moe-cache budget, true ub 4096) - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |"
    echo "|---|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in expertsoff4096; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "round11 complete: $RESULTS"
