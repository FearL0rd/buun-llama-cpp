#!/usr/bin/env bash
# Phase 2 round 13: kill the H2D expert streaming by shrinking the KV reservation.
# Discovery (round 12 profile): the Coder runs tiered ??? 27.5 GB CPU_Mapped experts while
# the V100s keep ~21 GB free each for the 512K-ctx KV pool; prefill streams 48 GB H2D.
#   ctx262k: ctx-size 524288 -> 262144   (halve the KV reservation)
#   ctx131k: ctx-size 524288 -> 131072   (quarter it)
# For each: record load_tensors CPU_Mapped size (did experts go resident?) + prefill/decode.
# Output: ~/.strata-bench/phase2-round18-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase2-round18-results.md"
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

start_server() {
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0 \
    GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 \
    LLAMA_HC_FUSED=1 \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$WORK/config-r18.ini" \
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
long_json() { python3 -c 'import json,sys; d=json.load(open(sys.argv[2])); d["model"]=sys.argv[1]; json.dump(d, sys.stdout)' "$1" "$WORK/long-prompt.json"; }

run_scenario() { # $1 = scenario, $2 = alias
    local scenario=$1 alias=$2 sjl sjs row i
    cp "$BASE_CONFIG" "$WORK/config-r18.ini"
    sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r18.ini"
    case "$scenario" in
        fitfree262k|fitfree131k)
            # unpin n-gpu-layers (the re-placement aborts on the user pin) and shrink
            # the ctx reserve so the (now fall-through) cache fit sees budget
            case "$scenario" in
                fitfree262k) ins="262144" ;;
                fitfree131k) ins="131072" ;;
            esac
            python3 ~/buun-llama-cpp/scripts/edit-coder-config.py "$WORK/config-r18.ini" "$WORK/config-r18.tmp" "$ins" \
                && mv "$WORK/config-r18.tmp" "$WORK/config-r18.ini" || { echo "| $scenario | $alias | CONFIG_EDIT_FAILED | | | | | | |" >> "$RESULTS"; return; } ;;
    esac
    sjs="$WORK/req-std.json"; std_json "$alias" > "$sjs"
    sjl="$WORK/req-long.json"; long_json "$alias" > "$sjl"
    kill_server
    start_server
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | | | $(vram) |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r18-$scenario-FAILED.log"
        kill_server; return
    fi
    { echo "### $scenario load:"; } >> "$RESULTS"
    grep -a 'load_tensors:' "$WORK/server-current.log" | grep -a 'model buffer size' | tail -6 | sed 's/^/    /' >> "$RESULTS"
    grep -a 'MoE cache' "$WORK/server-current.log" | grep -av 'accepted option\|option:' | sort -u | head -8 | sed 's/^/    /' >> "$RESULTS"
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
    cp "$WORK/server-current.log" "$WORK/server-r18-$scenario.log"
    kill_server
}

{
    echo "# Phase 2 round 18 (gate fix + ngl unpinned + reduced ctx) - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |"
    echo "|---|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in fitfree262k fitfree131k; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "round18 complete: $RESULTS"
