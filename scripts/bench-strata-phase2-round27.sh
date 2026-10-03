#!/usr/bin/env bash
# Phase 2 round 27: CUDA graph cache keyed on input pointers (the fix) —
# MTP n sweep {2, 3} with the fix active.
# Round 26b: the verify tail graph recaptured every cycle because the scheduler's
# input-copy ring (GGML_SCHED_MAX_COPIES=4) rotates nodes[0] src addresses once
# per graph rebuild, and the graph cache key mixed no data pointers. The fix
# folds nodes[0]'s data pointers into the O(1) shape key: each ring slot gets its
# own cache entry, captured once, then replayed. This round measures decode t/s
# with the fix and re-checks the draft-depth optimum (round 23 found n=2 best
# pre-fix, but the per-token verify cost changes with stable replays).
# Logging env stays on: round 26 proved it free, and the reason counts verify
# the recaptures actually stopped.
# Output: ~/.strata-bench/phase2-round27-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase2-round27-results.md"
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

start_server() { # $1 = draft n
    local n=$1
    cp "$BASE_CONFIG" "$WORK/config-r27.ini"
    sed -i 's/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r27.ini"
    python3 ~/buun-llama-cpp/scripts/edit-spec-n.py "$WORK/config-r27.ini" "$WORK/config-r27.tmp" RCO-Coder "$n" \
        && mv "$WORK/config-r27.tmp" "$WORK/config-r27.ini" || return 1
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="1,3,0" \
    GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 \
    LLAMA_HC_FUSED=1 \
    GGML_CUDA_LOG_GRAPH_UPDATE=1 \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$WORK/config-r27.ini" \
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

run_scenario() { # $1 = draft n, $2 = alias
    local n=$1 alias=$2 sjl sjs row i
    sjs="$WORK/req-std.json"; std_json "$alias" > "$sjs"
    sjl="$WORK/req-std.json"
    kill_server
    start_server "$n" || { echo "| n$n-fix | $alias | CONFIG_EDIT_FAILED | | | | | | |" >> "$RESULTS"; return; }
    if ! wait_ready "$alias"; then
        echo "| n$n-fix | $alias | SERVER_FAILED | | | | | | |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r27-n$n-FAILED.log"
        kill_server; return
    fi
    # warmup then 3 measured decode requests (short prompt, 384 tokens)
    curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$sjs" "http://127.0.0.1:$PORT/v1/chat/completions" || true
    for i in 1 2 3; do
        row=$(measure "$sjs")
        set -- $row
        echo "| n$n-fix | $alias | req$i | ${1:-0} | ${2:-0} | ${3:-0} | ${4:-0} | ${5:-0} |" >> "$RESULTS"
    done
    {
        echo "### n=$n fix-check (cycles $(grep -ac 'spec cycle' "$WORK/server-current.log"), resets $(grep -ac 'warmup reset' "$WORK/server-current.log"), reused $(grep -ac 'Graph id' "$WORK/server-current.log")):"
        echo "reason counts:"
        grep -a -o 'reason=[a-z_]*' "$WORK/server-current.log" | sort | uniq -c
        echo "spec cycles (last 6):"
        grep -a 'spec cycle' "$WORK/server-current.log" | tail -6 | sed 's/^/    /'
    } >> "$RESULTS"
    cp "$WORK/server-current.log" "$WORK/server-r27-n$n.log"
    kill_server
}

{
    echo "# Phase 2 round 27 (graph-cache pointer-key fix, MTP n sweep 2/3) - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for n in 2 3; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$n" "$alias"
    done
done

kill_server
echo "round27 complete: $RESULTS"
