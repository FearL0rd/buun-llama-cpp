#!/usr/bin/env bash
# Phase 0 round 3: the two measurements round 2 could not produce.
#   nospec3  - true no-speculation baseline: drop the md= draft sidecars so
#              auto-inference (common/arg.cpp) cannot re-enable MTP.
#   prefill3 - n3 spec config with cache-prompt = off and the 21K-token code
#              prompt: real prompt-processing speed (round 2's warmups had
#              cached the whole prompt, leaving only 4 uncached tokens).
# Output: ~/.strata-bench/phase0-round3-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder" "Qwen3.8-Flash-Next")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase0-round3-results.md"
mkdir -p "$WORK"

PROMPT='Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it.'
MAXTOK=384
LPFILE="$WORK/long-prompt.txt"

kill_server() {
    pkill -f llama-server 2>/dev/null || true
    for _ in $(seq 1 60); do
        pgrep -f llama-server >/dev/null || break
        sleep 2
    done
    pkill -9 -f llama-server 2>/dev/null || true
    sleep 3
}

make_config() {
    local variant=$1 out=$2
    cp "$BASE_CONFIG" "$out"
    case "$variant" in
        nospec3)  sed -i '/^spec-type/d; s/^md *=/; md_removed =/' "$out" ;;
        prefill3) sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$out" ;;
    esac
}

start_server() {
    local cfg=$1
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0 \
    GGML_CUDA_DISABLE_FUSION=1 LLAMA_HC_FUSED=1 \
    TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
    TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
    "$BIN" \
        --models-preset "$cfg" \
        --threads 40 --threads-batch 40 \
        --load-mode mlock \
        --host 0.0.0.0 --port "$PORT" \
        --models-max 1 \
        --main-gpu 0 \
        --split-mode layer \
        --models-autoload \
        --parallel 2 \
        -b 2048 -ub 512 -fa 1 \
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
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:" >&2; tail -5 "$WORK/server-current.log" >&2; return 1; }
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4,\"temperature\":0}" \
            "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || echo 000)
        [ "$code" = "200" ] && return 0
        sleep 6
    done
    echo "timeout; last log lines:" >&2; tail -5 "$WORK/server-current.log" >&2
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

run_scenario() { # $1=scenario $2=alias
    local scenario=$1 alias=$2 cfg="$WORK/config-r3-$1.ini" sj nw i row
    make_config "$scenario" "$cfg"
    kill_server
    start_server "$cfg"
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | | |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r3-$scenario-FAILED.log"
        kill_server; return
    fi
    if [ "$scenario" = "prefill3" ]; then
        sj="$WORK/req-long.json"; long_json "$alias" > "$sj"; nw=1
    else
        sj="$WORK/req-std.json"; std_json "$alias" > "$sj"; nw=2
    fi
    for i in $(seq 1 $nw); do
        curl -s -o /dev/null --max-time 900 -H 'Content-Type: application/json' -d @"$sj" "http://127.0.0.1:$PORT/v1/chat/completions" || true
        sleep 1
    done
    for i in $(seq 1 3); do
        row=$(measure "$sj")
        set -- $row
        echo "| $scenario | $alias | req$i | $1 | $2 | $3 | $4 | $5 |" >> "$RESULTS"
        sleep 1
    done
    cp "$WORK/server-current.log" "$WORK/server-r3-$scenario.log"
    kill_server
}

{
    echo "# Strata Phase 0 round 3 - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in nospec3 prefill3; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "round3 complete: $RESULTS"
