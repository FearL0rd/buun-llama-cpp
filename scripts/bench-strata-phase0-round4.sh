#!/usr/bin/env bash
# Phase 0 round 4: two configuration knobs from the Strata comparison.
#   fusionon - drop GGML_CUDA_DISABLE_FUSION=1 (Strata relies on fused kernels;
#              the production command line disables buun's CUDA fusion)
#   ub8k     - CLI -ub 8192 instead of 512 (the models config asks for
#              ubatch-size = 8192 but the command line overrides it; Strata
#              prefills in 8192-token chunks) + cache-prompt off, long prompt
# Output: ~/.strata-bench/phase0-round4-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder" "Qwen3.8-Flash-Next")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase0-round4-results.md"
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

start_server() { # $1 = config, $2 = ubatch, $3 = fusion (0 disables suppression)
    local cfg=$1 ub=$2 fusion=$3
    if [ "$fusion" = "1" ]; then
        unset GGML_CUDA_DISABLE_FUSION
        env -u GGML_CUDA_DISABLE_FUSION \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0 \
        LLAMA_HC_FUSED=1 \
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
            -b 2048 -ub "$ub" -fa 1 \
            --jinja \
            --spec-draft-threads 3 \
            --cache-ram -1 \
            --webui-mcp-proxy \
            --verbose \
            > "$WORK/server-current.log" 2>&1 &
    else
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
            -b 2048 -ub "$ub" -fa 1 \
            --jinja \
            --spec-draft-threads 3 \
            --cache-ram -1 \
            --webui-mcp-proxy \
            --verbose \
            > "$WORK/server-current.log" 2>&1 &
    fi
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

run_scenario() { # $1=scenario $2=alias
    local scenario=$1 alias=$2 cfg sj nw i row ub fusion
    if [ "$scenario" = "fusionon" ]; then
        cfg="$WORK/config-r4-fusion.ini"
        cp "$BASE_CONFIG" "$cfg"; sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/' "$cfg"
        ub=512; fusion=1; sj="$WORK/req-std.json"; std_json "$alias" > "$sj"; nw=2
    else # ub8k
        cfg="$WORK/config-r4-ub8k.ini"
        cp "$BASE_CONFIG" "$cfg"
        sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$cfg"
        ub=8192; fusion=0; sj="$WORK/req-long.json"; long_json "$alias" > "$sj"; nw=1
    fi
    kill_server
    start_server "$cfg" "$ub" "$fusion"
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | | |" >> "$RESULTS"
        cp "$WORK/server-current.log" "$WORK/server-r4-$scenario-FAILED.log"
        kill_server; return
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
    cp "$WORK/server-current.log" "$WORK/server-r4-$scenario.log"
    kill_server
}

{
    echo "# Strata Phase 0 round 4 - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in fusionon ub8k; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "round4 complete: $RESULTS"
