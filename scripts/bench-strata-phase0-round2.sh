#!/usr/bin/env bash
# Phase 0 round 2: scenarios round 1 could not measure.
#   nospec2 - truly disables speculation (spec-type = none; round 1's sed was
#             auto-overridden by MTP sidecar inference, see common/arg.cpp)
#   nmax4   - spec-draft-n-max 4 (round 1 showed n3 > n2)
#   long3   - n3 spec config + ~18K-token code prompt: prefill speed + long-ctx decode
# 3 measured requests each (2 warmups) to smooth first-request effects.
# Output: ~/.strata-bench/phase0-round2-results.md

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder" "Qwen3.8-Flash-Next")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
RESULTS="$WORK/phase0-round2-results.md"
mkdir -p "$WORK"

PROMPT='Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it.'
MAXTOK=384
NREQ=3

# build the long code prompt once (~70KB of C++/CUDA source)
LPFILE="$WORK/long-prompt.txt"
cat ~/buun-llama-cpp/src/llama-graph.cpp ~/buun-llama-cpp/ggml/src/ggml-cuda/fattn.cu ~/buun-llama-cpp/src/llama-kv-cache.cpp 2>/dev/null \
  | head -c 70000 > "$LPFILE"
LONGPROMPT="Below are three source files from a llama.cpp fork. Read them and summarize what the VBR KV-cache controller does and how fused attention handles the turbo quantization types. Be specific and cite function names.\n\n$(cat "$LPFILE")"

LPJSON="$WORK/long-prompt.json"
python3 - "$LPFILE" > "$LPJSON" <<'PYEOF'
import json, sys
body = open(sys.argv[1], encoding='utf-8', errors='replace').read()
q = ("Below are three source files from a llama.cpp fork. Read them and summarize "
     "what the VBR KV-cache controller does and how fused attention handles the "
     "turbo quantization types. Be specific and cite function names.\n\n" + body)
json.dump({"messages":[{"role":"user","content":q}], "max_tokens":256, "temperature":0}, sys.stdout)
PYEOF

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
        nospec2) sed -i '/^spec-type/d; /^\[\*\]/a spec-type = none' "$out" ;;
        nmax4)   sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 4/' "$out" ;;
        long3)   sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/' "$out" ;;
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
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died" >&2; return 1; }
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

measure() { # $1=alias $2=json-file(standard or long) -> row fields
    local alias=$1 json=$2 t
    RESP=$(mktemp)
    t=$(curl -sS -o "$RESP" -w '%{time_total}' --max-time 900 \
        -H 'Content-Type: application/json' \
        -d @- "http://127.0.0.1:$PORT/v1/chat/completions" < "$json")
    echo "$(g prompt_tokens) $(g completion_tokens) $(g prompt_per_second) $(g predicted_per_second) ${t:-0}"
}

std_json() { # $1=alias
    python3 -c 'import json,sys; json.dump({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":int(sys.argv[3]),"temperature":0}, sys.stdout)' "$1" "$PROMPT" "$MAXTOK"
}

long_json() { # $1=alias
    python3 -c 'import json,sys; d=json.load(open(sys.argv[2])); d["model"]=sys.argv[1]; json.dump(d, sys.stdout)' "$1" "$LPJSON"
}

run_scenario() { # $1=scenario $2=alias
    local scenario=$1 alias=$2 cfg="$WORK/config-r2-$1.ini" sj lj row
    make_config "$scenario" "$cfg"
    kill_server
    start_server "$cfg"
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | | |" >> "$RESULTS"
        kill_server; return
    fi
    sj="$WORK/req-std.json"
    if [ "$scenario" = "long3" ]; then
        sj="$WORK/req-long.json"; long_json "$alias" > "$sj"
    else
        std_json "$alias" > "$sj"
    fi
    # two warmups
    for _ in 1 2; do curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$sj" "http://127.0.0.1:$PORT/v1/chat/completions" || true; sleep 1; done
    for i in $(seq 1 $NREQ); do
        row=$(measure "$alias" "$sj")
        set -- $row
        echo "| $scenario | $alias | req$i | $1 | $2 | $3 | $4 | $5 |" >> "$RESULTS"
        sleep 1
    done
    cp "$WORK/server-current.log" "$WORK/server-r2-$scenario.log"
    kill_server
}

{
    echo "# Strata Phase 0 round 2 - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in nospec2 nmax4 long3; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "round2 complete: $RESULTS"
