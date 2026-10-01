#!/usr/bin/env bash
# Phase 0 baseline benchmark: measure decode t/s for Qwen3.8-Flash-Next-Coder and
# Qwen3.8-Flash-Next under scenario variants that isolate where decode time goes.
#
# Scenarios (each = one llama-server start with a sed-variant of models config):
#   base       - config as-is (draft-mtp n2, moe-cache auto, cpu-overlap auto, VBR t8->t4)
#   nospec     - spec-type removed (no speculation; isolates MTP contribution)
#   moeoff     - moe-cache off (isolates existing expert cache contribution)
#   nooverlap  - moe-cache-cpu-overlap = 0 (isolates CPU overlap contribution)
#   nmax3      - spec-draft-n-max 2 -> 3 (Strata's draft window)
#
# Output: ~/.strata-bench/phase0-results.md  (one line per request, plus VRAM + spec stats)
# NOTE: kills any running llama-server on this box and uses port 8091.

set -u

PORT=8091
ALIASES=("Qwen3.8-Flash-Next-Coder" "Qwen3.8-Flash-Next")
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"
mkdir -p "$WORK"
RESULTS="$WORK/phase0-results.md"

# measured request: greedy, fixed prompt, generous cap (thinking tokens count as output)
PROMPT='Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it.'
MAXTOK=384
NREQ=2   # measured requests per scenario (warmup excluded)

kill_server() {
    pkill -f 'llama-server' 2>/dev/null || true
    for _ in $(seq 1 60); do
        pgrep -f 'llama-server' >/dev/null || break
        sleep 2
    done
    pkill -9 -f 'llama-server' 2>/dev/null || true
    sleep 3
}

vram() {
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' '|'
}

make_config() { # $1 = variant name -> $2 = output path
    local variant=$1 out=$2
    cp "$BASE_CONFIG" "$out"
    case "$variant" in
        base)      : ;;
        nospec)    sed -i '/^spec-type/d' "$out" ;;
        moeoff)    sed -i 's/^moe-cache *= *auto/moe-cache = off/; s/^moe-cache *= *on/moe-cache = off/; s/^moe-cache *= *soft/moe-cache = off/' "$out" ;;
        nooverlap) sed -i 's/^moe-cache-cpu-overlap *= *auto/moe-cache-cpu-overlap = 0/' "$out" ;;
        nmax3)     sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/' "$out" ;;
    esac
}

start_server() { # $1 = config variant path
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

wait_ready() { # $1 = alias; polls warmup until model loaded (max ~15 min)
    local alias=$1
    for _ in $(seq 1 150); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server process died during startup; last log lines:" >&2
            tail -5 "$WORK/server-current.log" >&2
            return 1
        fi
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4,\"temperature\":0}" \
            "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || echo 000)
        [ "$code" = "200" ] && return 0
        sleep 6
    done
    return 1
}

measure() { # $1 = alias -> echoes "prompt_toks comp_toks seconds"
    local alias=$1 resp t p c
    resp=$(mktemp)
    t=$(curl -sS -o "$resp" -w '%{time_total}' --max-time 600 \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}" \
        "http://127.0.0.1:$PORT/v1/chat/completions")
    p=$(grep -o '"prompt_tokens":[0-9]*' "$resp" | head -1 | grep -o '[0-9]*')
    c=$(grep -o '"completion_tokens":[0-9]*' "$resp" | head -1 | grep -o '[0-9]*')
    rm -f "$resp"
    echo "${p:-0} ${c:-0} ${t:-0}"
}

run_scenario() { # $1 = scenario, $2 = alias
    local scenario=$1 alias=$2 cfg="$WORK/config-$1.ini" line p c s
    make_config "$scenario" "$cfg"
    kill_server
    start_server "$cfg"
    if ! wait_ready "$alias"; then
        echo "| $scenario | $alias | SERVER_FAILED | | | | $(vram) |" >> "$RESULTS"
        kill_server
        return
    fi
    # warmup already happened inside wait_ready ("ping" request)
    for i in $(seq 1 $NREQ); do
        line=$(measure "$alias")
        set -- $line
        p=$1; c=$2; s=$3
        if [ "${c:-0}" -gt 0 ] 2>/dev/null; then
            ts=$(awk -v c="$c" -v s="$s" 'BEGIN{printf "%.1f", c/s}')
            echo "| $scenario | $alias | req$i | $p | $c | $s | $ts | $(vram) |" >> "$RESULTS"
        else
            echo "| $scenario | $alias | req$i | $p | $c | $s | ERROR | $(vram) |" >> "$RESULTS"
        fi
        sleep 2
    done
    # speculative acceptance stats for spec-enabled scenarios
    if [ "$scenario" != "nospec" ]; then
        { echo; echo "### spec stats: $scenario / $alias"; } >> "$RESULTS"
        grep -iE 'accept|draft' "$WORK/server-current.log" | tail -15 >> "$RESULTS" || true
    fi
    # keep the full log per scenario
    cp "$WORK/server-current.log" "$WORK/server-$scenario.log"
    kill_server
}

{
    echo "# Strata Phase 0 baseline - $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "GPUs: $(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' ';')"
    echo "Server: $BIN | CUDA_VISIBLE_DEVICES=1,3,0 | port $PORT"
    echo
    echo "| scenario | model | run | prompt_tok | out_tok | wall_s | out_t/s | vram(idx,used,total) |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$RESULTS"

for scenario in base nospec moeoff nooverlap nmax3; do
    for alias in "${ALIASES[@]}"; do
        run_scenario "$scenario" "$alias"
    done
done

kill_server
echo "DONE" >> "$RESULTS"
echo "phase0 complete: $RESULTS"
