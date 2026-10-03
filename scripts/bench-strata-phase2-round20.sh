#!/usr/bin/env bash
# Round 20: dump the resolved tensor->device placement map (LLAMA_DUMP_PLACEMENT).
set -u
PORT=8091
ALIAS="Qwen3.8-Flash-Next-Coder"
BASE_CONFIG=/home/cesar/models/config.ini
WORK="$HOME/.strata-bench"
BIN="$HOME/buun-llama-cpp/build/bin/llama-server"

cp "$BASE_CONFIG" "$WORK/config-r20.ini"
sed -i 's/^spec-draft-n-max *= *2/spec-draft-n-max = 3/; s/^cache-prompt *= *on/cache-prompt = off/' "$WORK/config-r20.ini"

pkill -f llama-server 2>/dev/null || true; sleep 3; pkill -9 -f llama-server 2>/dev/null || true; sleep 2

LLAMA_DUMP_PLACEMENT=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3,0 \
GGML_CUDA_MMQ_MOE_ALL_BATCHES=1 LLAMA_HC_FUSED=1 \
TURBO_TCQ_CB="$HOME/buun-llama-cpp/codebooks/3bit/cb_50iter_finetuned.bin" \
TURBO_TCQ_CB2="$HOME/buun-llama-cpp/codebooks/2bit/tcq_2bit_100iter_s99.bin" \
"$BIN" \
    --models-preset "$WORK/config-r20.ini" \
    --threads 40 --threads-batch 40 --load-mode mlock \
    --host 0.0.0.0 --port "$PORT" --models-max 1 --main-gpu 0 --split-mode layer \
    --models-autoload --parallel 2 -b 2048 -ub 2048 -fa 1 --jinja \
    --spec-draft-threads 3 --cache-ram -1 --webui-mcp-proxy \
    > "$WORK/server-r20.log" 2>&1 &
SRV=$!

for _ in $(seq 1 100); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4,\"temperature\":0}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && break
    kill -0 "$SRV" 2>/dev/null || break
    sleep 6
done

pkill -f llama-server 2>/dev/null || true
for _ in $(seq 1 30); do pgrep -f 'build/bin/llama-server' >/dev/null || break; sleep 2; done
pkill -9 -f llama-server 2>/dev/null || true

OUT="$WORK/placement-map.txt"
grep -a 'placement: ' "$WORK/server-r20.log" > "$OUT"
echo "=== placement lines captured: $(wc -l < "$OUT")"
echo "=== per-buffer totals (MiB):"
awk '{s[$4]+=$5; n[$4]++} END {for (b in s) printf "%-14s %6d tensors %10.1f MiB\n", b, n[b], s[b]}' "$OUT"
echo "=== CPU_Mapped tensors (first 30):"
grep -a 'CPU_Mapped' "$OUT" | head -30
echo "=== CPU_Mapped layer histogram:"
grep -a 'CPU_Mapped' "$OUT" | grep -ao 'blk\.[0-9]*' | sort -V | uniq -c | head -40
echo "round20 complete"
