#!/usr/bin/env bash
# Live verification against the production server on :8080 (Coder).
set -u
W=~/.strata-bench
ALIAS="Qwen3.8-Flash-Next-Coder"

mk_json() { # $1 = prompt file or text, $2 = max_tokens, $3 = out file
    python3 - "$1" "$2" "$3" <<'PYEOF'
import json, sys
src, maxtok, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    body = open(src, encoding='utf-8', errors='replace').read()
    q = "Summarize what the following source files do.\n\n" + body[:70000]
except Exception:
    q = src
json.dump({"model":"Qwen3.8-Flash-Next-Coder",
           "messages":[{"role":"user","content":q}],
           "max_tokens":maxtok, "temperature":0}, open(out,"w"))
PYEOF
}

g() { grep -o "\"$1\":[0-9.]*" "$2" | head -1 | grep -o '[0-9.]*$'; }

req() { # $1 = json file, $2 = label
    local json=$1 label=$2 t resp
    resp=$(mktemp)
    t=$(curl -sS -o "$resp" -w '%{time_total}' --max-time 900 \
        -H 'Content-Type: application/json' -d @"$json" http://127.0.0.1:8080/v1/chat/completions)
    echo "$label: prompt_tok=$(g prompt_tokens "$resp") out_tok=$(g completion_tokens "$resp") pp_t/s=$(g prompt_per_second "$resp") tg_t/s=$(g predicted_per_second "$resp") wall=$t"
    rm -f "$resp"
}

mk_json "Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it." 384 "$W/prod-short.json"
mk_json "$W/long-prompt.txt" 256 "$W/prod-long.json"

# warmup (also confirms the server is serving)
curl -s -o /dev/null --max-time 600 -H 'Content-Type: application/json' -d @"$W/prod-short.json" http://127.0.0.1:8080/v1/chat/completions || true

req "$W/prod-short.json" "decode-short-req1"
req "$W/prod-short.json" "decode-short-req2"
req "$W/prod-long.json"  "prefill-21k-req1"
req "$W/prod-long.json"  "prefill-21k-req2"
req "$W/prod-long.json"  "prefill-21k-req3"
echo "VERIFY_DONE"
