#!/usr/bin/env bash
# Verify production after restart: config n, decode t/s, long-prompt prefill t/s.
CFG=/home/cesar/models/config.ini
ALIAS=SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF:IQ3_S

echo "=== config spec-draft-n-max lines:"
grep -n spec-draft-n-max $CFG

echo "=== warm-up (autoload):"
curl -sS --max-time 600 -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":4,\"temperature\":0}" \
    http://127.0.0.1:8080/v1/chat/completions \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("timings",{}); print("warmup tg:", t.get("predicted_per_second"))'

echo "=== decode 384 tokens:"
python3 -c '
import json,sys
req={"model":sys.argv[1],"messages":[{"role":"user","content":"Explain how flash-attention decoding works, then write a small CUDA kernel that implements one tile of it."}],"max_tokens":384,"temperature":0}
print(json.dumps(req))' "$ALIAS" > /tmp/req-dec.json
curl -sS --max-time 900 -H 'Content-Type: application/json' -d @/tmp/req-dec.json \
    http://127.0.0.1:8080/v1/chat/completions \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("timings",{}); print("decode tg:", t.get("predicted_per_second"), "draft_n:", t.get("draft_n"), "accepted:", t.get("draft_n_accepted"))'

echo "=== prefill ~20K-token prompt:"
python3 -c '
import json,sys
words = "the quick brown fox jumps over the lazy dog while programmers debate cache coherency and speculative decoding throughput in a small datacenter " * 1300
req={"model":sys.argv[1],"messages":[{"role":"user","content":words + " Summarize in one word."}],"max_tokens":8,"temperature":0}
print(json.dumps(req))' "$ALIAS" > /tmp/req-ppl.json
curl -sS --max-time 900 -H 'Content-Type: application/json' -d @/tmp/req-ppl.json \
    http://127.0.0.1:8080/v1/chat/completions \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); u=d.get("usage",{}); t=d.get("timings",{}); print("prefill pp:", t.get("prompt_per_second"), "prompt_tokens:", u.get("prompt_tokens"))'

echo VERIFY_DONE
