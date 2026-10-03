#!/usr/bin/env bash
curl -sS --max-time 300 -H 'Content-Type: application/json' -d @/tmp/req-dec.json \
    http://127.0.0.1:8080/v1/chat/completions \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("timings",{}); print("warm decode tg:", t.get("predicted_per_second"), "draft_n:", t.get("draft_n"), "accepted:", t.get("draft_n_accepted"))'
