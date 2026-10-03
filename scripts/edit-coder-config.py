#!/usr/bin/env python3
# Edit the Coder section of a models-config: insert ctx-size, remove n-gpu-layers.
# Usage: edit-config.py <in> <out> <ctx-size-value>
import sys, re

src, dst, ctx = sys.argv[1], sys.argv[2], sys.argv[3]
CODER = "[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF:IQ1_M]"

out = []
in_coder = False
removed_ngl = False
inserted_ctx = False
for line in open(src, encoding="utf-8", errors="replace"):
    if line.startswith("["):
        in_coder = line.strip() == CODER
        out.append(line)
        if in_coder:
            out.append("ctx-size = %s\n" % ctx)
            inserted_ctx = True
        continue
    if in_coder and re.match(r"^n-gpu-layers\s*=", line):
        removed_ngl = True
        continue
    out.append(line)

open(dst, "w", encoding="utf-8").write("".join(out))
if not inserted_ctx:
    sys.exit("ERROR: Coder section not found")
if not removed_ngl:
    sys.exit("ERROR: n-gpu-layers not found in Coder section")
print("config edit OK: ctx-size=%s inserted, n-gpu-layers removed" % ctx)
