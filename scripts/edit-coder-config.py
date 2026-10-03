#!/usr/bin/env python3
# Edit the Coder section of a models-config: insert ctx-size and an ot line,
# optionally removing n-gpu-layers.
# Usage: edit-coder-config.py <in> <out> <ctx-size-value> [ot-line|none] [drop-ngl|keep-ngl]
import sys, re

src, dst, ctx = sys.argv[1], sys.argv[2], sys.argv[3]
ot_line = sys.argv[4] if len(sys.argv) > 4 else "none"
drop_ngl = (sys.argv[5] if len(sys.argv) > 5 else "keep-ngl") == "drop-ngl"
CODER = "[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF:IQ1_M]"

out = []
in_coder = False
removed_ngl = False
inserted_ctx = False
inserted_ot = False
for line in open(src, encoding="utf-8", errors="replace"):
    if line.startswith("["):
        in_coder = line.strip() == CODER
        out.append(line)
        if in_coder:
            out.append("ctx-size = %s\n" % ctx)
            inserted_ctx = True
            if ot_line != "none":
                out.append("ot = %s\n" % ot_line)
                inserted_ot = True
        continue
    if in_coder and drop_ngl and re.match(r"^n-gpu-layers\s*=", line):
        removed_ngl = True
        continue
    out.append(line)

open(dst, "w", encoding="utf-8").write("".join(out))
if not inserted_ctx:
    sys.exit("ERROR: Coder section not found")
if drop_ngl and not removed_ngl:
    sys.exit("ERROR: n-gpu-layers not found in Coder section")
if ot_line != "none" and not inserted_ot:
    sys.exit("ERROR: ot line not inserted")
print("config edit OK: ctx=%s ot=%s ngl-removed=%s" % (ctx, ot_line, removed_ngl))
