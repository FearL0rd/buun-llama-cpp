#!/usr/bin/env python3
# Replace spec-draft-n-max in one section of a models-config file.
# Usage: edit-spec-n.py <in> <out> <section-substr> <n>
import sys

src, dst, sect, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
out = []
in_sect = False
replaced = 0
for line in open(src, encoding="utf-8", errors="replace"):
    if line.startswith("["):
        in_sect = sect in line
        out.append(line)
        continue
    if in_sect and line.strip().startswith("spec-draft-n-max"):
        out.append("spec-draft-n-max = %s\n" % n)
        replaced += 1
        continue
    out.append(line)

open(dst, "w", encoding="utf-8").write("".join(out))
if replaced != 1:
    sys.exit("ERROR: expected exactly 1 spec-draft-n-max in section %r, found %d" % (sect, replaced))
print("spec-n edit OK: section=%s n=%s" % (sect, n))
