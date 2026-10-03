# Round 26 pointer-diff results (round 26b, 2026-10-03)

Same run as round 26 but with the instrumentation extended to print old/new data and
src pointers at the first mismatching node (commit fe9be39d5). t/s unchanged
(61.6/61.7/61.4 - logging free). 589 cycles, 623 warmup resets.

The dominant recapture - the verify TAIL graph (node 0 = RMS_NORM "norm-48", the
~150-node final-norm + output-head graph), 2x per cycle:

    node.data:  0x7b60d3208200 -> 0x7b60d3208200   (STABLE)
    node.src0:  rotates among EXACTLY 4 fixed addresses, stepping 0x280000:
                0x7b60b0000000 / 0x7b60b0280000 / 0x7b60b0500000 / 0x7b60b0780000

src0 = the last layer's output activations, which land in the speculative-verify
position ring (n_max=3 -> 4 slots). The ring rotates one slot per cycle, so the
stored node_props from the previous capture never match the current cycle - the
memcmp chases the ring forever, forcing warmup + recapture every cycle.

Secondary recapturers (sidecar aux graphs, ~2-slot rings, low frequency):
- REPEAT "hc_init": src0 alternates 0x7b657a000000 / 0x7b657a00a000
- MUL "node_1989" / MUL "node_4380": src0 alternates 2 addresses each

The BIG layer graphs (1,985 / 2,391 / 2,162 nodes) never appear in the recapture
reasons - they capture once at load and replay stably. The per-cycle recapture tax
is confined to the tail graph (which contains the dominant kernel of the cycle:
the 4x ~310 MB output-head GEMV read) and small sidecar aux graphs.

## Fix design

The rotation is address-bounded (4 slots for the tail graph, 2 for the sidecar
aux). Two candidate fixes:

1. Multi-variant graph cache (contained in ggml-cuda.cu): per shape key, keep a
   small LRU set of captured variants discriminated by a pointer fingerprint
   (hash of all node data+src pointers, ~O(nodes) per call, tens of us - vs the
   ~5-15 ms recapture it avoids). Replay only when the existing full node-props
   memcmp passes against the matched variant, preserving the current safety
   guarantee. Ring-bounded rotations then hit stable replays; anything else
   captures as today.
2. Stop the rotation at the source: either pin the verify input to a fixed
   staging address (one extra ~2.5 MB D2D copy per cycle, ~microseconds) in the
   fork's speculative verify path, or restore scheduler-side input-copy pinning
   for graph-eligible backends (stock llama.cpp pins copy slots when CUDA graphs
   are active; verify whether the fork lost that).

Next session: read common/speculative.cpp verify-input handling and the sched
cur_copy logic to pick between the two; implement; rebuild; bench. Expected:
eliminates ~2 recaptures/cycle of the head-GEMV graph - the largest single
remaining decode overhead after the graph machinery itself works.

Raw: ~/.strata-bench/phase2-round26-results.md (r26b version);
server-r26-diag.log on AISERVER.
