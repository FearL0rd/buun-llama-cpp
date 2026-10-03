#pragma once

#include "ggml.h"
#include "../ggml-impl.h"

#include <cstdint>

// CUDA graph instances hard-code tensor shapes AND baked-in data pointers. This
// bounded O(1) key keeps alternating speculative verify widths — and the
// scheduler's input-copy ring slots, which rotate a split's nodes[0] source
// addresses once per graph rebuild (i.e. once per decode cycle) — in independent
// warmup/cache entries. Any residual collision remains safe because the
// existing graph update check compares every node property before capture or
// replay.
static inline uint64_t ggml_cuda_graph_shape_key(const ggml_cgraph * cgraph) {
    uint64_t key = (uint64_t) (uintptr_t) cgraph->nodes[0];
    const auto mix = [&key](uint64_t value) {
        key = (key ^ value) * 0x100000001b3ull;
    };

    mix(cgraph->n_nodes);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        mix(cgraph->nodes[0]->ne[d]);
        mix(cgraph->nodes[cgraph->n_nodes - 1]->ne[d]);
    }
    // Key on the first node's baked-in data pointers: split-input copy tensors
    // are selected per graph rebuild via the scheduler's cur_copy ring, so
    // without this every rebuild looks like a changed graph and the entry
    // recaptures forever. With it, each ring slot keeps its own entry.
    mix((uint64_t) (uintptr_t) cgraph->nodes[0]->data);
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        mix((uint64_t) (uintptr_t) (cgraph->nodes[0]->src[j] ? cgraph->nodes[0]->src[j]->data : nullptr));
    }
    return key;
}
