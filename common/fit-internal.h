#pragma once

#include "../src/llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <vector>

// Meta handles include model-specific split state. Repeated dry loads must
// compare the ordered physical inventory, not the identity of those wrappers.
inline bool common_fit_same_devices(
        const std::vector<ggml_backend_dev_t> & a,
        const std::vector<ggml_backend_dev_t> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        if (!ggml_backend_dev_is_meta(a[i]) || !ggml_backend_dev_is_meta(b[i])) return false;
        const size_t count = ggml_backend_meta_dev_n_devs(a[i]);
        if (count != ggml_backend_meta_dev_n_devs(b[i])) return false;
        for (size_t j = 0; j < count; ++j) {
            if (ggml_backend_meta_dev_simple_dev(a[i], j) != ggml_backend_meta_dev_simple_dev(b[i], j)) return false;
        }
    }
    return true;
}

// Tensor fit uses a balanced-equivalent aggregate. Ordinary child-device
// buffers (KV, compute, etc.) are separate from the Meta allocation. Charge
// each component at its largest child footprint, not an unsafe average.
inline llama_memory_breakdown_data common_fit_physical_memory(
        const std::vector<llama_memory_breakdown_data> & physical) {
    llama_memory_breakdown_data peak = {};
    for (const auto & mb : physical) {
        peak.model = std::max(peak.model, mb.model);
        peak.context = std::max(peak.context, mb.context);
        peak.compute = std::max(peak.compute, mb.compute);
        peak.context_fixed = std::max(peak.context_fixed, mb.context_fixed);
        peak.context_vbr_managed = std::max(peak.context_vbr_managed, mb.context_vbr_managed);
    }
    const size_t count = physical.size();
    peak.model *= count;
    peak.context *= count;
    peak.compute *= count;
    peak.context_fixed *= count;
    peak.context_vbr_managed *= count;
    return peak;
}
