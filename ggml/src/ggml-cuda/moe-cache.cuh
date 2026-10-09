#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Populates ggml_moe_cache (see ggml-backend-moe-cache.h).

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_moe_cache_register(const void * owner);

// Surrender the device's cache VRAM under allocator pressure; returns bytes freed.
size_t ggml_moe_cache_trim(int device);

#ifdef __cplusplus
}

#include <cstdint>

// Routed ids one routed launch logs: a larger op is not routed.
static constexpr int moe_cache_route_log_max = 512;

// Integer environment value in [min_value, max_value]; anything else is logged and ignored.
bool moe_cache_env_i64(const char * name, int64_t min_value, int64_t max_value, int64_t & value);

// Device-routed MUL_MAT_ID: table[e] points at expert e's bytes (VRAM cache slot
// or device-accessible host memory); kernels log the routed experts to log.
// EXL3 routes also carry VRAM copies of the op's host-resident src[3]/src[4] scales.
// Tiered routes stream from a mapped file: table[e] is null or a RAM tier copy until
// ggml_moe_tier_resolve runs before the launch (moe-tier.cuh).
struct ggml_moe_tier_route;
struct ggml_moe_cache_route_table {
    const void * const * table = nullptr;
    int32_t * log = nullptr;
    const void * scales[2] = {nullptr, nullptr};
    ggml_moe_tier_route * tier = nullptr;
};

// False when the host expert tensor at host_base is not device-routed.
bool ggml_moe_cache_route_find(const void * host_base, ggml_moe_cache_route_table & route);

// CUDA buffers this thread allocates inside the scope are VMM mappings that the
// cache can alias (cuMemRetainAllocationHandle). No-op without single-GPU VMM.
void ggml_cuda_compute_vmm_scope(bool enter);
#endif
