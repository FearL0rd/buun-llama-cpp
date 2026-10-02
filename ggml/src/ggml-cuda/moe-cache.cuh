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

// Device-routed MUL_MAT_ID: table[e] points at expert e's bytes (VRAM cache slot
// or device-accessible host memory); kernels log the routed experts to log.
struct ggml_moe_cache_route_table {
    const void * const * table = nullptr;
    int32_t * log = nullptr;
};

// False when the host expert tensor at host_base is not device-routed.
bool ggml_moe_cache_route_find(const void * host_base, ggml_moe_cache_route_table & route);
#endif
