#pragma once

#include "common.cuh"

// Deferred normalization must retain the graph's formula, not just its epsilon.
// Legacy L2 clamps the norm; GDN uses RMS(eps / width) followed by a scale.
struct ggml_cuda_gdn_norm {
    float eps = -1.0f;
    float post_scale = 1.0f;
    bool rms = false;

    __device__ float inverse(float sum, int width) const {
        return rms ? rsqrtf(sum / width + eps) : rsqrtf(fmaxf(sum, eps * eps));
    }

    __device__ float apply(float value, float inverse) const {
        return (inverse * value) * post_scale;
    }
};

static inline const ggml_tensor * ggml_cuda_gdn_norm_input(
        const ggml_tensor * output, ggml_cuda_gdn_norm & norm) {
    norm = {};
    if (!output) {
        return nullptr;
    }
    const ggml_tensor * base = output;
    if (output->op == GGML_OP_SCALE && output->src[0] && output->src[0]->op == GGML_OP_RMS_NORM) {
        if (ggml_get_op_params_f32(output, 1) != 0.0f) {
            return nullptr;
        }
        base = output->src[0];
        norm.rms = true;
        norm.post_scale = ggml_get_op_params_f32(output, 0);
        if (!(norm.post_scale > 0.0f) || !std::isfinite(norm.post_scale)) {
            return nullptr;
        }
    } else if (output->op != GGML_OP_L2_NORM) {
        return nullptr;
    }
    norm.eps = ggml_get_op_params_f32(base, 0);
    return norm.eps >= 0.0f && std::isfinite(norm.eps) ? base->src[0] : nullptr;
}
