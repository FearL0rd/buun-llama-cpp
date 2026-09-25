// Entropy-bound canvas generation for DiffusionGemma-style models, served inside llama-server.
// Port of PR #24423 (diffusion_generate_entropy_bound): the canvas is random-initialized, each
// step accepts the lowest-entropy positions within a mutual-information bound under a linear
// temperature schedule, renoises the rest, and stops adaptively once the argmax canvas is stable
// and confident. Runs block-autoregressively: a denoised block is committed to the prefix and a
// fresh canvas denoised, until an EOG token, a repetition loop, or the n_predict budget.
#pragma once

#include "llama.h"

#include <vector>

struct server_diffusion_eb_params {
    int32_t max_denoising_steps  = 48;
    float   t_min                = 0.4f;   // temperature at the last step
    float   t_max                = 0.8f;   // temperature at the first step
    float   entropy_bound        = 0.1f;   // accept lowest-entropy tokens within this MI bound
    int32_t stability_threshold  = 1;      // steps the argmax canvas must hold to count as stable
    float   confidence_threshold = 0.005f; // stop once mean canvas entropy drops below this
    int32_t seed                 = 0;
    bool    kv_cache             = false;  // prefix-KV prefill/decode path (single-device store)
};

// Denoise up to n_predict response tokens for one request. Batches use seq_id at positions
// 0.. so the caller's sequence bookkeeping applies; rows this call writes are transient (the
// caller clears them from the slot sequence afterwards). Returns the trimmed response tokens
// (prompt prefix not included); an empty result means generation failed (reason logged).
std::vector<llama_token> server_diffusion_generate_canvas(
        llama_context                    * ctx,
        llama_model                      * model,
        const std::vector<llama_token>   & prefix_tokens,
        llama_seq_id                       seq_id,
        const server_diffusion_eb_params & eb,
        int32_t                            canvas_length,
        int32_t                            n_predict);
