// Entropy-bound canvas generation for llama-server (see server-diffusion.h). Adapted from the
// upstream runner in PR #24423: batch construction uses the caller's sequence id and positions,
// the unified forward flags only the canvas rows (logits are packed, so row offset is 0), and
// the block-autoregressive commit loop from the reference cli lives here.
#include "server-diffusion.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// Denoise one canvas: [prefix | canvas] forward per step, entropy-bound acceptance, renoise the
// rest, adaptive stop. Writes the final argmax canvas into canvas_out (canvas_length tokens).
// Returns false on a decode failure.
static bool server_diffusion_denoise(
        llama_context                    * ctx,
        llama_model                      * model,
        const llama_token                * prefix,
        int32_t                            n_prefix,
        llama_seq_id                       seq_id,
        const server_diffusion_eb_params & eb,
        int32_t                            canvas_length,
        llama_token                      * canvas_out) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t      n_vocab  = llama_vocab_n_tokens(vocab);
    const int32_t      C        = canvas_length;
    const int32_t      S        = std::max(1, eb.max_denoising_steps);
    const int32_t      max_length = n_prefix + C;

    std::mt19937                           rng(eb.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<llama_token> current_canvas(C);
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);                  // random init (not mask)
    }

    std::vector<float>       sc_buffer((size_t) C * n_vocab, 0.0f); // host SC (gpu SC keeps them on-device)
    std::vector<llama_token> argmax_canvas(C, 0);
    std::vector<llama_token> prev_argmax(C, -1);
    std::vector<float>       entropy(C);
    std::vector<llama_token> denoiser(C);
    std::vector<int32_t>     order(C);
    std::vector<float>       u(C);
    std::vector<llama_token> renoise(C);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    llama_batch batch = llama_batch_init(std::max(max_length, (int32_t) llama_n_batch(ctx)), 0, 1);

    const bool use_kv = eb.kv_cache;
    if (use_kv) {
        // Chunked causal PREFILL: feed the prompt in ubatch-sized chunks, each writing its K/V
        // and attending causally over the prefix, so the buffer tracks the chunk, not the prompt.
        const int32_t U = std::max(1, (int32_t) llama_n_ubatch(ctx));
        for (int32_t s = 0; s < n_prefix; s += U) {
            const int32_t u = std::min(U, n_prefix - s);
            llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
            llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, n_prefix, /*off=*/s);
            batch.n_tokens = u;
            for (int32_t i = 0; i < u; i++) {
                batch.token[i]     = prefix[s + i];
                batch.pos[i]       = s + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = seq_id;
                // PREFILL logits are unused; flag one row per chunk so n_outputs > 0.
                batch.logits[i]    = (i == u - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: PREFILL chunk [%d,%d) decode failed\n", __func__, s, s + u);
                llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0, 0);
                llama_batch_free(batch);
                return false;
            }
        }
    }

    float   prev_temp_inv = 1.0f;
    int     held          = 0;
    bool    finished      = false;

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;
        const float   t        = eb.t_min + (eb.t_max - eb.t_min) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        if (use_kv) {
            llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, n_prefix, 0);
            batch.n_tokens = C;
            for (int32_t i = 0; i < C; i++) {
                batch.token[i]     = current_canvas[i];
                batch.pos[i]       = n_prefix + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = seq_id;
                batch.logits[i]    = 1;
            }
        } else {
            batch.n_tokens = max_length;
            for (int32_t i = 0; i < max_length; i++) {
                batch.token[i]     = (i < n_prefix) ? prefix[i] : current_canvas[i - n_prefix];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = seq_id;
                // only canvas rows need logits: the output buffer packs the flagged rows in
                // order, so row j below is canvas position j (no n_prefix row offset)
                batch.logits[i]    = (i >= n_prefix) ? 1 : 0;
            }
        }

        // self-conditioning = softmax(previous step's logits / previous t); gated off on the
        // first step. Host SC consumes the stashed canvas rows; device SC reads them on-device.
        llama_diffusion_set_sc(model, sc_buffer.data(), step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode at step %d\n", __func__, step_idx);
            break;
        }

        const float * logits = llama_get_logits(ctx);   // packed canvas rows: [C, n_vocab]

        // pre-draw the step's randomness single-threaded so the output is seed-reproducible
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        // per position: argmax, entropy of softmax(raw/t), and a multinomial sample; stash raw row for SC
        auto worker = [&](int32_t p0, int32_t p1) {
            for (int32_t pos = p0; pos < p1; pos++) {
                const float * row = logits + (size_t) pos * n_vocab;
                float m = -INFINITY; int32_t amax = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float z = row[v] * temp_inv;
                    if (z > m) { m = z; amax = v; }
                }
                float Z = 0.0f;
                for (int32_t v = 0; v < n_vocab; v++) {
                    Z += expf(row[v] * temp_inv - m);
                }
                const float target = u[pos] * Z;
                float   cum = 0.0f, H = 0.0f;
                int32_t sampled = n_vocab - 1; bool picked = false;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float e = expf(row[v] * temp_inv - m);
                    const float p = e / Z;
                    if (p > 0.0f) { H -= p * logf(p); }
                    cum += e;
                    if (!picked && cum >= target) { sampled = v; picked = true; }
                }
                entropy[pos]       = H;
                argmax_canvas[pos] = amax;
                denoiser[pos]      = sampled;
                std::memcpy(sc_buffer.data() + (size_t) pos * n_vocab, row, (size_t) n_vocab * sizeof(float));
            }
        };

        {
            std::vector<std::thread> pool;
            const int32_t chunk = (C + (int32_t) nth - 1) / (int32_t) nth;
            for (unsigned ti = 0; ti < nth; ti++) {
                const int32_t p0 = (int32_t) ti * chunk;
                const int32_t p1 = std::min(p0 + chunk, C);
                if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
            }
            for (auto & th : pool) { th.join(); }
        }

        if (step_idx == 0) {
            LOG_INF("%s: step 0 canvas[0] argmax=%d entropy=%.4f (decode rows=%d)\n",
                    __func__, argmax_canvas[0], entropy[0], C);
        }

        // accept the lowest-entropy positions within the MI bound (sum of strictly-earlier entropies <= bound)
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= eb.entropy_bound) { accepted[pos] = 1; }
        }

        // renoise: accepted -> sampled token, rest -> fresh random; the output canvas is the argmax
        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            current_canvas[pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
            canvas_out[pos]     = argmax_canvas[pos];
            entropy_sum        += entropy[pos];
        }

        // adaptive stop: argmax stable for stability_threshold steps AND confident (low mean entropy)
        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < eb.confidence_threshold;
        if (held >= eb.stability_threshold && confident) { finished = true; }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;
    }

    if (use_kv) {
        llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0, 0);
    }
    llama_batch_free(batch);
    return true;
}

std::vector<llama_token> server_diffusion_generate_canvas(
        llama_context                    * ctx,
        llama_model                      * model,
        const std::vector<llama_token>   & prefix_tokens,
        llama_seq_id                       seq_id,
        const server_diffusion_eb_params & eb,
        int32_t                            canvas_length,
        int32_t                            n_predict) {
    std::vector<llama_token> response;
    if (!ctx || !model || canvas_length <= 0 || n_predict <= 0 ||
            (int32_t) prefix_tokens.size() + canvas_length > (int32_t) llama_n_ctx_seq(ctx)) {
        return response;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t      n_batch  = (int32_t) llama_n_batch(ctx);
    const int32_t      n_ubatch = (int32_t) llama_n_ubatch(ctx);

    // unified forward (default): the whole [prefix | canvas] in one batch. The prefix-KV path is
    // only used when explicitly requested (single-GPU) and the canvas fits in one ubatch.
    const bool use_kv = eb.kv_cache && n_ubatch >= canvas_length;
    if (!use_kv && n_batch < (int32_t) prefix_tokens.size() + canvas_length) {
        LOG_ERR("%s: canvas batch too large: needs n_batch >= n_input(%d) + canvas(%d) "
                "(have n_batch=%d, n_ubatch=%d); raise --batch-size or use "
                "--diffusion-kv-cache on\n", __func__,
                (int) prefix_tokens.size(), canvas_length, n_batch, n_ubatch);
        return response;
    }

    // Trim a denoised canvas: cut at the first EOG token or (checkpoints often emit no stop
    // token) at the onset of a repetition loop (a token recurring at stride 1-2 for >= 6 steps).
    struct trim_result { size_t cut; bool eog; };
    auto trim_canvas = [&](const llama_token * canvas, size_t n) -> trim_result {
        for (size_t i = 0; i < n; i++) {
            if (llama_vocab_is_eog(vocab, canvas[i])) {
                return { i, true };
            }
        }
        for (size_t i = 0; i + 1 < n; i++) {
            bool loop = false;
            for (size_t stride = 1; stride <= 2 && !loop; stride++) {
                size_t reps = 0;
                for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) {
                    reps++;
                }
                loop = reps >= 6;
            }
            if (loop) {
                return { i, false };
            }
        }
        return { n, false };
    };

    llama_set_causal_attn(ctx, false);

    std::vector<llama_token> prefix(prefix_tokens);
    std::vector<llama_token> canvas(canvas_length);
    server_diffusion_eb_params eb_one = eb;
    eb_one.kv_cache                   = use_kv;
    int32_t n_budget                  = n_predict;

    while (n_budget > 0) {
        const int32_t prefix_len = (int32_t) prefix.size();
        if (prefix_len + canvas_length >= (int32_t) llama_n_ctx_seq(ctx)) {
            break;  // out of context room: keep what we have
        }
        if (!use_kv && prefix_len + canvas_length > n_batch) {
            break;  // out of batch room: keep what we have
        }

        eb_one.seed += 1;  // fresh randomness per block, reproducible from the request seed
        if (!server_diffusion_denoise(ctx, model, prefix.data(), prefix_len, seq_id,
                    eb_one, canvas_length, canvas.data())) {
            break;
        }

        const trim_result trimmed = trim_canvas(canvas.data(), (size_t) canvas_length);
        const int32_t take = (int32_t) std::min((size_t) n_budget, trimmed.cut);
        response.insert(response.end(), canvas.begin(), canvas.begin() + take);
        n_budget -= take;

        if (trimmed.eog && take == (int32_t) trimmed.cut) {
            // report the EOG token itself so the caller's delivery loop finishes with a clean EOS stop
            response.push_back(canvas[trimmed.cut]);
            break;
        }
        if (take < (int32_t) trimmed.cut || n_budget == 0) {
            break;  // n_predict budget exhausted
        }
        if (trimmed.cut < (size_t) canvas_length) {
            break;  // repetition loop: answer complete
        }
        prefix.insert(prefix.end(), canvas.begin(), canvas.begin() + trimmed.cut);  // commit the block
    }

    llama_set_causal_attn(ctx, true);
    return response;
}
