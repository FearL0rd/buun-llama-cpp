// A sequence state import into a dynamic-VBR cache grows the cache's physical backing itself.
// With the device exhausted it has to fail like any other rejected import: nothing installed,
// the context still usable, and the same import accepted once memory is back.
//
// Run with a model whose KV dynamic VBR can manage:
//   test-vbr-import-exhausted -m model.gguf -ctk vbr -fa on -ngl 99
// Exits 77 where there is no GPU or the controller did not arm.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

static constexpr int    SKIP     = 77;
static constexpr size_t MIB      = 1024*1024;
static constexpr size_t LEAVE    = 64*MIB;
static constexpr int    N_PROMPT = 4096;

static bool decode_range(llama_context * ctx, llama_batch & batch, int p0, int p1) {
    const int n_batch = llama_n_batch(ctx);
    for (int i = p0; i < p1; i += n_batch) {
        common_batch_clear(batch);
        for (int j = i; j < std::min(i + n_batch, p1); ++j) {
            common_batch_add(batch, 1 + j % 1000, j, {0}, j + 1 == p1);
        }
        if (llama_decode(ctx, batch)) {
            return false;
        }
    }
    return true;
}

static size_t free_bytes(ggml_backend_dev_t dev) {
    size_t free, total;
    ggml_backend_dev_memory(dev, &free, &total);
    return free;
}

// take the device's free memory down to LEAVE: whole pieces while they fit, then halves
static std::vector<ggml_backend_buffer_t> exhaust(ggml_backend_dev_t dev) {
    std::vector<ggml_backend_buffer_t> held;
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    for (size_t piece = 256*MIB; piece >= 2*MIB;) {
        ggml_backend_buffer_t buf = free_bytes(dev) < LEAVE + piece ? nullptr : ggml_backend_buft_alloc_buffer(buft, piece);
        if (buf == nullptr) {
            piece /= 2;
            continue;
        }
        held.push_back(buf);
    }
    return held;
}

// `import` installs the saved prompt into the empty sequence 0 and returns the bytes it read
static bool check_import(const char * name, llama_context * ctx, llama_batch & batch, ggml_backend_dev_t dev,
        size_t n_expected, const std::function<size_t()> & import) {
    llama_memory_t mem = llama_get_memory(ctx);

    // a short sequence in the prompt's place lets the mapped watermark fall back
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (!decode_range(ctx, batch, 0, 8)) {
        fprintf(stderr, "%s : failed to decode the short sequence\n", name);
        return false;
    }
    llama_memory_seq_rm(mem, 0, -1, -1);

    const size_t free_before = free_bytes(dev);
    std::vector<ggml_backend_buffer_t> held = exhaust(dev);
    fprintf(stderr, "%s : free device memory %zu -> %zu MiB, state is %zu MiB\n", name,
            free_before/MIB, free_bytes(dev)/MIB, n_expected/MIB);

    const size_t n_starved = import();

    for (ggml_backend_buffer_t buf : held) {
        ggml_backend_buffer_free(buf);
    }
    if (n_starved != 0) {
        fprintf(stderr, "%s : FAILED - the import was not starved (returned %zu), nothing was tested\n", name, n_starved);
        return false;
    }
    if (llama_memory_seq_pos_max(mem, 0) != -1) {
        fprintf(stderr, "%s : FAILED - the refused import left cells behind\n", name);
        return false;
    }
    if (import() != n_expected) {
        fprintf(stderr, "%s : FAILED - the import was refused with memory available again\n", name);
        return false;
    }
    if (llama_memory_seq_pos_max(mem, 0) != N_PROMPT - 1 || !decode_range(ctx, batch, N_PROMPT, N_PROMPT + 1)) {
        fprintf(stderr, "%s : FAILED - the restored sequence does not continue\n", name);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.kv_unified = true;
    params.n_parallel = 1;
    params.n_ctx      = 2*N_PROMPT;
    params.n_batch    = 512;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        fprintf(stderr, "%s : no GPU device, skipping\n", __func__);
        return SKIP;
    }

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

    if (!decode_range(ctx, batch, 0, N_PROMPT)) {
        fprintf(stderr, "%s : failed to decode the prompt\n", __func__);
        return 1;
    }
    if (llama_memory_vbr_state(llama_get_memory(ctx), 0, 0).checkpoint_epoch == 0) {
        fprintf(stderr, "%s : no dynamic-VBR controller on this cache, skipping\n", __func__);
        return SKIP;
    }

    std::vector<uint8_t> whole(llama_state_seq_get_size(ctx, 0));
    std::vector<uint8_t> range(llama_state_seq_get_size_range(ctx, 0, 0, N_PROMPT));
    if (llama_state_seq_get_data(ctx, whole.data(), whole.size(), 0) != whole.size() ||
        llama_state_seq_get_data_range(ctx, range.data(), range.size(), 0, 0, N_PROMPT) != range.size() || range.empty()) {
        fprintf(stderr, "%s : failed to save the sequence\n", __func__);
        return 1;
    }

    const bool ok =
        check_import("whole", ctx, batch, dev, whole.size(), [&]() {
            return llama_state_seq_set_data(ctx, whole.data(), whole.size(), 0);
        }) &&
        check_import("range", ctx, batch, dev, range.size(), [&]() {
            return llama_state_seq_append_data(ctx, range.data(), range.size(), 0, 0, N_PROMPT, N_PROMPT);
        });

    llama_batch_free(batch);

    if (!ok) {
        return 1;
    }
    fprintf(stderr, "%s : SUCCESS - exhausted imports are refused and the context keeps working\n", __func__);
    return 0;
}
