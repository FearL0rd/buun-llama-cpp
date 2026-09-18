#include "../src/models/models.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <cstdio>
#include <vector>

static void check(ggml_backend_buffer_type_t buft, bool own_device, bool no_alloc, bool tied, bool host_weights = false) {
    llama_model_dflash target(llama_model_default_params());
    ggml_context_ptr ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    target.tok_embd = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 128, 128);
    target.output = tied ? target.tok_embd : ggml_new_tensor_2d(ctx.get(), ggml_exl3_type(4, 2), 128, 128);
    target.output_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 128);
    target.output_in_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 128);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
    GGML_ASSERT(buffer);
    uint8_t value = 37;
    for (auto * t : { target.tok_embd, target.output, target.output_s, target.output_in_s }) {
        std::vector<uint8_t> data(ggml_nbytes(t), value++);
        ggml_backend_tensor_set(t, data.data(), 0, data.size());
    }
    target.adopt_buffer(std::move(ctx), ggml_backend_buffer_ptr(buffer));
    if (host_weights) {
        // Only the auxiliaries are foreign: sharing the weight alone must not
        // take the early return and leave the transforms on the wrong device.
        ggml_context_ptr host(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
        target.tok_embd = ggml_dup_tensor(host.get(), target.tok_embd);
        target.output = tied ? target.tok_embd : ggml_dup_tensor(host.get(), target.output);
        auto * host_buffer = ggml_backend_alloc_ctx_tensors_from_buft(host.get(), ggml_backend_cpu_buffer_type());
        GGML_ASSERT(host_buffer);
        ggml_backend_buffer_clear(host_buffer, 42);
        target.adopt_buffer(std::move(host), ggml_backend_buffer_ptr(host_buffer));
    }

    llama_model_dflash draft(llama_model_default_params());
    draft.hparams.no_alloc = no_alloc;
    if (own_device) {
        draft.devices.push_back({false, ggml_backend_buft_get_device(buft)});
    }
    llama_model_share_tensors(&draft, &target);
    const ggml_tensor * sources[] = {target.tok_embd, target.output, target.output_s, target.output_in_s};
    const ggml_tensor * dests[] = {draft.tok_embd, draft.output, draft.output_s, draft.output_in_s};
    for (int i = 0; i < 4; ++i) {
        const bool copied = !own_device && !ggml_backend_buffer_is_host(sources[i]->buffer);
        GGML_ASSERT(dests[i]);
        GGML_ASSERT((dests[i] != sources[i]) == copied);
        GGML_ASSERT(dests[i]->type == sources[i]->type);
        GGML_ASSERT(ggml_are_same_shape(dests[i], sources[i]));
        if (copied && no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_size(dests[i]->buffer) == 0);
        } else {
            std::vector<uint8_t> data(ggml_nbytes(dests[i]));
            std::vector<uint8_t> expected(ggml_nbytes(sources[i]));
            ggml_backend_tensor_get(dests[i], data.data(), 0, data.size());
            ggml_backend_tensor_get(sources[i], expected.data(), 0, expected.size());
            GGML_ASSERT(data == expected);
        }
    }
    GGML_ASSERT((draft.output == draft.tok_embd) == tied);
    // A materialized head or fully borrowed same-device bundle is stable on
    // repeat setup. Auxiliary-only copies are not covered by this assertion.
    if (draft.output != target.output || own_device) {
        llama_model_share_tensors(&draft, &target);
        GGML_ASSERT(draft.output == dests[1] && draft.output_s == dests[2] && draft.output_in_s == dests[3]);
    }

    // Sharing just the embedding must preserve a self-contained output bundle.
    llama_model_dflash independent(llama_model_default_params());
    independent.output = target.output_s;
    independent.output_s = target.output_in_s;
    independent.output_in_s = target.output;
    llama_model_share_tensors(&independent, &target);
    GGML_ASSERT(independent.output == target.output_s);
    GGML_ASSERT(independent.output_s == target.output_in_s);
    GGML_ASSERT(independent.output_in_s == target.output);

    // A plain head must not inherit stale scale fields from the draft shell.
    target.output_s = nullptr;
    target.output_in_s = nullptr;
    llama_model_dflash plain(llama_model_default_params());
    plain.output_s = independent.output_s;
    plain.output_in_s = independent.output_in_s;
    llama_model_share_tensors(&plain, &target);
    GGML_ASSERT(plain.output_s == nullptr && plain.output_in_s == nullptr);
}

int main() {
    ggml_backend_load_all();
    check(ggml_backend_cpu_buffer_type(), false, false, false);
    check(ggml_backend_cpu_buffer_type(), false, true, true);
    auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu) {
        for (bool own_device : {false, true}) {
            for (bool no_alloc : {false, true}) {
                for (bool tied : {false, true}) {
                    check(ggml_backend_dev_buffer_type(gpu), own_device, no_alloc, tied);
                    check(ggml_backend_dev_buffer_type(gpu), own_device, no_alloc, tied, true);
                }
            }
        }
    } else {
        puts("GPU placement cases skipped (no GPU backend)");
    }
    puts("draft shared tensor tests passed");
}
