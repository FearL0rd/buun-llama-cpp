// Byte ranges of a split meta tensor that cut its rows, as a state writer streaming in fixed-size
// pieces asks for them: every get must return the bytes a whole-tensor copy holds, and every set
// must change exactly its own bytes.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

static size_t n_devs = 0;

static ggml_backend_meta_split_state split_axis_0(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state state = {};
    state.axis = GGML_BACKEND_SPLIT_AXIS_0;
    // "fused": two segments along the row, each split across the devices
    const std::vector<int64_t> segments = strcmp(tensor->name, "fused") == 0 ?
        std::vector<int64_t>{ 64, 32 } : std::vector<int64_t>{ tensor->ne[0] };
    for (size_t s = 0; s < segments.size(); ++s) {
        for (size_t j = 0; j < n_devs; ++j) {
            state.ne[s*n_devs + j] = segments[s] / int64_t(n_devs);
        }
        state.nr[s] = 1;
    }
    state.n_segments = uint32_t(segments.size());
    return state;
}

int main() {
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count() && devices.size() < 2; ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devices.push_back(dev);
        }
    }
    if (devices.empty()) {
        fprintf(stderr, "SKIP: requires a GPU\n");
        return 77;
    }
    n_devs = devices.size();

    auto             dev = ggml_backend_meta_device(devices.data(), devices.size(), split_axis_0, nullptr);
    auto             buft = ggml_backend_dev_buffer_type(dev);
    ggml_context_ptr ctx(ggml_init({ ggml_tensor_overhead() * 4, nullptr, true }));
    ggml_tensor *    tensors[2] = {
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 96, 24),
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 96, 24),
    };
    ggml_set_name(tensors[0], "single");
    ggml_set_name(tensors[1], "fused");
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    GGML_ASSERT(buf);

    std::mt19937 rng(42);
    const auto   random_bytes = [&](size_t n) {
        std::vector<uint8_t> bytes(n);
        for (auto & b : bytes) {
            b = uint8_t(rng());
        }
        return bytes;
    };

    bool passed = true;
    for (ggml_tensor * tensor : tensors) {
        const size_t         nbytes = ggml_nbytes(tensor);
        const size_t         row    = tensor->nb[1];
        std::vector<uint8_t> mirror = random_bytes(nbytes);
        ggml_backend_tensor_set(tensor, mirror.data(), 0, nbytes);

        const std::pair<size_t, size_t> ranges[] = {
            { 0, 1 }, { 7, 1001 }, { row - 3, 6 }, { row, row }, { 5*row + 11, 3*row }, { nbytes - 13, 13 },
        };
        for (const auto & [offset, size] : ranges) {
            std::vector<uint8_t> got(size);
            ggml_backend_tensor_get(tensor, got.data(), offset, size);
            if (memcmp(got.data(), mirror.data() + offset, size) != 0) {
                fprintf(stderr, "FAIL: %s get [%zu, +%zu)\n", tensor->name, offset, size);
                passed = false;
            }

            const std::vector<uint8_t> put = random_bytes(size);
            ggml_backend_tensor_set(tensor, put.data(), offset, size);
            memcpy(mirror.data() + offset, put.data(), size);
            std::vector<uint8_t> whole(nbytes);
            ggml_backend_tensor_get(tensor, whole.data(), 0, nbytes);
            if (whole != mirror) {
                fprintf(stderr, "FAIL: %s set [%zu, +%zu)\n", tensor->name, offset, size);
                passed = false;
            }
        }
    }

    printf("%s: partial-row meta get/set on %zu device(s)\n", passed ? "OK" : "FAIL", n_devs);
    return passed ? 0 : 1;
}
