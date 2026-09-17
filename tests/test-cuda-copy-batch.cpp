#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

static void check(ggml_backend_t backend, int tokens, int copies, int channels, bool dependency) {
    constexpr int prefix = 3, sequences = 2, capacity = 4, head = 1;
    const int width = prefix + tokens, row = prefix * channels;
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, sequences);
    auto * states = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row, capacity * copies);
    auto * graph = ggml_new_graph_custom(ctx, 1024, false);
    std::vector<float> data(ggml_nelements(input)), expected(ggml_nelements(states), -123.0f);
    for (size_t i = 0; i < data.size(); ++i) data[i] = float(int(i % 1019) - 503) / 16;
    const uint32_t patterns[] = {0x7fc01234u, 0xff800000u, 0x80000000u};
    for (size_t i = 0; i < 3; ++i) std::memcpy(&data[i], &patterns[i], sizeof(float));
    for (int t = 1; t <= copies; ++t) {
        const int offset = std::max(0, tokens - copies + t), slot = copies - t;
        auto * src = ggml_view_3d(ctx, input, prefix, channels, sequences,
                input->nb[1], input->nb[2], offset * sizeof(float));
        auto * dst = ggml_view_2d(ctx, states, row, sequences, states->nb[1],
                (slot * capacity + head) * row * sizeof(float));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
        for (int s = 0; s < sequences; ++s)
            for (int c = 0; c < channels; ++c)
                for (int p = 0; p < prefix; ++p)
                    expected[(slot * capacity + head + s) * row + c * prefix + p] =
                        data[(s * channels + c) * width + offset + p];
    }
    if (dependency) {
        // Reads a preceding copy's destination; it cannot join that batch.
        auto * src = ggml_view_1d(ctx, states, row, head * row * sizeof(float));
        auto * dst = ggml_view_1d(ctx, states, row, 0);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
        std::copy_n(expected.begin() + head * row, row, expected.begin());
        // An identical source/destination must fall back without changing bytes.
        ggml_build_forward_expand(graph, ggml_cpy(ctx, dst, dst));
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
    const std::vector<float> initial(expected.size(), -123.0f);
    for (int iteration = 0; iteration < 3; ++iteration) {
        ggml_backend_tensor_set(states, initial.data(), 0, ggml_nbytes(states));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(states, actual.data(), 0, ggml_nbytes(states));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&actual[i], &expected[i], sizeof(float))) {
                std::fprintf(stderr, "mismatch tokens=%d copies=%d channels=%d dependency=%d at %zu: %g != %g\n",
                    tokens, copies, channels, dependency, i, actual[i], expected[i]);
                GGML_ABORT("copy batch mismatch");
            }
        }
    }
    std::printf("PASS tokens=%d copies=%d channels=%d dependency=%d\n", tokens, copies, channels, dependency);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto * backend = ggml_backend_cuda_init(0);
    if (!backend) return 77;
    for (int copies : {1,2,8,16,17}) {
        check(backend, 1, copies, 257, false);
        check(backend, copies, copies, 257, true);
    }
    check(backend, 8, 8, 10240, true);
    ggml_backend_free(backend);
}
