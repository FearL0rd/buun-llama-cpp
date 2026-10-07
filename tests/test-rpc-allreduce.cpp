// Run with two loopback rpc-server endpoints. The large case distinguishes
// full-precision replicated sums from rounding only the remote rank to BF16.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    if (argc == 3 && std::strcmp(argv[1], "--server") == 0) {
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(cpu);
        ggml_backend_rpc_start_server(argv[2], nullptr, 1, 1, &cpu);
        return 0;
    }
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s endpoint0 endpoint1 | --server endpoint\n", argv[0]);
        return 1;
    }
    ggml_backend_ptr ranks[] = {
        ggml_backend_ptr(ggml_backend_rpc_init(argv[1], 0)),
        ggml_backend_ptr(ggml_backend_rpc_init(argv[2], 0)),
    };
    GGML_ASSERT(ranks[0] && ranks[1]);
    ggml_backend_t backends[] = {ranks[0].get(), ranks[1].get()};
    auto reg = ggml_backend_rpc_reg();
    auto init = reinterpret_cast<ggml_backend_comm_init_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
    auto reduce = reinterpret_cast<ggml_backend_comm_allreduce_tensor_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
    auto free_comm = reinterpret_cast<ggml_backend_comm_free_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
    GGML_ASSERT(init && reduce && free_comm);
    void * comm = init(backends, 2);
    GGML_ASSERT(comm);
    bool ok = true;
    for (int count : {32767, 32768, 32769, 65536}) {
        ggml_context_ptr contexts[] = {
            ggml_context_ptr(ggml_init({ggml_tensor_overhead(), nullptr, true})),
            ggml_context_ptr(ggml_init({ggml_tensor_overhead(), nullptr, true})),
        };
        ggml_tensor * tensors[] = {
            ggml_new_tensor_1d(contexts[0].get(), GGML_TYPE_F32, count),
            ggml_new_tensor_1d(contexts[1].get(), GGML_TYPE_F32, count),
        };
        ggml_backend_buffer_ptr buffers[] = {
            ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors(contexts[0].get(), backends[0])),
            ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors(contexts[1].get(), backends[1])),
        };
        GGML_ASSERT(buffers[0] && buffers[1]);
        std::vector<float> expected(count), input(count), output(count);
        for (int rank = 0; rank < 2; ++rank) {
            tensors[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            for (int i = 0; i < count; ++i) {
                input[i] = rank == 0 ? 1.0009765625f + (i % 13)*0.0001220703125f : -0.998046875f;
                expected[i] += input[i];
            }
            ggml_backend_tensor_set(tensors[rank], input.data(), 0, count*sizeof(float));
        }
        GGML_ASSERT(reduce(comm, tensors));
        for (int rank = 0; rank < 2; ++rank) {
            ggml_backend_synchronize(backends[rank]);
            ggml_backend_tensor_get(tensors[rank], output.data(), 0, count*sizeof(float));
            ok = ok && output == expected;
        }
        std::printf("count=%d exact replicated F32 sum: %s\n", count, ok ? "PASS" : "FAIL");
    }
    free_comm(comm);
    return ok ? 0 : 1;
}
