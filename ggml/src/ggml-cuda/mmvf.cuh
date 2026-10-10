#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#define MMVF_F16_MULTI_MAX        4 // Max. matrices sharing src1 in one fused launch.
#define MMVF_F16_MULTI_MAX_TOKENS 4

// mm is a MUL_MAT whose f16 weights can be batched with others that read src1.
bool ggml_cuda_mul_mat_vec_f16_multi_supported(const ggml_tensor * mm, const ggml_tensor * src1);

// Computes n_mats MUL_MATs that share src1 in one launch; sigmoid (optional) also receives the
// sigmoid of mms[sigmoid_mat], which must be contiguous like it.
void ggml_cuda_mul_mat_vec_f16_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mms, int n_mats,
        const ggml_tensor * sigmoid = nullptr, int sigmoid_mat = 0);

// mm is a small-batch f16 MUL_MAT the KDA gate launch can compute.
bool ggml_cuda_mul_mat_vec_f16_kda_gate_supported(const ggml_tensor * mm);

// KDA decay gate in one launch: g1 = SCALE(SIGMOID(scale0(MUL(ADD(mm, bias), a)))), with a broadcast
// over equal row groups and bias over tokens. mm2 (optional, same token count) is computed alongside.
void ggml_cuda_mul_mat_vec_f16_kda_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
        const ggml_tensor * bias, const ggml_tensor * a, const ggml_tensor * scale0, ggml_tensor * g1,
        ggml_tensor * mm2 = nullptr);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);

void ggml_cuda_op_qwen35_recurrent_gates_f16(
        ggml_backend_cuda_context & ctx, const ggml_tensor * alpha_weight, const ggml_tensor * beta_weight,
        const ggml_tensor * input, const ggml_tensor * dt, const ggml_tensor * a,
        ggml_tensor * gate, ggml_tensor * beta);

void ggml_cuda_op_qwen35_recurrent_gates(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * alpha_weight, const ggml_tensor * beta_weight,
    const ggml_tensor * input, const ggml_tensor * dt, const ggml_tensor * a,
    ggml_tensor * gate, ggml_tensor * beta);

void ggml_cuda_op_qwen35_recurrent_gate_epilogue(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * alpha, const ggml_tensor * beta_input,
        const ggml_tensor * dt, const ggml_tensor * a,
        ggml_tensor * gate, ggml_tensor * beta);

void ggml_cuda_op_qwen35_recurrent_gate_epilogue_combined(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * mixed,
        const ggml_tensor * dt, const ggml_tensor * a,
        ggml_tensor * gate, ggml_tensor * beta);
