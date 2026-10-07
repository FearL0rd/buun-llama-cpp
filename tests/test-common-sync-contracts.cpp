#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "../src/llama-batch.h"

#include <algorithm>
#include <cstdio>
#include <limits>

// Model-free batch checks; the sampler tests use the repository's vocab-only
// llama-spm fixture (no inference weights or accelerator needed).
static void test_staged_batch() {
    common_batch batch;
    batch.batch.reset(new llama_batch_ext(2, 4, 8, 16, nullptr, 32, 4));
    batch.n_pos = 4;
    batch.seq_id_limit = 16;

    for (int i = 0; i < 5; ++i) {
        GGML_ASSERT(batch.add(i + 1, i, 9, i == 4) == i);
    }
    GGML_ASSERT(batch.size() == 5); // staging is not limited by render capacity
    GGML_ASSERT(batch.add_seq(3, 12));
    GGML_ASSERT(batch.add_seq(3, 12)); // duplicate membership stays unique
    GGML_ASSERT(batch.tokens[3].seq_ids.size() == 2);
    GGML_ASSERT(!batch.add_seq(3, 16));
    GGML_ASSERT(!batch.add_seq(-1, 1));
    GGML_ASSERT(batch.set_output(3, true));
    batch.tokens[3].decision_order = 2;

    auto * ext = batch.get_sub_batch(3, 2);
    GGML_ASSERT(ext->tokens.size() == 2);
    GGML_ASSERT(ext->tokens[0].id == 4 && ext->tokens[0].pos[0] == 3);
    GGML_ASSERT(ext->tokens[0].seq_ids.count(9) && ext->tokens[0].seq_ids.count(12));
    GGML_ASSERT(ext->tokens[0].output && ext->tokens[0].decision_order == 2);
    ext = batch.get_sub_batch(0, 2);
    GGML_ASSERT(ext->tokens.size() == 2 && ext->tokens[0].id == 1);
    GGML_ASSERT(ext->tokens[0].decision_order == 0);
    GGML_ASSERT(ext->tokens[0].seq_ids.size() == 1);

    batch.clear();
    float values[] = { 1, 2, 3, 4 };
    const llama_pos pos[] = { 11, 12, 13, 14 };
    const int idx = batch.add_embd({ values, 1, 4 }, pos, 9, false);
    GGML_ASSERT(batch.add_seq(idx, 12));
    GGML_ASSERT(!batch.set_embd(idx, { values, 1, 4 }));
    ext = batch.get();
    for (int axis = 0; axis < 4; ++axis) {
        GGML_ASSERT(ext->tokens[0].pos[axis] == pos[axis]);
    }
    GGML_ASSERT(ext->embd[2] == 3);
    values[2] = 7; // non-owning staged row, copied only during rendering
    ext = batch.get();
    GGML_ASSERT(ext->embd.size() == 4 && ext->embd[2] == 7);
    GGML_ASSERT(batch.set_output(0, true));
    GGML_ASSERT(batch.get()->tokens[0].output);

    batch.clear();
    const int token_idx = batch.add(4, 10, 9, true);
    GGML_ASSERT(batch.set_embd(token_idx, { values, 1, 4 }));
    ext = batch.get();
    GGML_ASSERT(ext->tokens[0].id == 4 && ext->tokens[0].has_embd);
    GGML_ASSERT(ext->embd.size() == 4 && ext->embd[2] == 7);
    batch.clear();
    GGML_ASSERT(batch.get()->tokens.empty());
}

static void test_proposal_prefix() {
    common_speculative_proposal p;
    GGML_ASSERT(p.matching_prefix_size(0, { 1, 2 }) == 0);
    p.seq_id = 3;
    p.selected = { 1, 2, 3 };
    p.q_covered_tokens = 3;
    p.exact_q = true;
    GGML_ASSERT(p.matching_prefix_size(3, { 1, 2 }) == 2); // truncated draft
    GGML_ASSERT(p.matching_prefix_size(3, { 1, 2, 3, 9 }) == 3); // CopySpec suffix
    GGML_ASSERT(p.matching_prefix_size(4, { 1, 2 }) == 0);
    GGML_ASSERT(p.matching_prefix_size(3, { 1, 9 }) == 0);
    p.clear();
    GGML_ASSERT(p.matching_prefix_size(3, { 1, 2 }) == 0);
}

static void test_sampler(const char * vocab_path) {
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model_ptr model(llama_model_load_from_file(vocab_path, mparams));
    GGML_ASSERT(model);
    const auto * vocab = llama_model_get_vocab(model.get());
    const llama_token eos = llama_vocab_eos(vocab);
    GGML_ASSERT(eos >= 0 && llama_vocab_is_eog(vocab, eos));
    llama_token plain = 0;
    while (llama_vocab_is_eog(vocab, plain)) {
        ++plain;
    }

    common_params_sampling params;
    params.seed = 19;
    params.temp = 0.8f;
    params.top_k = 8;
    params.no_perf = true;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    common_sampler_ptr source(common_sampler_init(model.get(), params));

    const llama_tokens interior = common_sampler_accept_draft(source.get(),
            { plain, eos, plain, plain }, { plain, eos, plain });
    GGML_ASSERT(interior == llama_tokens({ plain, eos }));
    common_sampler_reset(source.get());
    const llama_tokens terminal = common_sampler_accept_draft(source.get(),
            { plain, eos, plain }, { plain, eos });
    GGML_ASSERT(terminal == llama_tokens({ plain, eos, plain }));

    std::vector<float> logits(llama_vocab_n_tokens(vocab), -10.0f);
    for (int i = 0; i < 8; ++i) {
        logits[i] = 1.0f - 0.1f * i;
    }
    common_sampler_sample_from_logits(source.get(), logits.data(), logits.size());
    common_sampler_ptr clone(common_sampler_clone(source.get()));
    auto * original_candidates = common_sampler_get_candidates(source.get(), false);
    auto * cloned_candidates = common_sampler_get_candidates(clone.get(), false);
    GGML_ASSERT(original_candidates->data != cloned_candidates->data);
    GGML_ASSERT(original_candidates->size == cloned_candidates->size);
    const float clone_logit = cloned_candidates->data[0].logit;
    original_candidates->data[0].logit += 7;
    GGML_ASSERT(cloned_candidates->data[0].logit == clone_logit);

    common_sampler_ptr copied(common_sampler_init(model.get(), params));
    common_sampler_copy(source.get(), copied.get());
    GGML_ASSERT(common_sampler_get_candidates(copied.get(), false)->data != original_candidates->data);
    for (int i = 0; i < 16; ++i) {
        const auto a = common_sampler_sample_from_logits(source.get(), logits.data(), logits.size());
        const auto b = common_sampler_sample_from_logits(copied.get(), logits.data(), logits.size());
        GGML_ASSERT(a == b);
        common_sampler_accept(source.get(), a, true);
        common_sampler_accept(copied.get(), b, true);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s models/ggml-vocab-llama-spm.gguf\n", argv[0]);
        return 1;
    }
    test_staged_batch();
    test_proposal_prefix();
    test_sampler(argv[1]);
    std::puts("common sync contracts: PASS");
}
