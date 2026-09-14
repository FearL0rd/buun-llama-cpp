#include "arg.h"
#include "common.h"
#include "llama-kv-cache-iswa.h"
#include "llama-sha256.h"
#include "llama-vbr-swa-window.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct window_budget {
    size_t limit = 16*1024*1024;
    size_t used = 0;
    size_t attempts = 0;
};

struct window_charge {
    std::shared_ptr<window_budget> budget;
    size_t bytes;
    window_charge(std::shared_ptr<window_budget> b, size_t n) : budget(std::move(b)), bytes(n) { budget->used += n; }
    ~window_charge() { budget->used -= bytes; }
};

static std::shared_ptr<void> reserve_window(void * context, size_t bytes) {
    auto budget = *static_cast<std::shared_ptr<window_budget> *>(context);
    ++budget->attempts;
    if (budget->used > budget->limit || bytes > budget->limit-budget->used) { return {}; }
    return std::make_shared<window_charge>(budget, bytes);
}

static bool cancel_transfer(void * context) noexcept {
    return --*static_cast<int *>(context) > 0;
}

struct llama_kv_cache_vbr_epoch_test {
    static void verify(const llama_kv_cache & cache, const vbr_swa_window_image & image) {
        for (const auto & unit : image.units()) {
            const auto & layer = cache.layers.at(unit.logical_unit/2);
            const auto * t = unit.logical_unit & 1 ? layer.v : layer.k;
            check(unit.model_layer == layer.il && unit.generation.current_type == t->type, "wrong unit identity");
            check(unit.row_bytes == t->nb[1] && unit.payload.size() == image.rows().size()*unit.row_bytes, "wrong payload shape");
            llama_sha256 hash;
            hash.update(unit.payload.data(), unit.payload.size());
            check(hash.finish() == unit.checksum, "payload checksum differs");
            std::vector<uint8_t> row(unit.row_bytes);
            for (size_t i = 0; i < image.rows().size(); ++i) {
                ggml_backend_tensor_get(t, row.data(), image.rows()[i].physical_cell*unit.row_bytes, row.size());
                check(std::equal(row.begin(), row.end(), unit.payload.begin()+i*row.size()), "captured row differs");
            }
        }
    }
};

static void decode(llama_context * ctx, const std::vector<llama_token> & tokens, int first, int count) {
    auto batch = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) { common_batch_add(batch, tokens.at(first+i), first+i, {0}, i+1 == count); }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    check(result == 0, "decode failed");
    llama_synchronize(ctx);
}

int main(int argc, char ** argv) {
    try {
        common_params params;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) { return 1; }
        params.kv_unified = true;
        ggml_backend_load_all();
        auto mp = common_model_params_to_llama(params);
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mp));
        check(bool(model), "model load failed");
        auto cp = common_context_params_to_llama(params);
        cp.n_seq_max = 2;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        check(bool(ctx), "context creation failed");
        auto * tree = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
        check(tree != nullptr, "test requires an attention-only iSWA model");
        auto budget = std::make_shared<window_budget>();
        vbr_swa_window_capture_request request;
        request.sequence = 0; request.sequence_epoch = 1;
        request.execution_identity[0] = 1;
        static constexpr char build[] = "test-vbr-swa-window";
        request.representation = {build, sizeof(build)-1};
        request.reserve = reserve_window; request.capacity_context = &budget;
        std::string text;
        for (int i = 0; i < 1500; ++i) { text += " The river flows past the old stone bridge."; }
        auto tokens = common_tokenize(ctx.get(), text, true, true);

        // The Gemma4 E2B fixture has SWA512 and physical protected prefix128.
        decode(ctx.get(), tokens, 0, 64);
        request.frontier = 64;
        check(vbr_capture_swa_window(*ctx, request).status == vbr_swa_window_status::protected_rows, "protected source admitted");
        check(budget->attempts == 0 && budget->used == 0, "protected refusal charged storage");
        constexpr int frontier = 1191;
        for (int p = 64; p < frontier;) {
            const int n = std::min(512, frontier-p);
            decode(ctx.get(), tokens, p, n); p += n;
        }
        request.frontier = frontier;
        budget->limit = 1;
        check(vbr_capture_swa_window(*ctx, request).status == vbr_swa_window_status::capacity_refused, "capacity refusal failed");
        check(budget->used == 0, "refused reservation leaked");
        budget->limit = 16*1024*1024;
        int chunks = 3;
        request.continue_context = &chunks; request.continue_capture = cancel_transfer;
        auto cancelled = vbr_capture_swa_window(*ctx, request);
        check(cancelled.status == vbr_swa_window_status::cancelled && !cancelled.image && budget->used == 0, "partial capture published or leaked");
        request.continue_capture = nullptr;
        const auto start = std::chrono::steady_clock::now();
        auto result = vbr_capture_swa_window(*ctx, request);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        check(result.status == vbr_swa_window_status::ok && result.image, "capture failed");
        check(result.image->rows().size() == 512 && result.image->units().size() == 24, "unexpected fixture window");
        check(result.image->retained_bytes() == budget->used, "wrong live charge");
        std::map<int32_t, size_t> types;
        for (const auto & unit : result.image->units()) { ++types[unit.generation.current_type]; }
        for (auto [type, count] : types) {
            fprintf(stderr, "WINDOW capture codec=%s units=%zu\n", ggml_type_name(ggml_type(type)), count);
        }
        llama_kv_cache_vbr_epoch_test::verify(*tree->get_swa(), *result.image);
        const size_t retained = budget->used;
        auto reader = result.image;
        result.image.reset();
        check(budget->used == retained, "shared reader lost charge");
        budget->limit = retained;
        check(vbr_capture_swa_window(*ctx, request).status == vbr_swa_window_status::capacity_refused &&
            budget->used == retained, "new refusal displaced the retained image");
        budget->limit = 16*1024*1024;
        chunks = 3; request.continue_capture = cancel_transfer;
        check(vbr_capture_swa_window(*ctx, request).status == vbr_swa_window_status::cancelled &&
            budget->used == retained, "cancelled replacement displaced the retained image");
        request.continue_capture = nullptr;
        auto repeated = vbr_capture_swa_window(*ctx, request);
        check(repeated.status == vbr_swa_window_status::ok && budget->used == 2*retained, "second capture budget incorrect");
        for (size_t i = 0; i < reader->units().size(); ++i) {
            check(reader->units()[i].payload == repeated.image->units()[i].payload, "repeat capture changed bytes");
        }
        repeated.image.reset();
        check(budget->used == retained, "second capture charge leaked");
        check(reader->source_matches(*ctx, 1, request.execution_identity), "fresh source rejected");
        check(!reader->source_matches(*ctx, 2, request.execution_identity), "old lifetime accepted");
        auto other_execution = request.execution_identity; other_execution[1] = 1;
        check(!reader->source_matches(*ctx, 1, other_execution), "different execution accepted");

        const auto old_swa_epoch = tree->get_swa()->vbr_checkpoint_epoch(0);
        for (int p = frontier; p < frontier+2048; ++p) { decode(ctx.get(), tokens, p, 1); }
        check(!tree->get_swa()->can_share_live_prefix(0, 1, frontier), "historical rows still present");
        check(tree->get_swa()->vbr_checkpoint_epoch(0) != old_swa_epoch, "SWA lineage did not change");
        check(reader->source_matches(*ctx, 1, request.execution_identity), "recycling invalidated sealed window");
        check(vbr_capture_swa_window(*ctx, request).status == vbr_swa_window_status::unavailable, "captured stale frontier");
        check(tree->seq_rm(0, -1, -1), "source removal failed");
        decode(ctx.get(), tokens, 0, 64);
        check(!reader->source_matches(*ctx, 1, request.execution_identity), "reused source slot accepted");
        ctx.reset();
        check(budget->used == retained && !reader->units().front().payload.empty(), "image did not outlive context");
        reader.reset();
        check(budget->used == 0, "last reader did not release charge");
        fprintf(stderr, "WINDOW CAPTURE PASS bytes=%zu capture_ms=%.3f; protected/refusal/cancel/repeat/recycle/lifetime/accounting\n", retained, ms);
    } catch (const std::exception & e) {
        fprintf(stderr, "WINDOW CAPTURE FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
