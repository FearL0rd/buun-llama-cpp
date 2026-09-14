#include "arg.h"
#include "common.h"
#include "llama-kv-cache-iswa.h"
#include "llama-sha256.h"
#include "llama-vbr-swa-window.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
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
    static std::array<uint8_t, 32> required_bytes(const llama_kv_cache_iswa & tree, llama_seq_id seq) {
        llama_sha256 hash;
        for (auto * cache : {tree.get_base(), tree.get_swa()}) {
            const auto & cells = cache->v_cells[0];
            for (const auto & layer : cache->layers) {
                for (auto * t : {layer.k, layer.v}) {
                    std::vector<uint8_t> bytes(t->nb[1]);
                    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                        if (!cells.seq_has(cell, seq) || (cache->n_swa && llama_hparams::is_masked_swa(
                            cache->n_swa, cache->swa_type, cells.pos_get(cell), cells.seq_pos_max(seq)+1))) { continue; }
                        hash.update(&cell, sizeof(cell));
                        ggml_backend_tensor_get(t, bytes.data(), cell*t->nb[1], bytes.size());
                        hash.update(bytes.data(), bytes.size());
                    }
                }
            }
        }
        return hash.finish();
    }

    static void install_gates(llama_context & ctx, llama_kv_cache_iswa & tree,
            const std::shared_ptr<const vbr_swa_window_image> & image,
            const vbr_swa_window_plan_request & placement) {
        vbr_swa_window_install_request install {placement.source_epoch, placement.destination_epoch,
                                               placement.execution_identity};
        auto prepare = [&] {
            auto result = vbr_prepare_swa_window(ctx, image, placement);
            check(result.status == vbr_swa_window_status::ok, "install gate prepare failed");
            return std::move(result.plan);
        };
        const auto before = fingerprint(tree);
        const auto entries = tree.get_swa()->vbr_generation_tracker_get()->extent_store().live_entries();
        auto probe = prepare();
        int reads = 0;
        for (const auto & unit : image->units()) {
            const auto & cells = probe->destination_cells();
            for (size_t i = 0; i < cells.size();) {
                size_t end = i+1;
                while (end < cells.size() && cells[end] == cells[end-1]+1) { ++end; }
                reads += ((end-i)*unit.row_bytes+65535)/65536; i = end;
            }
        }
        probe.reset();
        // Entry cancellation, cancellation during backup, then cancellation
        // after one H2D chunk and after the entire H2D phase but before publish.
        for (int stop : {1, 3, reads+4, 2*reads+3}) {
            int remaining = stop;
            install.continue_context = &remaining; install.continue_install = cancel_transfer;
            auto plan = prepare();
            const auto status = vbr_install_swa_window(ctx, std::move(plan), install);
            check(status == (stop <= reads+2 ? vbr_swa_window_status::cancelled : vbr_swa_window_status::rolled_back),
                  "wrong cancellation/rollback outcome");
            check(!plan && before == fingerprint(tree), "cancel/rollback changed live bytes or metadata");
            check(tree.get_swa()->vbr_generation_tracker_get()->extent_store().live_entries() == entries,
                  "cancel/rollback leaked provenance extents");
            check(!vbr_recovery_owned_by(tree.get_base()->vbr_instance_id()) &&
                  !vbr_recovery_owned_by(tree.get_swa()->vbr_instance_id()), "cancel/rollback leaked recovery");
        }
        install.continue_install = nullptr;
        {
            auto & store = tree.get_swa()->vbr_generation_tracker_mut()->extent_store();
            std::vector<vbr_extent_handle> held;
            held.reserve(vbr_extent_store::CAPACITY);
            struct release {
                vbr_extent_store & store; std::vector<vbr_extent_handle> & held;
                ~release() { for (auto handle : held) { store.fail(handle); } }
            } release_guard {store, held};
            while (auto handle = store.reserve(vbr_mutation_family::import, vbr_operation_class::checkpoint_restore,
                                               0, 1, 0, image->frontier(), false)) { held.push_back(handle); }
            check(vbr_install_swa_window(ctx, prepare(), install) == vbr_swa_window_status::operation_refused,
                  "full extent slab was not refused");
            check(!store.exhausted_latched() && before == fingerprint(tree), "extent refusal changed live tracking");
        }
        {
            vbr_generation_tracker separate(1, 1, 1);
            vbr_scoped_operation busy(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 0, 0, 1,
                vbr_operation_class::state_api, separate.runtime_instance()));
            check(bool(busy), "recovery capacity fixture operation refused");
            std::vector<int32_t> slots;
            slots.reserve(64); // initial capacity only; fill until the registry refuses
            struct release {
                vbr_operation_id op; std::vector<int32_t> & slots;
                ~release() { for (auto slot : slots) { vbr_recovery_release_unused(slot, op); } }
            } release_guard {busy.id(), slots};
            for (int32_t slot; (slot = vbr_recovery_reserve(busy.id(), separate.runtime_instance())) >= 0;) {
                try { slots.push_back(slot); }
                catch (...) { vbr_recovery_release_unused(slot, busy.id()); throw; }
            }
            check(vbr_install_swa_window(ctx, prepare(), install) == vbr_swa_window_status::operation_refused,
                  "full recovery ring was not refused");
            check(before == fingerprint(tree), "recovery refusal changed live state");
        }
        const auto source_before = required_bytes(tree, 0);
        auto plan = prepare();
        const auto rows = plan->destination_cells();
        const auto start = std::chrono::steady_clock::now();
        check(vbr_install_swa_window(ctx, std::move(plan), install) == vbr_swa_window_status::ok, "install failed");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        check(required_bytes(tree, 0) == source_before, "install changed required source bytes");
        auto & swa = *tree.get_swa();
        check(tree.get_base()->v_cells[0].seq_has_prefix(1, image->frontier()), "base prefix not published");
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto cell = rows[i];
            check(swa.v_cells[0].seq_has(cell, 1) && swa.v_cells[0].pos_get(cell) == image->rows()[i].position &&
                  swa.v_cells[0].ext_get(cell).tok == image->rows()[i].token, "wrong installed cell metadata");
            const auto * tracker = swa.vbr_generation_tracker_get();
            const auto * extent = tracker->extent_store().lookup_committed(tracker->dependency_extent(0, cell));
            check(extent && extent->family == vbr_mutation_family::import && extent->seq_id == 1,
                  "missing committed import provenance");
        }
        for (const auto & unit : image->units()) {
            auto * t = (unit.logical_unit&1) ? swa.layers[unit.logical_unit/2].v : swa.layers[unit.logical_unit/2].k;
            std::vector<uint8_t> bytes(unit.row_bytes);
            for (size_t i = 0; i < rows.size(); ++i) {
                ggml_backend_tensor_get(t, bytes.data(), rows[i]*unit.row_bytes, unit.row_bytes);
                check(std::equal(bytes.begin(), bytes.end(), unit.payload.begin()+i*unit.row_bytes), "installed bytes differ");
            }
        }
        std::vector<uint32_t> indexed;
        check(swa.vbr_ownership_->enumerate_owned(0, 1, indexed), "installed ownership unavailable");
        auto expected = rows;
        std::sort(expected.begin(), expected.end());
        check(indexed == expected, "installed ownership index differs from cells");
        fprintf(stderr, "WINDOW INSTALL PASS ms=%.3f rows=%zu; rollback-before/partial/full-upload/extent/recovery/source/provenance\n", ms, rows.size());
    }

    static std::array<uint8_t, 32> fingerprint(const llama_kv_cache_iswa & tree) {
        llama_sha256 hash;
        const auto put = [&](const auto & x) { hash.update(&x, sizeof(x)); };
        for (const auto * cache : {tree.get_base(), tree.get_swa()}) {
            const auto * tracker = cache->vbr_generation_tracker_get();
            put(tracker->mutation_serial()); put(tracker->controller_generation());
            put(cache->vbr_representation_epoch()); put(cache->v_heads[0]); put(cache->vbr_stash_dirty_);
            put(cache->vbr_pools_[0].wm_cells);
            const auto & cells = cache->v_cells[0];
            for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                put(cells.pos_get(cell));
                put(tracker->dependency_generation(0, cell)); put(tracker->membership_generation(0, cell));
                const auto dep = tracker->dependency_extent(0, cell), mem = tracker->membership_extent(0, cell);
                put(dep.index); put(dep.expected_gen); put(mem.index); put(mem.expected_gen);
                put(tracker->dependency_provenance(0, cell)); put(tracker->membership_provenance(0, cell));
                const auto ext = cells.ext_get(cell);
                put(ext.x); put(ext.y); put(ext.tok);
                for (uint32_t seq = 0; seq < cache->n_seq_max; ++seq) { put(cells.seq_has(cell, seq)); }
            }
            for (const auto & layer : cache->layers) {
                for (const auto * tensor : {layer.k, layer.v}) {
                    std::vector<uint8_t> bytes(tensor->nb[1]*cache->vbr_pools_[0].wm_cells);
                    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
                    hash.update(bytes.data(), bytes.size());
                }
            }
        }
        return hash.finish();
    }

    static void check_all_owners(llama_kv_cache & cache) {
        // Exercise the shared production predicate with private metadata only;
        // restore by swap even if an assertion fails. No live tensor writes.
        llama_kv_cells fixture;
        fixture.resize(cache.v_cells[0].size());
        fixture.pos_set(128, 100); fixture.seq_add(128, 0); fixture.seq_add(128, 1);
        fixture.pos_set(129, 2000); fixture.seq_add(129, 0);
        std::swap(fixture, cache.v_cells[0]);
        struct restore {
            llama_kv_cells & live, & saved;
            ~restore() { std::swap(live, saved); }
        } guard {cache.v_cells[0], fixture};
        check(!cache.can_reuse_cell(0, 128), "paused shared owner lost its row");
        cache.v_cells[0].seq_add(129, 1);
        check(cache.can_reuse_cell(0, 128), "row not reusable after both owners advance");
        check(cache.can_reuse_cell(0, 130), "empty row not reusable");
    }

    static void check_capacity_refusal(llama_context & ctx, llama_kv_cache & cache,
            const std::shared_ptr<const vbr_swa_window_image> & image,
            const vbr_swa_window_plan_request & request) {
        // Metadata-only refusal fixture: every physical row is protected.
        const auto saved = cache.vbr_stash_rows_;
        cache.vbr_stash_rows_ = cache.v_cells[0].size();
        struct restore {
            uint32_t & live;
            uint32_t saved;
            ~restore() { live = saved; }
        } guard {cache.vbr_stash_rows_, saved};
        const auto result = vbr_prepare_swa_window(ctx, image, request);
        check(result.status == vbr_swa_window_status::insufficient_cells && !result.plan,
              "insufficient unprotected cells admitted");
    }

    static void change_representation_epoch(llama_kv_cache & cache) {
        // Exercise retier's invalidation hook without changing live bytes/types.
        cache.vbr_representation_changed();
    }

    static void verify_plan(const llama_kv_cache_iswa & tree, const vbr_swa_window_plan & plan,
                            const vbr_swa_window_image & image) {
        const auto & cache = *tree.get_swa();
        const auto & cells = cache.v_cells[0];
        const std::set<uint32_t> selected(plan.destination_cells().begin(), plan.destination_cells().end());
        check(selected.size() == image.rows().size(), "duplicate or missing destination row");
        std::array<llama_pos, LLAMA_MAX_SEQ> purge;
        purge.fill(-1);
        for (const auto cell : selected) {
            check(cell >= cache.vbr_stash_rows_ && cache.can_reuse_cell(0, cell), "unsafe destination row");
            cells.seq_for_each(cell, [&](llama_seq_id seq) { purge[seq] = std::max(purge[seq], cells.pos_get(cell)); });
        }
        size_t expected = 0;
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            cells.seq_for_each(cell, [&](llama_seq_id seq) {
                if (cells.pos_get(cell) > purge[seq]) { return; }
                const auto & removal = plan.removals().at(expected++);
                check(removal.cell == cell && removal.sequence == seq && removal.position == cells.pos_get(cell),
                      "incomplete older-owner purge");
                check(llama_hparams::is_masked_swa(cache.n_swa, cache.swa_type,
                    removal.position, cells.seq_pos_max(seq)+1), "purge removed a required row");
            });
        }
        check(expected == plan.removals().size(), "extra membership removals");
        check(plan.base_cells().size() == size_t(image.frontier()), "base prefix incomplete");
        check(*selected.rbegin() < plan.required_watermark() && plan.required_watermark() <= cells.size(), "wrong mapping endpoint");
    }

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

static void decode(llama_context * ctx, const std::vector<llama_token> & tokens, int first, int count, llama_seq_id seq = 0) {
    auto batch = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) { common_batch_add(batch, tokens.at(first+i), first+i, {seq}, i+1 == count); }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    check(result == 0, "decode failed");
    llama_synchronize(ctx);
}

static void shared_owner_gate(llama_model * model, llama_context_params cp, const std::vector<llama_token> & tokens,
                              vbr_swa_window_capture_request capture) {
    cp.n_seq_max = 3;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    check(bool(ctx), "three-owner context failed");
    auto & tree = *dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
    const int frontier = capture.frontier;
    for (int p = 0; p < frontier;) {
        const int n = std::min(512, frontier-p);
        decode(ctx.get(), tokens, p, n); p += n;
    }
    auto sealed = vbr_capture_swa_window(*ctx, capture);
    check(sealed.status == vbr_swa_window_status::ok, "three-owner capture failed");
    const int advanced = frontier+2048;
    for (int p = frontier; p < advanced; ++p) { decode(ctx.get(), tokens, p, 1); }
    tree.seq_cp(0, 2, 0, -1);
    const auto paused_bytes = llama_kv_cache_vbr_epoch_test::required_bytes(tree, 2);
    vbr_swa_window_plan_request request;
    request.source_epoch = capture.sequence_epoch;
    request.destination = 1; request.destination_epoch = 1;
    request.execution_identity = capture.execution_identity; request.representation = capture.representation;
    auto plan = vbr_prepare_swa_window(*ctx, sealed.image, request);
    check(plan.status == vbr_swa_window_status::ok, "three-owner prepare failed");
    const size_t removals = plan.plan->removals().size();
    check(removals >= 2*sealed.image->rows().size(), "test did not replace shared old memberships");
    vbr_swa_window_install_request install {request.source_epoch, 1, request.execution_identity};
    check(vbr_install_swa_window(*ctx, std::move(plan.plan), install) == vbr_swa_window_status::ok, "three-owner install failed");
    check(paused_bytes == llama_kv_cache_vbr_epoch_test::required_bytes(tree, 2), "install changed paused owner bytes");
    for (int i = 0; i < 16; ++i) {
        decode(ctx.get(), tokens, frontier+i, 1, 1);
        decode(ctx.get(), tokens, advanced+i, 1, 0);
    }
    check(paused_bytes == llama_kv_cache_vbr_epoch_test::required_bytes(tree, 2), "other decodes changed paused owner bytes");
    check(tree.seq_rm(0, -1, -1), "three-owner source release failed");
    for (int i = 0; i < 16; ++i) {
        decode(ctx.get(), tokens, advanced+i, 1, 2);
        const auto * logits = llama_get_logits_ith(ctx.get(), -1);
        check(std::all_of(logits, logits+llama_vocab_n_tokens(llama_model_get_vocab(model)),
            [](float v) { return std::isfinite(v); }), "paused owner continuation nonfinite");
    }
    decode(ctx.get(), tokens, frontier+16, 1, 1);
    fprintf(stderr, "WINDOW THREE-OWNER PASS shared_removals=%zu; paused-owner/source-release/independent-continuations\n", removals);
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

        vbr_swa_window_plan_request placement;
        placement.source_epoch = 1; placement.destination = 1; placement.destination_epoch = 1;
        placement.execution_identity = request.execution_identity;
        placement.representation = request.representation;
        const auto before_plan = llama_kv_cache_vbr_epoch_test::fingerprint(*tree);
        auto fresh_plan = vbr_prepare_swa_window(*ctx, reader, placement);
        // Only 345 empty cells remain in this 1536-cell fixture. Reusing any
        // expired source row would purge its original protected prefix too.
        check(fresh_plan.status == vbr_swa_window_status::protected_rows && !fresh_plan.plan,
              "indirect protected-prefix removal admitted");
        auto invalid = placement;
        invalid.destination = 0;
        check(vbr_prepare_swa_window(*ctx, reader, invalid).status == vbr_swa_window_status::destination_unavailable,
              "source-as-destination accepted");
        invalid = placement; invalid.representation = {"different-build", 15};
        check(vbr_prepare_swa_window(*ctx, reader, invalid).status == vbr_swa_window_status::representation_mismatch,
              "different executable codec admitted");
        check(before_plan == llama_kv_cache_vbr_epoch_test::fingerprint(*tree), "planning changed live state");
        llama_kv_cache_vbr_epoch_test::check_all_owners(*tree->get_swa());

        const auto old_swa_epoch = tree->get_swa()->vbr_checkpoint_epoch(0);
        for (int p = frontier; p < frontier+2048; ++p) { decode(ctx.get(), tokens, p, 1); }
        check(!tree->get_swa()->can_share_live_prefix(0, 1, frontier), "historical rows still present");
        check(tree->get_swa()->vbr_checkpoint_epoch(0) != old_swa_epoch, "SWA lineage did not change");
        check(reader->source_matches(*ctx, 1, request.execution_identity), "recycling invalidated sealed window");
        check(vbr_capture_swa_window(*ctx, request).status == vbr_swa_window_status::unavailable, "captured stale frontier");
        const auto before_recycled_plan = llama_kv_cache_vbr_epoch_test::fingerprint(*tree);
        const size_t reserve_attempts = budget->attempts;
        const auto plan_start = std::chrono::steady_clock::now();
        auto recycled = vbr_prepare_swa_window(*ctx, reader, placement);
        const double plan_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-plan_start).count();
        check(recycled.status == vbr_swa_window_status::ok, "recycled placement failed");
        check(recycled.plan->current(*ctx, 1, 1, request.execution_identity), "fresh placement is stale");
        check(!recycled.plan->current(*ctx, 1, 2, request.execution_identity), "destination lifetime ignored");
        check(!recycled.plan->current(*ctx, 2, 1, request.execution_identity), "source lifetime ignored");
        check(!recycled.plan->current(*ctx, 1, 1, other_execution), "execution identity ignored");
        llama_kv_cache_vbr_epoch_test::verify_plan(*tree, *recycled.plan, *reader);
        check(!recycled.plan->removals().empty(), "full-pool placement did not reuse occupied rows");
        llama_kv_cache_vbr_epoch_test::check_capacity_refusal(*ctx, *tree->get_swa(), reader, placement);
        check(reserve_attempts == budget->attempts && budget->used == retained, "planning duplicated image charge");
        check(before_recycled_plan == llama_kv_cache_vbr_epoch_test::fingerprint(*tree), "recycled planning changed live state");
        fprintf(stderr, "WINDOW PLAN prepared rows=%zu removals=%zu base=%zu endpoint=%u ms=%.3f\n",
            recycled.plan->destination_cells().size(), recycled.plan->removals().size(),
            recycled.plan->base_cells().size(), recycled.plan->required_watermark(), plan_ms);
        for (const auto * child : {tree->get_base(), tree->get_swa()}) {
            vbr_scoped_operation busy(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 1, 0, frontier,
                vbr_operation_class::state_api, child->vbr_instance_id()));
            check(bool(busy), "busy-context test operation refused");
            check(!recycled.plan->current(*ctx, 1, 1, request.execution_identity), "busy child admitted");
            check(vbr_prepare_swa_window(*ctx, reader, placement).status == vbr_swa_window_status::source_changed,
                  "prepared against a busy child");
            check(busy.close(vbr_operation_outcome::committed), "busy-context test close failed");
        }
        check(recycled.plan->current(*ctx, 1, 1, request.execution_identity), "no-write busy interval invalidated plan");
        for (auto * child : {tree->get_base(), tree->get_swa()}) {
            llama_kv_cache_vbr_epoch_test::change_representation_epoch(*child);
            check(!recycled.plan->current(*ctx, 1, 1, request.execution_identity), "representation epoch ignored");
            recycled = vbr_prepare_swa_window(*ctx, reader, placement);
            check(recycled.status == vbr_swa_window_status::ok, "replanning after epoch change failed");
        }
        decode(ctx.get(), tokens, frontier+2048, 1);
        check(!recycled.plan->current(*ctx, 1, 1, request.execution_identity), "decode did not invalidate plan");
        recycled = vbr_prepare_swa_window(*ctx, reader, placement);
        check(recycled.status == vbr_swa_window_status::ok, "replanning after decode failed");
        tree->get_base()->seq_cp(0, 1, 0, 1);
        check(!recycled.plan->current(*ctx, 1, 1, request.execution_identity), "occupied destination admitted");
        check(vbr_prepare_swa_window(*ctx, reader, placement).status == vbr_swa_window_status::destination_unavailable,
              "prepared an occupied destination");
        check(tree->get_base()->seq_rm(1, -1, -1), "destination removal failed");
        check(!recycled.plan->current(*ctx, 1, 1, request.execution_identity), "destination reuse accepted old plan");
        recycled = vbr_prepare_swa_window(*ctx, reader, placement);
        check(recycled.status == vbr_swa_window_status::ok, "replanning empty destination failed");
        llama_kv_cache_vbr_epoch_test::install_gates(*ctx, *tree, reader, placement);
        check(!recycled.plan->current(*ctx, 1, 1, request.execution_identity), "installation failed to stale old plan");
        for (int i = 0; i < 16; ++i) {
            decode(ctx.get(), tokens, frontier+i, 1, 1);
            const auto * logits = llama_get_logits_ith(ctx.get(), -1);
            check(std::all_of(logits, logits+llama_vocab_n_tokens(llama_model_get_vocab(model.get())),
                [](float value) { return std::isfinite(value); }), "destination continuation nonfinite");
            decode(ctx.get(), tokens, frontier+2049+i, 1);
        }
        check(tree->seq_rm(1, -1, -1), "installed destination removal failed");
        // A second restore may find only empty cells after expired history was
        // removed. It needs no old-owner event or corresponding manifest target.
        const auto source_bytes = llama_kv_cache_vbr_epoch_test::required_bytes(*tree, 0);
        check(tree->get_swa()->seq_rm(0, 0, frontier+2049+16-512), "expired history removal failed");
        auto empty_plan = vbr_prepare_swa_window(*ctx, reader, placement);
        check(empty_plan.status == vbr_swa_window_status::ok && empty_plan.plan->removals().empty(),
              "empty-row fixture did not select only empty cells");
        vbr_swa_window_install_request empty_install {1, 1, request.execution_identity};
        check(vbr_install_swa_window(*ctx, std::move(empty_plan.plan), empty_install) == vbr_swa_window_status::ok,
              "empty-row installation failed");
        check(source_bytes == llama_kv_cache_vbr_epoch_test::required_bytes(*tree, 0), "empty-row restore changed source");
        decode(ctx.get(), tokens, frontier, 1, 1);
        check(tree->seq_rm(1, -1, -1), "empty-row destination removal failed");
        fprintf(stderr, "WINDOW EMPTY-ROW PASS no-removal-target/source-preserved/continuation\n");
        check(tree->seq_rm(0, -1, -1), "source removal failed");
        decode(ctx.get(), tokens, 0, 64);
        check(!reader->source_matches(*ctx, 1, request.execution_identity), "reused source slot accepted");
        check(!recycled.plan->current(*ctx, 1, 1, request.execution_identity), "recycled plan survived source reuse");
        ctx.reset();
        check(budget->used == retained && !reader->units().front().payload.empty(), "image did not outlive context");
        const bool test_shared = reader->units().front().generation.current_type == GGML_TYPE_TURBO4_0;
        reader.reset();
        check(budget->used == retained, "plan did not retain image ownership");
        recycled.plan.reset();
        check(budget->used == 0, "last reader did not release charge");
        if (test_shared) {
            request.sequence_epoch = 2;
            shared_owner_gate(model.get(), cp, tokens, request);
            check(budget->used == 0, "three-owner image charge leaked");
        }
        fprintf(stderr, "WINDOW PLAN PASS no-write/all-owners/protected-purge/capacity/busy/stale-decode/epochs/identity/lifetime\n");
        fprintf(stderr, "WINDOW CAPTURE PASS bytes=%zu capture_ms=%.3f; protected/refusal/cancel/repeat/recycle/lifetime/accounting\n", retained, ms);
    } catch (const std::exception & e) {
        fprintf(stderr, "WINDOW CAPTURE FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
