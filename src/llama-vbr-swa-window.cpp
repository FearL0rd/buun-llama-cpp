#include "llama-vbr-swa-window.h"

#include "llama-hparams.h"
#include "llama-kv-cache-iswa.h"
#include "llama-sha256.h"
#include "llama-vbr-downward.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

namespace {
bool nonzero(const std::array<uint8_t, 32> & value) {
    return std::any_of(value.begin(), value.end(), [](uint8_t b) { return b != 0; });
}

bool add_bytes(size_t & total, size_t count, size_t stride) {
    if (stride && count > (SIZE_MAX-total)/stride) { return false; }
    total += count*stride;
    return true;
}
}

class vbr_swa_window_capture {
    using status = vbr_swa_window_status;
    static bool ready(const llama_kv_cache & cache) {
        const auto * tracker = cache.vbr_generation_tracker_get();
        return cache.vbr_operation_armed() && tracker && tracker->stable() &&
            !tracker->shadow_unavailable();
    }

public:
    static bool matches(const vbr_swa_window_image & image, llama_context & ctx,
                        uint64_t epoch, const std::array<uint8_t, 32> & execution) {
        auto * tree = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        if (!tree || epoch != image.sequence_epoch_ || execution != image.execution_identity_) { return false; }
        const auto & base = *tree->get_base();
        const auto & swa = *tree->get_swa();
        const vbr_controller_instance_id instances[] = {base.vbr_instance_id(), swa.vbr_instance_id()};
        return ready(base) && ready(swa) && base.vbr_instance_id() == image.base_instance_ &&
            swa.vbr_instance_id() == image.swa_instance_ &&
            vbr_operation_registry_quiescent_for(instances, 2) &&
            base.vbr_checkpoint_epoch(image.sequence_) == image.base_epoch_ &&
            base.v_cells.size() == 1 && base.v_cells[0].seq_has_prefix(image.sequence_, image.frontier_);
    }

    static vbr_swa_window_capture_result capture(
            llama_context & ctx, const vbr_swa_window_capture_request & request) {
        const auto fail = [](status s) { return vbr_swa_window_capture_result {s, nullptr}; };
        auto * tree = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(&ctx));
        if (!tree || request.sequence < 0 || request.sequence >= LLAMA_MAX_SEQ ||
            request.frontier <= 0 || request.sequence_epoch == 0 || !nonzero(request.execution_identity) ||
            !request.representation.build_identity || !request.representation.build_identity_len) {
            return fail(status::unsupported);
        }
        auto & base = *tree->get_base();
        auto & swa = *tree->get_swa();
        if (base.other || swa.other || base.n_stream != 1 || swa.n_stream != 1 ||
            swa.v_trans || swa.swa_type != LLAMA_SWA_TYPE_STANDARD || swa.n_swa == 0 ||
            base.n_swa != 0 || base.vbr_pools_.size() != 1 || swa.vbr_pools_.size() != 1 ||
            base.vbr_pools_[0].device < 0 || base.vbr_pools_[0].device != swa.vbr_pools_[0].device ||
            !base.vbr_vmm_active() || !swa.vbr_vmm_active()) {
            return fail(status::unsupported);
        }
        if (!request.reserve) { return fail(status::capacity_refused); }
        llama_synchronize(&ctx);
        const vbr_controller_instance_id instances[] = {base.vbr_instance_id(), swa.vbr_instance_id()};
        if (!ready(base) || !ready(swa) || !vbr_operation_registry_quiescent_for(instances, 2) ||
            !swa.vbr_capture_settle()) { return fail(status::unavailable); }
        const auto & cells = swa.v_cells[0];
        if (cells.seq_pos_max(request.sequence) != request.frontier-1 ||
            !base.v_cells[0].seq_has_prefix(request.sequence, request.frontier)) {
            return fail(status::unavailable);
        }
        const uint64_t base_serial = base.vbr_generation_tracker_get()->mutation_serial();
        const uint64_t swa_serial = swa.vbr_generation_tracker_get()->mutation_serial();
        const uint64_t base_repr = base.vbr_representation_epoch();
        const uint64_t swa_repr = swa.vbr_representation_epoch();
        auto image = std::shared_ptr<vbr_swa_window_image>(new vbr_swa_window_image);
        image->base_instance_ = instances[0]; image->swa_instance_ = instances[1];
        image->base_epoch_ = base.vbr_checkpoint_epoch(request.sequence);
        image->sequence_epoch_ = request.sequence_epoch;
        image->execution_identity_ = request.execution_identity;
        image->sequence_ = request.sequence; image->frontier_ = request.frontier;
        const size_t row_count = std::min<uint32_t>(swa.n_swa, request.frontier);
        image->rows_.reserve(row_count);
        for (uint32_t r = 0; r < cells.size(); ++r) {
            if (!swa.state_write_includes_cell(cells, r, request.sequence)) { continue; }
            if (r < swa.vbr_stash_rows_) { return fail(status::protected_rows); }
            const auto & ext = cells.ext_get(r);
            if (ext.x != 0 || ext.y != 0 || ext.tok == LLAMA_TOKEN_NULL || image->rows_.size() == row_count) {
                return fail(status::unsupported);
            }
            image->rows_.push_back({r, cells.pos_get(r), ext.tok});
        }
        if (image->rows_.size() != row_count) { return fail(status::unavailable); }
        // Pack in logical order, keeping the physical map for actual transfers.
        std::sort(image->rows_.begin(), image->rows_.end(), [](const auto & a, const auto & b) {
            return a.position < b.position;
        });
        for (size_t i = 0; i < row_count; ++i) {
            if (image->rows_[i].position != request.frontier-llama_pos(row_count)+llama_pos(i)) {
                return fail(status::unsupported);
            }
        }
        const auto * tracker = swa.vbr_generation_tracker_get();
        if (tracker->unit_count() != 2*swa.layers.size()) { return fail(status::unsupported); }
        image->units_.reserve(tracker->unit_count());
        size_t bytes = sizeof(vbr_swa_window_image);
        if (!add_bytes(bytes, image->rows_.capacity(), sizeof(vbr_swa_window_row)) ||
            !add_bytes(bytes, image->units_.capacity(), sizeof(vbr_swa_window_unit))) { return fail(status::capacity_refused); }
        std::vector<ggml_tensor *> tensors;
        for (uint32_t id = 0; id < tracker->unit_count(); ++id) {
            const auto & layer = swa.layers[id/2];
            const auto extents = swa.vbr_units_of(id/2, (id&1) != 0);
            if (extents.size() != 1) { return fail(status::unsupported); }
            const auto [pool, extent] = extents.front();
            auto * t = extent->t;
            const auto generation = tracker->unit_generation(id);
            vbr_downward_recipe recipe;
            if (!t || t != ((id&1) ? layer.v : layer.k) || t->ne[2] != 1 || t->ne[3] != 1 ||
                vbr_downward_resolve_recipe(t->type, t->type, t->type, true, recipe) != vbr_downward_recipe_status::equal_tier ||
                t->nb[1] != ggml_row_size(t->type, t->ne[0]) ||
                vbr_explicit_capture_validate_extent_generation(pool->wm_cells, t->type, extent->promote_hops, generation) !=
                    vbr_explicit_size_failure::none) { return fail(status::unsupported); }
            for (const auto & row : image->rows_) {
                if (row.physical_cell >= pool->wm_cells) { return fail(status::unavailable); }
            }
            vbr_swa_window_unit unit {};
            unit.logical_unit = id; unit.model_layer = layer.il;
            unit.generation = generation;
            unit.meansub_model_id = layer.turbo_meansub_ref.model_id;
            unit.meansub_layer = layer.turbo_meansub_ref.layer;
            if (!vbr_explicit_capture_representation_identity(&request.representation, t->type, (id&1) != 0,
                    unit.meansub_model_id, unit.codec)) { return fail(status::unavailable); }
            std::memcpy(unit.tensor_name.data(), t->name, unit.tensor_name.size());
            unit.columns = t->ne[0]; unit.row_bytes = t->nb[1];
            if (!add_bytes(bytes, row_count, unit.row_bytes)) { return fail(status::capacity_refused); }
            image->units_.push_back(std::move(unit));
            tensors.push_back(t);
        }
        image->capacity_ = request.reserve(request.capacity_context, bytes);
        if (!image->capacity_) { return fail(status::capacity_refused); }
        image->retained_bytes_ = bytes;
        const auto stable = [&] {
            return ready(base) && ready(swa) && base.vbr_generation_tracker_get()->mutation_serial() == base_serial &&
                swa.vbr_generation_tracker_get()->mutation_serial() == swa_serial &&
                base.vbr_representation_epoch() == base_repr && swa.vbr_representation_epoch() == swa_repr &&
                vbr_operation_registry_quiescent_for(instances, 2);
        };
        if (!stable()) { return fail(status::source_changed); }
        constexpr size_t chunk_bytes = 64*1024;
        for (size_t u = 0; u < image->units_.size(); ++u) {
            auto & unit = image->units_[u];
            unit.payload.resize(row_count*unit.row_bytes);
            llama_sha256 hash;
            for (size_t i = 0; i < row_count;) {
                size_t end = i+1;
                while (end < row_count && image->rows_[end].physical_cell == image->rows_[end-1].physical_cell+1) { ++end; }
                const size_t run_bytes = (end-i)*unit.row_bytes;
                for (size_t offset = 0; offset < run_bytes;) {
                    if (request.continue_capture && !request.continue_capture(request.continue_context)) { return fail(status::cancelled); }
                    if (!stable()) { return fail(status::source_changed); }
                    const size_t size = std::min(chunk_bytes, run_bytes-offset);
                    auto * dst = unit.payload.data()+i*unit.row_bytes+offset;
                    ggml_backend_tensor_get(tensors[u], dst, image->rows_[i].physical_cell*unit.row_bytes+offset, size);
                    hash.update(dst, size);
                    offset += size;
                }
                i = end;
            }
            unit.checksum = hash.finish();
        }
        if (!stable()) { return fail(status::source_changed); }
        if (request.continue_capture && !request.continue_capture(request.continue_context)) { return fail(status::cancelled); }
        return {status::ok, std::move(image)};
    }
};

bool vbr_swa_window_image::source_matches(llama_context & ctx, uint64_t epoch,
                                       const std::array<uint8_t, 32> & execution) const {
    return vbr_swa_window_capture::matches(*this, ctx, epoch, execution);
}

vbr_swa_window_capture_result vbr_capture_swa_window(
        llama_context & ctx, const vbr_swa_window_capture_request & request) {
    try {
        return vbr_swa_window_capture::capture(ctx, request);
    } catch (const std::bad_alloc &) {
        return {vbr_swa_window_status::allocation_failed, nullptr};
    }
}
