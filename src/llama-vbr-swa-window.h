#pragma once

#include "llama-vbr-explicit-capture.h"

#include <array>
#include <memory>
#include <vector>

struct llama_context;
class llama_kv_cache_iswa;

// Process-local partial window, deliberately not an artifact package or a
// payload_complete child. No serialized format and no live tensor pointers.
struct vbr_swa_window_row {
    uint32_t physical_cell;
    llama_pos position;
    llama_token token;
};

struct vbr_swa_window_unit {
    uint32_t logical_unit;
    uint32_t model_layer;
    vbr_unit_generation generation;
    vbr_explicit_representation_identity codec;
    int32_t meansub_model_id;
    int32_t meansub_layer;
    std::array<char, GGML_MAX_NAME> tensor_name {};
    uint64_t columns;
    size_t row_bytes;
    std::array<uint8_t, 32> checksum {};
    std::vector<uint8_t> payload;
};

struct vbr_swa_window_capture_request {
    llama_seq_id sequence = -1;
    llama_pos frontier = -1; // exclusive; capture only the committed frontier
    uint64_t sequence_epoch = 0; // caller's logical slot lifetime, not slot ID
    std::array<uint8_t, 32> execution_identity {};
    vbr_explicit_representation_policy representation;

    // Required admission adapter. Charge the quoted bytes once and return an
    // owning lease that releases the charge at destruction. The image retains
    // it across all shared readers. A null lease declines BEFORE payload reads.
    // Server integration must bind this to its host-cache budget, not a second
    // unaccounted allowance. No server caller is enabled by this module.
    void * capacity_context = nullptr;
    std::shared_ptr<void> (*reserve)(void *, size_t bytes) = nullptr;
    void * continue_context = nullptr;
    bool (*continue_capture)(void *) noexcept = nullptr;
};

enum class vbr_swa_window_status {
    ok,
    unsupported,
    unavailable,
    protected_rows,
    capacity_refused,
    cancelled,
    source_changed,
    allocation_failed,
};

class vbr_swa_window_image {
public:
    vbr_swa_window_image(const vbr_swa_window_image &) = delete;
    vbr_swa_window_image & operator=(const vbr_swa_window_image &) = delete;
    const std::vector<vbr_swa_window_row> & rows() const { return rows_; }
    const std::vector<vbr_swa_window_unit> & units() const { return units_; }
    // Payload plus owned metadata; allocator/control-block overhead excluded,
    // as with other logical host-cache byte charges.
    size_t retained_bytes() const { return retained_bytes_; }
    llama_pos frontier() const { return frontier_; }

    // Source identity/coverage ONLY, not authorization to install. Deliberately
    // ignores the live SWA content epoch: recycling that window is the use case.
    // Caller must still match execution and logical sequence lifetime, and the
    // destination transaction must validate its current representation/capacity.
    bool source_matches(llama_context & ctx, uint64_t sequence_epoch,
                        const std::array<uint8_t, 32> & execution_identity) const;

private:
    friend class vbr_swa_window_capture;
    vbr_swa_window_image() = default;
    // Declared first so the charge outlives payload/metadata destruction.
    std::shared_ptr<void> capacity_;
    vbr_controller_instance_id base_instance_ {}, swa_instance_ {};
    uint64_t base_epoch_ = 0, sequence_epoch_ = 0;
    std::array<uint8_t, 32> execution_identity_ {};
    llama_seq_id sequence_ = -1;
    llama_pos frontier_ = -1;
    size_t retained_bytes_ = 0;
    std::vector<vbr_swa_window_row> rows_;
    std::vector<vbr_swa_window_unit> units_;
};

struct vbr_swa_window_capture_result {
    vbr_swa_window_status status = vbr_swa_window_status::unavailable;
    std::shared_ptr<const vbr_swa_window_image> image;
};

// Requires the caller's exclusive scheduler boundary for this context, as do
// other llama memory operations. Synchronizes submitted compute. Not a lock
// allowing concurrent decode; callbacks must not reenter the context.
vbr_swa_window_capture_result vbr_capture_swa_window(
    llama_context & ctx, const vbr_swa_window_capture_request & request);
