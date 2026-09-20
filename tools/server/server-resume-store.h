#pragma once

// Durable store of the persistent server resume: one directory of objects per conversation,
// the manifest published last. Format: docs/development/server-resume-format.md §2.
//
// The store knows files, checksums and bounds. It does not know tokens, slots or the model: the
// ledger is opaque bytes here, and what a record means is the server's business.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define SERVER_RESUME_MANIFEST_VERSION 1
#define SERVER_RESUME_OBJECT_VERSION   1

enum class server_resume_reason : uint8_t {
    ok = 0,
    format_unsupported,
    manifest_corrupt,
    object_missing,
    object_size_mismatch,
    object_checksum_mismatch,
    store_locked,
    store_unwritable,
    no_space,
    io_error,
    _count,
};

const char * server_resume_reason_name(server_resume_reason reason) noexcept;

enum class server_resume_object_kind : uint32_t {
    base_chunk = 1, // base state of the token positions [p0, p1)
    tail_state = 2, // partial state taken when the sequence ended at p0; p1 is 0
};

// bounds of a manifest, checked before anything is allocated from its numbers
struct server_resume_limits {
    static constexpr uint64_t max_manifest_bytes = 64ull * 1024 * 1024;
    static constexpr uint64_t max_json_bytes     = 1ull * 1024 * 1024;
    static constexpr size_t   max_chunks         = 4096;
    static constexpr size_t   max_tail_states    = 64;
    static constexpr size_t   max_producers      = 16;
    static constexpr int32_t  max_tokens         = 1 << 24;
    static constexpr int      max_json_depth     = 6;
    static constexpr size_t   max_entries_listed = 1024;
    static constexpr uint64_t max_object_bytes   = 8ull * 1024 * 1024 * 1024;
};

struct server_resume_object_record {
    server_resume_object_kind kind = server_resume_object_kind::base_chunk;
    int32_t  p0    = 0;
    int32_t  p1    = 0;
    uint64_t gen   = 0;
    uint64_t bytes = 0; // payload bytes
    uint64_t xxh3  = 0; // of the payload

    std::string prefix_digest; // hex, bound by the server to the ledger's tokens [0, p1) or [0, pos)
    uint32_t    producer = 0;

    // tail states only
    int32_t     pos_min  = 0;
    int32_t     pos_max  = 0;
    int32_t     n_tokens = 0;
    std::string role; // frontier, turn, early

    int32_t pos() const { return p0; }
};

// provenance, never a gate
struct server_resume_producer {
    std::string model_name;
    std::string model_file; // basename
    std::string weight_type;
    std::string build;
    std::string cache_type_k;
    std::string cache_type_v;
    std::vector<std::string> adapters;
    int64_t host_unix_ms = 0;

    bool operator==(const server_resume_producer & other) const;
};

struct server_resume_manifest {
    uint64_t generation = 0;

    std::string resume_key; // hex
    std::string family_digest;
    std::string adapter_identity;

    int32_t n_tokens          = 0;
    int32_t chunk_tokens      = 0;
    int64_t saved_unix_ms     = 0;
    int64_t last_used_unix_ms = 0;
    int32_t slot_hint         = -1;

    std::vector<server_resume_object_record> chunks;
    std::vector<server_resume_object_record> tail_states;
    std::vector<server_resume_producer>      producers;

    std::vector<uint8_t> ledger;

    // index of an equal producer, added when there is none and room is left, else 0
    uint32_t producer_index(const server_resume_producer & producer);
};

// structure only: bounds, gapless tiling of [0, n_tokens), tail state positions, producer indices
bool server_resume_manifest_validate(const server_resume_manifest & manifest, std::string & error);

std::vector<uint8_t> server_resume_manifest_encode(const server_resume_manifest & manifest);

server_resume_reason server_resume_manifest_decode(
    const uint8_t * data, size_t size, server_resume_manifest & manifest, std::string & error);

struct server_resume_entry {
    std::string            id;
    server_resume_reason   reason = server_resume_reason::ok; // of the manifest
    std::string            error;
    server_resume_manifest manifest;
};

// test seam: called at named points of a write, a returned reason other than ok is injected there.
// points: object_write (half of the payload is on disk), object_publish, manifest_publish
using server_resume_fault_fn = server_resume_reason (*)(const char * point);
void server_resume_store_set_fault(server_resume_fault_fn fn) noexcept;

class server_resume_store {
public:
    // <cache root>/resume/<family digest>/, directories 0700, files 0600. takes the writer lock of the
    // namespace: a second server gets store_locked and runs without persistence
    static std::unique_ptr<server_resume_store> open(
        const std::string & cache_root, const std::string & family_digest,
        server_resume_reason & reason, std::string & error);

    ~server_resume_store();

    server_resume_store(const server_resume_store &) = delete;
    server_resume_store & operator=(const server_resume_store &) = delete;

    const std::string & directory() const { return dir; }

    static std::string new_entry_id();
    static bool        entry_id_valid(const std::string & id);

    // committed entries, newest last_used first. an unreadable manifest is listed with its reason
    std::vector<server_resume_entry> list() const;

    server_resume_reason read_manifest(const std::string & id, server_resume_manifest & manifest, std::string & error) const;

    // verifies the header against the record, the file size and the payload checksum.
    // the payload buffer is reused between calls: one chunk of staging
    server_resume_reason read_object(
        const std::string & id, const server_resume_object_record & record,
        std::vector<uint8_t> & payload, std::string & error) const;

    // stages, syncs and renames one object. fills bytes and xxh3 of the record
    server_resume_reason write_object(
        const std::string & id, server_resume_object_record & record,
        const uint8_t * payload, size_t size, std::string & error);

    // publishes the manifest: the entry is this generation from here on
    server_resume_reason commit(const std::string & id, const server_resume_manifest & manifest, std::string & error);

    // removes what the manifest does not name: replaced objects and staging leftovers
    void sweep(const std::string & id, const server_resume_manifest & manifest) const;

    // space-bounded replacement: take the entry out of the store before its objects go
    server_resume_reason uncommit(const std::string & id, std::string & error);

    void remove_entry(const std::string & id) const;

    // keeps the n entries used last, and drops damaged manifests and directories that never got one
    void prune(size_t n_keep) const;

    uint64_t free_bytes() const;

private:
    server_resume_store() = default;

    std::string dir; // the family namespace
    int lock_fd = -1;

    std::string entry_dir(const std::string & id) const;
};
