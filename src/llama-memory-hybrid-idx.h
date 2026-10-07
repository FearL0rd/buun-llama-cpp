#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <array>
#include <limits>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx,
    const llama_memory_vbr_params & vbr = {});

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool can_seq_rm_partial() const override {
        return llama_memory_hybrid::can_seq_rm_partial() &&
            (!mem_idx || mem_idx->can_seq_rm_partial());
    }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    bool seq_rm_attn(llama_seq_id seq_id,                            llama_pos p0, llama_pos p1) override;
    bool seq_rm_transient(llama_seq_id seq_id,                       llama_pos p0, llama_pos p1) override;
    bool seq_rm_attn_transient(llama_seq_id seq_id,                  llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    bool try_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    // Sharing attention alone would omit the indexer history.
    bool try_share_attn_prefix(llama_seq_id, llama_seq_id, llama_pos) override { return false; }
    bool can_share_attn_prefix(llama_seq_id, llama_seq_id, llama_pos) const override { return false; }
    bool try_seq_cp_transient(
            llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // Companion restore writes the child directly; invalidate derived pool
    // layouts after installing bytes or rolling a failed install back.
    void index_state_restored();

    // qwen4exp QSA keeps each block's pooled indexer key, already normed and rotated, across
    // ubatches, so a step re-pools only the blocks whose members changed (unified cache only).
    // F32 [idx_dim, max_blocks + 1]: row b holds block b, the last row absorbs padded updates.
    // nullptr when layer il has no indexer or the cache is not unified.
    ggml_tensor * get_qsa_pooled(int32_t il) const;

    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }
    struct kpool_layout;
    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;
    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();
    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }
    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

private:
    friend class llama_memory_hybrid_idx_context;

    // every pooled row is stale from here on: the indexer keys moved or were rewritten in bulk
    void qsa_invalidate();

    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;
    std::unique_ptr<kpool_layout> kpool_lay;
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;
    stale_pos_t mem_idx_stale = stale_pos_clean();

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> qsa_ctxs_bufs;
    std::map<int32_t, ggml_tensor *> qsa_pooled;

    // bumped whenever a cell's indexer key is written, so a pooled row can tell its inputs changed
    std::vector<uint32_t> qsa_cell_gen;

    // per ratio, the member cells and their generations each pooled row was computed from
    // a row whose cells are -1 is not valid
    struct qsa_row_keys {
        std::vector<int32_t>  cells;
        std::vector<uint32_t> gens;
    };
    std::map<uint32_t, qsa_row_keys> qsa_keys;

    // The direct QSA layout of one sequence: position p sits in slot p of block p/ratio, one cell
    // per slot. Kept across ubatches and patched from the cells' change log, so a decode step costs
    // the cells it wrote rather than a walk over every cell.
    struct qsa_layout {
        uint64_t rev    = 0;     // cells revision this layout reflects; 0 until built
        int64_t  n_kv   = 0;
        bool     direct = false; // false: a repeated or out-of-window position, no direct layout

        std::vector<int32_t> slot_cell; // [ratio*n_blocks] cell in each slot, -1 if none
        std::vector<int32_t> cell_slot; // [n_kv] slot of each cell, -1 if not in the layout
        std::vector<int32_t> cell_blk;  // [n_kv] block of each cell, n_blocks - 1 if not in the layout
        std::vector<uint8_t> fill;      // [n_blocks] occupied slots
    };
    std::map<std::pair<uint32_t, llama_seq_id>, qsa_layout> qsa_layouts;
    std::vector<uint32_t> qsa_dirty;

    // nullptr when the cells of seq have no direct layout over the first n_kv cells
    const qsa_layout * qsa_direct_layout(const llama_kv_cells & cells, llama_seq_id seq, uint32_t ratio, int64_t n_kv);
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    uint32_t get_n_kpool() const;
    uint32_t get_n_kpool_new() const;
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

    // A unified physical stream can expose one block layout only. Sparse selection is safe when
    // the current ubatch has one logical sequence; separate physical streams are independent.
    bool qsa_selection_safe(const llama_ubatch * ubatch) const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool causal_attn) const;

    // pooled rows to recompute this ubatch, or 0 when the graph must pool every block itself.
    // A step that changes no more than n_tokens/ratio + 2 blocks pads to that, so decode keeps one graph.
    int64_t qsa_n_upd(const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv) const;

    // for those rows: upd_rows I32 [n_upd], upd_cells I32 [ratio*n_upd, 1], upd_pos I32 [4*n_upd]
    // marks the rows valid, so the graph that reads these inputs must run
    void set_input_qsa_upd(ggml_tensor * upd_rows, ggml_tensor * upd_cells, ggml_tensor * upd_pos,
                           const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv) const;

    ggml_tensor * get_qsa_pooled(int32_t il) const;

private:
    // the stale pooled rows and their member cells, computed once per (ubatch, ratio)
    struct qsa_plan {
        size_t   i_ubatch = SIZE_MAX;
        uint32_t ratio    = 0;
        int64_t  n_kv     = 0;
        int64_t  n_upd    = 0;
        std::vector<int32_t> rows;
        std::vector<int32_t> cells;   // [ratio*rows], -1 for an empty slot of an incomplete block
    };
    const qsa_plan & plan_qsa(const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv) const;

    llama_memory_hybrid_idx * mem = nullptr;

    // a reserve context lays out no real cells; an update context moves them
    const bool is_full   = false;
    const bool is_update = false;

    // indexer cells each ubatch writes, for the pooled-row generations (unified cache only)
    // this and ns_ubatch are declared before ctx_idx, so they read sinfos_idx before it moves
    const std::vector<std::vector<uint32_t>> written_ubatch;

    mutable qsa_plan plan;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
