// Server-resume P0 probe: measures what a process restart costs and what a saved sequence state buys.
//
//   PROBE_MODE=save  prefill tokens [0, PROBE_NPREFIX) of -f <text>, write the sequence state to PROBE_STATE
//   PROBE_MODE=eval  reach PROBE_NPREFIX tokens either by a fresh prefill or by loading PROBE_STATE and
//                    replaying the tokens it does not cover, then teacher-force PROBE_NEVAL single-token
//                    decodes and write their logits to PROBE_OUT
//
//   PROBE_MODE=family  print llama_model_semantic_family_digest() of the loaded model
//
// A state saved short of PROBE_NPREFIX is the bounded-suffix replay of the resume plan: the saved history is
// kept, the last tokens are recomputed under the loaded model.
//
// Checkpoint companion (save: PROBE_CKPT + PROBE_NCKPT; eval: PROBE_NDIVERGE [+ PROBE_CKPT]): the save run also
// writes the partial (recurrent / SWA) state at PROBE_NCKPT tokens. The eval run then serves a request that
// leaves the saved history at PROBE_NDIVERGE: it rewinds there the way the server does (partial restore, then
// seq_rm), prefills PROBE_NSUFFIX tokens taken from PROBE_ALT_OFFSET and teacher-forces from there.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

static const uint32_t PROBE_STATE_MAGIC  = 0x52534d50; // "PMSR"
static const uint32_t PROBE_LOGITS_MAGIC = 0x4c534d50; // "PMSL"

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static const char * env_str(const char * name, const char * def) {
    const char * v = std::getenv(name);
    return v && *v ? v : def;
}

static int64_t env_int(const char * name, int64_t def) {
    const char * v = std::getenv(name);
    return v && *v ? std::atoll(v) : def;
}

// decode tokens [beg, end) as a prompt; logits only for the last token
static bool prefill(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t beg, int32_t end, int32_t n_batch) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int32_t i = beg; i < end; i += n_batch) {
        const int32_t n = std::min(n_batch, end - i);
        common_batch_clear(batch);
        for (int32_t j = 0; j < n; ++j) {
            common_batch_add(batch, tokens[i + j], i + j, { 0 }, i + j == end - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
    }
    llama_synchronize(ctx);
    llama_batch_free(batch);
    return true;
}

static bool write_all(int fd, const void * data, size_t size) {
    const char * p = (const char *) data;
    while (size > 0) {
        const ssize_t n = write(fd, p, std::min<size_t>(size, 1u << 30));
        if (n <= 0) {
            return false;
        }
        p    += n;
        size -= n;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.warmup = false;

    const std::string mode       = env_str("PROBE_MODE", "eval");
    const std::string state_path = env_str("PROBE_STATE", "");
    const std::string out_path   = env_str("PROBE_OUT", "");
    const int32_t     n_prefix   = (int32_t) env_int("PROBE_NPREFIX", 1024);
    const int32_t     n_eval     = (int32_t) env_int("PROBE_NEVAL", 256);
    const std::string ckpt_path  = env_str("PROBE_CKPT", "");
    const int32_t     n_ckpt     = (int32_t) env_int("PROBE_NCKPT", 0);
    const int32_t     n_diverge  = (int32_t) env_int("PROBE_NDIVERGE", -1);
    const int32_t     n_suffix   = (int32_t) env_int("PROBE_NSUFFIX", 64);
    const int32_t     alt_offset = (int32_t) env_int("PROBE_ALT_OFFSET", 0);

    llama_backend_init();
    llama_numa_init(params.numa);

    const double t_load0 = now_ms();
    auto init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        LOG_ERR("probe: failed to load model\n");
        return 1;
    }
    const double t_load = now_ms() - t_load0;

    if (mode == "family") {
        uint8_t digest[32];
        if (!llama_model_semantic_family_digest(model, digest)) {
            printf("PROBE_FAMILY unavailable\n");
            return 2;
        }
        char hex[65];
        for (int i = 0; i < 32; ++i) {
            snprintf(hex + 2 * i, 3, "%02x", digest[i]);
        }
        printf("PROBE_FAMILY %s\n", hex);
        llama_backend_free();
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    const int32_t n_need = n_prefix + (mode == "eval" ? n_eval : 0);
    if ((int32_t) tokens.size() < n_need) {
        LOG_ERR("probe: text has %zu tokens, need %d\n", tokens.size(), n_need);
        return 1;
    }

    printf("PROBE mode=%s n_prefix=%d n_eval=%d n_ctx=%u n_vocab=%d model_load_ms=%.1f\n",
            mode.c_str(), n_prefix, n_eval, llama_n_ctx(ctx), n_vocab, t_load);

    if (mode == "save") {
        double t0 = now_ms();
        const int32_t n_first = ckpt_path.empty() ? n_prefix : n_ckpt;
        if (!prefill(ctx, tokens, 0, n_first, params.n_batch)) {
            LOG_ERR("probe: prefill failed\n");
            return 1;
        }
        if (!ckpt_path.empty()) {
            // what the server keeps as a context checkpoint at this position
            const double t1 = now_ms();
            std::vector<uint8_t> part(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            const size_t n_part = llama_state_seq_get_data_ext(ctx, part.data(), part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            const double t_part = now_ms() - t1;
            const int fd = open(ckpt_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            const uint32_t hdr[2] = { PROBE_STATE_MAGIC, (uint32_t) n_ckpt };
            const uint64_t nbytes = n_part;
            const bool ok = fd >= 0 && write_all(fd, hdr, sizeof(hdr)) && write_all(fd, &nbytes, sizeof(nbytes)) &&
                write_all(fd, part.data(), n_part) && fsync(fd) == 0;
            if (fd >= 0) {
                close(fd);
            }
            if (!ok) {
                LOG_ERR("probe: failed to write %s\n", ckpt_path.c_str());
                return 1;
            }
            printf("PROBE_CKPT_SAVE tokens=%d bytes=%zu get_ms=%.1f pos_min=%d pos_max=%d\n", n_ckpt, n_part, t_part,
                    llama_memory_seq_pos_min(llama_get_memory(ctx), 0), llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
            if (!prefill(ctx, tokens, n_ckpt, n_prefix, params.n_batch)) {
                LOG_ERR("probe: prefill failed\n");
                return 1;
            }
        }
        const double t_prefill = now_ms() - t0;

        t0 = now_ms();
        const size_t size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> state(size);
        const size_t got = llama_state_seq_get_data(ctx, state.data(), state.size(), 0);
        const double t_get = now_ms() - t0;
        if (got == 0) {
            LOG_ERR("probe: llama_state_seq_get_data failed\n");
            return 1;
        }

        t0 = now_ms();
        const int fd = open(state_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        const uint32_t hdr[2] = { PROBE_STATE_MAGIC, (uint32_t) n_prefix };
        const uint64_t nbytes = got;
        bool ok = fd >= 0 &&
            write_all(fd, hdr, sizeof(hdr)) &&
            write_all(fd, tokens.data(), sizeof(llama_token) * n_prefix) &&
            write_all(fd, &nbytes, sizeof(nbytes)) &&
            write_all(fd, state.data(), got);
        const double t_write = now_ms() - t0;
        t0 = now_ms();
        ok = ok && fsync(fd) == 0;
        const double t_fsync = now_ms() - t0;
        if (fd >= 0) {
            close(fd);
        }
        if (!ok) {
            LOG_ERR("probe: failed to write %s\n", state_path.c_str());
            return 1;
        }
        printf("PROBE_SAVE tokens=%d prefill_ms=%.1f state_bytes=%zu bytes_per_token=%.1f get_ms=%.1f write_ms=%.1f fsync_ms=%.1f disk_MBps=%.1f\n",
                n_prefix, t_prefill, got, (double) got / n_prefix, t_get, t_write, t_fsync,
                got / 1e6 / ((t_write + t_fsync) / 1e3));
        printf("PROBE_POS stage=saved pos_min=%d pos_max=%d\n",
                llama_memory_seq_pos_min(llama_get_memory(ctx), 0), llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
        llama_backend_free();
        return 0;
    }

    // eval
    int32_t n_have = 0;
    double  t_read = 0.0, t_set = 0.0;
    if (!state_path.empty()) {
        double t0 = now_ms();
        FILE * f = fopen(state_path.c_str(), "rb");
        uint32_t hdr[2] = { 0, 0 };
        uint64_t nbytes = 0;
        std::vector<llama_token> saved;
        std::vector<uint8_t>     state;
        bool ok = f && fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == PROBE_STATE_MAGIC;
        if (ok) {
            saved.resize(hdr[1]);
            ok = fread(saved.data(), sizeof(llama_token), saved.size(), f) == saved.size() &&
                 fread(&nbytes, sizeof(nbytes), 1, f) == 1;
        }
        if (ok) {
            state.resize(nbytes);
            ok = fread(state.data(), 1, nbytes, f) == nbytes;
        }
        if (f) {
            fclose(f);
        }
        t_read = now_ms() - t0;
        if (!ok || (int32_t) saved.size() > n_prefix || std::memcmp(saved.data(), tokens.data(), sizeof(llama_token) * saved.size()) != 0) {
            LOG_ERR("probe: state file unreadable or not a prefix of the text\n");
            return 1;
        }
        // a torn write: hand the library a short buffer
        const int64_t n_cut = env_int("PROBE_TRUNCATE", 0);
        if (n_cut > 0 && (size_t) n_cut < state.size()) {
            state.resize(state.size() - n_cut);
        }
        t0 = now_ms();
        if (llama_state_seq_set_data(ctx, state.data(), state.size(), 0) == 0) {
            printf("PROBE_RESTORE_REFUSED\n");
            return 2;
        }
        llama_synchronize(ctx);
        t_set  = now_ms() - t0;
        n_have = (int32_t) saved.size();
        printf("PROBE_RESTORE tokens=%d state_bytes=%zu read_ms=%.1f set_ms=%.1f\n", n_have, state.size(), t_read, t_set);
        printf("PROBE_POS stage=restored pos_min=%d pos_max=%d\n",
                llama_memory_seq_pos_min(llama_get_memory(ctx), 0), llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
    }

    int32_t n_front = n_prefix;
    int32_t n_split = -1; // a fresh run cuts its prefill where the saving run did
    if (n_diverge >= 0) {
        if ((int32_t) tokens.size() < alt_offset + n_suffix + n_eval) {
            LOG_ERR("probe: text too short for the divergent request\n");
            return 1;
        }
        std::vector<llama_token> seq(tokens.begin(), tokens.begin() + n_diverge);
        seq.insert(seq.end(), tokens.begin() + alt_offset, tokens.begin() + alt_offset + n_suffix + n_eval);
        tokens.swap(seq);
        n_front = n_diverge + n_suffix;
        n_split = n_ckpt > 0 ? n_ckpt : n_diverge;

        if (n_have > 0) {
            int32_t n_keep = n_diverge;
            if (!ckpt_path.empty()) {
                FILE * f = fopen(ckpt_path.c_str(), "rb");
                uint32_t hdr[2] = { 0, 0 };
                uint64_t nbytes = 0;
                std::vector<uint8_t> part;
                bool ok = f && fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == PROBE_STATE_MAGIC && fread(&nbytes, sizeof(nbytes), 1, f) == 1;
                if (ok) {
                    part.resize(nbytes);
                    ok = fread(part.data(), 1, nbytes, f) == nbytes;
                }
                if (f) {
                    fclose(f);
                }
                if (!ok || (int32_t) hdr[1] > n_diverge) {
                    LOG_ERR("probe: checkpoint unreadable or past the divergence\n");
                    return 1;
                }
                const double t1 = now_ms();
                const size_t n_set = llama_state_seq_set_data_ext(ctx, part.data(), part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (n_set != part.size()) {
                    printf("PROBE_CKPT_REFUSED set=%zu size=%zu\n", n_set, part.size());
                    return 2;
                }
                n_keep = (int32_t) hdr[1];
                printf("PROBE_CKPT_RESTORE tokens=%d bytes=%zu set_ms=%.1f\n", n_keep, part.size(), now_ms() - t1);
            }
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, -1)) {
                printf("PROBE_RM_REFUSED keep=%d\n", n_keep);
                return 3;
            }
            n_have = n_keep;
            printf("PROBE_POS stage=rewound pos_min=%d pos_max=%d\n",
                    llama_memory_seq_pos_min(llama_get_memory(ctx), 0), llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
        }
    }

    double t0 = now_ms();
    const bool has_frontier = n_have < n_front;
    if (has_frontier && n_have < n_split && !prefill(ctx, tokens, n_have, n_split, params.n_batch)) {
        LOG_ERR("probe: prefill failed\n");
        return 1;
    }
    if (has_frontier && !prefill(ctx, tokens, std::max(n_have, n_split), n_front, params.n_batch)) {
        LOG_ERR("probe: prefill failed\n");
        return 1;
    }
    const double t_prefill = now_ms() - t0;
    printf("PROBE_PREFILL tokens=%d prefill_ms=%.1f\n", n_front - n_have, t_prefill);

    FILE * out = out_path.empty() ? nullptr : fopen(out_path.c_str(), "wb");
    if (out) {
        const uint32_t hdr[4] = { PROBE_LOGITS_MAGIC, (uint32_t) n_vocab, (uint32_t) n_eval, has_frontier ? 1u : 0u };
        fwrite(hdr, sizeof(hdr), 1, out);
        if (has_frontier) {
            fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, out);
        }
    }

    t0 = now_ms();
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int32_t i = 0; i < n_eval; ++i) {
        common_batch_clear(batch);
        common_batch_add(batch, tokens[n_front + i], n_front + i, { 0 }, true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("probe: decode failed at %d\n", i);
            return 1;
        }
        if (out) {
            fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, out);
        }
    }
    llama_batch_free(batch);
    const double t_decode = now_ms() - t0;
    if (out) {
        fclose(out);
    }
    printf("PROBE_EVAL tokens=%d decode_ms=%.1f ready_ms=%.1f\n", n_eval, t_decode, t_read + t_set + t_prefill);

    llama_backend_free();
    return 0;
}
