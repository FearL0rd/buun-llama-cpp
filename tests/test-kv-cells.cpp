#include "llama-kv-cells.h"

#include <cstdio>
#include <stdexcept>

int main() {
    try {
        llama_kv_cells cells;
        cells.resize(1);
        const auto check = [&](const std::vector<llama_seq_id> & expected) {
            std::vector<llama_seq_id> actual;
            cells.seq_for_each(0, [&](llama_seq_id seq) { actual.push_back(seq); });
            if (actual != expected) { throw std::runtime_error("owner iteration differs"); }
        };
        check({});
        // Every singleton, including high IDs, must be visited exactly once.
        for (llama_seq_id seq = 0; seq < LLAMA_MAX_SEQ; ++seq) {
            cells.pos_set(0, 0);
            cells.seq_add(0, seq);
            check({seq});
            cells.seq_rm(0, seq);
            check({});
        }
        cells.pos_set(0, 0);
        for (llama_seq_id seq : {LLAMA_MAX_SEQ-1, 64, 0, 63}) { cells.seq_add(0, seq); }
        check({0, 63, 64, LLAMA_MAX_SEQ-1});
        cells.seq_rm(0, 64);
        check({0, 63, LLAMA_MAX_SEQ-1});
        cells.seq_rm(0, LLAMA_MAX_SEQ-1);
        check({0, 63});
        cells.seq_rm(0, 0);
        cells.seq_rm(0, 63);
        check({});
        std::puts("KV cell owner iteration PASS");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
