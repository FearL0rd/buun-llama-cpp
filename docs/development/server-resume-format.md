# Persistent server resume — format and install contract (P1)

Status: **contract, implemented through P2** (§10 items 1 to 5, with the
measured results and the limits of v1 there). Where the build differs from the
first design the text says "as built".
Companion of `server-resume-plan.md` (objective, P0 results, phases). Frozen
2026-09-20 against `exp/server-resume`. Line anchors are from that branch; use the
symbols when lines move. Independent review is deferred by the maintainer until a
working product exists (one review is planned before the VBR phase), so every
statement below is the author's reading of the code, not a reviewed result.

Scope of v1: fixed-type KV (f16, q8_0, turbo, TCQ), dense, hybrid-recurrent and
SWA/iSWA models, one entry per conversation, direct install into a slot. Images
and audio are stored on models where a media chunk of n cells takes n consecutive
positions (§4a). Out of scope and refused with a reason: media under M-RoPE,
dynamic VBR (P3), drafter state (never saved), adapter changes between producer
and consumer (P4).

## 1. Review of the real owners

What P0 and the P1 code reading established, and what the design takes from it.

| Owner | Fact | Consequence |
|---|---|---|
| Library sequence-state file (`llama-context.cpp`, file v3) | 24-byte header with the declared total size and one FNV-1a-64 over the whole payload; `llama_state_seq_file_snapshot_prepare` reads the whole file into host memory and hashes it before anything is installed; the writer buffers the whole payload, publishes by rename, no `fsync` | Integrity exists but is all-or-nothing: no appending, no partial read, no bounded staging, no durability. A resume entry cannot live in this container |
| Per-sequence state blob (`llama_kv_cache::state_write`) | `n_stream`; per stream `cell_count`, a meta block (per cell `pos`, seq ids, optional ext), then per layer the K rows and the V rows of all written cells, then a TCQ footer when a TCQ type is present. Rows are contiguous per layer. Cells are written in cell-index order, which is not guaranteed to be position order | A token range is expressible as the same layout restricted to the cells of that range. No new blob grammar is needed, only a range filter and an ordering rule |
| Single-sequence reader (`state_read_meta`) | Clears the destination sequence first, places all `cell_count` cells through `find_slot`, all-or-nothing; validates layer count, `v_trans`, per-layer type and row size, TCQ fingerprint | Installing a second chunk needs an append mode that does not clear the sequence. The structural checks stay the single authority for "does this blob fit this cache" |
| Composite memories (hybrid, iSWA, hybrid-iSWA and relatives) | Write the base part (full-attention KV) unless `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, then always the partial part (recurrent state, SWA cache) | The base part is position-addressable and chunkable. The partial part is a point-in-time image: it is valid only at the position where it was taken. A server context checkpoint is exactly such an image |
| Slot semantic envelope `BUUNSLOT` v2 (`server-context.cpp`) | 224-byte header: counts, next position, five SHA-256 digests (runtime identity, adapter identity, token digest, serialized tokens, logits), then the serialized `server_tokens`, then optional logits. The parser refuses any other version, header size or unknown flag bit with `format_mismatch` | Reused as the entry's token ledger at version 3. Older builds refuse a v3 envelope at the header |
| Slot runtime identity (`buun.server.slot-file-runtime-identity/v2`) | Hashes the build label, context sizes, slot layout and index next to the family digest and the KV settings; any difference reports `model_family_mismatch` | Resume needs its own key (§5) and its own reason codes (§8). The legacy identity is not changed |
| `SERVER_TASK_TYPE_SLOT_RESTORE` | Defers while the slot is processing; parses the envelope; installs state; on failure runs `mandatory_recovery_reset(restore_failure)`; on success invalidates the frontier record, runs the slot's recovery-reset bookkeeping (drops the destination's checkpoints and draft state), sets the ledger, publishes the slot to the retention observer. It installs no checkpoints | This is the establishment path the resume installer reuses for the slot. Checkpoints need a second, existing door (next row) |
| Host-cache restore (`server_prompt_cache::commit_restore_delivery`, `server_prompt_cache_mirror_restore_retention`) | Delivers a whole `server_prompt` (tokens and checkpoints) to the slot, then admits the checkpoints as a batch through the publish authority (`admit_live_checkpoints`) and attaches their release operations on the retention observer. Fail-closed: checkpoints that cannot be admitted are retired and dropped, the restore still stands | The owner door for imported checkpoints. Resume does not insert into `slot.prompt.checkpoints` by hand |
| Checkpoint validity (`checkpoint_frontier_is_current`) | A checkpoint is usable only if its computation frontier carries this process's execution identity (random per model load), the slot's current sequence epoch, the current adapter identity, and token/position counts equal to the checkpoint's | A checkpoint read from disk can never pass as saved. Import is a named provenance transition that re-stamps it (§7) |
| `SERVER_TASK_TYPE_SLOT_SAVE` | Mutates: a one-token target decode to align frontier logits, clears the draft sequence, resets the DFlash ring | Resume capture is a separate read-only path. It saves no logits |
| Shutdown (`server.cpp`) | The inference loop returns, then `clean_up()` destroys the context. HTTP threads are still alive in between. A second signal exits from the handler. The router force-kills a child after 10 s | Save hook sits between the two. Per-entry commit, most recently used entry first, so a forced kill loses the tail of the list, not the store |
| Sleep (`handle_sleeping_state`, `load_model`) | State is destroyed on sleep and the model reloaded on wake inside one process | Same save and restore calls; the first P2 integration test, no restart needed |
| Streaming handlers (`server-http.cpp`) | A handler parked in the chunked provider is released only by the SSE ping timer (30 s) or a client disconnect | P2 prerequisite: a shutdown flag in the three `should_stop` closures. Without it a supervisor's grace period is spent waiting, not saving |
| Durable I/O already in the tree | `llama-repack-cache.cpp`: `flock`, staged files, streamed hashing with `sync_write()` every 64 MiB, directory `fsync`, rename. `llama-vram-ledger.cpp`: 0700 directory with ownership check, stale-owner detection by pid and start time. `fs_get_cache_file` forbids subdirectories and `fs_create_directory_with_parents` creates 0755 | The store reuses these patterns behind its own root helper (0700 directories, 0600 files) on `fs_get_cache_directory()` |

Correction to the plan's P0 notes: slot files do have a payload checksum (the
library file header above). What is missing is a checksum on the raw state API and
any integrity unit smaller than the whole file.

## 2. Container: a directory of objects, manifest published last

```text
<fs_get_cache_directory()>/resume/<semantic-family-digest>/
    writer.lock
    entries/<entry-id>/
        commit               manifest; atomic rename makes the entry visible
        c-<p0>-<p1>-<gen>    base-state chunk for token positions [p0, p1)
        t-<pos>-<gen>        tail state (recurrent / SWA part) taken at <pos>
        tmp-*                staging; removed by the next writer
```

An entry is one conversation: one slot's sequence at one captured boundary.
`<entry-id>` is 32 lowercase hex digits drawn at random when the conversation is
first saved. `<gen>` is the manifest generation that wrote the object.

Why not one file with appended chunks, which is how the chunk idea was first
sketched: a single file needs its own extent allocator, a double commit record
with torn-write reasoning, and truncation to give space back. With one object per
file the filesystem is the allocator, `rename` is the commit, `unlink` returns
space at once, and the existing repack-cache code is the template. Chunk overhead
is unchanged (one 64-byte header and one block of rounding per ≈54 MB chunk on a
27B TCQ cache, under 0.01 %). A 200k-token conversation is about 50 chunk files.
A resume entry was never meant to be a portable single file (plan §6).

Why not the library sequence file with a new envelope version inside it: see the
first row of §1. The "header that declares a resume manifest" of plan §6 is
therefore the manifest's own magic and version. Builds that predate it cannot
open an entry at all: given a manifest by name they fail the library magic check.

### 2.1 Object file (`c-*`, `t-*`)

64-byte little-endian header, then the payload, nothing after it.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `BUUNRSMO` |
| 8 | 4 | object format version = 1 |
| 12 | 4 | header size = 64 |
| 16 | 4 | kind: 1 = base chunk, 2 = tail state |
| 20 | 4 | flags = 0 (unknown bits refuse the object) |
| 24 | 8 | payload bytes |
| 32 | 8 | XXH3-64 of the payload |
| 40 | 4 | `p0` (chunk) or position (tail state) |
| 44 | 4 | `p1` (chunk) or 0 |
| 48 | 8 | manifest generation that wrote it |
| 56 | 8 | XXH3-64 of bytes 0..55 |

The file size must equal 64 + payload bytes. XXH3-64 (vendored under
`vendor/hash`) detects corruption at memory-copy speed; SHA-256 stays the hash
for identities. Neither authenticates: anyone who can write the store can forge
it, as plan §6 already says.

- **Base chunk payload** — the per-sequence state blob of §1 for the base part
  only, restricted to cells with `p0 <= pos < p1`, **cells in ascending position
  order**. Every chunk carries its own TCQ footer. Chunk boundaries are multiples
  of `chunk_tokens` (4096 in v1, recorded per entry; a reader accepts any gapless
  tiling). The last chunk of an entry may be short.
- **Tail state payload** — the `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` blob of the
  sequence taken when the sequence ended at `pos`. A server context checkpoint's
  `data_tgt` is this blob already; the state at the captured boundary is the same
  kind of image taken at save time. Dense models have no tail states.

So for every model class an entry is *base chunks covering [0, N)* plus *tail
states at some positions ≤ N*, and "the saved state" and "a checkpoint" stop being
different things on disk:

| Model class | Restorable positions |
|---|---|
| Dense | any `p <= N` |
| Hybrid, SWA, hybrid-SWA | each `p` that has a tail state; `N` always has one |

### 2.2 Manifest (`commit`)

| Part | Content |
|---|---|
| Header, 64 bytes LE | magic `BUUNRSMM`, manifest format version = 1, header size, flags = 0, JSON bytes, ledger bytes, generation, XXH3-64 of JSON + ledger, XXH3-64 of the header |
| JSON document | records below; UTF-8, at most 1 MiB, parsed with a depth limit |
| Ledger | `BUUNSLOT` envelope **version 3**: the v2 layout, flag `RESUME_KEY_BOUND` (bit 2) set, `RUNTIME_FAMILY_BOUND` and `HAS_LOGITS` clear, the identity field holding the resume compatibility key (§5). The v3 parser accepts v2 only on the legacy route and v3 only on the resume route |

JSON records:

- `resume_key`, `family_digest`, `adapter_identity` (hex), duplicated from the
  ledger for listing without parsing it.
- `n_tokens`, `chunk_tokens`, `saved_unix_ms`, `last_used_unix_ms`, `slot_hint`.
- `chunks[]`: `p0`, `p1`, `gen`, `bytes`, `xxh3`, `prefix_digest`, `producer`.
  `prefix_digest` is SHA-256 (domain `buun.server.resume-prefix/v1`) over the
  ledger's token ids `[0, p1)`: a chunk is bound to its whole prefix, because KV
  depends on it. The first cell of a media chunk adds the chunk's cell count and
  content id (§4a); a text-only ledger hashes the ids alone.
- `tail_states[]`: `pos`, `pos_min`, `pos_max`, `n_tokens`, `gen`, `bytes`,
  `xxh3`, `prefix_digest`, `producer`, `role` (`frontier`, `turn`, `early`).
- `producers[]`: provenance only, never a gate — model name and file basename as
  loaded, projector file basename (`mmproj_file`, optional), weight quantization, build label, KV types, adapter labels, host
  time. `producer` fields index this table; a conversation continued by a second
  model has chunks from both.
- Object names are derived from the records, never stored, so a manifest cannot
  name a path.

Bounded decode limits, checked before any allocation: manifest file ≤ 64 MiB;
≤ 4096 chunks; ≤ 64 tail states; ≤ 16 producers; `n_tokens` ≤ 2^24; chunks tile
`[0, n_tokens)` exactly; every tail state satisfies `pos <= n_tokens` and
`pos_max + 1 == pos == n_tokens(record)`; every declared size equals the file's
size; the ledger's token count equals `n_tokens` and its next position equals
`n_tokens` (a slot whose positions are not the identity, for instance after a
context shift, is not saved in v1).

## 3. Library additions (implemented in P2)

The range writer and the append reader are the only format work in the library.
Both reuse `state_write_meta` / `state_write_data` and their readers.

```c
#define LLAMA_STATE_SEQ_RANGE_VERSION 1

// size / write the base part of seq_id for positions [p0, p1), cells in
// ascending position order. 0 on failure
size_t llama_state_seq_get_size_range(ctx, seq_id, p0, p1);
size_t llama_state_seq_get_data_range(ctx, dst, size, seq_id, p0, p1);

// append a range blob: does not clear the sequence; p0 must be the sequence's
// pos_max + 1 in the base cache (0 on an empty sequence); cells at
// pos >= p_limit are read and dropped. All-or-nothing per call: on failure the
// cells of this call are removed and the earlier ones stay. 0 on failure
size_t llama_state_seq_append_data(ctx, src, size, seq_id, p0, p1, p_limit);
```

Range blob: 16 bytes `{magic "gqsr", LLAMA_STATE_SEQ_RANGE_VERSION, p0, p1}`,
then `cell_count`, the meta block and the data block of §1 for one stream.

- **No `n_stream` word.** A range belongs to one sequence, so the blob is the
  same under unified and split KV and installs under either. Tested both ways on
  a dense, a hybrid and an SWA model.
- **No `BASE_ONLY` flag and no `flags` argument**, as first sketched: the range
  calls are base-only by definition. Composite memories forward them to their
  attention / base cache. A purely recurrent memory writes an empty payload
  (blob size 16 means "no base part"); memory classes without an
  implementation throw, which the API reports as 0.
- **Ranges are strict.** The writer refuses a sequence that does not hold
  exactly the positions `p0 .. p1-1`; the reader requires `cell_count == p1 - p0`,
  `pos[i] == p0 + i` and the blob's `p0`/`p1` to equal the caller's. That bounds
  `cell_count` before the batch reservation it feeds.
- Ascending position order is what lets `p_limit` cut a chunk by row count: the
  reader places the first `p_limit - p0` cells and skips the remaining rows of
  every layer block.
- The reader does not check for trailing bytes; the store's exact file size and
  checksum cover that.
- A blob layout change bumps `LLAMA_STATE_SEQ_RANGE_VERSION`, which is part of
  the resume key. With the build label out of the key this constant and the
  reader's structural checks are what stop a stale layout.
- Dynamic VBR keeps refusing (the writer shares `state_write`'s settle/refuse
  step); P3.
- `llama_memory_seq_pos_max` on a composite memory is the minimum over both
  parts, so it stays -1 until the partial part is installed. Code that tracks an
  install in progress counts chunks; it does not ask the memory.

Two findings that shape the server side:

- `PARTIAL_ONLY` is ignored by a plain KV cache: on a dense model it writes the
  full state. The server decides "has a partial part" itself:
  `llama_model_is_hybrid || llama_model_is_recurrent || llama_model_n_swa > 0`.
- The SWA partial blob is an ordinary KV-cache blob and carries `n_stream`. It
  does **not** install across stream layouts. On a model with SWA the stream
  count is therefore part of the resume key (§5). Recurrent partial blobs have
  no such word, so dense and hybrid entries move between slot counts.

One flag was added for the `frontier` tail state of an SWA model:

```c
// write every cell the sequence still holds in an SWA cache, not only the cells
// inside the attention window
#define LLAMA_STATE_SEQ_FLAGS_SWA_HELD_CELLS 4
```

The default SWA image holds exactly the window. The server's reuse check is more
conservative than the window (`pos_min` must lie below
`pos_next - n_swa`, minus one when the request brings no new token), so a restored
exact-window image fails it on the first request and falls back to a checkpoint,
reprocessing the turn since. The live cache passes the check because it still
holds older cells. With the flag the frontier image carries those cells and the
restored slot behaves as the live one did: measured on a 9k conversation, the
first request after a restart reuses all 9047 tokens. The flag is additive and
off by default; context checkpoints and the legacy slot files keep the
exact-window image.

## 4. Capture (read-only)

Runs on the inference thread after the loop has returned (shutdown) or before
`destroy()` (sleep). Skipped when the server is already sleeping. Never from a
signal handler or a destructor, and never after a failed startup.

Per nonempty slot, most recently used first:

1. `llama_synchronize`. Read the ledger (`slot.prompt.tokens`) as it is. A slot
   that was generating is saved "one behind": the sampled-but-undecoded token is
   simply not part of the entry. No decode, no draft-sequence change, no ring
   reset, no logits.
2. Skip with a reason: media under M-RoPE or without a content id (§4a), dynamic VBR, a Qwen4 QSA
   index (the index image is outside the partial state; unverified), positions
   that are not the identity, or a memory whose `pos_max + 1` differs from the
   ledger length on a model with a partial part (`frontier_inconsistent`). On a
   dense model extra KV positions are clipped by the range writer.
3. Decide reuse: the slot remembers the entry id it was restored from or last
   saved as. A chunk of that entry is kept when its `prefix_digest`, the resume
   key and the adapter identity still match **and the live cells of its range
   serialize to the object's size and checksum**. Everything after the first
   mismatch is rewritten. Equal tokens do not prove an equal state: a request
   with `cache_prompt: false`, a legacy slot restore or another conversation in
   the slot recompute the cells, and after a restore from another model of the
   family the recomputed cells are another model's. Reading the ranges back
   costs 18 ms on a 9k-token hybrid 4B save (two 71 MB chunks and the short
   last one; 135 ms against 117 ms). A tail state that the slot still holds
   (the frontier, a checkpoint) is compared the same way. The `early` tail has
   no live counterpart once it left the ring, so it is kept only while every
   chunk under its position, the entry's shorter last chunk included, is the
   live state: the pair then is the state the entry held when both were
   written, which is what a prefix restore installs. Measured: base model
   saves, a fine-tune restores and refills cold, its save holds the fine-tune's
   chunks and tails (the bytes of a fresh fine-tune save); the same restore
   followed by an ordinary turn, and a restart after it, keep the base model's
   chunks and `early` tail.

   Which entry is a different question from which bytes. The entry of a
   conversation is the one whose chunks, all but the last, lead the ledger: the
   slot's own, else one no slot holds. That is how a conversation that came
   back from the host cache, or was sent again after a start that had no slot
   for it, goes on in its own entry. The last chunk is excused because a client
   re-renders the last turn; the price is that a new conversation sharing all
   but the last chunk of an entry takes it over, which costs the other
   conversation at most one chunk and its tail. An entry of a single chunk
   therefore goes with its slot. A shared first chunk is not enough: a second
   agent with the same long system prompt gets its own entry and the first
   keeps its history (measured: 5000 shared tokens, 12k and 6.5k conversations
   through one slot of two; both restored, the first with its whole history).
   One exception saves writes without costing anything: when the slot's
   previous entry would not outlive this save under the retention rule below
   (one slot, no host cache), the new conversation takes that entry and keeps
   whatever leading chunks are still the live state.
4. Stream new objects: range blob into a staging buffer of one chunk (≈54 MB for
   a 27B TCQ cache, ≈210 MB at f16), XXH3 while writing, `sync_write` every
   64 MiB, `fsync`, rename from `tmp-*`.
5. Tail states per §6, copied from the slot's checkpoints (`data_tgt` only) and
   one `PARTIAL_ONLY` read of the live sequence for the frontier.
6. Write the manifest to `tmp-*`, `fsync`, rename over `commit`, `fsync` the
   directory. The entry is now the new generation.
7. Unlink objects the new manifest does not reference.

Disk bound. What a save makes obsolete is decided before its bytes are written
and goes first, free space or not. A save that replaces more than the last
chunk of its entry (an edited history, a state that is no longer the stored
bytes) removes `commit` first (`fsync`), unlinks the objects it does not keep,
then writes. A save that needs a new entry applies the retention rule below
first, counting the new entry. An interruption then loses the affected entry,
which is the agreed trade; entries that are neither replaced nor due under
retention are never touched. Only the append keeps the old commit until the new
one is published, and it duplicates at most one chunk and the changed tail
states; the preflight takes that away too when it would not fit. Measured as
the peak of allocated bytes, sampled every 2 ms through the save, 13k-token
dense entry of 794 MB: history edited one chunk in, peak 794 MB (was 1308);
another conversation in the only slot, peak 794 MB (was 1527, both snapshots).

Retention, with no knob: the namespace keeps at most `n_parallel` entries of
this server's resume key, the ones slots hold first, then newest `last_used`,
and at most `max(8, 4 × n_parallel)` entries overall; the rest are unlinked,
before a new entry is written and after the live slots are committed. Entries of another key (a different KV type, another stream count on
an SWA model) cannot be read by this server and say nothing about its
conversations, so only the overall bound ends them: a run under other settings
does not delete what the usual settings saved. The same holds for entries of
this key whose conversation is known to be without a slot: the ones that found
no slot at install (§7), and, when there is a host cache, the one a slot held
before another conversation took the slot. They count against the overall bound
only, so a start with fewer slots does not end the conversations it had no slot
for. An entry is not deleted merely because its
conversation is not in a slot at save time. With unified KV and several slots
this is what keeps the conversations that were evicted to the host cache during
the run: their entries from the previous start survive, stale by the turns made
since, until `--resume-host-cache` (P4) saves them properly.

One writer per family namespace: `flock` on `writer.lock`, held for the life of
the process. As built there is no stale-owner detection by pid and start time:
the kernel drops the lock with its owner, so a killed server leaves nothing to
detect. The pid written into the file is for a person looking at the directory.
A second server runs without persistence and says so. Retention also removes an
entry whose manifest is damaged; one of an unsupported version is kept, it may
belong to a newer build. On Windows the store opens as `store_unwritable`
(encoding and decoding are portable, the durable I/O is POSIX).

### 4a. Media

The ledger already carries media: `server_tokens::serialize()` writes each image
or audio chunk as a placeholder (cell count, position type, content id, no
pixels or embeddings), a few hundred bytes per chunk. The content id is the
SHA-256 of the file the client sent. The KV cells of a chunk are ordinary cells
and go into the token ranges like any other.

- **Positions.** A token range names cells by position and requires
  `pos[i] == p0 + i`. That holds where a chunk of n cells takes n consecutive
  positions (Gemma 3/4, SmolVLM, LLaVA-style). Under M-RoPE the cells of a chunk
  share positions: a slot with media on a model whose RoPE type is MROPE, IMROPE
  or VISION is skipped with `unsupported_positions`, at capture and at install.
  Text-only conversations on such a model are stored as before.
- **Identity.** Media cells are `LLAMA_TOKEN_NULL` in the ledger, so ids alone
  would make two images equal. The prefix digest adds, at the first cell of a
  chunk, the chunk's cell count and its content id. A chunk without an id is
  `unsupported_media`.
- **Boundaries.** A chunk boundary may fall inside a media chunk: objects are
  data, and the digest of a boundary inside an image already covers the image. A
  restore never stops inside one. A checkpoint whose frontier lies inside a
  media chunk is not stored as a tail state, a tail state there is no restore
  candidate, and on a dense model the cap of a smaller context, or the last
  complete chunk after a failure, moves down to the first cell of the chunk.
- **Projector.** The key binds whether a projector is loaded, not which one. A
  follow-up request that sends the same file reuses the stored cells whichever
  projector encoded them, the same kind of handoff as between two models of one
  family (§5). `mmproj_file` in the producer record says which it was.

Measured on the 3090: a three-turn conversation with two
images, one of them across the first chunk boundary, over two restarts against
the same conversation in one process. SmolVLM2-500M (dense, q8_0 KV) and Gemma 4
E2B (SWA, turbo3_tcq KV, below and above the window): all turns token-identical,
`cache_n` equal to the one-process run after each restart, so no image was
encoded again. A 4096-token context that ends inside the first image restores
the 4033 tokens before it. Two media conversations saved under `-np 2` and
restarted under `-np 1` with a host cache: one `installed_host`, both continue
identically. Qwen3.5-4B with a projector: media slots skipped with
`unsupported_positions`, a text-only conversation identical across restarts.
Before this change `--resume` with a projector loaded hit an assert at the first
save, with or without media in the slot.

## 5. Resume compatibility key

SHA-256, domain `buun.server.resume-compat/v1`, over:

| In the key | Why |
|---|---|
| Semantic family digest (v4) | structure and tokenizer; MTP head and pad token already unbound |
| Manifest, object and `LLAMA_STATE_SEQ_RANGE_VERSION` numbers, byte order | portability versions instead of the build label |
| `cache_type_k`, `cache_type_v` | the bytes are in that codec |
| RoPE / YaRN parameters, `n_ctx_orig_yarn`, group-attention `n`/`w` | K is stored rotated |
| Control vectors | state-affecting, like adapters |
| `swa_full` | changes which SWA cells exist |
| mmproj presence | a ledger with media needs a projector to be read; which projector is provenance, not a gate (§4a) |
| Stream count, on a model with SWA only (1 under unified KV, else `n_parallel`; 0 without SWA) | the SWA partial blob records it (§3). Base chunks and recurrent partial blobs do not, so a dense or hybrid conversation saved under `-np 2` continues under `-np 1`, measured bit-exact |

| Left out | Becomes |
|---|---|
| Build label (`llama_commit()`) | producer provenance |
| `n_ctx`, `n_ctx_seq`, slot count, slot index | install-time fit (§8) |
| Flash-attention type, `v_trans`, layer count, per-layer type and row size, TCQ codebook fingerprint | the state reader's structural checks |
| `no_fused_gdn`, `logits_all` | provenance / irrelevant |
| Weight values, quantization, file name | provenance; fine-tunes stay eligible by decision |

The adapter identity stays a separate digest and must be equal in v1. The
producer-to-consumer adapter transition of plan §2 is P4.

Documented property of family reuse (plan §2, measured in P0): the restored
history is a mixed-model history. The typical next token follows the consumer
model (median KLD to the consumer's own prefill 3–6× below the distance between
the two models under f16 KV, about 2× under TCQ, top-1 agreement 0.93–0.98);
strongly history-determined predictions, mostly in the first ≈50 tokens, can
follow the producer. This is kept, not gated. The manifest records the producer
so that logs never call such a restore an exact-model hit.

## 6. Checkpoint set and budget

Tail states saved per entry on a model with a partial part:

1. `frontier` at `N` — mandatory; it *is* the state. Without it only a dense
   model could be restored.
2. `turn` — mandatory when the slot holds one: the slot's newest context
   checkpoint. P0: the first request after a restart re-renders the last reply
   and rewinds behind `N` even with thinking off; without this companion a
   hybrid reprocesses the whole history and an SWA model has no valid rewind.
3. `early` — at most one: the earliest tail state the entry already holds whose
   `prefix_digest` still matches and whose chunks below it are still the live
   state (§4 step 3), otherwise the slot's oldest retained
   checkpoint. Once persisted it outlives the live ring, so a conversation that
   has been saved since its start keeps a checkpoint near its preamble. It is
   what a partial-prefix restore into a smaller context lands on.

Budget: tail states 2 and 3 only. Each costs the model's fixed partial size
(52.7 MB on a 4B hybrid, 156.9 MB on a 27B; 6 MiB on the measured SWA model),
install 4–12 ms. More than three is opt-in and not part of v1. Pinning a
checkpoint at the preamble boundary in the live ring would make `early`
reliable; that belongs to the checkpoint owners and is only noted here.

Partial SWA images are valid only on the base chunks of the same entry; they are
never shared between entries.

## 7. Install and the checkpoint import transition

Runs at the tail of `load_model`, after the slots exist and before readiness
(startup and wake). Failure of any entry degrades that conversation to a cold
start; it never fails the load and never leaves half a slot.

1. Take the namespace lock; enumerate `entries/*/commit` (bounded); parse
   header, JSON and ledger; check the resume key and adapter identity. Sort by
   `last_used`.
2. Map entries to slots: `slot_hint` if free, otherwise any free slot. More
   entries of this key than slots: with a host prompt cache (`--cache-ram`,
   fixed-type KV) the oldest go there first, one at a time through a slot that
   is still empty: installed as below, saved by the call that saves an idle
   slot, cleared (`installed_host`, oldest first so that the cache evicts it
   first). The next request of such a conversation finds it by the ordinary
   prefix match. The cache applies its own budget; a state it does not take is
   `host_cache_rejected`. Without a host cache they are `no_free_slot`. In all
   three cases the entry stays in the store (§4).
3. Choose the install position `p`: the largest restorable position (§2.1) with
   `p < n_ctx_seq` of the destination. `p == N` is a full install, `0 < p < N` a
   prefix install, none is `context_too_small`.
4. For each chunk with `p0 < p`: verify header, size and XXH3 while reading into
   the one-chunk staging buffer, then `llama_state_seq_append_data(...,
   p_limit = p)`. Then the tail state at `p`, if the model has a partial part,
   through `llama_state_seq_set_data_ext(PARTIAL_ONLY)`.
   If the cache runs out of cells part-way (unified KV shares them between
   slots), or an object fails its check: a dense model keeps the complete chunks
   it has; a model with a partial part clears the sequence and retries at the
   largest tail-state position inside what did fit, and so on down the tail
   states. The outcome names the first failure as its `why`. No rewind of
   installed state is involved either way.
5. Establish the slot exactly as `SLOT_RESTORE` does after a successful state
   install (frontier-record invalidation, recovery-reset bookkeeping, ledger
   truncated to `p`, fresh sequence epoch, retention publication), with no
   frontier logits. The next request decodes at least one token and gets its own.
   Speculative state: event `target_restored_without_draft`; drafters rebind as
   in the `--mmproj-gpu-swap` path.
6. Select the newest tail states with `pos < p` within the configured checkpoint
   budget, then import them chronologically as context checkpoints — transition
   `resume_import`. This retains the recent turn before optional early history:
   - new `common_prompt_checkpoint` with `n_tokens`, `pos_min`, `pos_max` and
     `data_tgt` from the record; `data_dft`, `data_qsa`, `accel` empty; VBR
     epochs 0;
   - computation frontier filled exactly as the checkpoint creation site fills it
     (inline there today; P2 factors that into one helper used by both), so it
     carries this process's execution identity, the slot's new sequence epoch,
     the current adapter identity, `token_count = n_tokens` and
     `next_position = pos_max + 1`;
   - cache-family binding of the slot with the producer recorded, never relabeled
     as computed here;
   - admitted through the creation/host-restore door (retention publish,
     `admit_live_checkpoints`, release operations). Fail-closed: a checkpoint
     that is not admitted is dropped and reported; the install stands.
7. Any failure after step 4 began: `mandatory_recovery_reset(restore_failure)`
   on the slot, reason logged, next entry.

Rewinds of restored state happen only through the server's `pos_min` guard or a
restored checkpoint. Resume code never calls `llama_memory_seq_rm` on SWA or
recurrent state (P0: it succeeds and the output is silently wrong). It does not
need to: `p_limit` cuts the last chunk while it is read.

Open P2 check: a checkpoint restore with empty `data_dft` while a drafter is
active must take the target-only speculative transition.

## 8. Outcomes and reason codes

One log line and one `/slots` field per entry. The string
`model_family_mismatch` is never produced by resume.

| Outcome | Meaning |
|---|---|
| `installed_full` | `p == N` |
| `installed_prefix` | `0 < p < N`; reports `p`, `N` and why (`context_smaller`, `cells_exhausted`) |
| `installed_host` | no slot for the entry: restored through an empty slot into the host prompt cache; `restored` says full or prefix |
| `skipped` | destination untouched |
| `failed` | destination reset to empty |

| Reason | When |
|---|---|
| `resume_key_mismatch` | key differs (KV type, RoPE, versions …); the differing field is not recoverable from a hash, so the log prints the producer's provenance next to the current settings |
| `adapter_mismatch` | adapter identity differs |
| `format_unsupported` | unknown manifest/object/envelope version or flag |
| `manifest_corrupt`, `ledger_invalid` | checksum, bounds or token validation |
| `object_missing`, `object_size_mismatch`, `object_checksum_mismatch` | per object; a failed chunk ends the usable prefix at its `p0` if a restorable position remains below it, else `failed` |
| `companion_missing` | a model with a partial part and no tail state at or below the fit position |
| `context_too_small` | no restorable position fits |
| `no_free_slot` | more entries than slots and no host prompt cache; the entry is kept |
| `host_cache_rejected` | more entries than slots and the host prompt cache did not take the state (its size limit); the entry is kept |
| `entry_in_use` | the restore action named an entry another slot was restored from or saved as; two slots never write one entry |
| `state_rejected` | the library refused a blob (type, shape, TCQ fingerprint) |
| `unsupported_media` | a media chunk without a content id (§4a); capture-side skip, logged at save |
| `unsupported_positions` | cells that do not take consecutive positions: media under M-RoPE (§4a). At capture, and at install when the ledger has media and the loaded model shares positions |
| `unsupported_vbr`, `unsupported_qsa`, `frontier_inconsistent` | capture-side skips, logged at save |
| `store_locked`, `store_unwritable`, `no_space`, `io_error` | store level; the server runs without persistence |
| `checkpoints_dropped=<n>` | warning attached to an `installed_*` outcome |
| `provenance_limit` | capture skipped before disk mutation because all 16 producer records are occupied and the current producer is new; the previous entry is retained rather than misattributing new bytes |

**No silent downgrade.** An entry that fails a check is skipped or failed with
its reason. It is never installed by the legacy slot-file route, and the legacy
route never reads a manifest. Empty slots are skipped without a line.

`/slots/<id>?action=restore` routes by what it is given: a `filename` is a legacy
library file under `--slot-save-path` and takes the unchanged legacy route; a
`resume_entry` id is resolved inside the resume namespace and takes steps 3–7
above. That endpoint is the restart-free test entry for the installer.

As built: `POST /slots/<id>?action=restore` with `{"resume_entry": "<32 hex>"}`
is accepted whenever the server runs with `--resume`, with or without
`--slot-save-path`. A body carrying both `resume_entry` and `filename`, or an id
that is not 32 hex digits, is a 400. The manifest is read before the slot is
cleared. The response is the usual restore result plus a `resume` object
holding the outcome of this section; a missing entry is `skipped` /
`object_missing` with the destination unchanged. Paths of the
host appear in the server log only, never in the response.

Explicit slot `erase` ends the conversation on disk as well. The entry removed
is the one of the conversation in the slot by the rule of §4 step 3, found at
the time of the erase, not the id the slot saved into last: after another
conversation took the slot that id still names the previous one. Measured:
save, restart, erase, restart installs nothing; A saved and restored, B takes
the slot and is erased unsaved, A's entry survives and installs at the next
start; with both saved, erasing B's slot leaves A's entry alone. Within the
last-chunk tradeoff an entry of a single chunk goes with its slot here too.

## 9. Install route: recommendation and alternative

Recommended for v1, and what §7 describes: **direct slot install**. It reuses an
establishment path that P0 exercised across restart and sleep/wake, streams
chunks straight into the cache with one chunk of staging, and supports prefix
installs.

Alternative kept open: install entries as **host prompt-cache entries** and let
ordinary prefix matching pull them into whichever slot a request lands on. It
gets slot mapping, fewer-slots handling and checkpoint delivery from existing
code, and pinned requests now see host entries. It needs `--cache-ram`, holds
every entry in host memory, and loses chunk-level reads unless host entries
learn chunk lists. It is the natural route for `--resume-host-cache` in P4, where
the saved roots are host entries anyway.

As built, the two meet for the entries that get no slot (§7 step 2): a host
entry is a whole sequence image, a resume entry is range blobs and tail states,
so the entry is installed into an empty slot and saved from there by the
existing call. No second reader, and the host cache keeps its own format, budget
and eviction. Measured on the 3090: 5 ms (dense 0.6B, 647 tokens) and 54 ms
(hybrid 4B) per entry at start; the conversation's next turn reuses all of its
tokens from the host cache and is token-identical to the one-process run.

Open question for the maintainer: with `--kv-unified` and several slots, each
launch moves the other idle slots to the host cache, so at shutdown usually one
conversation is live. The retention rule of §4 keeps the others' previous
entries, but they are stale by the turns made since the last start. Exact
coverage for that configuration needs P4, or an earlier decision to save host
entries that were live slots.

## 10. What P2 builds from this

1. **Done.** Shutdown flag in the `should_stop` closures. SIGTERM mid-prefill on
   the 27B at 32k exits in 0.7–2.8 s (was 30 s), mid-decode in 0.3 s. Residual: a
   stream attached to a conversation pipe (`X-Conversation-Id`) polls only the
   pipe's cancel flag and still waits for its ping.
2. **Done.** Library: range writer, append reader, version constant; test 11 of
   `test-save-load-state` (three blobs plus the partial state equal the whole
   state byte for byte, cross-layout install, `p_limit`, refusals).
3. **Done.** Store (`tools/server/server-resume-store`): 0700/0600 root helper,
   lock, object and manifest I/O, bounded parsing (depth, counts, integer
   ranges), fault seams (write stopped at half a payload, `ENOSPC` before an
   object is published, kill between objects and manifest).
   `test-server-resume-store`: after each stopped write a reopened store lists
   the previous generation and every object of it verifies.
4. **Done.** Capture and install in the server, `--resume`, `--resume-path`
   (default: the llama.cpp cache directory), the `resume_entry` restore action,
   reason codes, the held-cells flag of §3. Self-tests of the v3 envelope in
   `test-server-prompt-cache`: ledger round trip, logits refused, the legacy and
   the resume route refuse each other's envelopes, a changed key is
   `resume_key_mismatch`.
5. **Done**, single RTX 3090, TCQ 3-bit KV, 9k-token conversation, greedy
   continuations compared token for token with the same conversation in one
   process:

   | | dense 0.6B | hybrid 4B | SWA (Gemma E2B) | hybrid 27B + MTP |
   |---|---|---|---|---|
   | entry size | 211 MB | 218 MB, 3 tail states | 16.5 MB | 582 MB |
   | save at shutdown | 0.19 s | 0.20 s | 0.09 s | 0.43 s |
   | install at startup | 54 ms | 70 ms | 4.6 ms | 184 ms (prefill: 9.6 s) |
   | restart, sleep/wake, restore action | identical | identical | see below | identical |
   | rewind | identical | identical | identical | |
   | smaller context | prefix at 4607 | `context_too_small` (no tail state fits) | | |
   | `-np 2` → `-np 1` | identical | identical | `resume_key_mismatch` by design | |
   | second save after one more turn | 28 MB | 113 MB | | 319 MB |

   A save killed part-way (SIGKILL 0.10–0.15 s after SIGTERM) leaves no entry on
   a first save and the previous generation on a later one; the next start
   installs that generation, continues identically and sweeps the leftovers.

   With MTP the restored turn is text-identical; draft acceptance of that turn
   is 0.89 against 0.96 in one process (code prompt, thinking off), because no
   drafter state is saved and the draft context refills as the turn runs.

Properties and limits of v1, as measured:

- **A restored SWA conversation above the window is row-exact, not
  logit-exact.** `tests/test-state-restore-swa-exact.cpp` (Gemma-4 E2B, window
  512, f16 and q8_0 KV) reads every K/V row back by position from both caches.
  After a whole-sequence restore and after the resume route (range append plus
  the held-cells tail), across a wrapped ring of 2185 tokens, each row equals
  the live row bit for bit. Two live runs give bit-equal logits, so nothing in
  the comparison is run-to-run noise. The logits of the same eight next tokens
  still differ from live (max KLD 7e-9, top-1 8/8), and the control shows why:
  below the window, where a restore is otherwise logit-exact, the same exact
  rows placed 200 cells further along differ by as much (max KLD 2e-6, top-1
  8/8). A restore compacts the sequence to the front of the cache while the
  live ring has wrapped, so the rows sit at other cell indices and the
  attention kernels reduce over them in another order and padding. Dense and
  hybrid continuations were identical in the measured cases, not proof for
  every restore path.
- A prefix install on a model with a partial part lands only on a tail-state
  position, so it needs an `early` or `turn` state inside the smaller context.
- Every save of a hybrid conversation rewrites the frontier and turn states
  (two partial images), whatever the number of new tokens.
- A restart with fewer slots keeps the entries it had no slot for (§4) and, with
  `--cache-ram`, serves them from the host cache (§7). What is saved of such a
  conversation is still only what a slot held at a save: turns made while it
  lived in the host cache alone reach the store when it is next in a slot at a
  save (P4 closes that).
- Conversations that live only in the host prompt cache at shutdown are not
  saved (P4).
- Slots with media under M-RoPE, dynamic VBR and a QSA index are skipped with a
  reason. Not measured for media: audio chunks, a chunk without a content id,
  and a turn made after a prefix restore that ended at an image.
