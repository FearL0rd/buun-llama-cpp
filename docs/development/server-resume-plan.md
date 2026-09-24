# Persistent server resume — design and implementation plan

Status: **P0 to P2 implemented on `exp/server-resume` (fixed-type KV, `--resume`);
P3 (dynamic VBR: every slot of a dense or hybrid cache, the most recent
conversation of an iSWA cache) implemented and not yet reviewed; P4 open; the blockers of the independent review (§11) are closed**. Contract, results and the limits
of v1: `server-resume-format.md`.
Created 2026-09-18 against master `ed774445c`.
This is an engineering plan, not documentation of an available feature.

## 1. Objective and agreed product decisions

Avoid repeating expensive prefill after a server restart or model swap by
persisting reusable conversation state. Reuse existing cache/checkpoint owners,
serializers, and restore transactions rather than introduce a second cache system.

Decisions agreed with the user:

- `--resume` is a Boolean flag, off by default. No path argument is required.
- Use the existing llama.cpp cache directory, including `LLAMA_CACHE` overrides.
- Default persistence includes **live slots and their required companions**.
  Idle slots still holding a prompt are live slots.
- Revised 2026-09-20 (user requirement: `--resume` gives back the server that
  was stopped): the host prompt cache is persisted too, by default. Under
  dynamic VBR every hosted conversation is saved at shutdown as a whole-pool
  image and comes back through a slot at load, at seconds each way (§9, P3
  stage 2). On the fixed route the shutdown saves the slots only: an entry a
  conversation left on disk when it left its slot stays there and comes back,
  a host-only conversation that was never saved is lost. Saving those is the
  fixed-route item still open in P4. The `--resume-host-cache` opt-in first
  proposed here was never built.
- Select snapshots by **semantic family**, not weight identity. Structurally
  compatible fine-tunes are intentionally eligible for experiments.
- Measure snapshot size and save/restore costs before choosing additional
  checkpoint retention, compression, or speculative-state persistence.
- Disk space takes priority over preserving an old cache. No backup generation
  by default: interrupted replacement may lose the affected entries and fall
  back to prefill. Independently valid old and new entries should remain usable.

The feature restores reusable model state, not a running process. It does not
resurrect HTTP connections, queued requests, tools, or automatically continue an
interrupted answer. Clients send their next request normally; ordinary token
prefix matching decides what can be reused. It does not reconstruct a chat UI.

Initial implementation target: `llama-server`. CLI integration is a later consumer
of the same storage layer, not part of the first server acceptance gate.

## 2. Meaning of family-compatible reuse

Use `llama_model_semantic_family_digest()` for the family key. It already binds
effective state-producing structure and tokenizer semantics while excluding
weight values, quantization, filenames, paths, and timestamps. Do not replace it
with a smaller test of layer count and head dimensions.

Family compatibility is **not numerical equivalence**. K/V and recurrent state
computed by fine-tune A generally differ from a fresh prefill under fine-tune B.
Reusing them is an intentional mixed-model history, also relevant to switching
LoRAs or weight quantizations. Measure this separately from same-model resume.

Measured in P0 (4B hybrid, stock vs. a fine-tune, identical prefill by one model
loaded by the other, four text windows, 4k and 16k histories) and kept as
documented behaviour, not gated: the typical next token follows the consumer
model — median KLD to the consumer's own prefill is 3–6× below the distance
between the two models under f16 KV and about 2× under turbo3_tcq, top-1
agreement 0.93–0.98 — while strongly history-determined predictions, mostly in
the first ≈50 tokens, can follow the model that produced the KV. Means are
tail-driven (2 of 24 cells exceed the model distance), so quote medians and
tails. The manifest records the producer.

Separate the following identities:

| Identity | Purpose |
|---|---|
| Semantic family | Find eligible snapshots; independent of source weight values |
| Saved-state representation | Validate codecs, codebooks/taps, layouts, state schemas, positions, and required companions |
| Execution configuration | Validate effective RoPE/context/window settings and other state-affecting options |
| Producer provenance | Record source model, adapters/scales, quantization and build for diagnostics and experiments; not a base-weight equality gate |
| Prompt/media identity | Establish actual reusable prefix using token and media identity, not merely family membership |

Do not claim weight identity from a filename, a model label, or matching tensor
shapes. Provenance can be unknown; logs must not then claim an exact-model hit.
No mandatory full-model hashing pass is proposed for family-based selection.

Existing LoRA identity checks in ordinary in-process caching remain intact.
If loading an A-produced snapshot for current adapter B, use an explicit resume
transition that records the producer change and installs the current runtime
ownership. Never disable ordinary adapter checks globally to make this work.
Keep original provenance on derived checkpoints/artifacts rather than silently
relabeling the entire history as freshly computed by B.

An incompatible tokenizer, state layout, RoPE policy, or media representation
still rejects restore even when the model names look related. Family matching
must never bypass representation or import validation.

## 3. Existing code to reuse

Anchors below are from `ed774445c`; use the symbols when lines move.

| Owner / file | Relevant seam and implication |
|---|---|
| `common/common.cpp:1120` | `fs_get_cache_directory()`: `LLAMA_CACHE`, otherwise platform cache root; on Linux XDG or `~/.cache/llama.cpp` |
| `common/common.cpp:1392` | `common_moe_cache_profile_file()`: existing per-family heatmap persistence, not persisted expert weight placement |
| `src/llama-model.cpp:4214`, `include/llama.h:721` | `llama_model_semantic_family_digest()`: existing family compatibility authority |
| `tools/server/server-context.cpp:239` | `server_slot_runtime_identity`: current slot-file identity also binds build revision and runtime configuration; do not assume it is a portable resume format |
| `tools/server/server-context.cpp:405` | `server_slot_envelope`: token ledger, adapter identity, positions and optional frontier logits |
| `include/llama.h:1090` | Full/sequence state APIs; useful fixed-state carriers, not complete server persistence |
| `tools/server/server-context.cpp:14610` | `SERVER_TASK_TYPE_SLOT_SAVE`: current idle-slot save and frontier alignment |
| `tools/server/server-context.cpp:15283` | `SERVER_TASK_TYPE_SLOT_RESTORE`: existing restore establishment and validation |
| `tools/server/server-context.cpp:7635` | Dynamic VBR disables ordinary slot files; bypassing this guard is not a resume implementation |
| `tools/server/server-context.cpp:9698` | `try_automatic_vbr_restore()`: existing artifact admission/import/restore path |
| `tools/server/server-prompt-cache-payload.{h,cpp}` | Fixed target/draft payloads, immutable VBR package leases, variant sets and allocation accounting |
| `src/llama-vbr-artifact.{h,cpp}` | Bounded stream encode/decode, representation metadata, payload verification, companion payloads |
| `src/llama-vbr-artifact-{capture,stage,adopt,catalog}.*` | Existing capture, staged import, publication and lifetime owners |
| `tools/server/server-context.cpp:5132,6192` | `destroy()` and `handle_sleeping_state()`: state must be saved before teardown, including explicit sleep/unload |
| `tools/server/server.cpp:520,573` | Shutdown notification and inference-loop return; avoid disk/GPU work inside a signal handler |

Relevant tests include `test-save-load-state`, `test-state-restore-fragmented`,
`test-recurrent-state-rollback`, `test-server-prompt-cache`, and the
`test-vbr-artifact*` family. Extend their real seams; do not duplicate their
serializers in test fixtures.

Implementation shape: a small server resume coordinator plus a durable object
store, calling existing owners for capture/import. Generic byte storage must not
learn model-specific recurrent, VBR, or speculative-decoding semantics.

## 4. Snapshot contents and default policy

| State | Default policy |
|---|---|
| Live-slot token/media ledger, positions, family/provenance | Save; required to match future requests correctly |
| Occupied attention KV | Save in its existing representation, never reserved empty capacity |
| Current recurrent state / retained SWA state | Save where required by the model |
| VBR layer-side tiers, domains, required stash/companions | Save through validated artifacts; do not expand quantized KV to F16 |
| Boundary rollback checkpoint needed to establish a usable frontier | Save as a required companion |
| Additional historical checkpoints | Select by measured reuse benefit; initially exclude those proven optional |
| MTP/DFlash target companions and draft state | Save what cannot be cheaply and correctly rebuilt; policy determined by gates |
| Host prompt-cache history | Save (revised 2026-09-20): conversations without a slot go through a slot at save and at load, within the entry bound |
| Allocator slack, scratch, graphs, streams, pointers, locks, leases | Never serialize as executable process state |
| HTTP requests, sampler/RNG state, tool invocations | Not resumed; new requests establish their own sampling settings |

Default saved roots are all nonempty slots, active or idle, at the captured
boundary, then the host cache's conversations that no slot holds a copy of,
newest first, up to the entry bound (§6). A hosted conversation is saved as a
slot's is, never as a second copy of shared live data.

Checkpoint count is not a safe proxy for optionality. A hybrid checkpoint or SWA
window may be necessary to reuse a prefix shorter than the saved frontier. Removing
it can turn a successful load into cold replay on the first real request. Preserve
required dependency closures and measure the reuse coverage lost by pruning.

Do not claim that persisting the existing final SWA window restores arbitrary
older tokens. Keep the existing retained-window and prefix-coverage checks.

### SWA / iSWA models (measured in P0, Gemma-4 E2B, `n_swa` = 512)

How the live server avoids cold prefill, and therefore what a snapshot must carry:

- The iSWA memory is two caches. The base cache (full-attention layers) holds every
  position and is never pruned. The SWA cache is sized `n_swa + n_ubatch` (padded)
  and pruned lazily — a masked cell is only overwritten when its slot is needed — so
  a live sequence usually retains about `n_swa + n_ubatch` positions, not `n_swa`.
- Prefix reuse requires `pos_min < pos_next - n_swa` (the server's `pos_min_thold`
  test is `>=`). The lazily retained cells are what let a plain append pass it.
- A rewind or mid-history edit falls back to context checkpoints. For iSWA these are
  partial (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`) images of the SWA cache alone —
  6 MiB each here, spanning the retained cells — and are valid only on top of the
  full base cache, which still holds the older positions. The server restores one
  and re-decodes from its `pos_max` to the divergence point.

Consequence for the per-sequence state writer: it filters cells through the SWA mask
(`state_write_includes_cell`), keeping only `pos_max - p0 < n_swa`. A restored
sequence therefore has `pos_min == pos_next - n_swa == pos_min_thold` by
construction and always fails the reuse test. Measured: after a slot-file restore a
pure append (LCP = cached, rewind 0) reprocessed the full 8k history, because the
new process had no checkpoint to fall back to.

Policy:

1. **Required:** save the slot's context checkpoints as companions of the sequence
   state (same rule as hybrid models). They are already self-describing byte images
   with `pos_min` / `pos_max` / `n_tokens`. A pure append then costs a restore of the
   newest checkpoint plus a re-decode of the tokens after it; rewinds resolve as they
   do in a live process. Partial checkpoints depend on the base cache of the same
   snapshot and must be published in the same dependency closure.
2. **Optional optimisation:** let a resume capture write the *retained* SWA cells
   rather than the masked window (the whole-cache path, `seq_id == -1`, already writes
   every cell). Masked cells are inert once restored, the cost is about `n_ubatch`
   extra cells, and a pure append then needs no checkpoint restore at all. This cannot
   replace (1): it does nothing for rewinds.

Do not change the `>=` threshold or the retention rule to make resume pass; those
belong to the checkpoint owners. Not yet tested: either policy after a restart, and
a Gemma host-cache rotation (host entries carry their checkpoints with the state).

### Frontier establishment and changed weights

Never blindly use saved next-token logits in family mode. They can be stale
after switching fine-tunes even if their byte envelope validates.

Proposed route: preserve a validated pre-frontier boundary plus the token/media
information needed to replay a bounded suffix under the loaded model. That replay
refreshes the prediction boundary while earlier cached history remains approximate.
It does **not** make cross-model history equivalent to fresh prefill.

For hybrid models, deleting one attention row is not enough: recurrent state
must also roll back to the corresponding point. For media, one token may not
describe a replayable boundary. For SWA, the necessary window must exist.

Phase 0 must establish the minimal valid boundary carrier for each family. If a
bounded suffix cannot be replayed safely, preserve more state or explicitly cold
replay that slot; do not silently pass through old logits. Any live bookkeeping
mutation needed for capture must use existing recovery/rollback rules.

### Speculative state

Separate the authoritative accepted-token frontier from unverified drafts.
Quiesce/cancel pending proposals before capture. No unaccepted token becomes part
of a durable prefix.

MTP may depend on recurrent backup/carry state. DFlash/DFlash2 may need target
hidden-state tape/ring data that cannot be reconstructed from attention KV alone.
Determine exact dependencies before dropping them for size. Rebuild into fresh
runtime contexts and invalidate stale proposal probabilities, logits and controller
samples. Reset adaptive-depth history unless measurements justify retaining it.

Different or missing drafters should not invalidate otherwise usable target state
if a safe target-only path exists. DSpark requires a separately measured path;
do not infer support from MTP/DFlash. Report any fallback and its first-request cost.

## 5. Lifecycle and transaction model

### Saving

1. Stop admitting new inference work and notify the inference owner to quiesce.
2. Reach a completed decode/prefill boundary; drain required device work. Do not
   wait for an arbitrarily long answer to finish just to snapshot it.
3. Freeze the saved-root inventory and obtain stable capture leases.
4. For each independently restorable entry, reuse unchanged payloads. Before
   reclaiming or overwriting bytes needed by its old version, durably invalidate
   that entry's old commit record. Do not reserve a second complete snapshot.
5. Stream changed payloads with bounded staging; validate lengths/digests and
   flush the complete dependency set for that entry.
6. Atomically publish the entry's small commit record, including directory
   durability where required. Reclaim unreferenced payloads; proceed to the next
   entry. Failure of one entry must not erase unrelated completed entries.
7. Release leases and continue teardown.

Signal handlers only request shutdown. Capture, CUDA/HIP synchronization and disk
I/O run in normal control flow while cache owners still exist. Destructors must
not perform a second save after explicit save/unload. Failed startup must never
overwrite a healthy snapshot with empty or partially restored state.

SIGTERM/Ctrl-C and explicit model unload/sleep are intended save boundaries.
SIGKILL, power failure and fatal GPU loss cannot guarantee a new snapshot.
Replacement may lose the entries being changed; unchanged old entries and fully
published new entries remain eligible independently. A forced second interrupt
may abandon current writes. Periodic/background snapshots are deferred, not implied.

For llama-swap or another supervisor, qualify the actual stop/start lifecycle and
allowed shutdown grace period. Do not assume its default timeout is long enough.

### Loading

1. Load the model, resolve its family and actual execution configuration, and fit
   the new process normally. Do not blindly reserve the old process's budget.
2. Enumerate bounded entry commit records and select compatible complete entries.
   They may come from different shutdowns; validate each dependency closure before
   mutating live state. An inventory index is a hint, not an all-or-nothing commit.
3. Use existing staged imports and allocate fresh process-local ownership.
4. Map saved slots into available slots, preserving token/position and shared-state
   relationships. Slot IDs are placement hints, not external conversation identity.
5. Establish usable frontiers and initialize compatible speculative contexts.
6. Publish usable slots/cache entries and advertise readiness. Log what restored,
   what was skipped, and what needs replay. Do not advertise a hit before commit.

The next request still passes normal token/media prefix checks. Family equality
must not cause one agent's conversation to match another's different prompt.

Restore errors must leave an empty/valid destination or invoke the existing
mandatory recovery reset. A corrupt/incomplete entry is skipped with cold replay
for that prefix; a corrupt shared dependency invalidates every entry depending on
it. Never publish half a slot. Older valid entries remain subject to normal prefix
matching; being older does not make different tokens a match.

New context sizes, slot counts, GPU splits, KV codec settings and smaller memory
budgets require explicit compatibility/admission decisions. Reuse an existing
validated projection/transcode route if available; otherwise skip with a reason.
Do not promise arbitrary cross-backend or cross-build binary state portability.

## 6. Storage layout, safety, and concurrency

Namespace (schema frozen in `server-resume-format.md`):

```text
<fs_get_cache_directory()>/resume/<semantic-family-digest>/
    writer.lock
    entries/<entry-id>/commit              manifest, published last
    entries/<entry-id>/c-<p0>-<p1>-<gen>   base-state chunk, token range
    entries/<entry-id>/t-<pos>-<gen>       recurrent/SWA state at a position
```

Each independently published entry record references payload objects and carries
format versions, compatibility metadata, producer provenance, token ledgers,
checkpoint dependencies, and sizes/digests. Entry kinds include live-slot roots
and optional retained-prefix roots. A small commit record distinguishes completed
entries from interrupted replacement. A content-addressed object pool shared
between entries (`objects/<content-digest>`) is deferred: v1 stores a prefix that
several conversations share once per entry.

Files are not independent just because they are separate. A slot's KV, recurrent
state, token ledger, and required checkpoint/speculative companions must describe
one consistent captured boundary. Bind that closure in one entry commit. Never
assemble a slot from old recurrent state and new KV after an interrupted save.
Conversely, separate valid slots/prefixes need not have the same shutdown epoch.

Deduplicate identical immutable payloads. Prefer existing VBR artifact unit IDs
and ownership over rehashing/reserializing equivalent representations. For fixed
state, first prove occupied-state capture before adding shared-prefix factoring.
Different encodings of the same logical prefix must not be incorrectly merged.
Choose bounded units so appending a few rows does not rewrite a whole layer; first
measure what existing artifact units allow without unsafe format surgery.

Never overwrite a shared object while any valid committed entry still references
it. Reuse unchanged objects; reclaim changed objects only after their old references
are durably invalidated. If an object spans several roots, schedule that dependency
group explicitly or skip its update for lack of space. Prefer a smaller failure
domain; do not silently invalidate every slot to update one prefix. Reuse freed
storage without violating content identity, and remove orphan staging data on restart.

One cooperating writer per family namespace initially, using an OS-backed lock
with stale-process recovery, not just a PID file. If occupied, leave that store
untouched and run without persistence with a clear warning. Read/GC/write ordering
must prevent removing an object still in use by a restore. Independent same-family
server namespaces are a later extension if needed; never last-writer-wins silently.

Use private directory/file permissions (0700/0600 on POSIX and appropriate Windows
ACL policy). Snapshots reveal conversation content and are not encrypted by default.
Validate relative paths, lengths, integer arithmetic, section counts and digests;
never trust saved pointers or let a manifest direct writes outside its namespace.
Digest checks detect corruption, not malicious rewriting by someone who can alter
the entire store. Do not introduce an executable/pickle-style serialization format.

Preflight the estimated final size, incremental writes and bounded staging space
separately. Target approximately one retained inventory plus staging, not two
full snapshots. Peak occupancy and bytes written are distinct measurements: an
old 8-GiB snapshot plus an 8-GiB replacement means 16 GiB occupied but only 8 GiB
newly written. Neither duplication nor a no-data-loss backup is the default.

ENOSPC/short writes/sync errors may leave the affected entry invalid; do not republish
it until all dependencies verify. Recovery must never accept partially overwritten
state. Preserve unrelated entries and report losses/skips. Explicit cache clearing
must durably retire its entry records so scanning does not resurrect cleared state.
Physical freed-space behavior, shared-object lifetimes and filesystem allocation
must be measured; bounded staging alone does not prove a bounded disk footprint.

### Relationship to `--slot-save-path` files (decided 2026-09-19)

`--slot-save-path` is the same serializer driven by hand: `POST /slots/<id>?action=
save|restore` writes and reads one slot's tokens plus its per-sequence state behind
a versioned `BUUNSLOT` header that the restore handler authenticates before the
state blob reaches the library. P0 measured what it lacks for resume: no lifecycle
hook, no checkpoint companion (hybrid and SWA models reprocess the history, and an
SWA rewind of restored state is silently wrong), one checksum over the whole file
that is only checkable after reading all of it into host memory (no partial read,
no appending, no bounded staging, no `fsync`), a runtime identity bound to the
build label, context size and slot layout, no file when dynamic VBR is active,
and a save that is not read-only.

Resume does not fork the format. One envelope family, two install routes:

- A library sequence-state file (the current container) takes the legacy route
  unchanged: strict runtime identity, sequence state only.
- A resume manifest (its own magic and version, holding a version-3 slot
  envelope as the token ledger) takes the resume installer: resume compatibility
  key, per-object checksums, required companions, rewinds only through the
  server's `pos_min` guard or a restored checkpoint, resume reason codes. P1
  settled that the declaration is the manifest's own header rather than a flag
  inside the library file, because that container's declared length and
  whole-payload checksum rule out chunks.
- Both routes share the state serializer and the state blob layout: a chunk is
  that layout restricted to a token range. The manifest, checksums and companions
  are additive; there is one reader to maintain.
- **No silent downgrade.** A file that declares a manifest and fails any resume
  check (checksum, missing companion, key mismatch) is refused with its reason. It
  is never installed through the legacy route as state only.
- A resume entry is a set of objects under the namespace above, not one file. The
  `/slots` restore action accepts a resume entry by name and resolves its objects
  inside the resume namespace; it does not make a resume entry a single portable
  file, and builds that predate the new header version refuse it at the header.
- The `/slots` save action may gain an explicit way to publish a resume entry on
  demand. Its default output stays the legacy file.

The manual endpoint is therefore the test and operator entry point for the code
that `--resume` runs at startup and wake: restore a named entry into a slot
without restarting the server.

## 7. Memory and disk cost controls

- No mandatory 64-GiB host-cache dump: the opt-in is independent of ordinary
  `--cache-ram`. Existing host-memory admission limits still apply on restore.
- Stream payloads with bounded staging, not a snapshot-sized RAM vector.
- Avoid touching unrelated pageable host weights while saving cache state.
- Restore optional host entries lazily where the existing carrier interface can
  support it safely; otherwise admit only a bounded subset. Do not mmap everything
  and claim the page-cache cost disappeared.
- Preserve present KV representations; don't add a lossy re-encoding step merely
  to shrink snapshots in the baseline implementation.
- Defer compression, incremental autosave and generic block-level dedup until
  measured. Quantized KV may compress poorly, while recurrent/checkpoint payloads
  may differ. Compare wall-clock benefit, not just compressed size.
- Measure model-load time separately from resume import and first-request replay.
  Resume removes repeated prefill work, not model weight loading.

## 8. Experiments and acceptance gates

Use the same saved prompt/continuation and fixed configuration for controls.
Archive full commands, build IDs, model revisions, sampling settings, raw results,
and whether the OS file cache was warm. Do not use unrelated engines as controls.

### Save-policy experiment

Compare: (A) minimal valid live state; (B) live state plus selected historical
checkpoints and speculative accelerators; (C) B plus host cache (explicit opt-in).
Use exact continuation, partial-prefix extension/rewind, and multi-agent rotation
requests. Include a declared large host budget with both sparse and populated
entries to prove default saves don't traverse the entire host cache.

Record unique bytes, apparent/allocated disk usage, bytes read/written, peak RSS,
VRAM peak, shutdown delay, import time, suffix-replay tokens, first-token latency,
uncached PP, warm TG, and later reuse coverage. Retain raw repetitions; separate
warm/cold filesystem effects. Require no material steady-state regression with
resume off, and investigate measured overhead with it on.

### Same-model fidelity

- Reproduce the cold-restart loss before implementation using the same prompts.
- Compare uninterrupted vs saved/restarted same-model execution at an identical
  accepted-token boundary. Require exact-zero anchors under deterministic matched
  execution; investigate nonzero drift rather than redefining the anchor tolerance.
- Pin tokens, positions, recurrent state, cache representations and dependency
  lineage with focused tests. A coherent answer alone is not a sufficient gate.
- For changed topology/kernel selection, report numerical differences separately
  from serialization fidelity; never call those runs exact anchors.
- Test stochastic serving at normal temperatures, but do not demand the same
  random sequence after a new request. Use matched teacher-forced logits for
  fidelity and separately check sampler behavior/acceptance.

### Cross-fine-tune / LoRA experiment

Compare B fresh-prefill against A-prefill -> save -> B-resume, alongside A->A and
B->B controls. Include base-to-fine-tune, fine-tune-to-fine-tune, LoRA changes,
and changed weight quantization, only when family/representation checks pass.
Measure mean/median/tail KLD under B on identical tokens, task outcomes, and how
the difference evolves as B appends tokens. Avoid interpreting different generated
strings as a KLD comparison. Label this approximate-history reuse, not a save/load
correctness failure or a claim that old KV becomes B's KV after boundary replay.

### Failure and feature matrix

- SIGTERM during prefill/decode, forced interruption during save, restart after
  publication, unload/sleep/wake, empty slots and startup failure.
- Disk full, read-only cache root, truncated manifest/payload, wrong checksum,
  unsupported schema, missing dependency, concurrent writer and orphan cleanup.
- Interrupt before invalidation, during payload replacement, and before/after
  each entry commit. Recover a mix of untouched old and completed new entries;
  skip incomplete ones. Prove no mixed-epoch dependency closure is accepted.
- Update one shared prefix while other slots reference it; measure the affected
  dependency group and ensure unrelated entries survive. Include whole-cache VBR
  tier changes and demonstrate that backup-sized disk duplication is not required.
- Fixed F16/Turbo, dynamic VBR after multiple degrades, hybrid recurrent models,
  SWA/iSWA, shared live prefixes, and host-cache on/off.
- Different tokenizers/RoPE/codebooks rejected; smaller contexts/budgets and fewer
  slots handled without over-allocation or damaged surviving entries.
- No draft, MTP, DFlash/DFlash2; target-only fallback when optional draft differs.
- Image/audio/video ledgers and replay boundaries: either supported end-to-end
  or explicitly skipped with cold replay, never restored using text-only identity.
- CUDA first, then HIP correctness using the same storage/owner contracts;
  Windows compile plus platform-specific atomic-file/locking tests before claiming
  Windows support. Do not put CUDA-specific persistence logic in the shared layer.

## 9. Implementation phases and checklist

### P0 — Inventory and minimal-size proof (no shipping flag yet)

- [x] Trace ownership for fixed/VBR live state, host entries, checkpoints, media
  and each speculative companion; identify serializable vs process-local fields.
- [x] Prove family fingerprint/tokenizer and representation coverage; list missing
  receipts (e.g. media/projector identity) rather than assuming them present.
  Finding: the semantic family digest separated a fine-tune from its base when
  the base carried an integrated MTP layer or a different pad token id.
  Resolved in digest v4: an appended MTP head and the pad token id are unbound.
- [x] Prove bounded frontier replay for a dense and a hybrid model, including
  configurations with checkpoints disabled and a changed producer model.
- [x] Measure A/B save policies using existing serializers and a restart harness.
  Policy B used a partial-state blob written by the test probe; no file
  serializer for server checkpoints exists yet.
- [x] Record a cold-restart baseline and default host-cache exclusion test.

Not covered in P0, carried forward: dynamic-VBR disk round trip (P3), DFlash
companions and LoRA transitions (P4), termination during a save (P2 fault tests).

Gate: a valid minimal snapshot with measured size/load/first-request cost. Resolve
mandatory companions before freezing the format or promising broad coverage.

### P1 — Design review and format contract

The contract is `server-resume-format.md`; section numbers below refer to it.

- [x] Review lifecycle ordering, identity, checkpoint dependencies and failure
  recovery against the real owners identified in P0. (§1, §4, §7)
- [x] Freeze a versioned manifest/object contract, bounded decode limits and
  provenance transition rules; separate portability versions from build labels.
  (§2, §5, §7 `resume_import`)
- [x] Decide checkpoint selection, payload chunking, entry boundaries and
  space-bounded replacement ordering using P0 results; no backup generation.
  Token-range chunks of 4096 tokens, one entry per conversation, at most three
  recurrent/SWA states per entry, replaced objects as the only transient cost,
  invalidate-first when even that does not fit. (§2, §4, §6)
- [x] Specify unsupported configurations and observable cold-fallback outcomes.
  (§4 step 2, §8)
- [x] Specify the envelope contract of §6 "Relationship to `--slot-save-path`
  files": what declares a resume manifest, per-object checksums, route selection
  in the `/slots` restore handler, the no-silent-downgrade rule, and the resume
  reason codes (distinct from `model_family_mismatch`). (§2, §8)
- [x] Define the resume compatibility key: what it hashes, what it leaves out,
  and which of those become install-time checks. A smaller context is not a
  refusal: the entry installs up to the largest restorable position that fits
  (`installed_prefix`). (§5, §7, §8)
- [x] Define the checkpoint companion as the partial (recurrent/SWA)
  per-sequence state blob measured in P0, mandatory for hybrid and SWA models.
  The state at the saved boundary is the same kind of object. (§2.1, §6)

Open for the maintainer (§9 of the contract): direct slot install is recommended
for v1 with host-cache entry install kept for P4; unified KV with several slots
saves one live conversation per shutdown and keeps the others' earlier entries.

Gate: no unresolved correctness/ownership findings known to the author;
performance questions have explicit experiments (P2 tests listed in §10 of the
contract). Independent adversarial review is deferred by the maintainer until a
working product exists, with one review planned before the VBR phase; the
contract is unreviewed until then.

### P2 — Durable store and fixed-state vertical slice

- [x] Implement root resolution, private files, locking, streamed I/O, integrity
  checks, per-entry publication/invalidation and reference-safe garbage collection.
- [x] Add flags and coordinator through normal shutdown/startup control flow.
- [x] Fixed-state single-slot dense and hybrid save/restore; establish frontiers.
- [x] Save and restore the slot's context checkpoints as required companions
  (hybrid: full; iSWA: partial images tied to the snapshot's base cache). Test a pure
  append and a rewind after restart on a hybrid and on an SWA model.
- [x] Route `/slots` restore by header: legacy files unchanged, resume entries
  through the resume installer, refusal (never legacy install) on a failed
  resume check. Use it as the restart-free test entry point.
- [x] Fault-injection tests and uninterrupted-vs-resumed regression tests.

Gate: end-to-end same-model restart and deliberate family reuse, with resume-off
cost effectively zero. Failed writes may lose affected entries, never admit corrupt
state or destroy unrelated valid entries; record measured peak disk occupancy.

Initial measurements: dense and hybrid models had token-identical continuations
across restart, sleep/wake, rewind, fewer slots and a killed save. Family reuse
was measured separately in P0. These do not close the review findings in §11. SWA
above the window is row-exact, not logit-exact (`server-resume-format.md` §10). Without `--resume`
no resume code runs. Peak disk during a second save of a 221 MB hybrid entry:
332 MB, i.e. the entry plus the objects being replaced. The retained-cell capture
listed under P3 was needed already here and is in (`SWA_HELD_CELLS`).

### P3 — VBR, multiple slots and checkpoint selection

- [x] Reuse artifact capture/stream/import; preserve tiers, companions and lineage.
- [x] Reconstruct fresh accounting/leases; no serialized live process handles.
- [ ] Save shared payloads once, restore under current budgets, handle partial
  inventory admission without partial-slot publication.
- [ ] Implement measured historical checkpoint policy and qualify SWA/iSWA,
  including the optional retained-cell capture (§4) if the checkpoint restore plus
  re-decode on a pure append proves worth removing.

Gate: degraded VBR + hybrid multi-slot restart, prefix extension/rewind, and shared
prefix tests pass. This is required for the fork's default cache configuration.

#### P3 route (design, traced against the code before implementation)

Under dynamic VBR a fixed-type state blob is not a restore source: the library
refuses a save taken after any degrade, and an entry-tier save stops restoring once
the target cache has degraded. The artifact system is the owner of tiered KV, so a
resume entry under VBR holds **one artifact object** instead of chunks and tails.

Save, per slot, at the same two capture points as the fixed route:

1. The slot-capture control route's **exact** capture (`build_capture_request` →
   `server_vbr_artifact_store::capture`). It already covers plain-attention slots
   and carries the recurrent/SWA companions. The projected idle route is not a wire
   form: its unit ids live in the projected domains and its shards carry Merkle
   roots, not payloads.
2. The store rebuilds the package from the catalog view (the builder the view's
   own `validate()` uses, factored out) and runs `vbr_artifact_encode` into a
   streamed resume object. The catalog reference is released after the write.
3. The manifest names the object and the artifact's manifest digest; an unchanged
   slot is not rewritten. A changed one replaces the entry's object, old commit
   given up first (the disk bound of §7).

Load, per entry and free slot:

1. `vbr_artifact_decode` from the streamed object under bounded limits, the payload
   consumer staging segment chains; nothing is kept on a failed verdict.
2. The store re-ingests through the catalog's capture-sink door (`begin_capture` /
   `begin_unit` / `accept_verified_segment` / `seal_unit` /
   `accept_verified_companion` / `publish_reference`): budget admission, accounting
   and dedup are the catalog's, freshly made in this process.
3. The existing control import (`server_vbr_artifact_store::import`) installs the
   reference into the empty slot: schedule quote, destination pricing under the
   current budget, transcode, precision gate, companions, no-fail publication.
   Anything but `ok` leaves the slot empty and the request prefills cold.

Process-bound values:

- `frontier_execution_identity` is digest-covered in three layers of the artifact and
  compared at import. With resume active under VBR the server persists the identity
  in the family namespace and adopts it at model load, before anything derives from
  it. Everything else bearing the identity is an in-process object that does not
  survive the process. A second server of the family does not get the writer lock,
  runs without persistence, and keeps a random identity.
- `sequence_epoch` comes with the artifact and the slot adopts it, so the counter of
  the new process starts above the highest epoch in the store.
- Codebook and companion identities hash the build commit: an entry of another
  build is refused by validation and the slot starts cold. Resume under VBR is
  same-build.

Import never lowers the live cache to the artifact's tiers: it prices the
destination from the live tiers and the memory fit, transcodes the artifact up to
that, and refuses a unit more than one rung below its destination. The running
server meets the same case whenever an emptied cache resets to its entry tiers
before a host restore. Whether a multi-slot restart reaches the saved tiers (the
first slot is imported into a cache the others have not filled yet) is measured in
the gate before any policy is added.

Not in the first slice: partial-prefix restore of an artifact, checkpoints of a VBR
entry (VBR host entries carry none today), conversations beyond the slot count
(the VBR host cache takes projected packages), media.

#### P3 first slice as built (2026-09-20; contract in the format document §11)

Built as designed, with the tenantless capture trio (`prepare_host_payload` →
`transfer_host_payload` → `publish_host_payload`) in place of the control route's
`capture`, and `export_host_payload` / `ingest_host_payload` as the two new doors
of the artifact store. What the gate found:

- **One conversation per cache (first slice only; lifted by the second slice
  below).** The exact capture is an image of the whole pool with one sequence's
  placement, and the empty import is a whole-cache contract of its owners. The
  first slice restored the most recently used conversation and skipped the rest
  (`cache_shared`).
- **Target emptiness includes the pool watermark.** The warmup leaves one; the
  install runs the idle boundary (`breathe`) first, as the host restore does.
- **The pool's side stream is created lazily** by the first tier change or
  capture. A cache restored right after a start has seen neither, so the import's
  `prepare_backing` now creates it.
- **Hashing dominated**: about 13 SHA-256 passes over the payload per save or
  install. `llama_sha256` takes whole blocks in place and uses the SHA extensions
  on x86; a 353 MB artifact went from 13.6 s to 2.6 s (save) and 16.2 s to 2.9 s
  (install). The pass count itself is unchanged.
- **Sleep/wake under VBR aborted on wake, with or without `--resume`**: the
  artifact store and the host prompt cache outlived the accounting ledger they
  report to, because a reload replaces the authority first. `destroy()` now
  releases both.
- **Tier behaviour, measured**: same budget on both sides restores at every
  depth (`native_import`, `live_rebased`, `downward_rebase`); a degraded artifact
  against an unconstrained target is `precision_refused` (two rungs), a
  full-precision artifact against a budget that cannot hold it `state_rejected`.
  A `downward_rebase` is lossy by one rung and its greedy continuation can differ
  from an unrestarted server; the others were token-identical.
- **Idle displacement emptied the slot before a save.** With one slot the idle
  capture publishes the conversation to the host cache and clears the live slot
  about 0.4 s after a turn; the projected package has no wire form, so a stop or
  a sleep after any idle moment saved nothing and released the previous entry.
  The first gate missed it: every scenario stopped right after a turn, and the
  sleep scenario matched the reference text on a cold prefill. The first slice
  kept the source live under a persistent resume (the multi-slot behaviour); that
  exception was keyed on the VBR configuration, not on an active resume, so it
  changed every dynamic-VBR server (review of 2026-09-23, R1). It is now keyed
  on an active resume of a dynamic cache and nothing else: a server without
  `--resume`, or one whose store did not open, displaces as the owners' code
  does. Removing it altogether was measured too (stage 2 below saves what the
  host cache holds, and the owners capture an idle source before displacing
  it): the hosted conversations then come back cold on half the restarts,
  because a package captured from displaced rows is refused on the empty door
  (the owners' offset finding in stage 2). With the exception in place under a
  resume the hosted round trip is whole (6 of 6) and one further case is lost
  instead (`idle`, 2 of 3): a conversation installed from the store, extended
  and then displaced by a stranger's request is refused on its way back
  (`representation_mismatch` in the occupied replacement, for the owners). The
  exception stays until both owners' doors take such packages. The harness has
  an `idle` scenario whose oracle is `cache_n`, not the text, and `idle_ctl`,
  the same without `--resume`. The same rule covers the second place the owners
  clear a live source: after an exact capture on a windowed, recurrent or hybrid
  memory, the finisher clears every live conversation once all are proven
  durable. The owners' pre-displacement capture (merged from master after
  stage 2) runs such captures when a request launches, so three live hybrid
  conversations were saved as one placement and two hosted, and the hosted two
  came back cold (a hybrid hosted entry carries no checkpoints past its last
  one, the open item below). Under a resume the finisher keeps the sources in
  their slots; the captures still publish host copies.
- **`--cache-ram 0` has no artifact store**: nothing is persisted, warned at start.
- The envelope carries drafter/accelerator companions (27B with MTP: three
  companions, identical tokens, acceptance unchanged).

Gate status: restart ×2, sleep/wake ×2, idle-then-stop, rewind, live restore
action, degraded restore, smaller context, fewer slots and overflow pass on dense;
restart, sleep, idle-then-stop and degraded restore on hybrid; MTP on the 27B. On
hybrid, a conversation that returns from the VBR host cache after another took the
slot restores (`cache_n` 9344 of 9696) but its greedy continuation differs from the
reference, identically without `--resume`: a host-cache observation for its owners,
not a resume result. Fixed-route regression rerun
clean on dense and hybrid. `test-vbr-artifact*` and `test-server-resume-store`
pass. Open from the P3 list: shared payloads saved once and partial inventory
admission (both need the multi-conversation import), checkpoint policy for VBR
entries.

#### P3 second slice: every slot of the cache (2026-09-20; format document §11)

The first slice kept one conversation, which is not what `--resume` is for: a
restart should give back the server that was stopped. Three routes were weighed:

1. *One artifact per slot.* The exact capture is a pool image, so N slots cost N
   images of the same rows on disk, and the import of the second one needs an
   occupied destination that preserves foreign rows: new owners' machinery.
2. *Persist the VBR host cache and let slots restore from it.* The projected
   package of the host cache has no wire form; that is P4 and larger.
3. *One image, co-resident placements (built).* The rows of every slot are
   already in the image of the most recently used one. Saving where the other
   sequences lie is 16 bytes per token, and the owners' empty import needs one
   addition: a list of co-resident sequences whose rows validation authorizes,
   the image install assigns and the tracker stamps. The wire format of the
   artifact is unchanged, and nothing about tiers, budget or the host cache is.

Library additions, all behind an empty default: `vbr_sequence_placement` (cache),
`vbr_explicit_co_resident_placements` (capture side, refuses a child that holds
one sequence only; `vbr_explicit_pool_single_sequence` says so ahead of a save),
`co_residents` in the validation policy and the validated
manifest, co-resident rows in `build_live_image`, `resident_cells` in the
destination projection. The store passes `co_residents` and `unowned_cells`.

What the gate found:

- **A move lost the co-residents.** The hand-written move assignment of the
  validated manifest did not carry the new member, so adoption saw none and the
  placed slots came back without rows (`pos_max` -1). One line, and a unit check.
- **A lone image was priced for one conversation.** A restart with fewer slots
  imports an image whose watermark covers every conversation; under a tight
  budget the destination was `exhausted`. The rows no sequence takes are now
  priced (`unowned_cells`).
- **iSWA admits one sequence per capture.** With three live slots every capture
  was refused and nothing was saved. The save passes are terminal, so the other
  slots leave the cache and the most recent conversation is saved alone.
- **A save a few milliseconds after a request was refused** about one run in
  three (`transfer_failed`, ring unavailable): an idle capture was in flight.
  The save pass cancels and drains it first; 8 of 8 runs then saved. This
  predates the second slice and affected every VBR save.
- **Not a resume result, for the VBR owners:** iSWA with dynamic VBR and three
  slots reuses no prefix on the next turn of any slot (`cache_n` 0 of 3047),
  without `--resume` and without a restart; the same run on fixed-type KV
  reuses 3047.
- **A placement with rows missing installed as whole** (review of 2026-09-23,
  R7). The import admits the rows a placement names, as a checkpoint may hold a
  part of a sequence, and the install checked only the last position. A
  placement object with a middle row removed, and every checksum recomputed,
  came back as a conversation with a hole in it. The server now refuses a
  placement that is not the whole of its sequence: one placement per attention
  child of the tree in its order, its frontier at the end, one row for each
  position (harness scenario `mutate`: a hole, a duplicated position, a missing
  child, a short frontier; each refused, the pool and the other slots restored,
  the conversation cold).

Gate (RTX 3090, three slots, harness scenario `group`: fill three slots, restart,
every slot must reuse its held prefix, `cache_n >= n_held - 1`; then an unchanged
restart, a changed one, and a restart with one slot):

| cell | result |
|------|--------|
| fixed-type KV, dense and hybrid: restart, sleep, rewind, smaller, fewer | pass |
| dynamic VBR, dense and hybrid: the same, plus tier and group | pass |
| dynamic VBR under `--vbr-vram 120M`, dense and hybrid: group | pass |
| dynamic VBR, iSWA, one slot: group | pass |
| dynamic VBR, iSWA, three slots: group | the 7 multi-slot reuse compares fail as in the owners' finding above; the save is `sole` |
| fixed-type KV, iSWA: restart, sleep, fewer | f16 KV passes; turbo3_tcq reuses the full prefix (`cache_n` 9047 of 9047) and its text differs from the one-process reference, as it did before this slice |

Limits: a placed conversation that gets no slot after a restart with fewer slots
stays in the store until the image is next rewritten, then it is dropped. iSWA
restores one conversation in a slot; the others come back through the host cache
when there is one (next section).

#### P3 second slice, stage 2: the VBR host cache (2026-09-20; format document §11)

A server with fewer slots than conversations keeps most of them in the host
cache, and a restart that gives back the slots only loses them. Two routes:

1. *Write the host cache's packages.* A hosted artifact is a projected package:
   packed rows, no pool image, and no wire form. Giving it one is a new format
   and a new import door in the owners' store.
2. *Round trip through a slot (built).* The owners already turn a hosted
   package into a live conversation (`try_automatic_vbr_restore`) and a live
   conversation into a hosted package (the idle capture). The save pass is
   terminal, so after the slots are saved one of them is a stage: each hosted
   conversation is restored into it, saved as a slot's conversation is, and
   replaced by the next. At load every artifact entry after the first is
   installed into an empty staging slot, published by the owners' idle capture
   run synchronously, and cleared, oldest first, as the fixed route does with
   `resume_install_host`. No library change, no new file kind, and serving is
   untouched.

What is saved: the hosted states of the running execution identity and adapter
configuration, without media, that are not an earlier state of a saved slot or
of another hosted state; the newest of them up to the entry bound, which is now
`max(8, 4 x slots)`.

What the gate found:

- **A resumed slot never reached the host cache.** The install did not publish
  the restored tokens to the retention owner, so the idle capture saw nothing
  to keep. One call (`slot_restored_tokens_publish`).
- **The restore goes through the occupied door.** The stage keeps what it holds
  and each restore replaces it (the owners' occupied replacement), after the
  stage's own conversation is made durable in the host cache so that the
  one-slot handoff gate does not divert to the empty door. A cache with no room
  for two conversations refuses the replacement; the stage is then cleared and
  the restore retried into it.
- **For the VBR owners, not a resume result:** the empty-destination import
  refuses a hosted package whose rows were not at physical cell 0 when it was
  captured (`stage_failed`, `source_hash_mismatch`). `stage_child` in
  `llama-vbr-artifact-stage.cpp` takes `first_physical_cell * row_bytes` as the
  offset into a payload that is packed (measured: first cell 3047, 3047 rows,
  payload of 3159 rows). Any package captured after an occupied restore has
  such rows. In plain serving this is a silent cold prefill; the occupied door
  takes the same packages without complaint (6 of 6 in a run without restarts).
- **A pool image is as long as the cache has been written, not as the
  conversation is.** After an occupied replacement the rows lie above the ones
  they replaced, and the image of a 3161-token conversation covers 6400 cells
  (734 MB against 353 MB). This holds for a live slot in ordinary serving too,
  it does not grow past two conversations, and the watermark only shrinks once
  the upper cells are free. Disk and load time of such an entry double.
- **Under a tight budget the owners' occupied restore declines**
  (`destination=exhausted`) with or without a restart, so the hosted
  conversations are kept across the restart and reuse nothing in either case.
- **A hybrid's hosted state ends at its last checkpoint** (2996 of 3047), with
  or without a restart.

Gate (harness scenario `host`: one slot, three conversations in turn, then two
rounds of restart and one more turn of each; `P2_HOST_CTL=1` is the same without
the restarts):

| cell | result |
|------|--------|
| dynamic VBR, dense, one slot, three conversations | 6 of 6 reuse the full prefix; every save pass saves both hosted conversations |
| dynamic VBR, dense, two slots, five conversations | 10 of 10; four hosted conversations saved per pass |
| dynamic VBR, hybrid | reuse equals the run without restarts (2996, then 3108 to 3110); the live conversation reuses more (3047, 3161) |
| dynamic VBR, dense, `--vbr-vram 120M` | both hosted conversations saved and republished on every pass; reuse 0 as in the run without restarts |
| regressions: group (dense, hybrid, tight, iSWA one slot), restart, sleep; fixed-type restart, fewer, overflow | pass |

Limits: a wake from sleep with requests already queued refuses the idle session
the publication needs, and the hosted entries stay on disk until the next start.
Hosted media conversations are not saved.

#### Stage 2 review round (2026-09-23)

The R1 and R7 findings are recorded with their slices above. The rest:

- **A hosted candidate could change between the choice and the restore** (R2).
  The save pass picked the hosted states, then restored each by its tokens, and
  a state the owners replaced in between was restored in its newer form or not
  at all. Each candidate is now pinned (`recovery_pins`) when it is chosen and
  restored by the pin, so the pass saves the states it counted.
- **A busy slot at shutdown was skipped** (R3). A graceful stop with a request
  still generating saved the idle slots and left the busy one out. Once the
  loop has returned, a request it left mid-generation is ended as a cancelled
  one is (the slot released), then the save pass runs, so the conversation is
  saved to its last generated token and continues after the restart (harness
  scenario `busy`: two hosted conversations, one generating when SIGTERM
  lands, a fourth idle in a second slot; all saved and restored, on dense and
  hybrid). An idle capture whose worker is still reading a slot when the loop
  returns is cancelled and joined first, as the stop-token path does before
  it releases a slot; cancelling alone does not stop the worker, and a release
  drops the speculative rows and child cells it may be reading. The shutdown
  line reports whether a capture was in flight and how many slots it held
  (harness scenario `drain`: SIGTERM ladders across the idle capture after a
  response, a cut prompt, a busy second slot and a fresh restart, on dense,
  hybrid, SWA and speculative models, with a resume-off control). Every stop
  drained before it released and saved the conversation to its last token;
  none landed inside an exact-background transfer, because the idle pass
  captures on the queue thread before the loop can return and the SWA frontier
  capture that holds a slot lasts tens of milliseconds. The in-flight branch is
  held by the order of the helper, not by a measurement: a delay seam in the
  owners' worker would make it measurable.
- **An unchanged entry was kept by its tokens** (R4). A save skipped the
  artifact write when the slot held the entry's tokens, so a slot refilled with
  the same tokens under another model of the family, or under
  `cache_prompt: false`, kept an image of state it no longer had. The slot now
  carries a lineage certificate: the entry's image is the slot's state only
  while the cache's checkpoint epochs are the ones noted when the two were last
  the same; a restore or a refill breaks it. The hosted round trip carries the
  same certificate. (Harness scenario `handoff`: save under one model, restart
  under a fine-tune of the family, refill the exact ledger with the cache off,
  save: recaptured, `artifact_kept` false, for a primary and a placed entry;
  the untouched control is kept.)
- **An oversized artifact was allocated before it was refused** (R5). The host
  cache admitted a manifest's artifact after reading it whole. The ingest now
  prices the manifest's byte count against the cache's budget first, and an
  artifact above it is refused before any of it is read
  (`accounting_refused`); the bytes staged so far count back into the headroom
  the package is priced against. The budget the ingest prices against is the
  host's physical headroom, as for every other host package; `--cache-ram` is
  the retention bound, not the admission ceiling.
- **A restored slot displaced before its first idle pass lost its projection.**
  The first request after a restart usually diverges from the slot it lands
  on, and a slot displaced before the idle pass has captured it goes to the
  host by the exact route, which the owners' restore cannot project a
  diverging request onto (a cold prefill for the very conversation the client
  was in the middle of). The install now ends with the idle capture run over
  the restored slots, wave after wave, until each has a host copy; one
  device-to-host copy per live slot at startup. A wave keeps the owners'
  bounds (eight manifests, one stateful or speculative candidate), so the pass
  runs at most one wave per slot and stops at the first wave that covers no
  further slot (refused, cancelled, displaced). `install_done` reports `live`,
  `exact` (slots with a durable copy: a continuation is warm) and `projectable`
  (a diverging request is warm too), and warns about the difference. A
  stateful conversation (hybrid model, a drafter, a speculative slot: the gate
  of the idle pass's checkpoint cut) is exact only: its copy is cut at a sealed
  checkpoint, which a restored slot has none of, and the owners' restore
  projects onto a copy without checkpoints only — the ring checkpoint
  persistence of the next slice closes that.

### P4 — Accelerator integration and remaining host-cache work

- [x] Host-cache persistence under dynamic VBR, bounded by the entry count, no
  duplication of live payloads (built by default in P3; the opt-in was dropped).
- [ ] Host-cache persistence on the fixed route: the shutdown keeps the entries
  already on disk but does not save the host-only conversations still in RAM.
- [ ] Host-cache publication on a wake with requests queued (P3 stage 2 limit).
- [ ] MTP and DFlash/DFlash2 companion save/rebuild policies, then DSpark when a
  representative model/hardware is available.
- [ ] Media-aware payload identity and replay; compatible LoRA transition tests.
- [ ] Sleep/unload and actual supervisor-driven stop/start validation.

Gate: measured first-request behavior, not just successful file loading. Explicit
fallbacks for any deferred feature must be documented before release.

### P5 — Qualification and cleanup

- [ ] Full failure/feature matrix, family-reuse experiment report, and P0–P4
  regression comparison (including ordinary PP/TG and resume-off behavior).
- [ ] HIP/runtime and Windows/platform gates appropriate to supported scope.
- [ ] Simplification/correctness review; remove failed experiments and temporary
  environment gates, keep only the two agreed product flags unless justified.
- [ ] Public usage/privacy/storage guidance and reproducible scripts/results.
- [ ] Commit reviewable units; merge/push only when requested.

## 10. Progress log

- 2026-09-18: Initial plan written from current source and user decisions.
  No implementation, hardware tests, or independent review have been performed.
  First executable task is P0, not enabling ordinary slot saves under VBR.
- 2026-09-18: Revised storage policy after user feedback: no backup generation;
  permit cache loss on interrupted shutdown in exchange for bounded disk usage.
  Publish slots/prefix entries independently with complete dependency closures,
  allowing recovery of valid old and new entries from the same store.
- 2026-09-19: P0 measurements run (single RTX 3090, 1d0f493c7). SWA finding added
  to §4: the per-sequence writer emits exactly the masked window, so a restored
  iSWA sequence always sits on the reuse threshold and a pure append reprocesses
  the whole history unless checkpoints are restored with it. Checkpoint companions
  moved into the P2 vertical slice for both hybrid and SWA models.
- 2026-09-19: P0 gap closure (cbaf04301). Gate met for fixed-type KV.
  - Sequence state plus one partial-state (recurrent/SWA) blob restores hybrid
    4B and 27B models bit-exactly across a process restart, f16 and turbo3_tcq;
    the companion is needed with thinking off as well (reply re-render rewind).
  - On an SWA model, `llama_memory_seq_rm` on restored state succeeds and the
    continuation is silently wrong. Install must go through the server's
    `pos_min` guard or a checkpoint; resume code never rewinds SWA state itself.
  - The state API refuses a wrong KV type, wrong model, too-small context and a
    truncated blob without crashing. The raw state API has no payload checksum
    (corrected in P1: the library sequence file does carry one FNV-1a-64 over
    the whole payload). f16 and undegraded VBR states are mutually accepted.
  - The semantic family digest hashes `n_layer_all`, `n_layer_nextn` and the pad
    token id, so a fine-tune without the base's MTP layer is a different family.
    §2 needs a KV-compatibility key or an explicit override to keep fine-tunes
    eligible.
    Resolved (decided by the maintainer): digest v4 hashes the trunk layers
    only when the model has an appended MTP head (the target context never
    allocates those layers; router layers and all-NextN sidecars stay bound)
    and drops the pad token id. Measured on a 4B hybrid, stock with MTP head
    vs. a fine-tune without it, f16 and turbo3_tcq KV, both directions: the
    slot file installs (4380 tokens, 0.10–0.22 s), the follow-up reuses the
    whole prefix, and every inherited-KV reply is a coherent answer that
    diverges from both the producer's and the consumer's own-KV reply after a
    shared opening, as §2 expects of a mixed-model history.
  - Slot files refuse context-size, KV-layout and slot-index mismatches under the
    label `model_family_mismatch`; resume needs its own reason codes.
  - With `--kv-unified` and several slots (fixed KV types), launching a task
    moves every other idle slot into the host cache, so at shutdown only one
    slot is live and the rest are host entries; an emptied slot cannot be saved
    and must be skipped. Split KV saves and restores every slot, and a file
    restores into a different slot index.
  - The 30 s shutdown delay is the SSE ping interval: a streaming handler parked
    in the chunked provider is only released by the ping timer or a client
    disconnect. Save hook: after `start_loop()` returns and before `clean_up()`
    in `server.cpp`. The wait is removable with a shutdown flag in the
    `should_stop` closures of `server-http.cpp`.
  - Sleep/wake restores through a slot file today (first P2 integration test).
    A restored reply equals the uninterrupted reply; a cold re-prefill reply
    does not, so regression tests compare restored against uninterrupted.
  - Found outside resume: with split target KV the MTP draft context is sized
    for one sequence but shared by all slots, failing requests at `-np 2`; a
    request pinned by `id_slot` to an empty slot never consults the host cache
    (slot selection leaves `update_cache` false), which under `--kv-unified`
    means every pinned follow-up is a cold replay. Entries restored into the
    host cache rather than a slot would be invisible to pinned requests.
    Both fixed in 08826ad6e and re-measured: the draft context follows the
    target's total capacity, and a pinned empty slot performs the host lookup
    (pinned follow-ups reuse the full prefix), so host-cache restores are
    visible to pinned requests.
- 2026-09-20: P1 contract written (`server-resume-format.md`), unreviewed by
  maintainer decision until a working product exists.
  - Family handoff measured over four text windows and two history lengths; the
    behaviour is kept and documented in §2 (an earlier single-window reading
    that mean reuse error stays below the model distance was too strong).
  - An entry is a directory of objects: base-state chunks by token range, each
    the existing state blob layout restricted to that range, plus recurrent/SWA
    states at positions, plus a manifest published last by rename. Chunking
    costs under 0.01 % in size. A smaller context installs a prefix instead of
    refusing: any position on a dense model, a saved recurrent/SWA position on
    the others.
  - The library needs a base-only flag, a position-range writer and an append
    reader; the state serializer itself is unchanged.
  - Checkpoints read from disk cannot pass the server's validity test (per-load
    execution identity, sequence epoch), so import re-stamps them as a named
    transition and admits them through the door host-cache restores use.
  - The build label, context size, slot count and index leave the identity; a
    resume key keeps the family, the KV types, RoPE/YaRN and format versions.
  - Correction: slot files carry a whole-file FNV-1a-64; the missing pieces are
    smaller integrity units, streaming and durability.
- 2026-09-20: P3 first slice (dynamic VBR), unreviewed. Details under P3 in §9
  and in `server-resume-format.md` §11.
  - A VBR entry is one streamed object holding the VBR artifact envelope,
    companions included; capture and import go through the artifact store's
    existing doors, and the envelope's own validation decides precision.
  - The execution identity the envelope is bound to is kept in the namespace,
    and the sequence-epoch counter starts above every stored epoch.
  - An exact capture images the whole pool and an empty import takes the whole
    cache, so a save keeps the most recently used conversation of a shared
    cache and an install refuses (`cache_shared`) once another slot holds
    tokens. Several conversations per cache need an import beside live rows
    and a capture narrowed to one sequence, both in the VBR artifact system.
  - Found on the way: a reload after `--sleep-idle-seconds` under VBR aborted
    without `--resume` too (the artifact store and the host cache outlived the
    ledger they report to); a cache that never degraded had no side stream for
    the import; SHA-256 of a payload ran scalar at about 0.3 GB/s.
  - Measured on a 3090: dense and hybrid restart, sleep, rewind, action restore,
    fewer slots, overflow, a smaller context, degraded budgets, tier mismatch in
    both directions (refused, never silently rebased up), MTP companions; the
    fixed route re-run unchanged.
- 2026-09-20: P3 second slice, unreviewed. Stage 1 restores every slot of a
  dynamic cache from one image; stage 2 takes the VBR host cache through a
  staging slot in both directions. Details and gates under P3 in §9.

## 11. P0–P2 independent review — before VBR

Review base: `08826ad6e`; reviewed head: `655037231`. Three independent
code reviews covered structure, resource costs, and correctness. The object
directory/container and existing state-owner integration are worth retaining;
no wholesale rewrite is proposed. Implementation is not yet accepted for P3.

First cleanup batch (focused gates tracked in the private review ledger):

- [x] Store retirement is idempotent for already-pruned entries. Connecting it
  to explicit slot erase awaits the conversation-ownership fix below.
- [x] Unreported overflow entries can fill slots left empty by failed imports.
  Host-cache staging retains its existing path; test that separately.
- [x] Limited checkpoint budgets select recent companions before optional early
  history, then import in chronological order.
- [x] Release chunk staging on failed captures as well as successful ones.
- [x] Refuse producer-table overflow instead of assigning new bytes to producer
  zero. Report `provenance_limit` before changing disk state.
- [x] Reject over-depth JSON rather than silently dropping its nested fields;
  avoid signed overflow in tail-position validation.
- [x] Exclude the POSIX research probe from Windows example builds.

Still required before moving on:

- [x] **State identity:** token-prefix equality does not prove that stored KV or
  recurrent bytes describe the current live state. Cold refill, family handoff,
  legacy restore, and slot replacement can invalidate this assumption. Verify
  serialized bytes or carry a proven lineage before reusing objects; never join
  old base state to a newly computed unrelated recurrent frontier. Include
  `cache_prompt:false` as a negative control.
  Done by bytes (contract §4 step 3): an object is reused only when the live
  range or tail serializes to its size and checksum. The re-review removed
  disk-only `early` reuse: base bytes cannot authenticate partial state. Early
  checkpoints now come from the validated live ring. Gates: cold refill under a fine-tune after a family restore
  saves the fine-tune's bytes; an inherited append and a restart after it keep
  the producer's chunks. Known cost: the range blob names the sequence id, so a
  conversation that comes back in another slot is rewritten once.
- [x] **Conversation ownership:** the own-slot entry shortcut accepts a match of
  only one chunk and can overwrite much more than the documented last-chunk
  tradeoff. Separate entry adoption from object identity; qualify long shared
  system prompts and returning host-cache conversations.
  Done: one rule for every entry, all chunks but the last lead the ledger; the
  slot's previous entry is taken over only when retention would drop it in this
  save anyway. Gate: 5000 shared tokens, two conversations through one slot of
  two, both restored whole. Host-cache overflow and fewer-slots scenarios
  repeat token-identical. Open: a single-chunk entry still goes with its slot.
- [x] **Explicit erase:** retire the current conversation's disk entry, not a
  stale entry ID left attached to its slot. Qualify both restart → erase →
  restart and A → replace with B → erase B (A's entry must survive).
  Done, both gates, B unsaved and B saved.
- [x] **Disk bound:** edited histories and replacement conversations can retain
  almost two inventories until post-save pruning. Honor invalidate-first even
  with plentiful free space, without deleting unrelated/held entries. Measure
  peak allocated bytes, not just appended-turn bytes written.
  Done (contract §4): retention runs before a new entry is written and keeps
  slot-held entries first; a save replacing more than its last chunk gives up
  its commit first. Peak allocated bytes 794 MB on a 794 MB entry for both an
  edited and a replaced history (were 1308 and 1527). A kill inside such a save
  loses that entry and leaves nothing behind. Appends keep the old commit and
  duplicate at most one chunk and the tails.
- [x] **Recoverable VMM import:** the newly reached mapping helper aborts on
  physical exhaustion. State import must use the recoverable mapping operation
  and existing rollback, with a forced allocation-failure gate.
  Done: whole-sequence import and range append map recoverably and fail through
  their existing rollback; decode's mid-batch backstop is unchanged. Gate:
  `tests/test-vbr-import-exhausted.cpp` takes the device down to 64 MiB free and
  imports 448 MiB both ways — refused, no cells left, accepted again with the
  memory back; each path aborted before. No server route reaches this map today
  (slot files are off under dynamic VBR, the host restore declines an exhausted
  destination before importing), so the test is at the library API. It becomes
  a server path when resume takes dynamic VBR.
  Extended before dynamic-VBR resume (asked for by the re-review): the gate now
  starves the import with something to lose. Another sequence of 1024 tokens
  lives beside the destination, and a third arm appends `[1024, 4096)` onto a
  live prefix. Device down to 4 MiB free; after each refusal the prefix rows and
  the other sequence's rows read back byte for byte as before, both sequences
  keep decoding across the boundary where the controller settles the refusal,
  and the import is accepted with memory back and reads back as saved. Passes on
  a dense model, a hybrid one and Gemma 4 (two caches, a 30 MiB import); on the
  latter two a range alone is not a decodable sequence, so those arms compare
  rows and do not decode the destination. Controls, each a one-line library
  break: the append unwind dropping the whole sequence fails the prefix arm, the
  whole-import unwind clearing the cache fails on the other sequence.
  The gate found one defect: a context freed with a refused import still
  unsettled aborted in the tracker's destructor (the refusal is settled at the
  next decode boundary, and there was none) — what a server does when it frees
  its context right after a refused restore. The cache now runs the same
  boundary drain once more when it is destroyed; the gate's last arm frees the
  context on a refusal, and aborts without that call.
- [x] **Wrapped SWA state fidelity:** establish the cause with exact row/position
  comparisons and same-token logits. Different batch sizes and coherent output
  are not an acceptance substitute.
  Re-reviewed: `tests/test-state-restore-swa-exact.cpp` on Gemma-4 E2B (window 512,
  f16, q8_0 and turbo3_tcq KV). All saved rows, including older held rows on the
  ranged route, read back by position from both caches, equal the live rows bit for bit, below the window and across a
  wrapped ring of 2185 tokens, for the whole-sequence restore and for the
  resume route. Two live runs give bit-equal logits. Same-token logits after
  a wrapped restore differ (max KLD 7e-9, top-1 8/8); the control reproduces
  that below the window with exact rows moved 200 cells along (max KLD 2e-6),
  where an unmoved restore is logit-exact. The cause is cell placement, which
  changes the attention kernels' reduction order and padding, not restored
  data in that comparison. The original test excluded older held-row payloads;
  the re-review covers those too and requires successful zero-offset restores,
  held-position bounds and finite logits. Failed arms can no longer silently
  disappear. Matching the live ring's layout on restore
  is not planned. The TCQ server continuation difference is still a limitation,
  not a passing fidelity gate.
- [x] **Qualification:** repeat dense/hybrid restart, rewind, sleep/wake and
  media gates after cleanup; verify missing-entry action leaves a live slot
  intact, host-cache overflow, resume-off PP/TG, and interrupted replacements.
  Done on the cleaned-up head, RTX 3090, TCQ 3-bit KV unless noted. All 14
  review gates pass (missing-entry action, interrupted replacement among them).
  Greedy continuations against one process: hybrid 17/17 (restart, sleep,
  rewind, fewer slots, host-cache overflow, restore action, killed save,
  fine-tune handoff); dense 18/18 with the kill 0.04 s after SIGTERM, the 0.6B
  save finishing inside the default delay; media 3/3 on SmolVLM2 (q8_0 KV) and
  on Gemma-4 E2B. The two-conversation `-np 2` → `-np 1` media restart is
  identical on SmolVLM2; on Gemma-4 it is a `resume_key_mismatch` by design, so
  both conversations prefill cold: identical under f16 KV, one second turn in
  other words under TCQ 3-bit. M-RoPE media is skipped as designed, so that cell restores
  nothing: its second turn is a cold prefill of the history and differs in text
  from the reference's cached turn under TCQ 3-bit, q8_0 and f16 alike, which
  is the server without resume. SWA 8/8 with f16 KV; with TCQ 3-bit the turns
  after a restart or wake diverge in text (contract §10).
  The earlier dense smaller-context continuation had never run: both requests
  overflowed the smaller context and the harness compared two failures as equal.
  With a history that fits it is token-identical, against a live and a cold
  reference; the harness now refuses an empty reference. Resume off, against
  master at the review base, 9000-token prefill and 256 generated tokens, three
  runs in each build order: dense 2000 vs 2000 t/s prefill, 83.4 vs 83.7 t/s
  generation; hybrid 3169 vs 3164 and 114.6 vs 114.7. The build that runs first
  in a pair is ahead by 0.2–0.6 % in either order.
- [x] **Remaining design decisions:** explicitly qualify projector-independent
  approximate reuse; compact unreferenced producer records before the table
  fills; measure range-enumeration cost at long context. Keep deferred host-only,
  VBR, speculative-companion and platform support distinct from passing P2 tests.
  Done. Projector: presence stays in the key, the file stays provenance; one
  model under its projector at two precisions continues token-identically
  (contract §4a), differently trained projectors were not available. Producers:
  a save keeps the records of the objects it carries over, so a conversation
  refilled by another model lists that model alone. Range enumeration at 131072
  tokens, 32 chunks: asking every chunk's size, which is the cell scan alone,
  takes 13 ms (1.3 ms at 32768); reading all chunks takes 271 ms against 191 ms
  for the whole sequence on the dense 0.6B (3.06 GB) and 88 ms against 61 ms on
  the hybrid 4B (0.88 GB). The scan is 2–15 % of a chunked read, so no cell
  inventory is added. Still deferred, and not covered by any P2 result:
  host-only conversations at shutdown, dynamic VBR, drafter and speculative
  companions, non-POSIX platforms.

### Re-review of `7502c100a`

The second independent review found and corrected three remaining cases:

- Optional disk-only early tails could survive a cold refill without proof of
  recurrent/SWA identity. Early now comes from a validated live checkpoint.
  This trades indefinite historical checkpoint retention for a sound identity
  rule; ordinary inherited chunks and live inherited checkpoints still reuse
  their objects.
- Erase ignored filesystem errors. Retirement now requires a durable uncommit,
  preserves the live slot on failure, and retries a failed directory sync even
  after the commit was unlinked. HTTP errors do not expose storage paths.
- Unslotted entries could outrank an older live entry under the overall cap.
  Live slots now have first priority; the same per-key and total caps apply.

The SWA test now compares all held rows and requires every normal restore arm.
Mutation controls demonstrate why this matters: the previous test succeeded
when range append was forcibly refused and when older held rows were omitted;
the revised test fails both controls. F16, Q8 and TCQ 3-bit pass the real
state/finite-logit gates on Gemma-4 E2B. The known TCQ free-generation difference
is not reclassified as an exact continuation pass.

Fresh CUDA SM86 build: store unit test, 16 focused server gates (including
failed unlink, repeated failed directory sync, disk bounds and interrupted
replacement), and dense/hybrid range-state tests pass. The new early-tail and
erase-unlink controls fail on `7502c100a` and pass with the fixes. Peak allocated
bytes for the edited/replaced-history gates remain about 794 MB, not two
inventories. No wire-format change, new flag, or inference-path optimization.
The hardened Python harness also passes restart, rewind, sleep/wake,
smaller-context, fewer-slot and host-overflow scenarios on dense and hybrid
models. Failed requests and empty outputs are now errors, not matching results.
