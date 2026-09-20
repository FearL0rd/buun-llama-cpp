# Persistent server resume — design and implementation plan

Status: **proposal; not implemented or reviewed for acceptance**.
Created 2026-09-18 against master `ed774445c`.
This is an engineering plan, not documentation of an available feature.

## 1. Objective and agreed product decisions

Avoid repeating expensive prefill after a server restart or model swap by
persisting reusable conversation state. Reuse existing cache/checkpoint owners,
serializers, and restore transactions rather than introduce a second cache system.

Decisions agreed with the user:

- `--resume` is a Boolean flag, off by default. No path argument is required.
- Use the existing llama.cpp cache directory, including `LLAMA_CACHE` overrides.
- Default persistence includes **live slots and their required companions**, not
  the entire host prompt cache. Idle slots still holding a prompt are live slots.
- `--resume-host-cache` additionally includes retained host prompt-cache entries.
  Proposed CLI rule: this implies `--resume`; it must not be silently ignored.
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
| Host prompt-cache history | Excluded unless `--resume-host-cache` is present |
| Allocator slack, scratch, graphs, streams, pointers, locks, leases | Never serialize as executable process state |
| HTTP requests, sampler/RNG state, tool invocations | Not resumed; new requests establish their own sampling settings |

Default saved roots are all nonempty slots, active or idle, at the captured
boundary. Traverse only their dependency closure. The host-cache flag adds host
entries as roots, not another independent copy of shared live data.

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

Proposed namespace:

```text
<fs_get_cache_directory()>/resume/<semantic-family-digest>/
    writer.lock
    entries/<entry-id>/commit
    objects/<content-digest>
```

Each independently published entry record references payload objects and carries
format versions, compatibility metadata, producer provenance, token ledgers,
checkpoint dependencies, and sizes/digests. Entry kinds include live-slot roots
and optional retained-prefix roots. A small commit record distinguishes completed
entries from interrupted replacement. Exact on-disk schema is a Phase 2 deliverable,
not frozen by this illustration.

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
SWA rewind of restored state is silently wrong), no payload checksum, a runtime
identity bound to the build label, context size and slot layout, no file when
dynamic VBR is active, and a save that is not read-only.

Resume does not fork the format. One envelope family, two install routes:

- The header is versioned. A file without a resume manifest (the current version)
  takes the legacy route unchanged: strict runtime identity, sequence state only.
- A header that declares a resume manifest (next version or a flag bit) takes the
  resume installer: resume compatibility key, per-section checksums, required
  companions, rewinds only through the server's `pos_min` guard or a restored
  checkpoint, resume reason codes.
- Both routes share the state serializer and the state blob layout. The manifest,
  checksums and companions are additive sections; there is one reader to maintain.
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

- [ ] Review lifecycle ordering, identity, checkpoint dependencies and failure
  recovery against the real owners identified in P0.
- [ ] Freeze a versioned manifest/object contract, bounded decode limits and
  provenance transition rules; separate portability versions from build labels.
- [ ] Decide checkpoint selection, payload chunking, entry boundaries and
  space-bounded replacement ordering using P0 results; no backup generation.
- [ ] Specify unsupported configurations and observable cold-fallback outcomes.
- [ ] Specify the envelope contract of §6 "Relationship to `--slot-save-path`
  files": header version or flag that declares a resume manifest, section table
  with per-section checksums, route selection in the `/slots` restore handler,
  the no-silent-downgrade rule, and the resume reason codes (distinct from
  `model_family_mismatch`).
- [ ] Define the resume compatibility key: what it hashes, what it leaves out
  (build label, context size, slot count and index, KV layout, pad token id,
  presence of an MTP layer), and which of those become install-time checks
  (enough cells, KV type, layer shapes) instead of identity.
- [ ] Define the checkpoint companion section as the partial (recurrent/SWA)
  per-sequence state blob measured in P0, mandatory for hybrid and SWA models.

Gate: no unresolved correctness/ownership findings; performance questions have
explicit experiments. Independent adversarial review is owed, not yet performed.

### P2 — Durable store and fixed-state vertical slice

- [ ] Implement root resolution, private files, locking, streamed I/O, integrity
  checks, per-entry publication/invalidation and reference-safe garbage collection.
- [ ] Add flags and coordinator through normal shutdown/startup control flow.
- [ ] Fixed-state single-slot dense and hybrid save/restore; establish frontiers.
- [ ] Save and restore the slot's context checkpoints as required companions
  (hybrid: full; iSWA: partial images tied to the snapshot's base cache). Test a pure
  append and a rewind after restart on a hybrid and on an SWA model.
- [ ] Route `/slots` restore by header: legacy files unchanged, resume entries
  through the resume installer, refusal (never legacy install) on a failed
  resume check. Use it as the restart-free test entry point.
- [ ] Fault-injection tests and uninterrupted-vs-resumed regression tests.

Gate: end-to-end same-model restart and deliberate family reuse, with resume-off
cost effectively zero. Failed writes may lose affected entries, never admit corrupt
state or destroy unrelated valid entries; record measured peak disk occupancy.

### P3 — VBR, multiple slots and checkpoint selection

- [ ] Reuse artifact capture/stream/import; preserve tiers, companions and lineage.
- [ ] Reconstruct fresh accounting/leases; no serialized live process handles.
- [ ] Save shared payloads once, restore under current budgets, handle partial
  inventory admission without partial-slot publication.
- [ ] Implement measured historical checkpoint policy and qualify SWA/iSWA,
  including the optional retained-cell capture (§4) if the checkpoint restore plus
  re-decode on a pure append proves worth removing.

Gate: degraded VBR + hybrid multi-slot restart, prefix extension/rewind, and shared
prefix tests pass. This is required for the fork's default cache configuration.

### P4 — Optional host-cache persistence and accelerator integration

- [ ] Host-cache root opt-in, bounded admission, no duplication of live payloads.
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
    truncated blob without crashing. It has no payload checksum: the file format
    needs one. f16 and undegraded VBR states are mutually accepted.
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
