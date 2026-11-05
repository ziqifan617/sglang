# KVCR HiCacheStorage backend

This optional backend uses KVCR's indexed-region public API and NIXL/UCX
transport. Install a compatible KVCR build separately; selecting another
HiCache backend does not import KVCR.

## Ownership and data path

SGLang owns GPU memory, HiCache host pages, scheduling, and host-to-GPU copies.
KVCR owns its deposited DRAM cache:

- Backup: GPU -> HiCache host pages -> KVCR-owned DRAM.
- Restore: local or peer KVCR DRAM -> locked HiCache host pages -> GPU.

This is not a DirectLinker or direct-HBM restore. Peer delivery does not require
an intermediate allocation in the target's KVCR DRAM cache. Unpinned SGLang
host pages are never offered as remote sources.

## Configuration

Enable hierarchical caching, choose `--hicache-storage-backend kvcr`, and use
`--hicache-mem-layout page_first` or `page_first_direct`. Supply inline JSON
through `--hicache-storage-backend-extra-config`, for example:

```json
{
  "control_advertise_host": "<reachable-worker-host>",
  "control_port": 32100,
  "local_dram_bytes": 17179869184,
  "combine_page_reads": true,
  "combine_page_writes": true,
  "prefetch_batch_pages": 512,
  "align_storage_prefix": true
}
```

This is an illustrative configuration, not a model-specific capacity
recommendation. Account separately for GPU cache, HiCache host staging, and
KVCR DRAM. The default rank-local mode allocates KVCR storage on each rank.
Control ports must accommodate the rank offsets and be reachable by peers.
The control endpoint's TCP scheme does not select the NIXL payload transport.

Combined reads/writes are opt-in and apply only to primary KV plus
KV-derived `ALL_PAGES` sidecars. Independent and trailing/checkpoint pools
retain their own transfer and resume-boundary handling. Read batching defaults
to 128 pages; configure the same value on ranks participating in the same
prefetch collective. Query and backup batches remain 128 pages.

Prefix-recency alignment additionally requires
`--hicache-storage-pass-prefix-keys`. It adjusts eviction preference; it does
not reserve source blocks or guarantee residency.

An optional positive integer `startup_timeout_seconds` requires the companion
KVCR startup budget change. Omitting it preserves KVCR's normal startup budget.
Transfer and abandonment deadlines are separate from this initialization setting.

## Remote hints and replicated MLA

Cross-replica discovery requires request-scoped source hints, such as those
provided by the companion in-process Dynamo SGLang integration. A query is
advisory; terminal delivery determines which pages are usable. Missing hints
do not authorize fetching from an unrelated replica.

Move hints require `"enable_move_hint": true` and a KVCR version that supports
move advice (tested against KVCR revision `419ac8b`). The default remains copy.
A move first copy-fetches every required physical layout into the receiver's
KVCR storage and holds its claims through local host-page delivery. Only pages
complete in every required pool of this rank are re-fetched with move advice.
Those pinned local hits request source invalidation; the receiver releases the
claims afterward and keeps the cached data. Move enables combined KV/indexer
reads even when `combine_page_reads` is false. Keys the receiver cannot retain or
deliver locally fall back to ordinary copy delivery into the same host pages,
without invalidation. Terminal local-delivery failures release their fetch claims
first, allowing a delayed invalidation to drain before the copy retry.
Capacity rejection therefore does not reduce restore coverage relative to a copy
read with the same source availability. KVCR `419ac8b` rejects both fetch and
direct delivery while a receiver key is pending removal; fallback cannot bypass
that barrier if another operation still holds it. A page moves only if every required component was retained and
delivered.
Hybrid restores hold copies under
the controller's operation scope across KV batches and trailing-state delivery;
they invalidate only after the confirmed prefix and checkpoints are all local.
Cancellation, scope replacement and errors release the held claims. Trailing
pools without a confirmed operation scope use copy-only behavior.
A scoped restore that uses copy fallback stays copy-only for the entire scope.
If the controller trims an already-confirmed prefix head, the scope still expects
that head and safely leaves the source intact; trimmed scopes do not commit moves.

TP and CP ranks move independently. If ranks diverge, a logical KVCR page can become
unusable on both workers even though each moved shard reached its destination.
The controller's cross-rank completion reduction never admits a partial page;
the request recomputes and unclaimed leftover shards become eligible for KVCR
LRU eviction under pressure. This is a recoverable cache miss, not a bound of
one recomputation over the page's lifetime. Cross-rank confirmation after the
controller's reduction is a follow-up. Shared replicated MLA storage remains
unsupported for move because another rank may still need the source pages.

Move applies to KVCR-owned host storage. Source GPU and HiCache cache entries
remain governed by SGLang's existing cache policy; this option neither evicts
them nor marks them relinquished. A later framework backup can recreate the
source's KVCR copy.

In that KVCR revision, invalidation is per physical key and has no deletion
acknowledgment. The adapter's copy/pin/deliver phase closes the within-rank
partial-pool loss window before requesting invalidation. Lost invalidation can
leave both workers with a copy. It provides no cross-rank commit protocol,
crash durability, or framework-wide stale-GPU relinquish guard.

For backend-owned routing inventory, opt in with `"publish_inventory": true`
in the backend extra configuration and configure a ZMQ KV-event publisher
with `"format": "dynamo"` when using Dynamo routing.
Framework GPU and HiCache-host events remain unowned; only committed KVCR DRAM
stores/removals carry `ownership="kvcr"`. A logical page is advertised after
all required physical pools and attention shards own it, and withdrawn when
any required shard/pool removes it. Trailing/checkpoint admission is still
verified by the storage query and terminal read.
Existing prefetch metadata also lets retained move reads publish destination
Stored events; ordinary copy delivery retains no KV/indexer data to advertise.
With `diagnostic_telemetry`, request/rank move records report complete local
pages, failed pages and requested invalidations. Requested invalidation does
not acknowledge source removal; observe the published ownership events too.

The companion Dynamo integration requires its KV state-agent host, started
with `DYN_NAMESPACE=<worker-namespace> python -m dynamo.kv_state_agent` before
the workers. It consumes SGLang's reported event/control endpoints and fresh
engine-lifetime owner IDs. Do not run the legacy publisher in parallel with
this mode. The reported raw topic includes that owner identity, isolating
replacement engines even when they reuse the same TCP port. Inventory is not
persistent across worker shutdown; it is withdrawn
on detach. A detected event-stream gap fails the source closed; this live stream
is not a durable replay journal. Runtime storage attach/detach requires a worker
restart in inventory mode so startup source metadata cannot silently go stale.
The default `publish_inventory: false` preserves existing behavior.
Enable this mode only with matching Dynamo frontend, worker and state-agent
builds supporting `sglang_hicache_v1`; upgrading the worker alone is not enough.
It requires remote hints enabled and an explicit `control_advertise_host`.

`share_replicated_mla: true` is an explicit assertion that KV and indexer bytes
are identical across attention TP ranks. It stores one copy on TP0 while every
rank restores into its own host buffers. It requires physical MLA KV, CP=1,
PP=1, and no heterogeneous TP head splitting, and rejects pools other than KV
and indexer. Do not enable it for sharded hybrid state or SWA pools.

The adapter preserves valid resume-boundary sets for trailing/checkpoint
pools. The hybrid controller intersects these sets across ranks instead of
assuming that the minimum scalar hit count is a valid checkpoint.

## Failure handling and diagnostics

Successful or missed operations must reach terminal native completion before
HiCache can reuse their buffers. If native transfer state is uncertain, the
worker exits because HiCache cannot quarantine potentially in-flight pages.
Startup/progress/shutdown failures likewise must not release registered
memory while native access may remain possible.

`diagnostic_telemetry` and `verify_transfer_bytes` are disabled by default.
They are validation tools, not free performance instrumentation. Diagnostic
logging can include storage-key and request identities; retain those logs
according to the deployment's data-handling policy.

## Validation

CPU adapter/controller contracts:

```bash
python test/registered/unit/mem_cache/test_kvcr_storage.py -v
python test/registered/unit/mem_cache/test_buffer_mode_sidecar.py -v
python test/registered/unit/mem_cache/test_pp_prefetch_ticket.py -v
```

Real NIXL host-byte round trips, including multi-pool and shared-MLA cases:

```bash
python test/manual/test_kvcr_storage_native.py -v
```

The native gate requires Linux, compatible KVCR/NIXL/UCX, and adequate shared
memory; provision `/dev/shm` explicitly in containers. It verifies host bytes,
not model accuracy or inference performance. End-to-end validation was run
with GLM5.3, two independently complete TP4 replicas, and in-process Dynamo.
That does not establish arbitrary hybrid-model or multi-node-engine support.
