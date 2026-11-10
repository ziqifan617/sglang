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

Hinted reads normally copy KV/indexer pages straight into HiCache host pages
without retaining them in the receiver's KVCR. Hybrid lookup confirmation
separately fetches and pins the trailing checkpoint in receiver KVCR. With `"retain_hinted_reads": true`, a copy-mode read also
queues retention after an ordinary copy delivery into the host pages. The
background progress owner fetches delivered keys from the hinted source into
the receiver's KVCR. Hybrid scopes defer submission until the controller
releases the delivery scope; unscoped reads can progress after their IO lock
is released. The read never waits for the queued fetch to finish. A missing receiver key requires
a second network transfer; a checkpoint already fetched by confirmation is a
local hit.
Background work still shares CPU, locks and transport resources with requests,
so this does not guarantee identical TTFT to ordinary copy. Once the fetch completes, its claims are released and the
retained copy stays cached and advertised. Retaining duplicates
the source's copy; the default remains off. A failed follow-up fetch only means
the receiver keeps no copy; it never fails the read.

Move hints require `"enable_move_hint": true` and a KVCR whose public `fetch`
accepts `source_endpoint` and `move` (tested against KVCR main `6d54642` with
that API exposed). A move is a retained read followed by source invalidation:
once the follow-up fetch holds every required pool of a page on this rank, those
keys are fetched again with `move=True` from the hinted source. They are local
hits, so KVCR sends the source a one-way invalidation, and nothing waits for it.
For an isolated one-way move, this orders local pool completeness before
requesting source cleanup on each rank. Move enables combined KV/indexer reads
even when `combine_page_reads` is false. A page
whose follow-up fetch misses any pool, for example under capacity pressure or
while the receiver key is pending removal, keeps its source copy.
Hybrid restores track the controller's operation scope across KV batches and
trailing-state delivery; they invalidate only after the confirmed prefix and
checkpoints are all held locally. If the controller releases the scope while
retention is queued or running, the scope's checkpoint claims are kept until
the follow-up fetches settle. Cancellation before every piece is delivered, scope replacement and
errors keep the source. Trailing pools without a confirmed operation scope use
copy-only behavior. Any failed delivery or follow-up fetch in a scope keeps the
source for the entire scope. A scope that mixes copy and move actions also
keeps its source. If the controller trims an already-confirmed prefix
head, the scope still expects that head and safely leaves the source intact;
trimmed scopes do not commit moves.

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

KVCR invalidation is per physical key and has no deletion
acknowledgment. Requiring every pool of a page to be held locally first closes the
within-rank partial-pool loss window for an isolated one-way move. Lost
invalidation can leave both workers with a copy. Opposing moves can instead
lose both KVCR copies when delayed invalidations reach reused local residencies.
The protocol has no migration epoch, cross-rank commit, crash durability or
framework-wide stale-GPU relinquish guard.

For backend-owned routing inventory, opt in with `"publish_inventory": true`
in the backend extra configuration and configure a ZMQ KV-event publisher
with `"format": "dynamo"` when using Dynamo routing.
Framework GPU and HiCache-host events remain unowned; only committed KVCR DRAM
stores/removals carry `ownership="kvcr"`. A logical page is advertised after
all required physical pools and attention shards own it, and withdrawn when
any required shard/pool removes it. Trailing/checkpoint admission is still
verified by the storage query and terminal read.
Existing prefetch metadata also lets retained and moved pages publish destination
Stored events once the follow-up fetch lands; the adapter keeps that metadata
until then. Ordinary copy delivery retains no KV/indexer data to advertise.
With `diagnostic_telemetry`, request/rank move records report retained pages,
moved pages, requested invalidations and the follow-up window (`start_epoch` to
`epoch`). Requested invalidation does
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
