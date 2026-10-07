# Hybrid KVCR linker follow-up

This follow-up targets Ishan's `idhanani/kvcr-direct-linker` branch, not SGLang
main. It ports the reusable changes from the DeepSeek-V4.1 and Kimi K3 linker
experiments onto that branch's newer cache interfaces. It does not add model
implementations or weights.

## What changed

- Expose DeepSeek C1/C2 KV and indexer buffers in logical tree-page units. A
  smaller physical indexer page is grouped into the same logical object; absent
  C4/C128 pools are not dereferenced.
- Expose hybrid full-attention and Mamba/KDA state, including the current pool's
  transferable sibling state. Preserve pipeline-stage layer numbering and
  rank-qualified keys for TP-sharded state. Restore a tree checkpoint separately
  from the request's mutable state. Duplicate checkpoint insertion must not
  launch DMA into the slot that was just freed.
- Size sparse checkpoint capacity according to the checkpoint grid instead of
  reserving one large state object for every KV page. The byte budget remains
  per rank; rounding includes the final partial checkpoint interval.
- Optionally restore whole objects directly from a peer into registered HBM,
  with destination-descriptor reuse and page chunking. The direct path now
  completes reads into private slots **before** publishing a radix-tree hit.
  All attention ranks must agree on allocation and read success; a normal
  source miss frees only drained, unpublished slots and recomputes.
- A direct HBM restore is not a local KVCR DRAM deposit. Keep restored nodes
  eligible for write-through so a later worker can fetch them from this worker.

The companion KVCR branch is `ai-dynamo/kvcr:codex/linker-hybrid-followup`, based
on `idhanani/framework-gpu-regions`. It adds a disjoint-layout eviction index,
ordered named-span delivery, and immediate progress-thread source submission.
`deliver()` retains its existing signature; `deposit()` and `fetch()` still
represent whole objects. Local delivery supports the same projection because
KVCR may select a target-local copy before consulting its peer hint.

## Enabling the experimental path

Keep the existing linker configuration, remote hints, advertised control
endpoints, memory registration and per-rank budgets. Add these fields to
`--hicache-storage-backend-extra-config` to test direct peer restoration:

```json
{
  "direct_remote_restore": true,
  "direct_remote_descriptor_cache": true,
  "fetch_chunk_pages": 32
}
```

This JSON is an **addition**, not a complete launch configuration.
`direct_remote_restore` defaults to false: the existing claimed target-DRAM
staging path remains the default. A page here is one pool object key, not
necessarily one physical buffer or one token. `progressive_remote_restore`,
`direct_remote_chunk_pages`, and `direct_remote_inflight_layers` are retained
for legacy configuration compatibility but are not used by the precommit path.
Progressive transfer/compute overlap is intentionally disabled for unreserved
direct reads; the startup log explicitly reports this. Staged restore behavior
is unchanged.

Direct restore charges ALL_PAGES pools for the whole prefix and TRAILING_PAGES
pools only for their required tail. Lookup still intersects valid resume
boundaries, including sparse checkpoint gaps.

## Safety and review limits

Query remains advisory; there is **no reservation/lease spanning lookup and
delivery**. KVCR protects sources while each submitted operation reads them.
If a source was evicted, has not finished offloading, or only holds the prefix
in HBM, a drained precommit read can return a miss before admission. It never
publishes partially populated slots. A native transfer with uncertain DMA, an
owner-thread failure, or a read that does not drain is still fatal, not a safe
recompute case: destination storage must remain owned until quiescence.

The correctness gate blocks scheduler admission and waits for all layers.
Recovering progressive overlap requires a reviewed source reservation API and
its cancellation/expiry lifecycle; bypassing the gate restores the old race.
The old failure was reproducible as A -> B -> C: B received the prefix directly
in HBM, was incorrectly marked externally stored, and skipped its own DRAM
deposit. C then trusted a hint naming B and failed after publishing its slots.

The precommit path transfers whole object layouts rather than layer subsets.
The earlier per-layer ablation required both endpoints to support named-span
subset delivery; its helpers remain for isolated protocol regression tests.
Mamba external-linker MTP draft pools remain explicitly unsupported. The new
Mamba assembly uses the current branch's transfer-entry iterator rather than
copying the older benchmark snapshot's memory-pool implementation.

Not carried forward: already-upstream KV-hint transport/RLock fixes; batched
control (`deliver_many`) and multi-layer grouping ablations; experimental leases;
hard-coded cluster endpoints, model paths or global polling/GIL tuning; Lin's
separate HiCacheStorage adapter. These PRs cover the linker path.

## Validation and reproducibility

Tests were run in an isolated directory of the existing Linux/aarch64 GB300
container, with this SGLang checkout and the companion KVCR checkout first on
`PYTHONPATH`. No serving processes were restarted or benchmark settings changed.

```bash
PYTHONPATH=python:/path/to/kvcr/src SGLANG_RUST_BUILD_MODE=never \
python3 -m pytest \
  test/registered/unit/mem_cache/test_kvcr_direct_linker.py \
  test/registered/unit/mem_cache/test_kvcr_hybrid_followup.py \
  test/registered/unit/mem_cache/test_linker_pool_assembler.py \
  test/registered/unit/mem_cache/test_unified_cache_linker.py \
  -q -k 'not Rust'
```

The tests use real CPU tensors and controlled transports to verify bytes,
claims, geometry, checkpoint deduplication, partial completion and failure
draining. Seven Rust-backed cases are outside this invocation because a matching
native extension was not available in the isolated checkout. This is not a new
end-to-end model or performance validation of the rebased commits.

Historical experiments used 116,000-token prompts, four output tokens and serial
requests, with source population, target remote restoration, immediate HBM reuse,
and a distinct cold recompute control. DeepSeek-V4.1 used two independent TP4/EP4,
DP1 workers and 256-token pages; Kimi K3 used TP8 with DP2 attention across two
four-GPU nodes and 64-token pages. Those GPU runs motivated these changes but
used older runtime overlays. Do not attribute their timing numbers to these
new rebased commits without repeating the workload.

## Diagnose the NIXL / UCX path

The default `device_copy=true` uses KVCR's CUDA-runtime engine for GPU offload;
`direct_restore=true` uses the linker's CUDA-runtime engine for local restores
when available. Those payload copies do **not** exercise NIXL/UCX. To inspect
the complete staged path through NIXL, add these fields to the configuration:

```json
{
  "device_copy": false,
  "direct_restore": false,
  "direct_remote_restore": false
}
```

Set these variables before launching **both** workers:

```bash
export NIXL_LOG_LEVEL=DEBUG
export UCX_LOG_LEVEL=DEBUG
export UCX_PROTO_INFO=y
```

This tests source GPU -> source KVCR DRAM (self), source DRAM -> target DRAM
(peer), then target DRAM -> target GPU (self). Setting
`direct_remote_restore=true` instead tests source DRAM -> target GPU directly
through NIXL; the cross-node transfer is expected to use network transport,
not `cuda_copy`.

The startup log names the configured backend, requested offload engine,
effective local-restore engine, and remote-restore path. One scheduler rank
owns one data agent; it registers both that rank's GPU pools and KVCR DRAM.
The backend availability probe is a temporary agent, not another payload path.
KVCR currently requests four NIXL worker threads. Same NIXL agent does not
imply same UCX worker/interface or guarantee `cuda_copy` selection.

Use the companion KVCR `tests/manual/nixl_ucx_self_copy.py` to isolate local
copy selection and descriptor geometry. Check the actual CUDA/host memory-pair
protocol table, not simply whether `cuda_copy` appears in the available lanes.
Do not disable peer-error handling globally as a production performance fix.
Keep DEBUG/protocol runs separate from quiet TTFT measurements.
