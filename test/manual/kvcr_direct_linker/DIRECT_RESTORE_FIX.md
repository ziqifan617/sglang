# Direct KVCR restore: multi-turn correctness fix

This fix is stacked on `ziqifan617/sglang:codex/kvcr-linker-followup`, the head
of [SGLang fork PR #11](https://github.com/ishandhanani/sglang/pull/11).
It fixes that branch's experimental `direct_remote_restore=true` path.
It is not a port to `mkhazraee/sglang:moein/kvcr_linker_integration_2` or a
claim of compatibility with a different KVCR API revision.

## Failure and ownership

An A -> B -> C relay exposed two separate problems:

1. B reads A's KVCR DRAM directly into HBM. Marking the restored tree path
   externally stored incorrectly suppresses B's own write-through deposit.
   B can serve the prefix locally from HBM, but that does not make its KVCR
   DRAM a source for C.
2. A query/hint is advisory, not a source-residency lease. Publishing C's
   destination indices before the actual read succeeds makes a missing or
   evicted source an admitted-load failure instead of a recoverable cache miss.

The new ordering is:

```text
advisory lookup
  -> allocate unpublished HBM slots on every attention rank
  -> collectively confirm allocation
  -> complete direct reads into those slots
  -> collectively confirm read success
  -> PREPARE / publish radix entry / COMMIT
  -> leave restored pages eligible for local DRAM write-through
```

A drained source miss aborts the unpublished allocation and recomputes.
An uncertain DMA, owner failure, or timeout is fatal: the destination slots
must not be recycled while a transfer could still write to them.

The remote path remains NIXL/UCX peer DRAM -> target HBM. There is no target
DRAM staging before computation. Subsequent write-through is a separate
operation that creates a serveable local DRAM copy.

The tradeoff is deliberate: direct reads block scheduler admission until all
layers complete. Legacy progressive-direct flags do not bypass this gate.
Recovering safe progressive overlap requires a source-reservation contract
and its cancellation/expiry lifecycle. The default staged path is unchanged.

## Configuration

Add to an otherwise complete linker configuration:

```json
{
  "direct_remote_restore": true,
  "direct_remote_descriptor_cache": true,
  "fetch_chunk_pages": 32
}
```

See [HYBRID_FOLLOWUP.md](HYBRID_FOLLOWUP.md) for compatibility flags and
transport diagnostics. This example is not a complete launch configuration.

## Retained validation evidence

The following tests were performed on the original fix commit
`34ea1441975394b7bf9dc2cc69dbae6d4d646332`, with unchanged KVCR commit
`7f564b1c25e7d2db8ee6cea9c6afcd2d54a67854`. The serving experiments ran on
October 6, 2026 (Pacific time).

| Check | Result |
|---|---|
| CPU regression tests | 82 passed, 14 skipped |
| GPU tree tests, Python backend | 45 passed; 7 Rust-backed cases deselected |
| TP1 A -> B -> C relay | Both hops restored 1,984 / 24,512 cached tokens for 2,048 / 24,576-token prompts; outputs matched |
| Stale-hint canaries | Missing source, partial source, and partial source with a preserved local prefix safely recomputed; subsequent HBM reuse worked |
| TP2 relay | Three TP2 replicas on one B200 host; both hops and output identity passed at both prompt lengths |
| Initial routed multi-turn run | 6,476 / 6,476 successful requests; zero output-length mismatches; 33 safe misses |
| Follow-up comparison run | 6,509 / 6,509 successful KVCR requests; zero output-length mismatches; 109 successful precommit loads and 11 safe misses |

Asymmetric rank allocation/read failure is covered by unit tests, not by live
GPU fault injection. Rust tree support and hybrid-model end-to-end validation
are not established by these Qwen experiments.

The PR reapplies the fix on `f5616f71ef13c2070a8025739c4b1f9af155191a`,
preserving that base's Mamba checkpoint COMMIT fix. The recorded GPU runs
predate this stacking step; they are not new measurements of the final PR head.
Fresh local syntax and diff checks passed. The local Python environment lacks
pytest, PyTorch, and KVCR, so the runtime tests were not rerun when preparing
this PR.

### Performance context, not a patch-only speedup claim

The follow-up comparison used Qwen3-32B-FP8, one 8-B200 host with eight
independent TP1 replicas, 64-token pages, and 32 GiB of store DRAM per worker.
Both backends used linker mode. The workload was agent-loadgen v0.1.82,
stock `swe-bench.toml`, seed 42, 32 concurrent root streams, and a 15-minute
dispatch window followed by draining in-flight requests.

| Moved-turn metric | Fixed KVCR direct linker | Mooncake direct linker |
|---|---:|---:|
| Count | 117 | 201 |
| TTFT p50 | 246.8 ms | 202.2 ms |
| TTFT p90 | 987.1 ms | 1,549.5 ms |

These are different realized moved-turn populations, not paired requests.
Mooncake had a larger long-prompt share; the aggregate p90 is not evidence of
an intrinsic KVCR tail-latency advantage. Moved means a worker change, not
proof of a remote cache hit. This comparison is not a before/after ablation
of the correctness patch, a pure transport comparison, or a reproduction of
the separate H100 NVL staged-linker / Mooncake HiCacheStorage report.
