# SPDX-License-Identifier: Apache-2.0
"""KVCR as a direct external linker: GPU pools <-> KVCR-owned DRAM over NIXL.

Ownership model
---------------
* The scheduler thread calls the ``UnifiedCacheLinker`` API. It enqueues work
  and reads snapshots under ``_lock``; it never touches the KVCR core.
* One owner thread (``KVCRAdapter``) performs every KVCR call, polls
  completions, and runs deadline/deferred-submission tickers.
* In staged mode, a source hint is never a hit. ``prepare_request`` queries KVCR and issues
  ``fetch`` for useful candidates, which acquires residency claims; ``lookup``
  then exposes only pages whose claims are held. Claims live until the page is
  delivered to the GPU (or the request is released), so a lookup result cannot
  be evicted underneath the load.
* Direct remote mode has no source lease. Its lookup is advisory; the wrapper
  calls ``load_before_commit`` to finish NIXL delivery into unpublished HBM
  before inserting a tree hit. A quiescent source miss recomputes. Restored HBM
  is not local DRAM residency and remains eligible for write-through offload.
"""

from __future__ import annotations

import collections
import logging
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from operator import attrgetter
from typing import Any, Optional

import msgspec
import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.kvcr.kvcr_adapter import KVCRAdapter
from sglang.srt.mem_cache.storage.kvcr.kvcr_config import KVCRLinkerConfig
from sglang.srt.mem_cache.storage.kvcr.kvcr_layout import (
    CapacityPlan,
    PoolObjectLayout,
    build_pool_object_layouts,
    carve_local_dram,
    compatibility_digest_for,
    compatibility_identity,
    page_descriptors,
    plan_capacity,
    restorable_boundaries,
    unique_allocations,
)
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    KVCRFetchHint,
    KVCRLinkerKeyAdapter,
    encode_object_key,
    offset_control_endpoint,
    page_hash_to_int64,
    parse_fetch_hint,
    unique_page_hashes_from_keys,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LinkerRequestContext,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
    get_spec,
    mamba_track_grid,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import freeze_gc, get_device_module
from sglang.srt.utils.common import is_hip

logger = logging.getLogger(__name__)
device_module = get_device_module()

_span_addr = attrgetter("addr")
_span_size = attrgetter("size")

_SUPPORTED_NIXL_BACKENDS = frozenset({"UCX"})
# How long reset/close wait for outstanding KVCR operations before treating
# them as late work (they are still drained, never assumed finished).
_DRAIN_POLL_S = 0.005


class _State:
    FETCHING = "fetching"
    READY = "ready"
    MISS = "miss"


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        self._lock = threading.Lock()

    def update_producer(self) -> int:
        with self._lock:
            self.producer_index += 1
            self.futures[self.producer_index] = [
                Future() for _ in range(self.num_layers)
            ]
            return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        with self._lock:
            futures = self.futures.get(index)
        if futures is not None and not futures[layer].done():
            futures[layer].set_result(None)

    def complete_all(self, index: int) -> None:
        for layer in range(self.num_layers):
            self.complete(index, layer)

    def fail(self, index: int, error: BaseException) -> None:
        with self._lock:
            futures = self.futures.get(index, ())
        for future in futures:
            if not future.done():
                future.set_exception(error)

    def fail_all(self, error: BaseException) -> None:
        with self._lock:
            indices = list(self.futures)
        for index in indices:
            self.fail(index, error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        with self._lock:
            futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("KVCR layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                with self._lock:
                    self.futures.pop(index, None)

    def reset(self) -> None:
        with self._lock:
            self.producer_index = -1
            self.consumer_index = -1
            self.futures.clear()


# Pause between offload deposits on the owner thread; each deposit holds the
# GIL for a few milliseconds of per-key work, and the scheduler thread
# launching a forward needs uncontended slices in between.
_OFFLOAD_DEPOSIT_PAUSE_S = 0.001


class _LinkerTelemetry:
    """KVCR ``TelemetryStats`` sink folded into the linker's stats log.

    Counters sum, gauges keep the last value, histograms keep count, sum, and
    max, all keyed by metric name and label values. KVCR reports from both the
    owner thread and its progress thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = collections.defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = collections.defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )

    @staticmethod
    def _key(name: str, labelvalues: tuple[str, ...]) -> str:
        return f"kvcr_{name}" + (
            "[" + ",".join(labelvalues) + "]" if labelvalues else ""
        )

    def increase_counter(
        self, name: str, value: int | float = 1, labelvalues: tuple[str, ...] = ()
    ) -> None:
        with self._lock:
            self._counters[self._key(name, labelvalues)] += value

    def set_gauge(
        self, name: str, value: int | float, labelvalues: tuple[str, ...] = ()
    ) -> None:
        with self._lock:
            self._gauges[self._key(name, labelvalues)] = value

    def observe_histogram(
        self, name: str, value: int | float, labelvalues: tuple[str, ...] = ()
    ) -> None:
        with self._lock:
            entry = self._histograms[self._key(name, labelvalues)]
            entry[0] += 1
            entry[1] += value
            entry[2] = max(entry[2], value)

    def reduce(self) -> dict[str, int | float]:
        with self._lock:
            flat: dict[str, int | float] = dict(self._counters)
            flat.update(self._gauges)
            for key, (count, total, peak) in self._histograms.items():
                flat[f"{key}_count"] = count
                flat[f"{key}_sum"] = total
                flat[f"{key}_max"] = peak
            return flat

    def is_empty(self) -> bool:
        with self._lock:
            return not (self._counters or self._gauges or self._histograms)


class _NoFrameworkPinning:
    """Decline every peer pin request: this rank never serves live GPU pages.

    Peers are served from completed KVCR-owned residency only. Registering the
    GPU pools as NIXL endpoints does not grant permission to read them
    asynchronously, so the pin callbacks answer "nothing held".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0
        self._declined: list[tuple[int, None]] = []

    def request_pin(self, keys) -> int:
        with self._lock:
            request = self._next
            self._next += 1
            self._declined.append((request, None))
            return request

    def poll_pin_results(self):
        with self._lock:
            declined, self._declined = self._declined, []
        return declined

    def cancel_pin_request(self, request: int) -> None:
        with self._lock:
            self._declined = [entry for entry in self._declined if entry[0] != request]

    def release_pin(self, pin_handle) -> bool:
        logger.error("KVCR linker asked to release framework pin %r", pin_handle)
        return False


class _PoolPlan(msgspec.Struct):
    """One physical pool's part of a prepared request."""

    pool: str
    policy: str
    window: int
    present: list[bool]


class _Preparation:
    """Asynchronous residency preparation for one request attempt."""

    __slots__ = (
        "handle",
        "request_id",
        "page_hashes",
        "page_index",
        "pools",
        "hint",
        "state",
        "deadline",
        "outstanding_ops",
        "claims",
        "restorable",
        "started_at",
        "bytes_requested",
        "bytes_confirmed",
        "peer_hinted",
        "miss_reason",
        "ready_observed",
        "spans",
        "owner_started_at",
        "query_done_at",
        "fetch_issued_at",
        "fetched_at",
    )

    def __init__(
        self,
        handle: CacheRequestHandle,
        request_id: str,
        page_hashes: list[str],
        pools: dict[str, _PoolPlan],
        hint: Optional[KVCRFetchHint],
        deadline: float,
    ) -> None:
        self.handle = handle
        self.request_id = request_id
        self.page_hashes = page_hashes
        self.page_index = {page: index for index, page in enumerate(page_hashes)}
        self.pools = pools
        self.hint = hint
        self.state = _State.FETCHING
        self.deadline = deadline
        self.outstanding_ops = 0
        # (pool, page index) -> KVCR release handle.
        self.claims: dict[tuple[str, int], int] = {}
        # (pool, page index) -> the claimed slots' descriptors, in subpool
        # order, as fetch reported them. A claim pins the residency, so these
        # spans stay readable until the handle is released.
        self.spans: dict[tuple[str, int], list] = {}
        self.restorable: list[int] = []
        self.started_at = time.monotonic()
        self.bytes_requested = 0
        self.bytes_confirmed = 0
        self.peer_hinted = hint is not None
        self.miss_reason: Optional[str] = None
        self.ready_observed = False
        # Timeline for the per-request log: owner pickup, last fetch issued,
        # last fetch completed.
        self.owner_started_at = 0.0
        self.query_done_at = 0.0
        self.fetch_issued_at = 0.0
        self.fetched_at = 0.0


class _LoadPool:
    __slots__ = (
        "pool",
        "page_hashes",
        "indices",
        "claims",
        "spans",
        "request_id",
    )

    def __init__(
        self,
        pool: str,
        page_hashes: list[str],
        indices: torch.Tensor,
        claims: list[int],
        spans: list,
        request_id: str,
    ):
        self.pool = pool
        self.page_hashes = page_hashes
        self.indices = indices
        self.claims = claims
        # Claimed slot descriptors per page (None when fetch reported none).
        self.spans = spans
        # KVCR request scope retaining the peer hint for direct remote delivery.
        self.request_id = request_id


class _LoadBatch:
    __slots__ = (
        "counter_index",
        "rids",
        "pools",
        "ready_event",
        "outstanding",
        "success",
        "bytes",
        "started_at",
        "requested_at",
        "submit_at",
        "handles",
        "next_handle",
        "layer_outstanding",
        "precommit_done",
        "error",
    )

    def __init__(
        self, counter_index: int, rids: list[str], pools: list[_LoadPool], ready_event
    ):
        self.counter_index = counter_index
        self.rids = rids
        self.pools = pools
        self.ready_event = ready_event
        self.outstanding = 0
        self.success = True
        self.bytes = 0
        self.started_at = time.monotonic()
        # Earliest load() call folded into this batch, and when the owner
        # thread began building copies; the gap is the ready-event wait plus
        # owner-thread queueing.
        self.requested_at = self.started_at
        self.submit_at = 0.0
        # Direct restore: (logical layers, copy handle) in submission order.
        self.handles: list[tuple[list[int], Any]] = []
        self.next_handle = 0
        self.layer_outstanding: dict[int, int] = {}
        self.precommit_done: Optional[threading.Event] = None
        self.error: Optional[BaseException] = None


class _OffloadTask:
    __slots__ = (
        "transfers",
        "ready_event",
        "bytes",
        "outstanding",
        "success",
        "done",
        "started_at",
        "chunks",
        "next_chunk",
        "submitted_all",
    )

    def __init__(self, transfers: list[PoolTransfer], ready_event, nbytes: int):
        self.transfers = transfers
        self.ready_event = ready_event
        self.bytes = nbytes
        self.outstanding = 0
        self.success = True
        self.done = False
        self.started_at = time.monotonic()
        # One deposit per chunk: (pool, page hashes, rows) segments covering
        # every pool's pages for that chunk index. Built once; submission may
        # span several owner-loop iterations.
        self.chunks: Optional[list[list[tuple[str, list[str], list[int]]]]] = None
        self.next_chunk = 0
        self.submitted_all = False


def _cpu_indices(indices: torch.Tensor) -> torch.Tensor:
    return indices.detach().to(device="cpu", dtype=torch.int64).flatten()


def _ready_event():
    """Event marking the producer stream's position, or None without a device.

    Recorded on the scheduler thread; the owner thread submits the transfer
    only after ``query()`` reports it complete, so KV bytes and index tensors
    are final before NIXL reads them.
    """
    if not device_module.is_available():
        return None
    event = device_module.Event()
    event.record()
    return event


def _pool_policy(transfer: PoolTransfer) -> tuple[str, int]:
    if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
        return ("trailing_pages", max(1, len(transfer.keys or ())))
    return ("all_pages", 0)


def _default_copy_engine_factory(regions):
    """KVCR's CUDA copy engine over this rank's registered GPU pools."""
    from kvcr.device_copy import create_device_copy_engine

    return create_device_copy_engine(regions)


class _RestorePlan(msgspec.Struct, frozen=True):
    """Per-pool span geometry for direct restores.

    ``sizes`` is in layout span order. ``order`` lists the layout's span
    indices layer-major (the spans landing logical layer 0 first, then layer
    1, and so on) and ``ordered_*`` are the geometry permuted the same way, so
    a page batch laid out span-major over pages has every layer as one
    contiguous slice; ``layer_slices`` gives each layer's [start, end) span
    range in that order. A span shared by several logical layers (packed draft
    mappings) sits under the smallest one, so no layer completes before every
    span it reads landed; spans no layer maps to are landed with the last
    layer.
    """

    device_id: int
    sizes: np.ndarray
    order: np.ndarray
    ordered_bases: np.ndarray
    ordered_strides: np.ndarray
    ordered_sizes: np.ndarray
    layer_slices: tuple[tuple[int, int, int], ...]


def _build_restore_plans(
    pool_group, layouts, num_layers: int
) -> dict[str, _RestorePlan]:
    plans: dict[str, _RestorePlan] = {}
    for entry in pool_group.entries:
        layout = layouts[str(entry.name)]
        by_buffer: dict[int, int] = {}
        for logical, mapped in entry.layer_mapping.items():
            for buffer_index in [mapped] if isinstance(mapped, int) else mapped:
                by_buffer[buffer_index] = min(
                    by_buffer.get(buffer_index, logical), logical
                )
        span_layers: list[int] = []
        for component in entry.buffer_meta:
            for buffer_index in range(len(component)):
                span_layers.append(by_buffer.get(buffer_index, num_layers - 1))
        if len(span_layers) != len(layout.spans):
            raise ValueError(
                f"KVCR linker pool {entry.name} layout has {len(layout.spans)} spans "
                f"but its buffers describe {len(span_layers)}."
            )
        order_parts: list[np.ndarray] = []
        layer_slices: list[tuple[int, int, int]] = []
        for layer in sorted(set(span_layers)):
            columns = np.fromiter(
                (i for i, l in enumerate(span_layers) if l == layer), dtype=np.intp
            )
            start = sum(len(part) for part in order_parts)
            layer_slices.append((layer, start, start + len(columns)))
            order_parts.append(columns)
        order = np.concatenate(order_parts)
        bases = np.fromiter((b for b, _, _ in layout.spans), dtype=np.uint64)
        strides = np.fromiter((s for _, s, _ in layout.spans), dtype=np.uint64)
        sizes = np.fromiter((z for _, _, z in layout.spans), dtype=np.uint64)
        plans[str(entry.name)] = _RestorePlan(
            device_id=layout.device_id,
            sizes=sizes,
            order=order,
            ordered_bases=bases[order],
            ordered_strides=strides[order],
            ordered_sizes=sizes[order],
            layer_slices=tuple(layer_slices),
        )
    return plans


class KVCRDirectLinker(UnifiedCacheLinker):
    prepares_requests = True
    publishes_external_events = True

    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        _kvcr_factory: Optional[Callable[..., Any]] = None,
        _nixl_probe: Optional[Callable[[str], set[str]]] = None,
        _copy_engine_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        self.config = KVCRLinkerConfig.from_extra_config(extra_config)
        self.requires_precommit_load = self.config.direct_remote_restore
        self.loaded_pages_are_stored = not self.config.direct_remote_restore
        # Never recycle destinations after a transfer with unproven quiescence.
        self._uncertain_transfer_error: Optional[BaseException] = None
        if self.config.gil_switch_interval_ms is not None:
            # Process-wide: the owner thread and the scheduler share this
            # interpreter, and the default 5 ms slice is long against a
            # single request's launch path.
            sys.setswitchinterval(self.config.gil_switch_interval_ms / 1000.0)
            logger.info(
                "KVCR linker set the interpreter switch interval to %.3f ms",
                self.config.gil_switch_interval_ms,
            )
        self.page_size = params.page_size
        self._params = params
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        if self.num_layers <= 0:
            raise ValueError("KVCR linker requires at least one logical layer.")
        self.layouts: dict[str, PoolObjectLayout] = build_pool_object_layouts(
            self.pool_group
        )
        self._object_bytes = sum(
            layout.object_bytes for layout in self.layouts.values()
        )

        self._resolve_ranks(params)
        self._validate_platform(_nixl_probe)
        self._validate_speculative(params)
        self.digest = compatibility_digest_for(self._compatibility_identity())

        capacity_divisors = {}
        if str(PoolName.MAMBA) in self.layouts:
            track_grid = mamba_track_grid(self.page_size)
            if track_grid % self.page_size:
                raise ValueError(
                    "Mamba checkpoint grid must be divisible by the linker page "
                    f"size: {track_grid} vs {self.page_size}."
                )
            capacity_divisors[str(PoolName.MAMBA)] = track_grid // self.page_size
        self.plan: CapacityPlan = plan_capacity(
            self.layouts,
            self._rank_budget_bytes(),
            capacity_divisors=capacity_divisors,
        )
        self._local_dram = torch.empty(
            self.plan.total_bytes,
            dtype=torch.uint8,
            pin_memory=self.config.pin_local_dram,
        )
        self._framework_regions = self._build_framework_regions()
        self._kvcr_factory = _kvcr_factory
        self.agent_name = self._new_agent_name()
        self._key_adapter = KVCRLinkerKeyAdapter()
        self._pinning = _NoFrameworkPinning()
        self._telemetry = _LinkerTelemetry() if self.config.enable_telemetry else None
        self._control = self._build_control_channel()
        # Restores copy claimed KVCR slots straight into GPU pages from this
        # process, one logical layer at a time, so compute can start on a
        # layer while later ones are still landing. Without an engine (no CUDA
        # runtime, or disabled) restores go through KVCR deliver instead.
        self._copy_engine = (
            (_copy_engine_factory or _default_copy_engine_factory)(
                self._framework_regions
            )
            if self.config.direct_restore
            else None
        )
        self._restore_plans = _build_restore_plans(
            self.pool_group, self.layouts, self.num_layers
        )
        # MemDescriptor geometry is immutable while these registered HBM pools
        # live. The owner thread is the sole reader/writer of this lazy cache.
        self._direct_remote_descriptors: dict[tuple[str, int], tuple[Any, ...]] = {}

        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        # Reentrant: owner-thread completions hold it while releasing claims,
        # and the synchronous release path takes it again to count results.
        self._lock = threading.RLock()
        self._generation = 0
        self._preparations: dict[CacheRequestHandle, _Preparation] = {}
        self._by_rid: dict[str, _Preparation] = {}
        self._pending_loads: dict[str, list[_LoadPool]] = {}
        # rid -> when load() accepted it; peer-hinted rids log their restore.
        self._load_requested_at: dict[str, float] = {}
        self._hinted_rids: set[str] = set()
        self._completed_loads: collections.deque[list[str]] = collections.deque()
        self._offload_tasks: collections.deque[_OffloadTask] = collections.deque()
        self._offload_results: collections.deque[bool] = collections.deque()
        self._deferred: list[tuple[Any, Callable[[], None]]] = []
        self._removed_pages: list[str] = []
        # Per physical pool, the page hashes this rank knows to be in its KVCR
        # DRAM: added when a deposit or fetch confirms them, dropped on KVCR
        # removal events. A request whose first tail page is absent from an
        # all-pages pool cannot restore anything, so it is marked a miss on the
        # scheduler thread without a round trip through the owner thread.
        # Staleness only ever costs a hit, never correctness: presence is still
        # confirmed by fetch.
        self._resident_pages: dict[str, set[str]] = collections.defaultdict(set)
        # Direct restores whose copies are still landing (owner thread only).
        self._direct_batches: list[_LoadBatch] = []
        self._inflight_prepare_bytes = 0
        self._inflight_offload_bytes = 0
        self._abandoned_bytes = 0
        self._unhealthy: Optional[BaseException] = None
        self._closed = False
        self._gc_frozen = False
        self._quarantine: list[tuple[Any, torch.Tensor]] = []
        self._index_stream = None
        self.stats: dict[str, float] = collections.defaultdict(float)
        self._next_stats_log = time.monotonic() + self.config.stats_log_interval_s

        self._kvcr = self._build_kvcr()
        self._adapter = self._start_adapter(self._kvcr)
        self._log_startup()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _resolve_ranks(self, params: CacheInitParams) -> None:
        parallel = get_parallel()
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        self.tp_rank = 0
        self.tp_size = parallel.tp_size
        self.world_rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.tp_rank = torch.distributed.get_rank(group=tp_group)
            self.tp_size = torch.distributed.get_world_size(group=tp_group)
            self.world_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        self.attn_cp_rank = params.attn_cp_rank
        self.attn_cp_size = params.attn_cp_size
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        if self.pp_size > 1:
            raise ValueError(
                "KVCR linker does not support pipeline parallelism: stages would "
                "share a control port block and a key namespace."
            )
        dp_size = parallel.dp_size or 1
        if dp_size > 1 and not parallel.enable_dp_attention:
            raise ValueError(
                "KVCR linker requires --enable-dp-attention when --dp-size > 1: "
                "independent data-parallel replicas cannot derive distinct control "
                "ports from their rank coordinate."
            )
        nnodes = parallel.nnodes or 1
        self.local_ranks = max(1, world_size // nnodes)
        self.rank_replicated = self.pool_group.rank_replicated

    def _rank_budget_bytes(self) -> int:
        return self.config.local_dram_bytes_per_worker // self.local_ranks

    def _validate_platform(self, probe: Optional[Callable[[str], set[str]]]) -> None:
        backend = self.config.nixl_backend
        if backend not in _SUPPORTED_NIXL_BACKENDS:
            raise ValueError(
                f"KVCR linker nixl_backend {backend!r} is not validated; supported: "
                f"{sorted(_SUPPORTED_NIXL_BACKENDS)}."
            )
        if is_hip():
            raise RuntimeError(
                "KVCR linker has not been validated on ROCm; the NIXL VRAM path "
                "needs an explicit ROCm capability check and an MI355X run before "
                "it can be enabled there."
            )
        mem_types = {layout.mem_type for layout in self.layouts.values()}
        supported = probe(backend) if probe is not None else self._probe_nixl(backend)
        missing = {mem for mem in mem_types if f"{mem}_SEG" not in supported}
        if missing:
            raise RuntimeError(
                f"NIXL backend {backend} does not support memory types {sorted(missing)}; "
                f"advertised: {sorted(supported)}."
            )

    @staticmethod
    def _probe_nixl(backend: str) -> set[str]:
        from nixl._api import nixl_agent, nixl_agent_config

        agent = nixl_agent(
            f"kvcr-linker-probe-{uuid.uuid4().hex[:8]}",
            nixl_agent_config(backends=[backend], enable_listen_thread=False),
        )
        try:
            if backend not in agent.get_plugin_list():
                raise RuntimeError(f"NIXL plugin {backend} is not available.")
            return set(agent.get_plugin_mem_types(backend))
        finally:
            del agent

    def _validate_speculative(self, params: CacheInitParams) -> None:
        algorithm = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
        self.spec_algorithm = algorithm
        if algorithm.is_none() or algorithm.is_ngram():
            return
        if not params.mtp_draft_device_pools:
            raise ValueError(
                f"KVCR linker cannot restore draft state for speculative algorithm "
                f"{get_spec().speculative_algorithm}: the device pool layout carries "
                "no draft pools, so a restored target prefix would leave the draft "
                "KV uninitialized."
            )

    def _compatibility_identity(self) -> dict[str, Any]:
        model = get_model()
        spec = get_spec()
        shard = {
            "tp_size": self.tp_size,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_cp_size": self.attn_cp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
        }
        if not self.rank_replicated:
            shard["tp_rank"] = self.tp_rank
        return compatibility_identity(
            model_path=str(model.model_path),
            revision=model.revision,
            dtype=str(model.dtype),
            kv_cache_dtype=str(model.kv_cache_dtype),
            quantization=model.quantization,
            page_size=self.page_size,
            is_eagle=bool(self._params.is_eagle),
            layouts=self.layouts,
            shard=shard,
            speculative={
                "algorithm": spec.speculative_algorithm,
                "draft_model_path": spec.speculative_draft_model_path,
                "draft_model_revision": spec.speculative_draft_model_revision,
                "num_draft_pools": len(self._params.mtp_draft_device_pools),
            },
        )

    def _build_framework_regions(self):
        from kvcr.config import FrameworkMemoryRegion

        return tuple(
            FrameworkMemoryRegion(
                address=address,
                length=length,
                mem_type=mem_type,
                device_id=device_id,
                owner=owner,
            )
            for address, length, mem_type, device_id, owner in unique_allocations(
                self.pool_group
            )
        )

    def _new_agent_name(self) -> str:
        return f"kvcr-sgl-r{self.world_rank}-{uuid.uuid4().hex[:8]}"

    def _control_port(self) -> int:
        configured = self.config.control_port
        if configured <= 0:
            return 0
        highest = configured + self.local_ranks * (get_parallel().nnodes or 1) - 1
        if highest > 65535:
            raise ValueError(
                f"KVCR control_port {configured} leaves no room for this engine's "
                f"ranks: rank {highest - configured} would bind {highest}."
            )
        return configured + self.world_rank

    def _build_control_channel(self):
        if not self.config.enable_remote_hint:
            return None
        from kvcr.control_channels import ZmqPeerControlChannel

        port = self._control_port()
        advertise = self.config.control_advertise_host
        channel = ZmqPeerControlChannel(self.config.control_host, port, advertise)
        self.control_endpoint = channel.endpoint
        return channel

    def _build_kvcr(self):
        from kvcr import KVCR, KVCRBindings
        from kvcr.config import (
            KVCRBackendConfigs,
            KVCRConfig,
            LocalDramOptions,
            RemoteFWDramOptions,
        )

        from sglang.srt.mem_cache.storage.kvcr.kvcr_policy import resolve_policy

        config = KVCRConfig(
            nixl_agent_name=self.agent_name,
            pool_layouts=list(self.plan.pool_layouts),
            enable_telemetry=self.config.enable_telemetry,
            operation_timeout_ms=self.config.operation_timeout_ms,
            abandon_timeout_ms=self.config.abandon_timeout_ms,
            nixl_listen_port=_ephemeral_port(),
        )
        bindings = KVCRBindings(
            request_pin=self._pinning.request_pin,
            poll_pin_results=self._pinning.poll_pin_results,
            release_pin=self._pinning.release_pin,
            cancel_pin_request=self._pinning.cancel_pin_request,
            framework_control=self._control,
            key_adapter=self._key_adapter,
            inventory_sink=self._on_inventory_event,
            policy=resolve_policy(self.config.policy),
            on_resilience_event=self._on_resilience_event,
            stats_factory=(
                (lambda: self._telemetry) if self._telemetry is not None else None
            ),
        )
        backend_configs = KVCRBackendConfigs(
            framework_regions=self._framework_regions,
            local_dram=LocalDramOptions(
                carve_local_dram(self.plan, self._local_dram),
                backend=self.config.nixl_backend,
                device_copy=self.config.device_copy,
            ),
            remote_fw_dram=RemoteFWDramOptions(
                eager_ctrl_connect=self.config.eager_ctrl_connect,
                opportunistic_query=self.config.opportunistic_query,
                metadata_retry_interval_ms=self.config.metadata_retry_interval_ms,
                backend=self.config.nixl_backend,
            ),
        )
        factory = self._kvcr_factory or KVCR
        try:
            return factory(config, bindings, backend_configs)
        except Exception as error:
            # NIXL registers GPU memory with every transport UCX selected; an
            # InfiniBand device without GPUDirect RDMA peer memory refuses it.
            raise RuntimeError(
                "KVCR linker could not start its KVCR core (registering "
                f"{len(self._framework_regions)} framework regions over "
                f"{self.config.nixl_backend}). If the log shows ibv_reg_mr "
                "failing on a cuda address, the host has no GPUDirect RDMA peer "
                "memory: set UCX_TLS=cuda_copy,cuda_ipc,sm,tcp for same-node "
                "transfers or enable nvidia_peermem."
            ) from error

    def _start_adapter(self, kvcr) -> KVCRAdapter:
        adapter = KVCRAdapter(
            kvcr,
            poll_interval_s=self.config.poll_interval_ms / 1000.0,
            name=f"r{self.world_rank}",
            on_unhealthy=self._on_unhealthy,
            idle_poll_interval_s=self.config.idle_poll_interval_ms / 1000.0,
        )
        adapter.add_ticker(self._tick)
        # The ticker reads self._adapter and the stats clock from the owner
        # thread, so both must be in place before the first iteration runs.
        self._adapter = adapter
        self._next_stats_log = time.monotonic() + self.config.stats_log_interval_s
        adapter.start()
        return adapter

    def _log_startup(self) -> None:
        logger.info(
            "KVCRDirectLinker rank=%d/%d agent=%s digest=%s pools=%s page_bytes=%d "
            "page_capacity=%d pool_capacities=%s dram_bytes=%d "
            "unused_tail_bytes=%d remote_hint=%s "
            "control=%s",
            self.world_rank,
            self.local_ranks,
            self.agent_name,
            self.digest,
            {name: list(layout.span_sizes) for name, layout in self.layouts.items()},
            self._object_bytes,
            self.plan.page_capacity,
            self.plan.pool_capacities,
            self.plan.total_bytes,
            self.plan.unused_bytes,
            self.config.enable_remote_hint,
            self._control.endpoint if self._control is not None else None,
        )
        logger.info(
            "KVCRDirectLinker copy paths rank=%d nixl_backend=%s "
            "device_copy_requested=%s local_restore=%s remote_restore=%s "
            "(NIXL self-copy uses this rank's agent; UCX chooses the transport)",
            self.world_rank,
            self.config.nixl_backend,
            self.config.device_copy,
            "cuda_runtime" if self._copy_engine is not None else "nixl_self",
            "nixl_peer_to_hbm"
            if self.config.direct_remote_restore
            else "nixl_peer_to_dram_then_local_restore",
        )
        if self.requires_precommit_load:
            logger.warning(
                "KVCR direct remote reads are confirmed before radix-tree commit; "
                "progressive_remote_restore is not used without source leases. "
                "Restored HBM pages remain eligible for local DRAM offload."
            )

    # ------------------------------------------------------------------
    # Keys and descriptors
    # ------------------------------------------------------------------

    def _key(self, page_hash: str, pool: str):
        return encode_object_key(page_hash, self.digest, pool)

    def _claimed_spans(self, pool: str, descriptors) -> Optional[np.ndarray]:
        """Slot addresses of a claimed page in layout span order.

        None when the claim does not match the pool layout; such a page falls
        back to KVCR deliver rather than being copied from the wrong spans.
        """
        plan = self._restore_plans[pool]
        count = len(plan.sizes)
        if len(descriptors) != count:
            return None
        sizes = np.fromiter(map(_span_size, descriptors), dtype=np.uint64, count=count)
        if not np.array_equal(sizes, plan.sizes):
            return None
        return np.fromiter(map(_span_addr, descriptors), dtype=np.uint64, count=count)

    def _descriptors(self, pool: str, row: int) -> list:
        from kvcr.types import MemDescriptor

        return page_descriptors(self.layouts[pool], row, self.agent_name, MemDescriptor)

    def _direct_remote_page_descriptors(self, pool: str, row: int) -> tuple:
        if not self.config.direct_remote_descriptor_cache:
            return tuple(self._descriptors(pool, row))
        cache_key = (pool, row)
        descriptors = self._direct_remote_descriptors.get(cache_key)
        if descriptors is None:
            descriptors = tuple(self._descriptors(pool, row))
            self._direct_remote_descriptors[cache_key] = descriptors
            self.stats["direct_remote_descriptor_cache_misses"] += 1
        else:
            self.stats["direct_remote_descriptor_cache_hits"] += 1
        return descriptors

    def _rows(self, pool: str, indices: torch.Tensor) -> list[int]:
        return self.pools[PoolName(pool)].prepare_locations(
            self._snapshot_indices(indices)
        )

    def _rows_for_pools(self, pools: list[tuple[str, torch.Tensor]]) -> list[list[int]]:
        """Rows for several pools' device indices with one copy and one sync.

        A DeepSeek V4 page touches six pools; snapshotting them one at a time
        costs six device round trips per restore or offload.
        """
        device = [
            (index, indices)
            for index, (_, indices) in enumerate(pools)
            if indices.device.type == "cuda"
        ]
        rows: list[Optional[list[int]]] = [None] * len(pools)
        if len(device) > 1:
            flat = self._snapshot_indices(
                [indices.detach().flatten() for _, indices in device]
            )
            offset = 0
            for index, indices in device:
                count = indices.numel()
                rows[index] = self.pools[PoolName(pools[index][0])].prepare_locations(
                    flat[offset : offset + count]
                )
                offset += count
        for index, (pool, indices) in enumerate(pools):
            if rows[index] is None:
                rows[index] = self._rows(pool, indices)
        return rows  # type: ignore[return-value]

    def _snapshot_indices(
        self, indices: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        """CPU copy of device indices, taken on the owner thread's own stream.

        A copy on this thread's default stream would wait for every queued
        compute kernel. The producer event already guarantees the indices are
        final, so a private stream only waits for its own copy. Several
        tensors are concatenated on that stream and copied with one sync.
        """
        if isinstance(indices, torch.Tensor):
            if indices.device.type != "cuda":
                return _cpu_indices(indices)
            sources = [indices.detach().flatten()]
        else:
            sources = indices
        device = sources[0].device
        if self._index_stream is None:
            self._index_stream = device_module.Stream(device=device)
        total = sum(source.numel() for source in sources)
        pinned = torch.empty(total, dtype=torch.int64, device="cpu", pin_memory=True)
        with device_module.stream(self._index_stream):
            for source in sources:
                source.record_stream(self._index_stream)
            source = sources[0] if len(sources) == 1 else torch.cat(sources)
            pinned.copy_(source, non_blocking=True)
        self._index_stream.synchronize()
        return pinned

    # ------------------------------------------------------------------
    # Preparation (scheduler thread -> owner thread)
    # ------------------------------------------------------------------

    def prepare_request(
        self, context: LinkerRequestContext, transfers: list[PoolTransfer]
    ) -> None:
        handle = context.request
        expanded = self.pool_group.resolve_transfers(transfers)
        kv = next((t for t in transfers if t.name == PoolName.KV), None)
        page_hashes = list(kv.keys or []) if kv is not None else []
        hint = self._aligned_hint(context.router_hint)
        with self._lock:
            previous = self._by_rid.pop(handle.rid, None)
            if previous is not None:
                self._preparations.pop(previous.handle, None)
                self._retire_preparation_locked(previous, "superseded")
        if not expanded or not page_hashes:
            return
        deadline = time.monotonic() + self.config.preparation_deadline_ms / 1000.0
        pools = {
            str(transfer.name): _PoolPlan(
                pool=str(transfer.name),
                policy=_pool_policy(transfer)[0],
                window=_pool_policy(transfer)[1],
                present=[False] * len(page_hashes),
            )
            for transfer in expanded
        }
        prep = _Preparation(
            handle=handle,
            request_id=f"{handle.rid}#{handle.attempt_id}#{self._generation}",
            page_hashes=page_hashes,
            pools=pools,
            hint=hint,
            deadline=deadline,
        )
        with self._lock:
            self._preparations[handle] = prep
            self._by_rid[handle.rid] = prep
            self.stats["prepare_requests"] += 1
            self.stats["queried_pages"] += len(page_hashes)
            if hint is not None:
                self.stats["hinted_requests"] += 1
                self._hinted_rids.add(handle.rid)
            declined = self._decline_reason_locked()
            if declined is not None:
                self._mark_miss_locked(prep, declined)
                return
            anchor = next(
                (plan for plan in pools.values() if plan.policy == "all_pages"), None
            )
            if (
                hint is None
                and anchor is not None
                and page_hashes[0] not in self._resident_pages[anchor.pool]
            ):
                # Nothing local can satisfy the prefix and no peer was named:
                # a certain miss, decided here so the request is admitted in
                # this scheduling pass. Any all-pages pool works as the anchor
                # because every one of them must hold page 0 for a restore.
                self._mark_miss_locked(prep, "no_local_candidates")
                return
        self._adapter.post(lambda adapter: self._start_preparation(prep))

    def _aligned_hint(self, envelope: object) -> Optional[KVCRFetchHint]:
        if envelope is None or not self.config.enable_remote_hint:
            return None
        hint = parse_fetch_hint(envelope)
        if hint is None:
            with self._lock:
                self.stats["hints_malformed"] += 1
            return None
        # The router resolves the source's DP rank; the within-group attention
        # rank is this rank's own offset because each rank holds its own shard.
        offset = self.attn_cp_rank * self.tp_size + self.tp_rank
        endpoint = offset_control_endpoint(hint.source_control_endpoint, offset)
        if endpoint is None:
            with self._lock:
                self.stats["hints_undialable"] += 1
            return None
        return KVCRFetchHint(
            source_control_endpoint=endpoint, block_hashes=hint.block_hashes
        )

    def _decline_reason_locked(self) -> Optional[str]:
        if self._unhealthy is not None or self._closed:
            return "unhealthy"
        fetching = sum(
            1 for p in self._preparations.values() if p.state == _State.FETCHING
        )
        if fetching > self.config.max_inflight_prepare_requests:
            return "backpressure_requests"
        if self._abandoned_bytes >= self.config.max_abandoned_bytes:
            return "backpressure_abandoned"
        return None

    def _mark_miss_locked(self, prep: _Preparation, reason: str) -> None:
        prep.state = _State.MISS
        prep.restorable = []
        prep.miss_reason = reason
        self.stats[f"miss_{reason}"] += 1

    # ---- owner thread ----

    def _start_preparation(self, prep: _Preparation) -> None:
        from kvcr.types import QueryStatus

        kvcr = self._adapter.kvcr
        prep.owner_started_at = time.monotonic()
        if prep.hint is not None:
            try:
                kvcr.submit_hint(prep.hint.to_kvcr_hint(), request_id=prep.request_id)
            except Exception:
                logger.warning("KVCR submit_hint failed", exc_info=True)
                prep.hint = None
        num_pages = len(prep.page_hashes)
        candidates: dict[str, list[bool]] = {}
        for pool in prep.pools:
            keys = [self._key(page, pool) for page in prep.page_hashes]
            statuses = kvcr.query(keys, request_id=prep.request_id)
            candidates[pool] = [
                status is not QueryStatus.MISS for status, _ in statuses
            ]
        policies = {
            pool: (plan.policy, plan.window) for pool, plan in prep.pools.items()
        }
        boundaries = restorable_boundaries(candidates, policies, num_pages)
        prep.query_done_at = time.monotonic()
        limit = boundaries[-1] if boundaries else 0
        # Bound one request's restore footprint. A direct remote load transfers
        # every ALL_PAGES object but only the selected tail window for a
        # TRAILING_PAGES pool. Charging the latter on every anchor page makes a
        # K3 Mamba checkpoint look 1,812 times larger than the one state object
        # that is actually restored.
        if self.config.direct_remote_restore:
            all_pages_bytes = sum(
                self.layouts[pool].object_bytes
                for pool, plan in prep.pools.items()
                if plan.policy == "all_pages"
            )
            trailing_bytes = sum(
                self.layouts[pool].object_bytes * max(1, plan.window)
                for pool, plan in prep.pools.items()
                if plan.policy == "trailing_pages"
            )
            available = max(
                0, self.config.max_prepare_bytes_per_request - trailing_bytes
            )
            max_pages = (
                available // all_pages_bytes
                if all_pages_bytes
                else (
                    limit
                    if self.config.max_prepare_bytes_per_request >= trailing_bytes
                    else 0
                )
            )
        else:
            # Staged preparation may materialize every candidate pool object.
            max_pages = self.config.max_prepare_bytes_per_request // max(
                1, self._object_bytes
            )
        limit = min(limit, max_pages)
        if self.config.direct_remote_restore:
            # Query is advisory, not a residency claim. The wrapper must use
            # load_before_commit once private device slots exist; only a
            # completed delivery can turn these candidates into a tree hit.
            with self._lock:
                if limit <= 0:
                    self._finish_preparation_locked(prep, reason="no_candidates")
                    self._discard_hint(prep)
                    return
                for pool, plan in prep.pools.items():
                    plan.present[:limit] = candidates[pool][:limit]
                prep.fetch_issued_at = prep.query_done_at
                prep.fetched_at = prep.query_done_at
                self._finish_preparation_locked(prep, reason=None)
                self.stats["direct_remote_prepared_pages"] += limit
            return
        with self._lock:
            room = self.config.max_inflight_prepare_bytes - self._inflight_prepare_bytes
            limit = min(limit, max(0, room // max(1, self._object_bytes)))
            if limit <= 0:
                reason = "no_candidates" if not boundaries else "backpressure_bytes"
                self._finish_preparation_locked(prep, reason=reason)
                self._discard_hint(prep)
                return
            prep.bytes_requested = limit * self._object_bytes
            self._inflight_prepare_bytes += prep.bytes_requested
            self.stats["prepare_inflight_bytes_hwm"] = max(
                self.stats["prepare_inflight_bytes_hwm"], self._inflight_prepare_bytes
            )
        # One KVCR operation per page chunk covers every pool's key for those
        # pages: one control message, one peer write, and one notification
        # per chunk instead of one per pool (DeepSeek V4 has six pools).
        chunk = self.config.fetch_chunk_pages
        pool_layouts = {pool: self.layouts[pool].expected_layout for pool in prep.pools}
        for start in range(0, limit, chunk):
            entries: list[tuple[str, int, Any]] = []
            layouts: dict[Any, list[str]] = {}
            for pool in prep.pools:
                present = candidates[pool]
                for page in range(start, min(start + chunk, limit)):
                    if present[page]:
                        key = self._key(prep.page_hashes[page], pool)
                        entries.append((pool, page, key))
                        layouts[key] = pool_layouts[pool]
            if not entries:
                continue
            first_layout = entries[0][2]
            op = kvcr.fetch(
                [key for _, _, key in entries],
                request_id=prep.request_id,
                expected_layout=layouts[first_layout],
                expected_layouts=layouts,
            )
            prep.outstanding_ops += 1
            self._adapter.track(op, self._fetch_completion(prep, entries))
        prep.fetch_issued_at = time.monotonic()
        if prep.outstanding_ops == 0:
            with self._lock:
                self._finish_preparation_locked(prep, reason="no_candidates")
            self._discard_hint(prep)

    def _fetch_completion(
        self, prep: _Preparation, fetched: list[tuple[str, int, Any]]
    ):
        """Completion for one fetch operation over ``(pool, page, key)`` entries."""

        def completion(entries: Mapping[Any, Any]) -> None:
            confirmed_bytes = 0
            requested_bytes = 0
            late = prep.state != _State.FETCHING
            with self._lock:
                for pool, page, key in fetched:
                    object_bytes = self.layouts[pool].object_bytes
                    requested_bytes += object_bytes
                    entry = entries.get(key)
                    if entry is None or not entry.success:
                        continue
                    if late or (pool, page) in prep.claims:
                        # Nobody waits: release immediately so no claim leaks.
                        self._release_handles([entry.release_handle])
                        self.stats["late_claims_released"] += 1
                        continue
                    prep.claims[(pool, page)] = entry.release_handle
                    if entry.descriptors:
                        spans = self._claimed_spans(pool, entry.descriptors)
                        if spans is not None:
                            prep.spans[(pool, page)] = spans
                    prep.pools[pool].present[page] = True
                    self._resident_pages[pool].add(prep.page_hashes[page])
                    confirmed_bytes += object_bytes
                if late:
                    self._abandoned_bytes = max(
                        0, self._abandoned_bytes - requested_bytes
                    )
                    self.stats["late_completions"] += 1
                    return
                prep.bytes_confirmed += confirmed_bytes
                prep.outstanding_ops -= 1
                if prep.outstanding_ops == 0:
                    prep.fetched_at = time.monotonic()
                    self._finish_preparation_locked(prep, reason=None)
                    finished = True
                else:
                    finished = False
            if finished:
                self._discard_hint(prep)

        return completion

    def _finish_preparation_locked(
        self, prep: _Preparation, *, reason: Optional[str]
    ) -> None:
        """Compute the confirmed restorable set and release unusable claims."""
        if prep.state != _State.FETCHING:
            return
        self._inflight_prepare_bytes -= prep.bytes_requested
        num_pages = len(prep.page_hashes)
        present = {pool: plan.present for pool, plan in prep.pools.items()}
        policies = {
            pool: (plan.policy, plan.window) for pool, plan in prep.pools.items()
        }
        restorable = (
            restorable_boundaries(present, policies, num_pages)
            if reason is None
            else []
        )
        limit = restorable[-1] if restorable else 0
        unusable = [
            claim for (pool, page), claim in prep.claims.items() if page >= limit
        ]
        prep.claims = {k: v for k, v in prep.claims.items() if k[1] < limit}
        self._release_handles(unusable)
        elapsed = time.monotonic() - prep.started_at
        self.stats["prepare_latency_s_sum"] += elapsed
        self.stats["prepare_latency_s_max"] = max(
            self.stats["prepare_latency_s_max"], elapsed
        )
        if limit > 0:
            prep.state = _State.READY
            prep.restorable = restorable
            self.stats["prepared_requests"] += 1
            self.stats["prepared_pages"] += limit
            if prep.peer_hinted:
                self.stats["peer_prepared_pages"] += limit
        else:
            self._mark_miss_locked(prep, reason or "absent")

    def _discard_hint(self, prep: _Preparation) -> None:
        if prep.hint is None:
            return
        try:
            self._adapter.kvcr.discard_hint(prep.request_id)
        except Exception:
            logger.debug("KVCR discard_hint failed", exc_info=True)

    def _release_handles(self, handles: Iterable[int]) -> None:
        """Release claims on the owner thread; safe to call from any thread."""
        handles = [h for h in handles if h is not None]
        if not handles:
            return

        def release(adapter: KVCRAdapter) -> None:
            results = adapter.kvcr.release(handles)
            failed = sum(1 for _, ok in results if not ok)
            with self._lock:
                self.stats["claims_released"] += len(handles) - failed
                self.stats["claims_release_failed"] += failed

        if threading.current_thread() is self._adapter._thread:
            release(self._adapter)
        else:
            try:
                self._adapter.post(release)
            except RuntimeError:
                logger.error(
                    "KVCR linker cannot release %d claims: owner dead", len(handles)
                )

    def _retire_preparation_locked(self, prep: _Preparation, reason: str) -> None:
        """Drop a preparation the scheduler no longer waits for."""
        if prep.state == _State.FETCHING:
            # Outstanding ops keep draining; their claims release on arrival.
            self._inflight_prepare_bytes -= prep.bytes_requested
            self._abandoned_bytes += max(0, prep.bytes_requested - prep.bytes_confirmed)
            self.stats["abandoned_bytes_hwm"] = max(
                self.stats["abandoned_bytes_hwm"], self._abandoned_bytes
            )
        self.stats[f"prepare_retired_{reason}"] += 1
        prep.state = _State.MISS
        prep.miss_reason = reason
        self._release_handles(list(prep.claims.values()))
        prep.claims = {}
        self._discard_hint(prep)

    def _tick(self, now: float) -> bool:
        """Owner-thread periodic work: deadlines, deferred submissions, stats."""
        worked = False
        expired: list[_Preparation] = []
        with self._lock:
            for prep in self._preparations.values():
                if prep.state == _State.FETCHING and now >= prep.deadline:
                    expired.append(prep)
            for prep in expired:
                remaining = prep.outstanding_ops
                self._finish_preparation_locked(prep, reason=None)
                if prep.state == _State.MISS:
                    prep.miss_reason = "deadline"
                self.stats["prepare_deadlines"] += 1
                self._abandoned_bytes += max(
                    0, prep.bytes_requested - prep.bytes_confirmed
                )
                self.stats["abandoned_bytes_hwm"] = max(
                    self.stats["abandoned_bytes_hwm"], self._abandoned_bytes
                )
                prep.outstanding_ops = 0
                worked = worked or remaining > 0
        for prep in expired:
            self._discard_hint(prep)
        if self._direct_batches:
            # Restores are the TTFT-critical path: poll their events every
            # iteration instead of sleeping until the next command.
            self._poll_direct_batches()
            worked = True
        if self._deferred:
            still = []
            for event, submit in self._deferred:
                if event is None or event.query():
                    submit()
                else:
                    still.append((event, submit))
            self._deferred = still
            # A restore waiting on its ready event is TTFT-critical: keep
            # polling instead of parking for the idle interval.
            worked = True
        if now >= self._next_stats_log:
            self._next_stats_log = now + self.config.stats_log_interval_s
            self.log_stats()
        return worked

    # ------------------------------------------------------------------
    # UnifiedCacheLinker: preparation observation and lookup
    # ------------------------------------------------------------------

    def preparation_ready(self, request: CacheRequestHandle) -> bool:
        with self._lock:
            prep = self._preparations.get(request)
            if prep is None:
                return True
            if self._unhealthy is not None and prep.state == _State.FETCHING:
                self._mark_miss_locked(prep, "unhealthy")
            ready = prep.state != _State.FETCHING
            if ready and not prep.ready_observed:
                # Scheduler-observed wait from enqueue to admissibility; the
                # owner-thread preparation time is tracked separately.
                prep.ready_observed = True
                waited = time.monotonic() - prep.started_at
                self.stats["admission_wait_s_sum"] += waited
                self.stats["admission_wait_s_max"] = max(
                    self.stats["admission_wait_s_max"], waited
                )
                if self.stats["prepare_requests"] <= 3 or prep.peer_hinted:
                    # Peer-hinted requests log their whole timeline: enqueue to
                    # owner pickup, owner work until the last fetch is issued,
                    # fetch completion, and the scheduler's observation lag.
                    now = time.monotonic()
                    marks = (
                        prep.started_at,
                        prep.owner_started_at or now,
                        prep.query_done_at or prep.owner_started_at or now,
                        prep.fetch_issued_at or now,
                        prep.fetched_at or now,
                        now,
                    )
                    logger.info(
                        "KVCR linker preparation observed ready: rid=%s state=%s "
                        "pages=%d restorable=%d waited=%.3fs reason=%s "
                        "queue=%.3fs query=%.3fs issue=%.3fs fetch=%.3fs observe=%.3fs",
                        prep.handle.rid,
                        prep.state,
                        len(prep.page_hashes),
                        prep.restorable[-1] if prep.restorable else 0,
                        waited,
                        prep.miss_reason,
                        *(max(0.0, b - a) for a, b in zip(marks, marks[1:])),
                    )
            return ready

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        kv = next((t for t in transfers if t.name == PoolName.KV), None)
        current = list(kv.keys or []) if kv is not None else []
        with self._lock:
            prep = self._by_rid.get(rid)
            if prep is None or not current:
                self.stats["lookup_unprepared"] += 1
                return []
            if prep.state != _State.READY:
                return []
            start = prep.page_index.get(current[0])
            if start is None:
                # The device prefix shrank below the prepared tail; the pages in
                # between were never prepared, so the tail recomputes.
                self._retire_preparation_locked(prep, "tail_shrunk")
                return []
            shifted = [p - start for p in prep.restorable if p > start]
            shifted = [p for p in shifted if p <= len(current)]
            if start > 0:
                # Pages now resident on the device are no longer needed here.
                dropped = [c for (pool, page), c in prep.claims.items() if page < start]
                prep.claims = {k: v for k, v in prep.claims.items() if k[1] >= start}
                self._release_handles(dropped)
            if not shifted:
                self._retire_preparation_locked(prep, "empty_after_shift")
                return []
            self.stats["lookup_hits"] += 1
            return shifted

    def release_request(self, rid: str) -> None:
        with self._lock:
            self._hinted_rids.discard(rid)
            self._load_requested_at.pop(rid, None)
            prep = self._by_rid.pop(rid, None)
            if prep is None:
                return
            self._preparations.pop(prep.handle, None)
            self._retire_preparation_locked(prep, "released")

    def finish_request(self, rid: str) -> None:
        self.release_request(rid)

    # ------------------------------------------------------------------
    # Load: claims -> deliver into adopted GPU indices
    # ------------------------------------------------------------------

    def load(
        self, rid: str, transfers: list[PoolTransfer], *, _precommit: bool = False
    ) -> bool:
        if self.requires_precommit_load and not _precommit:
            raise RuntimeError("Direct KVCR reads require load_before_commit.")
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        with self._lock:
            if rid in self._pending_loads:
                raise RuntimeError(f"KVCR load for rid={rid} is already queued.")
            prep = self._by_rid.pop(rid, None)
            if prep is None or prep.state != _State.READY:
                raise RuntimeError(
                    f"KVCR load for rid={rid} has no prepared residency; lookup "
                    "must not have reported a hit."
                )
            self._preparations.pop(prep.handle, None)
            pools: list[_LoadPool] = []
            for transfer in expanded:
                pool = str(transfer.name)
                pages = list(transfer.keys or [])
                if not pages or transfer.host_indices is None:
                    continue
                claims = []
                spans = []
                for page in pages:
                    index = prep.page_index.get(page)
                    if self.config.direct_remote_restore:
                        if index is None:
                            raise RuntimeError(
                                f"KVCR direct remote load for rid={rid} pool={pool} "
                                "contains a page outside the prepared prefix."
                            )
                        spans.append(None)
                    else:
                        claim = (
                            prep.claims.pop((pool, index), None)
                            if index is not None
                            else None
                        )
                        if claim is None:
                            raise RuntimeError(
                                f"KVCR load for rid={rid} pool={pool} page has no "
                                "held claim; residency was not prepared for this page."
                            )
                        claims.append(claim)
                        spans.append(prep.spans.pop((pool, index), None))
                pools.append(
                    _LoadPool(
                        pool,
                        pages,
                        transfer.host_indices,
                        claims,
                        spans,
                        prep.request_id,
                    )
                )
            leftover = list(prep.claims.values())
            prep.claims = {}
            prep.state = _State.MISS
            self._release_handles(leftover)
            if not self.config.direct_remote_restore:
                self._discard_hint(prep)
            self._pending_loads[rid] = pools
            self._load_requested_at[rid] = time.monotonic()
            # The widest pool is the all-pages prefix; trailing-window pools
            # cover a subset of it.
            self.stats["admitted_pages"] += max(
                (len(pool.page_hashes) for pool in pools), default=0
            )
        return True

    def load_before_commit(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Read directly into unpublished HBM; safely miss if the source is gone.

        No target DRAM staging is introduced. Until KVCR offers a source lease,
        this synchronous gate deliberately trades progressive overlap for safe
        admission. Component PREPARE/COMMIT and tree publication happen later.
        """
        assert self.requires_precommit_load
        if not self.load(rid, transfers, _precommit=True):
            return False
        with self._lock:
            pools = self._pending_loads.pop(rid)
        batch = _LoadBatch(-1, [rid], pools, _ready_event())
        batch.precommit_done = threading.Event()
        batch.bytes = sum(
            len(pool.page_hashes) * self.layouts[pool.pool].object_bytes
            for pool in pools
        )
        batch.requested_at = self._load_requested_at.pop(rid, batch.started_at)
        with self._lock:
            self.stats["load_batches"] += 1
            self.stats["precommit_loads"] += 1
        self._adapter.post(
            lambda adapter: self._defer(
                batch.ready_event, lambda: self._submit_precommit_load(batch)
            )
        )
        deadline = time.monotonic() + self.config.operation_timeout_ms / 1000.0
        while not batch.precommit_done.wait(timeout=0.01):
            self._adapter.raise_if_failed()
            if self._uncertain_transfer_error is not None:
                raise RuntimeError(
                    "KVCR precommit DMA is uncertain; device slots must stay owned"
                ) from self._uncertain_transfer_error
            if time.monotonic() >= deadline:
                # Never return False on timeout: DMA may still own these slots.
                raise RuntimeError(
                    "KVCR precommit read did not drain; device slots must stay owned"
                )
        if self._uncertain_transfer_error is not None:
            raise RuntimeError(
                "KVCR precommit DMA is uncertain; device slots must stay owned"
            ) from self._uncertain_transfer_error
        if batch.error is not None:
            raise RuntimeError("KVCR precommit submission failed") from batch.error
        return batch.success

    def _submit_precommit_load(self, batch: _LoadBatch) -> None:
        """Copy complete objects; no layer is exposed before all reads finish."""
        batch.submit_at = time.monotonic()
        try:
            operations = []
            pool_rows = self._rows_for_pools(
                [(pool.pool, pool.indices) for pool in batch.pools]
            )
            for pool, rows in zip(batch.pools, pool_rows):
                if len(rows) != len(pool.page_hashes):
                    raise RuntimeError("KVCR precommit page/index count mismatch")
                chunk = self.config.fetch_chunk_pages
                for start in range(0, len(rows), chunk):
                    blocks = {
                        self._key(page, pool.pool): list(
                            self._direct_remote_page_descriptors(pool.pool, row)
                        )
                        for page, row in zip(
                            pool.page_hashes[start : start + chunk],
                            rows[start : start + chunk],
                        )
                    }
                    operations.append((pool.request_id, blocks))
            for request_id, blocks in operations:
                op = self._adapter.kvcr.deliver(blocks, request_id=request_id)
                batch.outstanding += 1
                self._adapter.track(op, self._load_completion(batch, list(blocks)))
            with self._lock:
                self.stats["restore_direct_remote_batches"] += 1
                self.stats["restore_direct_remote_operations"] += len(operations)
        except Exception as error:
            # Retain any submitted operations and wait for their completions.
            batch.success = False
            batch.error = error
            logger.exception("KVCR precommit read submission failed")
        if batch.outstanding == 0:
            self._finish_load(batch)

    def cancel_queued_load(self, rid: str) -> bool:
        # The tree already published the destination indices; dropping the
        # transfer would leave a device hit over slots never populated.
        return False

    def num_completed_loads(self) -> int:
        with self._lock:
            return len(self._completed_loads)

    def pop_completed_load(self) -> list[str]:
        with self._lock:
            return self._completed_loads.popleft()

    def start_layer_wise_loading(self) -> int:
        with self._lock:
            if not self._pending_loads:
                return -1
            pending, self._pending_loads = self._pending_loads, {}
        self._freeze_gc_once()
        counter_index = self.layer_done_counter.update_producer()
        ready_event = _ready_event()
        pools = [pool for loads in pending.values() for pool in loads]
        batch = _LoadBatch(counter_index, list(pending), pools, ready_event)
        batch.bytes = sum(
            len(pool.page_hashes) * self.layouts[pool.pool].object_bytes
            for pool in pools
        )
        batch.requested_at = min(
            (self._load_requested_at.pop(rid, batch.started_at) for rid in pending),
            default=batch.started_at,
        )
        with self._lock:
            self.stats["load_batches"] += 1
        self._adapter.post(
            lambda adapter: self._defer(
                batch.ready_event, lambda: self._submit_load(batch)
            )
        )
        return counter_index

    def _defer(self, event, submit: Callable[[], None]) -> None:
        if event is None or event.query():
            submit()
        else:
            self._deferred.append((event, submit))

    def _submit_load(self, batch: _LoadBatch) -> None:
        if self.config.direct_remote_restore:
            self._submit_direct_remote_load(batch)
            return
        if self._copy_engine is not None and all(
            span is not None for pool in batch.pools for span in pool.spans
        ):
            self._submit_direct_load(batch)
            return
        kvcr = self._adapter.kvcr
        try:
            chunk = self.config.fetch_chunk_pages
            for pool in batch.pools:
                rows = self._rows(pool.pool, pool.indices)
                if len(rows) != len(pool.page_hashes):
                    raise RuntimeError(
                        f"KVCR load pool={pool.pool} rows={len(rows)} pages={len(pool.page_hashes)}"
                    )
                for start in range(0, len(rows), chunk):
                    blocks = {
                        self._key(page, pool.pool): self._descriptors(pool.pool, row)
                        for page, row in zip(
                            pool.page_hashes[start : start + chunk],
                            rows[start : start + chunk],
                        )
                    }
                    op = kvcr.deliver(blocks)
                    batch.outstanding += 1
                    self._adapter.track(op, self._load_completion(batch, list(blocks)))
            if batch.outstanding == 0:
                self._finish_load(batch)
        except Exception as error:  # noqa: BLE001 - propagated through the counter
            batch.success = False
            logger.exception("KVCR load submission failed")
            self._finish_load(batch, error=error)

    def _submit_direct_remote_load(self, batch: _LoadBatch) -> None:
        """Pull selected object spans from the hinted peer straight into HBM.

        One KVCR deliver operation covers one request and logical layer.  The
        destination descriptor layout is an ordered subset of the stored
        object's full layout; the paired KVCR ablation patch resolves the
        matching source spans by their ``info`` labels.  Completion therefore
        releases the model's layer future without a target-DRAM staging copy.
        """
        kvcr = self._adapter.kvcr
        started = time.perf_counter()
        batch.submit_at = time.monotonic()
        try:
            grouped: dict[int, dict[str, dict[Any, list[Any]]]] = {}
            pool_rows = self._rows_for_pools(
                [(pool.pool, pool.indices) for pool in batch.pools]
            )
            rows_done = time.perf_counter()
            for pool, rows in zip(batch.pools, pool_rows):
                plan = self._restore_plans[pool.pool]
                if len(rows) != len(pool.page_hashes):
                    raise RuntimeError(
                        f"KVCR direct remote load pool={pool.pool} rows={len(rows)} "
                        f"pages={len(pool.page_hashes)}"
                    )
                # A page's full descriptor list and encoded key do not change
                # across logical layers. Build them once, then partition the
                # descriptor references according to the precomputed plan.
                page_entries = [
                    (
                        self._key(page, pool.pool),
                        self._direct_remote_page_descriptors(pool.pool, row),
                    )
                    for page, row in zip(pool.page_hashes, rows)
                ]
                for layer, start, end in plan.layer_slices:
                    span_indices = plan.order[start:end]
                    blocks = grouped.setdefault(layer, {}).setdefault(
                        pool.request_id, {}
                    )
                    for key, descriptors in page_entries:
                        blocks[key] = [
                            descriptors[int(index)] for index in span_indices
                        ]
            assembled = time.perf_counter()
            operations: list[tuple[list[int], str, dict[Any, list[Any]]]] = []
            for layer in range(self.num_layers):
                request_batches = grouped.get(layer)
                if not request_batches:
                    # Shared spans are delivered at their earliest consumer.
                    # Sequential forward execution has already waited there.
                    if self.config.progressive_remote_restore:
                        self.layer_done_counter.complete(batch.counter_index, layer)
                    continue
                for request_id, blocks in request_batches.items():
                    items = list(blocks.items())
                    if not items:
                        continue
                    chunk_pages = self.config.direct_remote_chunk_pages or len(items)
                    for start in range(0, len(items), chunk_pages):
                        chunk = dict(items[start : start + chunk_pages])
                        operations.append(([layer], request_id, chunk))
                        batch.layer_outstanding[layer] = (
                            batch.layer_outstanding.get(layer, 0) + 1
                        )

            next_operation = 0
            window = self.config.direct_remote_inflight_layers or len(operations)

            def make_completion(layers, keys, *, refill: bool):
                def completion(entries) -> None:
                    ok = all(
                        (entry := entries.get(key)) is not None and entry.success
                        for key in keys
                    )
                    batch.success = batch.success and ok
                    batch.outstanding -= 1
                    for layer in layers:
                        batch.layer_outstanding[layer] -= 1
                        if (
                            self.config.progressive_remote_restore
                            and batch.layer_outstanding[layer] == 0
                            and batch.success
                        ):
                            with self._lock:
                                self.stats[
                                    f"restore_layer_completion_s_sum[{layer}]"
                                ] += time.monotonic() - batch.submit_at
                                self.stats[
                                    f"restore_layer_completion_count[{layer}]"
                                ] += 1
                            self.layer_done_counter.complete(batch.counter_index, layer)
                    if batch.success and refill:
                        try:
                            submit_more()
                        except Exception:
                            # Keep existing deliveries tracked until DMA drains.
                            batch.success = False
                            logger.exception("KVCR direct remote refill failed")
                    if batch.outstanding == 0 and (
                        not batch.success or next_operation == len(operations)
                    ):
                        self._finish_load(batch)

                return completion

            def submit_more() -> None:
                nonlocal next_operation
                while (
                    batch.success
                    and next_operation < len(operations)
                    and batch.outstanding < window
                ):
                    layers, request_id, blocks = operations[next_operation]
                    next_operation += 1
                    op = kvcr.deliver(blocks, request_id=request_id)
                    batch.outstanding += 1
                    keys = list(blocks)
                    self._adapter.track(
                        op,
                        make_completion(layers, keys, refill=True),
                    )

            submit_more()
            submitted = time.perf_counter()
            with self._lock:
                self.stats["restore_direct_remote_batches"] += 1
                self.stats["restore_direct_remote_operations"] += len(operations)
                self.stats["restore_direct_remote_chunk_pages"] += (
                    self.config.direct_remote_chunk_pages
                )
                self.stats["restore_direct_remote_keys"] += sum(
                    len(blocks) for _, _, blocks in operations
                )
                self.stats["restore_direct_remote_descriptors"] += sum(
                    sum(len(descriptors) for descriptors in blocks.values())
                    for _, _, blocks in operations
                )
                self.stats["restore_direct_remote_initial_inflight"] += min(
                    len(operations), window
                )
                self.stats["restore_rows_s_sum"] += rows_done - started
                self.stats["restore_assemble_s_sum"] += assembled - rows_done
                self.stats["restore_submit_s_sum"] += submitted - assembled
                self.stats["restore_build_s_sum"] += submitted - started
            if not operations:
                self._finish_load(batch)
        except Exception as error:  # noqa: BLE001 - propagated through the counter
            batch.success = False
            logger.exception("KVCR direct remote restore submission failed")
            if batch.outstanding == 0:
                self._finish_load(batch, error=error)

    def _load_completion(self, batch: _LoadBatch, keys: list):
        def completion(entries: Mapping[Any, Any]) -> None:
            ok = all((e := entries.get(k)) is not None and e.success for k in keys)
            batch.success = batch.success and ok
            batch.outstanding -= 1
            if batch.outstanding == 0:
                self._finish_load(batch)

        return completion

    def _submit_direct_load(self, batch: _LoadBatch) -> None:
        """Copy claimed slots into GPU pages, one logical layer per request.

        Layers are submitted in ascending order, so their completion events
        fire in the order attention waits for them and the forward pass can
        begin on layer 0 while later layers are still landing.
        """
        engine = self._copy_engine
        started = time.perf_counter()
        batch.submit_at = time.monotonic()
        try:
            per_layer: dict[int, list[tuple[np.ndarray, ...]]] = {}
            device_ids: set[int] = set()
            pool_rows = self._rows_for_pools(
                [(pool.pool, pool.indices) for pool in batch.pools]
            )
            rows_done = time.perf_counter()
            for pool, rows_list in zip(batch.pools, pool_rows):
                plan = self._restore_plans[pool.pool]
                device_ids.add(plan.device_id)
                rows = np.asarray(rows_list, dtype=np.uint64)
                pages = len(pool.page_hashes)
                if len(rows) != pages:
                    raise RuntimeError(
                        f"KVCR load pool={pool.pool} rows={len(rows)} pages={pages}"
                    )
                # Span-major over pages in layer order, so each layer's
                # operands are one contiguous slice and no per-layer gather
                # runs in Python. Claimed spans were checked against the
                # layout when the claim arrived.
                dst = (
                    plan.ordered_bases[None, :]
                    + rows[:, None] * plan.ordered_strides[None, :]
                ).T.ravel()
                src = np.stack(pool.spans)[:, plan.order].T.ravel()
                sizes = np.repeat(plan.ordered_sizes, pages)
                for layer, start, end in plan.layer_slices:
                    low, high = start * pages, end * pages
                    per_layer.setdefault(layer, []).append(
                        (dst[low:high], src[low:high], sizes[low:high])
                    )
            if len(device_ids) != 1:
                raise RuntimeError(
                    f"KVCR load spans several devices: {sorted(device_ids)}"
                )
            device_id = device_ids.pop()
            assemble_done = time.perf_counter()
            # Consecutive layers whose operands are small are merged into one
            # batch: each submission costs a launch and an event regardless of
            # size, and a merged batch still completes in layer order.
            min_bytes = self.config.direct_restore_min_batch_bytes
            group_layers: list[int] = []
            group_parts: list[tuple[np.ndarray, ...]] = []
            group_bytes = 0
            ordered_layers = sorted(per_layer)
            for position, layer in enumerate(ordered_layers):
                parts = per_layer[layer]
                group_layers.append(layer)
                group_parts.extend(parts)
                group_bytes += int(sum(int(part[2].sum()) for part in parts))
                last = position == len(ordered_layers) - 1
                if group_bytes < min_bytes and not last:
                    continue
                if len(group_parts) == 1:
                    dst, src, sizes = group_parts[0]
                else:
                    dst = np.concatenate([part[0] for part in group_parts])
                    src = np.concatenate([part[1] for part in group_parts])
                    sizes = np.concatenate([part[2] for part in group_parts])
                request = engine.request(device_id, dst, sizes, src, sizes)
                if isinstance(request, str):
                    raise RuntimeError(f"KVCR load layers={group_layers}: {request}")
                batch.handles.append((list(group_layers), engine.submit(request)))
                group_layers = []
                group_parts = []
                group_bytes = 0
            finished = time.perf_counter()
            with self._lock:
                self.stats["restore_direct_batches"] += 1
                self.stats["restore_build_s_sum"] += finished - started
                # Build phases: device index snapshot, numpy operand assembly,
                # copy submission (requests, streams, events).
                self.stats["restore_rows_s_sum"] += rows_done - started
                self.stats["restore_assemble_s_sum"] += assemble_done - rows_done
                self.stats["restore_submit_s_sum"] += finished - assemble_done
            self._direct_batches.append(batch)
            self._poll_direct_batches()
        except Exception as error:  # noqa: BLE001 - propagated through the counter
            batch.success = False
            logger.exception("KVCR direct restore submission failed")
            self._finish_load(batch, error=error)

    def _poll_direct_batches(self) -> bool:
        """Advance in-flight direct restores; True while any is still landing.

        Layers complete as their events fire. A failed layer fails the batch,
        but the remaining copies are still awaited so no slot is released
        while a DMA can read it.
        """
        engine = self._copy_engine
        still: list[_LoadBatch] = []
        for batch in self._direct_batches:
            while batch.next_handle < len(batch.handles):
                layers, handle = batch.handles[batch.next_handle]
                result = engine.poll(handle)
                if result is None:
                    break
                batch.next_handle += 1
                if not result:
                    batch.success = False
                elif batch.success:
                    for layer in layers:
                        self.layer_done_counter.complete(batch.counter_index, layer)
            if batch.next_handle >= len(batch.handles):
                self._finish_load(batch)
            else:
                still.append(batch)
        self._direct_batches = still
        return bool(still)

    def _finish_load(
        self, batch: _LoadBatch, error: Optional[BaseException] = None
    ) -> None:
        now = time.monotonic()
        elapsed = now - batch.started_at
        if self.config.direct_remote_restore:
            for request_id in {pool.request_id for pool in batch.pools}:
                try:
                    self._adapter.kvcr.discard_hint(request_id)
                except Exception:
                    logger.debug("KVCR discard_hint failed", exc_info=True)
        claims = [claim for pool in batch.pools for claim in pool.claims]
        if batch.success and error is None:
            if batch.precommit_done is None:
                self.layer_done_counter.complete_all(batch.counter_index)
            submit_at = batch.submit_at or batch.started_at
            with self._lock:
                self.stats["restored_pages"] += max(
                    (len(pool.page_hashes) for pool in batch.pools), default=0
                )
                self.stats["gpu_restore_bytes"] += batch.bytes
                self.stats["gpu_restore_s_sum"] += elapsed
                self.stats["restore_wait_s_sum"] += submit_at - batch.requested_at
                self.stats["restore_copy_s_sum"] += now - submit_at
                hinted = [rid for rid in batch.rids if rid in self._hinted_rids]
                self._hinted_rids.difference_update(batch.rids)
            if hinted:
                logger.info(
                    "KVCR linker restore done: rids=%s bytes=%d wait=%.3fs "
                    "copy=%.3fs total=%.3fs",
                    ",".join(hinted),
                    batch.bytes,
                    submit_at - batch.requested_at,
                    now - submit_at,
                    now - batch.requested_at,
                )
        elif batch.precommit_done is not None:
            # Private slots have not been published and all submitted reads
            # have completed. A normal absent-source failure can recompute.
            # An uncertain native transfer is separately fatal at the gate.
            batch.error = batch.error or error
            with self._lock:
                self.stats["precommit_misses"] += 1
                self._hinted_rids.difference_update(batch.rids)
        else:
            # After admission a failed transfer is not a miss: the tree already
            # exposes the destination slots. Fail the counter so the worker stops
            # instead of computing over uncertain bytes.
            failure = error or RuntimeError("KVCR deliver reported a failed entry")
            self.layer_done_counter.fail(batch.counter_index, failure)
            with self._lock:
                self.stats["uncertain_loads"] += 1
                self._unhealthy = self._unhealthy or failure
            logger.error(
                "KVCR linker GPU load failed; worker must stop", exc_info=error
            )
        self._release_handles(claims)
        if batch.precommit_done is not None:
            batch.precommit_done.set()
        else:
            with self._lock:
                self._completed_loads.append(batch.rids)

    # ------------------------------------------------------------------
    # Offload: GPU pages -> KVCR DRAM through deposit
    # ------------------------------------------------------------------

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        nbytes = sum(
            len(t.keys or ()) * self.layouts[str(t.name)].object_bytes for t in expanded
        )
        with self._lock:
            if self._unhealthy is not None or self._closed:
                return False
            if (
                self._inflight_offload_bytes + nbytes
                > self.config.max_inflight_offload_bytes
            ):
                self.stats["offload_declined_backpressure"] += 1
                return False
            self._inflight_offload_bytes += nbytes
            self.stats["offload_inflight_bytes_hwm"] = max(
                self.stats["offload_inflight_bytes_hwm"], self._inflight_offload_bytes
            )
        self._freeze_gc_once()
        task = _OffloadTask(expanded, _ready_event(), nbytes)
        with self._lock:
            self._offload_tasks.append(task)
            self.stats["offload_tasks"] += 1
        self._adapter.post(
            lambda adapter: self._defer(
                task.ready_event, lambda: self._submit_offload(task)
            )
        )
        return True

    def _offload_chunks(
        self, task: _OffloadTask
    ) -> list[list[tuple[str, list[str], list[int]]]]:
        """Group every pool's pages into deposits of ``offload_chunk_pages``.

        Chunk ``i`` carries pages ``i*n .. (i+1)*n`` of each pool that has
        them, so one KVCR operation (one copy batch, one completion) covers a
        page range across all physical pools, and each owner-thread call stays
        short enough to hand the GIL back to the scheduler between chunks.
        """
        chunk = self.config.offload_chunk_pages
        chunks: list[list[tuple[str, list[str], list[int]]]] = []
        pool_rows = self._rows_for_pools(
            [(str(t.name), t.host_indices) for t in task.transfers]
        )
        for transfer, rows in zip(task.transfers, pool_rows):
            pool = str(transfer.name)
            pages = list(transfer.keys or [])
            if len(rows) != len(pages):
                raise RuntimeError(
                    f"KVCR offload pool={pool} rows={len(rows)} pages={len(pages)}"
                )
            for index, start in enumerate(range(0, len(rows), chunk)):
                if index == len(chunks):
                    chunks.append([])
                chunks[index].append(
                    (pool, pages[start : start + chunk], rows[start : start + chunk])
                )
        return chunks

    def _submit_offload(self, task: _OffloadTask) -> None:
        """Submit an offload one deposit at a time, yielding to queued commands.

        Preparations and restores posted while an offload is being submitted
        are on the TTFT path, so when a command is waiting the remaining
        deposits are deferred to the next owner-loop iteration, after it runs.
        """
        kvcr = self._adapter.kvcr
        try:
            build_started = time.perf_counter()
            if task.chunks is None:
                task.chunks = self._offload_chunks(task)
            while task.next_chunk < len(task.chunks):
                deposited: list[tuple[str, str, Any]] = []
                blocks = {}
                for pool, chunk_pages, chunk_rows in task.chunks[task.next_chunk]:
                    for page, row in zip(chunk_pages, chunk_rows):
                        key = self._key(page, pool)
                        blocks[key] = self._descriptors(pool, row)
                        deposited.append((pool, page, key))
                op = kvcr.deposit(blocks)
                # Filling pages are fetchable: a fetch waits for the fill, so
                # they count as candidates from submission on.
                with self._lock:
                    for pool, chunk_pages, _ in task.chunks[task.next_chunk]:
                        self._resident_pages[pool].update(chunk_pages)
                task.outstanding += 1
                task.next_chunk += 1
                self._adapter.track(op, self._offload_completion(task, deposited))
                if task.next_chunk < len(task.chunks):
                    if self._adapter.has_pending_commands():
                        self._deferred.append(
                            (None, lambda: self._submit_offload(task))
                        )
                        break
                    # Offloads are off the TTFT path; give the scheduler a
                    # guaranteed window between deposits instead of holding
                    # the GIL for the whole task while a forward is being
                    # launched. A zero-length yield only offers the GIL; a
                    # millisecond lets the launch loop actually progress.
                    time.sleep(_OFFLOAD_DEPOSIT_PAUSE_S)
            else:
                task.submitted_all = True
            with self._lock:
                self.stats["descriptor_build_s_sum"] += (
                    time.perf_counter() - build_started
                )
        except Exception:  # noqa: BLE001 - reported as a failed offload
            logger.exception("KVCR offload submission failed")
            task.success = False
            task.submitted_all = True
        if task.submitted_all and task.outstanding == 0:
            self._finish_offload(task)

    def _offload_completion(
        self, task: _OffloadTask, deposited: list[tuple[str, str, Any]]
    ):
        def completion(entries: Mapping[Any, Any]) -> None:
            failed = [
                (pool, page)
                for pool, page, key in deposited
                if (e := entries.get(key)) is None or not e.success
            ]
            if failed:
                task.success = False
                with self._lock:
                    for pool, page in failed:
                        self._resident_pages[pool].discard(page)
            task.outstanding -= 1
            if task.outstanding == 0 and task.submitted_all:
                self._finish_offload(task)

        return completion

    def _finish_offload(self, task: _OffloadTask) -> None:
        with self._lock:
            task.done = True
            self._inflight_offload_bytes -= task.bytes
            self.stats["offload_bytes" if task.success else "offload_failed_bytes"] += (
                task.bytes
            )
            self.stats["offload_s_sum"] += time.monotonic() - task.started_at
            # Results are consumed in submission order even when KVCR finishes
            # tasks out of order.
            while self._offload_tasks and self._offload_tasks[0].done:
                self._offload_results.append(self._offload_tasks.popleft().success)

    def num_completed_offloads(self) -> int:
        with self._lock:
            return len(self._offload_results)

    def pop_completed_offload(self) -> bool:
        with self._lock:
            return self._offload_results.popleft()

    # ------------------------------------------------------------------
    # Inventory, health, stats
    # ------------------------------------------------------------------

    def _on_inventory_event(self, event) -> None:
        if not event.removed:
            return
        pages = unique_page_hashes_from_keys(event.keys)
        with self._lock:
            self._removed_pages.extend(pages)
            # Keys name their pool, but a page evicted from one pool can no
            # longer be restored anyway, so drop it everywhere.
            for resident in self._resident_pages.values():
                resident.difference_update(pages)
            self.stats["inventory_removed_pages"] += len(pages)

    def take_removed_page_hashes(self) -> list[int]:
        """Event hashes (int64) of pages KVCR evicted since the last call."""
        with self._lock:
            pages, self._removed_pages = self._removed_pages, []
        return [page_hash_to_int64(page) for page in pages]

    def _on_resilience_event(self, error: Exception) -> None:
        with self._lock:
            self.stats["resilience_events"] += 1
            if getattr(error, "state", None) == "uncertain":
                self._uncertain_transfer_error = error
        logger.warning("KVCR resilience event: %s", error)

    def _on_unhealthy(self, error: BaseException) -> None:
        with self._lock:
            self._unhealthy = error
            for prep in self._preparations.values():
                if prep.state == _State.FETCHING:
                    self._mark_miss_locked(prep, "unhealthy")
        self.layer_done_counter.fail_all(error)

    def _freeze_gc_once(self) -> None:
        if not self._gc_frozen:
            freeze_gc("KVCR direct linker")
            self._gc_frozen = True

    def snapshot_stats(self) -> dict[str, float]:
        with self._lock:
            snapshot = dict(self.stats)
            snapshot["outstanding_claims"] = sum(
                len(p.claims) for p in self._preparations.values()
            )
            snapshot["preparations_tracked"] = len(self._preparations)
            snapshot["prepare_inflight_bytes"] = self._inflight_prepare_bytes
            snapshot["offload_inflight_bytes"] = self._inflight_offload_bytes
            snapshot["abandoned_bytes"] = self._abandoned_bytes
        snapshot["kvcr_pending_ops"] = self._adapter.pending_ops
        snapshot["kvcr_inflight_ops_hwm"] = self._adapter.inflight_high_water
        snapshot["dram_bytes"] = self.plan.total_bytes
        if self._telemetry is not None:
            snapshot.update(self._telemetry.reduce())
        return snapshot

    def log_stats(self) -> None:
        snapshot = self.snapshot_stats()
        logger.info(
            "KVCRDirectLinker stats rank=%d: %s",
            self.world_rank,
            " ".join(
                f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(snapshot.items())
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _drain(self, timeout_s: float) -> bool:
        """Wait for tracked KVCR operations; False if some are still pending."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._adapter.healthy:
                return False
            if (
                self._adapter.pending_ops == 0
                and not self._deferred
                and not self._direct_batches
            ):
                return True
            time.sleep(_DRAIN_POLL_S)
        return self._adapter.pending_ops == 0 and not self._direct_batches

    def _release_everything(self) -> None:
        with self._lock:
            preps = list(self._preparations.values())
            self._preparations.clear()
            self._by_rid.clear()
            pending = self._pending_loads
            self._pending_loads = {}
            handles = [c for p in preps for c in p.claims.values()]
            handles.extend(
                c for loads in pending.values() for pool in loads for c in pool.claims
            )
            for prep in preps:
                prep.claims = {}
                prep.state = _State.MISS
            self._completed_loads.clear()
            self._offload_tasks.clear()
            self._offload_results.clear()
            self._resident_pages.clear()
            self._inflight_prepare_bytes = 0
            self._inflight_offload_bytes = 0
        self._release_handles(handles)

    def reset(self) -> None:
        """Quiesce, drop all claims and residency, and rebuild the local tier.

        KVCR has no clear operation, so a flush closes the core and constructs
        a new one over the same GPU registrations. A core that cannot prove
        quiescence keeps its DRAM: the replacement gets a fresh buffer.
        """
        if self._closed:
            return
        # Bounded by the core's own operation deadline: work still pending
        # afterwards is quarantined below, never assumed finished.
        drained = self._drain(self.config.operation_timeout_ms / 1000.0)
        self._release_everything()
        self.layer_done_counter.reset()
        self._adapter.stop(timeout_s=5.0)
        # The owner thread is stopped; anything still listed missed the drain
        # and its counter was already reset above.
        self._direct_batches.clear()
        old_kvcr = self._kvcr
        quiescent = drained and self._close_core(old_kvcr)
        if not quiescent:
            logger.error(
                "KVCR linker reset could not prove quiescence; quarantining the "
                "old core and its DRAM tier"
            )
            self._quarantine.append((old_kvcr, self._local_dram))
            self._local_dram = torch.empty(
                self.plan.total_bytes,
                dtype=torch.uint8,
                pin_memory=self.config.pin_local_dram,
            )
        with self._lock:
            self._generation += 1
            self._unhealthy = None
            self._abandoned_bytes = 0
            self._deferred = []
            self.stats["resets"] += 1
        self.agent_name = self._new_agent_name()
        if self._control is not None:
            self._control = self._build_control_channel()
        self._kvcr = self._build_kvcr()
        self._adapter = self._start_adapter(self._kvcr)

    def _close_core(self, kvcr) -> bool:
        try:
            kvcr.close()
        except BaseException:  # noqa: BLE001 - reported, resources retained
            logger.exception("KVCR core did not close cleanly")
            return False
        return True

    def close(self) -> None:
        if self._closed:
            return
        drained = self._drain(self.config.operation_timeout_ms / 1000.0)
        self._release_everything()
        with self._lock:
            self._closed = True
        self.log_stats()
        self._adapter.stop(timeout_s=5.0)
        self._direct_batches.clear()
        if self._copy_engine is not None and drained:
            self._copy_engine.close()
        if not (drained and self._close_core(self._kvcr)):
            logger.error(
                "KVCR linker close left the core in place: outstanding transfers "
                "may still reference registered memory"
            )
            self._quarantine.append((self._kvcr, self._local_dram))


def _ephemeral_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]
