"""Contract and fault tests for the KVCR direct linker.

The real ``KVCRDirectLinker`` runs over CPU tensors with a fake NIXL agent that
moves bytes with ``memmove`` and can hold or fail individual transfers, so the
owner-thread preparation, claim, load, offload, and teardown paths execute
exactly as in production without a GPU.
"""

from __future__ import annotations

import ctypes
import json
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("kvcr")

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.kvcr import kvcr_direct_linker as linker_module
from sglang.srt.mem_cache.storage.kvcr.kvcr_config import KVCRLinkerConfig
from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import KVCRDirectLinker
from sglang.srt.mem_cache.storage.kvcr.kvcr_layout import restorable_boundaries
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    KVCRLinkerKeyAdapter,
    encode_object_key,
    normalize_block_hash,
    parse_fetch_hint,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LinkerRequestContext,
)
from sglang.srt.mem_cache.utils import hash_str_to_int64
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PAGE = 2
LAYERS = 3
ROW_BYTES = 8
TIMEOUT_S = 10.0


class FakeNixlAgent:
    """Loopback NIXL agent: WRITE to self copies bytes; transfers can be held."""

    def __init__(self):
        self.name = ""
        self.registrations = []
        self.deregistered = []
        self.xfers = []
        self.released = []
        self.notifs = {}
        # handle -> forced state; default DONE. Tests set "PROC" to hold a
        # transfer and "ERR" to fail it.
        self.states: dict[int, str] = {}
        self.default_state = "DONE"
        self.transferred = []

    def register_memory(self, descs, mem_type="DRAM"):
        self.registrations.append((list(descs), mem_type))
        return len(self.registrations)

    def deregister_memory(self, handle):
        self.deregistered.append(handle)

    def get_agent_metadata(self):
        return b"metadata"

    def add_remote_agent(self, metadata):
        return b"remote"

    def get_xfer_descs(self, descs, mem_type="DRAM"):
        return list(descs)

    def initialize_xfer(
        self, op, local_descs, remote_descs, remote_agent, notif_msg=b"", backends=None
    ):
        local_descs, remote_descs = list(local_descs), list(remote_descs)
        assert len(local_descs) == len(remote_descs)
        self.xfers.append((op, local_descs, remote_descs, remote_agent))
        return len(self.xfers)

    def transfer(self, handle, notif_msg=b""):
        op, local_descs, remote_descs, remote_agent = self.xfers[handle - 1]
        if self.states.get(handle, self.default_state) == "ERR":
            return "ERR"
        if op == "WRITE" and remote_agent == self.name:
            for (src, size, _), (dst, _, _) in zip(local_descs, remote_descs):
                ctypes.memmove(dst, src, size)
        self.transferred.append(handle)
        return "PROC"

    def check_xfer_state(self, handle):
        return self.states.get(handle, self.default_state)

    def release_xfer_handle(self, handle):
        self.released.append(handle)

    def send_notif(self, agent_name, notif_msg):
        return None

    def get_new_notifs(self, backends=None):
        notifs, self.notifs = self.notifs, {}
        return notifs


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pool_group(*, with_swa: bool, rows: int = 64) -> tuple[DevicePoolGroup, dict]:
    """CPU stand-ins for device pools: rows are token slots, pages are PAGE rows."""
    buffers = {
        "k": [torch.zeros(rows, ROW_BYTES, dtype=torch.uint8) for _ in range(LAYERS)],
        "v": [torch.zeros(rows, ROW_BYTES, dtype=torch.uint8) for _ in range(LAYERS)],
    }
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[buffers["k"], buffers["v"]],
            layer_mapping={i: i for i in range(LAYERS)},
            page_size=PAGE,
            rows_are_pages=False,
        )
    ]
    if with_swa:
        buffers["swa"] = [torch.zeros(rows, ROW_BYTES, dtype=torch.uint8)]
        entries.append(
            DevicePoolEntry(
                name=PoolName.SWA,
                indices_from_pool=PoolName.SWA,
                device_pool=None,
                components=[buffers["swa"]],
                layer_mapping={0: 0},
                page_size=PAGE,
                rows_are_pages=False,
            )
        )
    return DevicePoolGroup(entries, LAYERS, PAGE), buffers


def _publish_args(extra: dict) -> None:
    config = {
        "local_dram_bytes_per_worker": 1 << 20,
        "pin_local_dram": False,
        "preparation_deadline_ms": 2000,
        "operation_timeout_ms": 500,
        "abandon_timeout_ms": 1000,
        "poll_interval_ms": 0.2,
        "fetch_chunk_pages": 2,
        "offload_chunk_pages": 2,
        # Fake pages are a few bytes; keep one copy batch per layer so the
        # layer-ordering tests see every layer land separately.
        "direct_restore_min_batch_bytes": 0,
    }
    config.update(extra)
    args = ServerArgs(
        model_path="dummy",
        page_size=PAGE,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="kvcr",
        hicache_storage_backend_extra_config=json.dumps(config),
    )
    set_global_server_args_for_scheduler(args)


class Harness:
    """A linker over fake pools plus the fake agent it talks to."""

    def __init__(
        self,
        *,
        with_swa: bool = False,
        extra: dict | None = None,
        copy_engine=None,
    ):
        _publish_args(extra or {})
        self.copy_engine = copy_engine
        self.agent = FakeNixlAgent()
        self.group, self.buffers = _pool_group(with_swa=with_swa)
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=1,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        import kvcr.progress as kvcr_progress
        from kvcr import KVCR

        agent = self.agent

        def factory(config, bindings, backend_configs):
            def make_agent(name, *_):
                agent.name = name
                return agent

            with patch.multiple(
                kvcr_progress,
                nixl_agent=make_agent,
                nixl_agent_config=lambda **kwargs: kwargs,
            ):
                return KVCR(config, bindings, backend_configs)

        with patch.object(
            linker_module, "resolve_hybrid_device_pool_group", return_value=self.group
        ):
            self.linker = KVCRDirectLinker(
                None,
                params,
                components=set(),
                _kvcr_factory=factory,
                _nixl_probe=lambda backend: {"DRAM_SEG", "VRAM_SEG"},
                _copy_engine_factory=(
                    (lambda regions: copy_engine) if copy_engine is not None else None
                ),
            )

    # -- helpers --------------------------------------------------------

    def page_indices(self, first_page: int, num_pages: int) -> torch.Tensor:
        return torch.arange(first_page * PAGE, (first_page + num_pages) * PAGE)

    def fill(self, first_page: int, num_pages: int, seed: int) -> None:
        for name, layers in self.buffers.items():
            for layer, buffer in enumerate(layers):
                rows = buffer[first_page * PAGE : (first_page + num_pages) * PAGE]
                rows.copy_(
                    torch.arange(rows.numel(), dtype=torch.int64)
                    .add(seed * 7 + layer * 13 + hash(name) % 5)
                    .remainder(251)
                    .to(torch.uint8)
                    .reshape(rows.shape)
                )

    def snapshot(self, first_page: int, num_pages: int) -> dict:
        return {
            name: [
                buffer[first_page * PAGE : (first_page + num_pages) * PAGE].clone()
                for buffer in layers
            ]
            for name, layers in self.buffers.items()
        }

    def offload(self, hashes: list[str], first_page: int, *, swa_tail: int = 0) -> None:
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=self.page_indices(first_page, len(hashes)),
                keys=list(hashes),
            )
        ]
        if swa_tail:
            transfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self.page_indices(
                        first_page + len(hashes) - swa_tail, swa_tail
                    ),
                    keys=list(hashes[-swa_tail:]),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        assert self.linker.offload(transfers)

    def wait_offloads(self, count: int) -> list[bool]:
        self.wait(lambda: self.linker.num_completed_offloads() >= count)
        return [self.linker.pop_completed_offload() for _ in range(count)]

    def lookup_transfers(self, hashes: list[str], *, swa_window: int = 0):
        transfers = [PoolTransfer(name=PoolName.KV, keys=list(hashes))]
        if swa_window:
            transfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=list(hashes[-swa_window:]),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        return transfers

    def prepare(
        self,
        rid: str,
        hashes: list[str],
        *,
        attempt: int = 0,
        hint=None,
        swa_window: int = 0,
    ):
        handle = CacheRequestHandle(rid=rid, attempt_id=attempt)
        self.linker.prepare_request(
            LinkerRequestContext(request=handle, router_hint=hint),
            self.lookup_transfers(hashes, swa_window=swa_window),
        )
        return handle

    def wait_ready(self, handle: CacheRequestHandle) -> None:
        self.wait(lambda: self.linker.preparation_ready(handle))

    def load(
        self, rid: str, hashes: list[str], first_page: int, *, swa_tail: int = 0
    ) -> int:
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=self.page_indices(first_page, len(hashes)),
                keys=list(hashes),
            )
        ]
        if swa_tail:
            transfers.append(
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self.page_indices(
                        first_page + len(hashes) - swa_tail, swa_tail
                    ),
                    keys=list(hashes[-swa_tail:]),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        assert self.linker.load(rid, transfers)
        return self.linker.start_layer_wise_loading()

    def wait_loads(self, count: int) -> list[list[str]]:
        self.wait(lambda: self.linker.num_completed_loads() >= count)
        return [self.linker.pop_completed_load() for _ in range(count)]

    def public_claims(self) -> int:
        return len(self.linker._kvcr._core._local_dram._public_claims)

    @staticmethod
    def wait(predicate, timeout: float = TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        raise AssertionError("condition not reached")

    def close(self) -> None:
        self.linker.close()


def _hashes(prefix: str, count: int) -> list[str]:
    """Page hashes shaped like production ones: 64 hex chars each."""
    import hashlib

    return [hashlib.sha256(f"{prefix}{i}".encode()).hexdigest() for i in range(count)]


@pytest.fixture
def harness():
    created = []

    def make(**kwargs):
        h = Harness(**kwargs)
        created.append(h)
        return h

    yield make
    for h in created:
        h.close()


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_startup_logs_nixl_only_copy_paths(harness, caplog):
    with caplog.at_level("INFO"):
        harness(extra={"device_copy": False, "direct_restore": False})
    assert "device_copy_requested=False local_restore=nixl_self" in caplog.text
    assert "remote_restore=nixl_peer_to_dram_then_local_restore" in caplog.text


def test_offload_prepare_lookup_load_round_trip_moves_bytes(harness):
    h = harness()
    hashes = _hashes("a", 4)
    h.fill(0, 4, seed=1)
    expected = h.snapshot(0, 4)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]

    handle = h.prepare("r1", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r1", h.lookup_transfers(hashes)) == [1, 2, 3, 4]
    # Claims are held between lookup and load so the pages cannot be evicted.
    assert h.public_claims() == 4

    index = h.load("r1", hashes, first_page=8)
    assert index >= 0
    h.linker.layer_done_counter.set_consumer(index)
    assert h.wait_loads(1) == [["r1"]]
    h.linker.layer_done_counter.wait_until(LAYERS - 1)
    restored = h.snapshot(8, 4)
    for name in expected:
        for got, want in zip(restored[name], expected[name]):
            assert torch.equal(got, want), name
    h.wait(lambda: h.public_claims() == 0)
    stats = h.linker.snapshot_stats()
    assert stats["restored_pages"] == 4
    assert stats["offload_bytes"] > 0
    assert stats["gpu_restore_bytes"] == stats["offload_bytes"]


class FakeCopyEngine:
    """Stand-in for KVCR's CUDA copy engine over CPU buffers.

    Copies land with ``memmove`` when a handle has been polled ``latency``
    times, so layers become visible one request at a time in submission order.
    """

    def __init__(self, *, latency: int = 1, fail_layer: int | None = None):
        self.latency = latency
        self.fail_layer = fail_layer
        self.submitted: list = []
        self.closed = False
        # When set, requests beyond this submission index stay in flight.
        self.release_up_to: int | None = None

    @staticmethod
    def request(device_id, dst_addresses, dst_sizes, src_addresses, src_sizes):
        from kvcr.device_copy import DeviceCopyEngine

        return DeviceCopyEngine.request(
            device_id, dst_addresses, dst_sizes, src_addresses, src_sizes
        )

    def submit(self, request):
        handle = SimpleNamespace(
            request=request, remaining=self.latency, index=len(self.submitted)
        )
        self.submitted.append(handle)
        return handle

    def poll(self, handle):
        if self.release_up_to is not None and handle.index > self.release_up_to:
            return None
        if handle.remaining > 0:
            handle.remaining -= 1
            return None
        if handle.index == self.fail_layer:
            return False
        request = handle.request
        if request is not None:
            for dst, src, size in zip(
                request.dst_addresses.tolist(),
                request.src_addresses.tolist(),
                request.sizes.tolist(),
            ):
                ctypes.memmove(dst, src, size)
            handle.request = None
        return True

    def close(self):
        self.closed = True


def test_direct_restore_lands_layers_in_order_from_claimed_slots(harness):
    engine = FakeCopyEngine(latency=2)
    h = harness(copy_engine=engine)
    hashes = _hashes("dr", 4)
    h.fill(0, 4, seed=21)
    expected = h.snapshot(0, 4)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("rd", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("rd", h.lookup_transfers(hashes)) == [1, 2, 3, 4]
    index = h.load("rd", hashes, first_page=8)
    h.linker.layer_done_counter.set_consumer(index)
    assert h.wait_loads(1) == [["rd"]]
    # One request per logical layer, ascending, each covering every page's
    # K and V span of that layer.
    assert len(engine.submitted) == LAYERS
    for layer, submitted in enumerate(engine.submitted):
        assert submitted.request is None, f"layer {layer} never landed"
    h.linker.layer_done_counter.wait_until(LAYERS - 1)
    restored = h.snapshot(8, 4)
    for name in expected:
        for got, want in zip(restored[name], expected[name]):
            assert torch.equal(got, want), name
    h.wait(lambda: h.public_claims() == 0)
    stats = h.linker.snapshot_stats()
    assert stats["restore_direct_batches"] == 1
    assert stats["restored_pages"] == 4
    assert stats["gpu_restore_bytes"] == stats["offload_bytes"]
    assert stats.get("load_batches", 0) == 1
    h.close()
    assert engine.closed


def test_direct_restore_merges_small_layers_into_one_batch(harness):
    # Layers whose operands stay under the threshold share one copy batch;
    # the counter still releases every layer and the bytes still land.
    engine = FakeCopyEngine(latency=1)
    h = harness(copy_engine=engine, extra={"direct_restore_min_batch_bytes": 1 << 30})
    hashes = _hashes("dm", 4)
    h.fill(0, 4, seed=24)
    expected = h.snapshot(0, 4)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("rm", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("rm", h.lookup_transfers(hashes)) == [1, 2, 3, 4]
    index = h.load("rm", hashes, first_page=8)
    h.linker.layer_done_counter.set_consumer(index)
    assert h.wait_loads(1) == [["rm"]]
    assert len(engine.submitted) == 1
    assert engine.submitted[0].request is None
    h.linker.layer_done_counter.wait_until(LAYERS - 1)
    restored = h.snapshot(8, 4)
    for name in expected:
        for got, want in zip(restored[name], expected[name]):
            assert torch.equal(got, want), name
    h.wait(lambda: h.public_claims() == 0)
    stats = h.linker.snapshot_stats()
    assert stats["restore_direct_batches"] == 1
    assert stats["gpu_restore_bytes"] == stats["offload_bytes"]
    h.close()


def test_direct_peer_path_also_restores_locally_resident_objects(harness):
    # Core delivery chooses local DRAM before remote hints. The layer-subset
    # protocol must therefore work for a target's existing local copy too.
    h = harness(
        with_swa=True,
        extra={"direct_remote_restore": True, "direct_remote_chunk_pages": 1},
    )
    hashes = _hashes("peer-local", 4)
    h.fill(0, 4, seed=42)
    expected = h.snapshot(0, 4)
    h.offload(hashes, first_page=0, swa_tail=2)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("local-peer", hashes, swa_window=2)
    h.wait_ready(handle)
    assert (
        h.linker.lookup("local-peer", h.lookup_transfers(hashes, swa_window=2))[-1] == 4
    )
    assert h.public_claims() == 0
    assert h.linker.requires_precommit_load
    assert not h.linker.loaded_pages_are_stored
    transfers = [
        PoolTransfer(
            name=PoolName.KV, keys=hashes, device_indices=h.page_indices(8, 4)
        ),
        PoolTransfer(
            name=PoolName.SWA,
            keys=hashes[-2:],
            device_indices=h.page_indices(10, 2),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        ),
    ]
    assert h.linker.load_before_commit("local-peer", transfers)
    assert h.linker.num_completed_loads() == 0
    assert h.linker.start_layer_wise_loading() == -1
    restored = h.snapshot(8, 4)
    for name in ("k", "v"):
        for got, want in zip(restored[name], expected[name]):
            assert torch.equal(got, want)


def test_direct_read_cannot_queue_into_published_slots(harness):
    h = harness(extra={"direct_remote_restore": True})
    with pytest.raises(RuntimeError, match="load_before_commit"):
        h.linker.load("unreserved", [])


def test_direct_precommit_missing_source_is_a_safe_miss(harness):
    from kvcr.types import QueryStatus

    h = harness(extra={"direct_remote_restore": True})
    hashes = _hashes("stale-source", 2)
    # Simulate stale advisory metadata without inventing source residency.
    h.linker._resident_pages[str(PoolName.KV)].update(hashes)
    with patch.object(
        h.linker._kvcr, "query", return_value=[(QueryStatus.FETCHABLE, None)] * 2
    ):
        h.wait_ready(h.prepare("stale", hashes))
    assert h.linker.lookup("stale", h.lookup_transfers(hashes)) == [1, 2]
    before = h.snapshot(8, 2)
    assert not h.linker.load_before_commit(
        "stale",
        [
            PoolTransfer(
                name=PoolName.KV, keys=hashes, device_indices=h.page_indices(8, 2)
            )
        ],
    )
    assert h.linker._unhealthy is None
    assert h.linker.snapshot_stats()["precommit_misses"] == 1
    assert h.linker.snapshot_stats()["kvcr_pending_ops"] == 0
    assert h.linker.num_completed_loads() == 0
    for name, layers in before.items():
        for got, want in zip(h.snapshot(8, 2)[name], layers):
            assert torch.equal(got, want)
    # A subsequent ordinary offload/load still works; the worker stays usable.
    h.fill(0, 2, seed=93)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("next", hashes))
    assert h.linker.load_before_commit(
        "next",
        [
            PoolTransfer(
                name=PoolName.KV, keys=hashes, device_indices=h.page_indices(8, 2)
            )
        ],
    )


def test_direct_precommit_uncertain_dma_is_not_a_recomputable_miss(harness):
    from kvcr.types import TransferError

    h = harness(extra={"direct_remote_restore": True})
    hashes = _hashes("uncertain", 1)
    h.fill(0, 1, seed=94)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    h.wait_ready(h.prepare("uncertain", hashes))
    h.linker._on_resilience_event(TransferError("test", 100, state="uncertain"))
    with pytest.raises(RuntimeError, match="slots must stay owned"):
        h.linker.load_before_commit(
            "uncertain",
            [
                PoolTransfer(
                    name=PoolName.KV, keys=hashes, device_indices=h.page_indices(8, 1)
                )
            ],
        )


def test_direct_restore_layer_lands_before_later_layers(harness):
    # Only the first layer's copy is allowed to land, so the counter must
    # release layer 0 while the last layer is still in flight.
    engine = FakeCopyEngine(latency=1)
    engine.release_up_to = 0
    h = harness(copy_engine=engine)
    hashes = _hashes("dl", 2)
    h.fill(0, 2, seed=22)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("rl", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("rl", h.lookup_transfers(hashes)) == [1, 2]
    index = h.load("rl", hashes, first_page=4)
    h.linker.layer_done_counter.set_consumer(index)
    h.wait(lambda: len(engine.submitted) == LAYERS)
    futures = h.linker.layer_done_counter.futures[index]
    h.wait(lambda: futures[0].done())
    assert not futures[LAYERS - 1].done(), "later layers must not complete early"
    assert h.linker.num_completed_loads() == 0
    engine.release_up_to = None
    assert h.wait_loads(1) == [["rl"]]
    h.linker.layer_done_counter.wait_until(LAYERS - 1)
    h.wait(lambda: h.public_claims() == 0)


def test_direct_restore_failure_fails_the_counter_and_awaits_remaining_copies(
    harness,
):
    engine = FakeCopyEngine(latency=1, fail_layer=1)
    h = harness(copy_engine=engine)
    hashes = _hashes("df", 2)
    h.fill(0, 2, seed=23)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("rf", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("rf", h.lookup_transfers(hashes)) == [1, 2]
    index = h.load("rf", hashes, first_page=4)
    h.linker.layer_done_counter.set_consumer(index)
    assert h.wait_loads(1) == [["rf"]]
    with pytest.raises(RuntimeError):
        h.linker.layer_done_counter.wait_until(LAYERS - 1)
    # Every layer's copy was awaited before the claims were released.
    assert all(handle.remaining == 0 for handle in engine.submitted)
    h.wait(lambda: h.public_claims() == 0)
    stats = h.linker.snapshot_stats()
    assert stats["uncertain_loads"] == 1
    assert not h.linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV, device_indices=h.page_indices(0, 1), keys=hashes[:1]
            )
        ]
    ), "an unhealthy linker declines new work"


def test_unprepared_and_unknown_pages_never_hit(harness):
    h = harness()
    hashes = _hashes("b", 3)
    # Nothing offloaded: the first tail page is not resident, so the miss is
    # decided on the calling thread and the request is admissible at once.
    handle = h.prepare("r2", hashes)
    assert h.linker.preparation_ready(handle)
    assert h.linker.lookup("r2", h.lookup_transfers(hashes)) == []
    assert h.public_claims() == 0
    stats = h.linker.snapshot_stats()
    assert stats["miss_no_local_candidates"] == 1
    assert stats.get("prepared_requests", 0) == 0
    # A lookup for a request that was never prepared is a miss, not a hang.
    assert h.linker.lookup("never-prepared", h.lookup_transfers(hashes)) == []


def test_resident_first_page_takes_the_owner_thread_path(harness):
    h = harness()
    hashes = _hashes("b2", 3)
    h.fill(0, 3, seed=2)
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    # Only the tail beyond a resident first page is a candidate: a request
    # starting at a page KVCR never saw is a certain miss, one starting at a
    # resident page is prepared through KVCR and confirmed by fetch.
    stale = h.prepare("r2a", _hashes("b3", 2) + hashes[:1])
    assert h.linker.preparation_ready(stale)
    handle = h.prepare("r2b", hashes[1:])
    h.wait_ready(handle)
    assert h.linker.lookup("r2b", h.lookup_transfers(hashes[1:])) == [1, 2]
    stats = h.linker.snapshot_stats()
    assert stats["miss_no_local_candidates"] == 1
    assert stats["prepared_requests"] == 1
    h.linker.reset()
    # A rebuilt core holds nothing, so the resident set must be empty too.
    after = h.prepare("r2c", hashes)
    assert h.linker.preparation_ready(after)
    assert h.linker.snapshot_stats()["miss_no_local_candidates"] == 2


def test_hinted_but_absent_pages_never_become_hits(harness):
    port = _free_port()
    h = harness(
        extra={
            "enable_remote_hint": True,
            "control_port": port,
            "control_advertise_host": "127.0.0.1",
            "preparation_deadline_ms": 300,
        }
    )
    hashes = _hashes("c", 2)
    hint = {
        "protocol_version": "0.1",
        "message_id": "m",
        "actions": [
            {
                "action_id": "a",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                # A peer that does not exist: nothing ever arrives.
                "payload": {
                    "source_control_endpoint": f"tcp://127.0.0.1:{_free_port()}",
                    "block_hashes": [hash_str_to_int64(x) for x in hashes],
                },
            }
        ],
    }
    handle = h.prepare("r3", hashes, hint=hint)
    started = time.monotonic()
    h.wait_ready(handle)
    assert time.monotonic() - started < TIMEOUT_S
    assert h.linker.lookup("r3", h.lookup_transfers(hashes)) == []
    stats = h.linker.snapshot_stats()
    assert stats["hinted_requests"] == 1
    assert stats.get("prepared_pages", 0) == 0
    assert stats["prepare_deadlines"] == 1
    # The dead-peer fetch is abandoned, not assumed finished: it stays tracked
    # as late work, counted against the abandoned bound, until KVCR's own
    # operation timeout resolves it as failed. Only then is the accounting
    # released; the core keeps the destination slots quarantined internally.
    assert stats["abandoned_bytes"] > 0
    h.wait(lambda: h.linker.snapshot_stats().get("late_completions", 0) >= 1, timeout=5)
    late = h.linker.snapshot_stats()
    assert late["abandoned_bytes"] == 0
    assert late["kvcr_pending_ops"] == 0
    assert h.public_claims() == 0
    # New requests are not blocked by the quarantined work below the bound.
    other = h.prepare("r3b", _hashes("c2", 1))
    h.wait_ready(other)
    assert h.linker.lookup("r3b", h.lookup_transfers(_hashes("c2", 1))) == []


def test_partial_pool_success_selects_only_valid_boundaries(harness):
    h = harness(with_swa=True)
    hashes = _hashes("d", 4)
    h.fill(0, 4, seed=3)
    # Store KV for every page but SWA only for the trailing page of the node.
    h.offload(hashes, first_page=0, swa_tail=1)
    assert h.wait_offloads(1) == [True]

    handle = h.prepare("r4", hashes, swa_window=1)
    h.wait_ready(handle)
    # Only the boundary whose trailing SWA window is present is restorable.
    assert h.linker.lookup("r4", h.lookup_transfers(hashes, swa_window=1)) == [4]


def test_restorable_boundaries_are_sparse_for_trailing_pools():
    present = {"kv": [True] * 5, "swa": [False, True, False, True, False]}
    policies = {"kv": ("all_pages", 0), "swa": ("trailing_pages", 1)}
    assert restorable_boundaries(present, policies, 5) == [2, 4]
    policies["swa"] = ("trailing_pages", 2)
    assert restorable_boundaries(present, policies, 5) == []
    assert restorable_boundaries(
        {"kv": [True, False, True]}, {"kv": ("all_pages", 0)}, 3
    ) == [1]


def test_out_of_order_offload_completions_keep_fifo_results(harness):
    h = harness()
    first, second = _hashes("e", 2), _hashes("f", 2)
    # Hold the first offload's transfer; the second completes immediately.
    h.agent.states[1] = "PROC"
    h.offload(first, first_page=0)
    h.wait(lambda: len(h.agent.xfers) >= 1)
    h.offload(second, first_page=4)
    h.wait(lambda: len(h.agent.transferred) >= 2)
    time.sleep(0.05)
    assert h.linker.num_completed_offloads() == 0
    h.agent.states[1] = "DONE"
    assert h.wait_offloads(2) == [True, True]


def test_offload_submission_yields_between_chunks_and_still_completes(harness):
    h = harness(extra={"fetch_chunk_pages": 1, "offload_chunk_pages": 1})
    hashes = _hashes("y", 4)
    h.fill(0, 4, seed=31)
    expected = h.snapshot(0, 4)
    # Pretend a command is always waiting: every deposit after the first is
    # deferred to a later owner-loop iteration, and the task must still finish
    # exactly once with every page resident.
    with patch.object(h.linker._adapter, "has_pending_commands", return_value=True):
        h.offload(hashes, first_page=0)
        assert h.wait_offloads(1) == [True]
    assert h.linker.num_completed_offloads() == 0
    handle = h.prepare("ry", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("ry", h.lookup_transfers(hashes)) == [1, 2, 3, 4]
    index = h.load("ry", hashes, first_page=8)
    h.linker.layer_done_counter.set_consumer(index)
    assert h.wait_loads(1) == [["ry"]]
    h.linker.layer_done_counter.wait_until(LAYERS - 1)
    restored = h.snapshot(8, 4)
    for name in expected:
        for got, want in zip(restored[name], expected[name]):
            assert torch.equal(got, want), name
    stats = h.linker.snapshot_stats()
    assert stats["offload_tasks"] == 1
    assert stats["offload_inflight_bytes"] == 0


def test_release_and_finish_drain_claims(harness):
    h = harness()
    hashes = _hashes("g", 3)
    h.fill(0, 3, seed=5)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r5", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r5", h.lookup_transfers(hashes)) == [1, 2, 3]
    assert h.public_claims() == 3
    h.linker.release_request("r5")
    h.wait(lambda: h.public_claims() == 0)
    assert h.linker.lookup("r5", h.lookup_transfers(hashes)) == []

    handle = h.prepare("r6", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r6", h.lookup_transfers(hashes)) == [1, 2, 3]
    h.linker.finish_request("r6")
    h.wait(lambda: h.public_claims() == 0)


def test_stale_attempt_cannot_satisfy_a_new_attempt(harness):
    h = harness()
    hashes = _hashes("h", 2)
    h.fill(0, 2, seed=6)
    # Hold the deposit so the pages are still filling: both attempts' fetches
    # then wait on the same fill, and only the live attempt may keep claims.
    h.agent.default_state = "PROC"
    h.offload(hashes, first_page=0)
    h.wait(lambda: len(h.agent.transferred) >= 1)
    first = h.prepare("r7", hashes, attempt=0)
    h.wait(lambda: h.linker.snapshot_stats()["kvcr_pending_ops"] >= 2)
    second = h.prepare("r7", hashes, attempt=1)
    assert h.linker.preparation_ready(first)  # retired, never blocks admission
    h.agent.default_state = "DONE"
    h.wait_ready(second)
    assert h.wait_offloads(1) == [True]
    assert h.linker.lookup("r7", h.lookup_transfers(hashes)) == [1, 2]
    stats = h.linker.snapshot_stats()
    assert stats["prepare_retired_superseded"] == 1
    assert stats["late_claims_released"] == 2
    # Exactly the new attempt's claims remain; the stale ones were released.
    h.wait(lambda: h.linker.snapshot_stats()["kvcr_pending_ops"] == 0)
    assert h.public_claims() == 2


def test_lookup_realigns_to_a_grown_device_prefix(harness):
    h = harness()
    hashes = _hashes("i", 4)
    h.fill(0, 4, seed=7)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r8", hashes)
    h.wait_ready(handle)
    # Another request inserted the first two pages meanwhile: the tail is
    # shorter, boundaries shift, and the now-resident pages' claims drop.
    assert h.linker.lookup("r8", h.lookup_transfers(hashes[2:])) == [1, 2]
    h.wait(lambda: h.public_claims() == 2)


def test_lookup_after_device_eviction_recomputes_instead_of_exposing_gaps(harness):
    h = harness()
    hashes = _hashes("j", 4)
    h.fill(0, 4, seed=8)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r9", hashes[1:])
    h.wait_ready(handle)
    # The device prefix shrank: page 0 was never prepared, so nothing is hit.
    assert h.linker.lookup("r9", h.lookup_transfers(hashes)) == []
    h.wait(lambda: h.public_claims() == 0)
    assert h.linker.snapshot_stats()["prepare_retired_tail_shrunk"] == 1


def test_deadline_yields_miss_and_late_claims_are_released(harness):
    h = harness(extra={"preparation_deadline_ms": 200})
    hashes = _hashes("k", 2)
    h.fill(0, 2, seed=9)
    # A held deposit keeps the pages filling, so the fetch cannot confirm
    # before the preparation deadline.
    h.agent.default_state = "PROC"
    h.offload(hashes, first_page=0)
    h.wait(lambda: len(h.agent.transferred) >= 1)
    handle = h.prepare("r10", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r10", h.lookup_transfers(hashes)) == []
    stats = h.linker.snapshot_stats()
    assert stats["prepare_deadlines"] == 1
    assert stats["abandoned_bytes"] > 0
    h.agent.default_state = "DONE"
    assert h.wait_offloads(1) == [True]
    h.wait(lambda: h.linker.snapshot_stats().get("late_completions", 0) >= 1)
    h.wait(lambda: h.public_claims() == 0)
    assert h.linker.snapshot_stats()["late_claims_released"] == 2
    assert h.linker.snapshot_stats()["abandoned_bytes"] == 0


def test_failed_gpu_load_fails_the_layer_counter_and_stops_new_work(harness):
    h = harness()
    hashes = _hashes("l", 2)
    h.fill(0, 2, seed=10)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    handle = h.prepare("r11", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r11", h.lookup_transfers(hashes)) == [1, 2]
    # The deliver transfer errors: after admission this is not a miss.
    h.agent.default_state = "ERR"
    index = h.load("r11", hashes, first_page=8)
    h.linker.layer_done_counter.set_consumer(index)
    h.wait_loads(1)
    with pytest.raises(RuntimeError, match="KVCR layer-wise KV load failed"):
        h.linker.layer_done_counter.wait_until(0)
    assert h.linker.snapshot_stats()["uncertain_loads"] == 1
    # The backend refuses further work rather than recomputing over it.
    h.agent.default_state = "DONE"
    assert not h.linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV, device_indices=h.page_indices(0, 2), keys=hashes
            )
        ]
    )
    new = h.prepare("r12", hashes)
    assert h.linker.preparation_ready(new)
    assert h.linker.lookup("r12", h.lookup_transfers(hashes)) == []


def test_reset_clears_local_residency(harness):
    h = harness()
    hashes = _hashes("m", 2)
    h.fill(0, 2, seed=11)
    h.offload(hashes, first_page=0)
    h.wait_offloads(1)
    h.linker.reset()
    handle = h.prepare("r13", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r13", h.lookup_transfers(hashes)) == []
    # The rebuilt core is live: a new offload round-trips again.
    h.offload(hashes, first_page=0)
    assert h.wait_offloads(1) == [True]
    handle = h.prepare("r14", hashes)
    h.wait_ready(handle)
    assert h.linker.lookup("r14", h.lookup_transfers(hashes)) == [1, 2]


def test_inventory_removal_reports_page_event_hashes(harness):
    # A tiny tier: the third page evicts the first, and the eviction surfaces
    # as an EXTERNAL removal keyed by the page's event hash.
    h = harness(
        extra={"local_dram_bytes_per_worker": ROW_BYTES * PAGE * (2 * LAYERS) * 2}
    )
    hashes = _hashes("n", 3)
    h.fill(0, 3, seed=12)
    h.offload(hashes[:2], first_page=0)
    assert h.wait_offloads(1) == [True]
    h.offload(hashes[2:], first_page=2)
    assert h.wait_offloads(1) == [True]
    h.wait(lambda: h.linker.snapshot_stats()["inventory_removed_pages"] >= 1)
    removed = h.linker.take_removed_page_hashes()
    assert hash_str_to_int64(hashes[0]) in removed
    assert h.linker.take_removed_page_hashes() == []


def test_offload_backpressure_declines_beyond_inflight_bytes(harness):
    h = harness(extra={"max_inflight_offload_bytes": ROW_BYTES * PAGE * (2 * LAYERS)})
    h.agent.default_state = "PROC"
    h.offload(_hashes("o", 1), first_page=0)
    assert not h.linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=h.page_indices(2, 1),
                keys=_hashes("p", 1),
            )
        ]
    )
    assert h.linker.snapshot_stats()["offload_declined_backpressure"] == 1
    h.agent.default_state = "DONE"
    h.wait_offloads(1)


# ---------------------------------------------------------------------------
# Identity, hints, and configuration
# ---------------------------------------------------------------------------


def test_full_key_identity_survives_event_hash_conversion():
    page = "7f" * 32
    for pool in ("kv", "swa"):
        key = encode_object_key(page, "digest", pool)
        decoded = KVCRLinkerKeyAdapter().decode(key)
        assert decoded == normalize_block_hash(hash_str_to_int64(page))
        assert key.decode().startswith(page + "#kvcr-linker-v1#digest#")
    # Different digests keep incompatible layouts apart on the full key.
    assert encode_object_key(page, "a", "kv") != encode_object_key(page, "b", "kv")


def test_hint_parser_reads_kv_fetch_and_ignores_unknown_actions():
    envelope = {
        "protocol_version": "0.1",
        "message_id": "m",
        "actions": [
            {
                "action_id": "x",
                "action_type": "kv.other",
                "action_version": "1.0",
                "payload": {},
            },
            {
                "action_id": "y",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://h:1",
                    "block_hashes": [5, -1],
                },
            },
        ],
    }
    hint = parse_fetch_hint(envelope)
    assert hint is not None
    assert hint.source_control_endpoint == "tcp://h:1"
    assert hint.block_hashes == (5, (1 << 64) - 1)
    assert (
        parse_fetch_hint(
            {
                "actions": [
                    {"action_type": "kv.fetch", "action_version": "2.0", "payload": {}}
                ]
            }
        )
        is None
    )
    assert parse_fetch_hint(None) is None
    assert parse_fetch_hint({"actions": "nope"}) is None
    assert (
        parse_fetch_hint(
            {
                "actions": [
                    {
                        "action_type": "kv.fetch",
                        "action_version": "1.0",
                        "payload": {
                            "source_control_endpoint": "tcp://h:1",
                            "block_hashes": ["zz"],
                        },
                    }
                ]
            }
        )
        is None
    )


def test_config_rejects_retired_unknown_and_unsafe_options():
    with pytest.raises(ValueError, match="local_dram_bytes_per_worker"):
        KVCRLinkerConfig.from_extra_config({"local_dram_bytes": 1})
    with pytest.raises(ValueError, match="unknown options"):
        KVCRLinkerConfig.from_extra_config(
            {"local_dram_bytes_per_worker": 1, "bogus": 1}
        )
    with pytest.raises(ValueError, match="requires local_dram_bytes_per_worker"):
        KVCRLinkerConfig.from_extra_config({})
    with pytest.raises(ValueError, match="explicit control_port"):
        KVCRLinkerConfig.from_extra_config(
            {"local_dram_bytes_per_worker": 1, "enable_remote_hint": True}
        )
    with pytest.raises(ValueError, match="cannot advertise"):
        KVCRLinkerConfig.from_extra_config(
            {
                "local_dram_bytes_per_worker": 1,
                "enable_remote_hint": True,
                "control_port": 25000,
                "control_advertise_host": "0.0.0.0",
            }
        )
    with pytest.raises(ValueError, match="abandon_timeout_ms"):
        KVCRLinkerConfig.from_extra_config(
            {
                "local_dram_bytes_per_worker": 1,
                "operation_timeout_ms": 1000,
                "abandon_timeout_ms": 1000,
            }
        )


def test_startup_rejects_unsupported_arrangements(harness):
    with pytest.raises(RuntimeError, match="does not support memory types"):
        Harness(extra={}) if False else None
        _publish_args({})
        agent = FakeNixlAgent()
        group, _ = _pool_group(with_swa=False)
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=1,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        with patch.object(
            linker_module, "resolve_hybrid_device_pool_group", return_value=group
        ):
            KVCRDirectLinker(
                None, params, components=set(), _nixl_probe=lambda b: {"VRAM_SEG"}
            )
    with pytest.raises(ValueError, match="pipeline parallelism"):
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=2,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        with patch.object(
            linker_module, "resolve_hybrid_device_pool_group", return_value=group
        ):
            KVCRDirectLinker(
                None,
                params,
                components=set(),
                _nixl_probe=lambda b: {"DRAM_SEG", "VRAM_SEG"},
            )
    with pytest.raises(RuntimeError, match="ROCm"):
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_rank=0,
            attn_cp_size=1,
            pp_rank=0,
            pp_size=1,
            is_eagle=False,
            mtp_draft_device_pools=(),
        )
        with (
            patch.object(
                linker_module, "resolve_hybrid_device_pool_group", return_value=group
            ),
            patch.object(linker_module, "is_hip", return_value=True),
        ):
            KVCRDirectLinker(
                None,
                params,
                components=set(),
                _nixl_probe=lambda b: {"DRAM_SEG", "VRAM_SEG"},
            )


def test_startup_rejects_speculative_without_draft_pools():
    args = ServerArgs(
        model_path="dummy",
        page_size=PAGE,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="kvcr",
        hicache_storage_backend_extra_config=json.dumps(
            {"local_dram_bytes_per_worker": 1 << 20, "pin_local_dram": False}
        ),
        speculative_algorithm="EAGLE",
        speculative_draft_model_path="dummy-draft",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=2,
    )
    set_global_server_args_for_scheduler(args)
    group, _ = _pool_group(with_swa=False)
    params = SimpleNamespace(
        page_size=PAGE,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
        attn_tp_cache_group=None,
        tp_cache_group=None,
        attn_cp_rank=0,
        attn_cp_size=1,
        pp_rank=0,
        pp_size=1,
        is_eagle=True,
        mtp_draft_device_pools=(),
    )
    with patch.object(
        linker_module, "resolve_hybrid_device_pool_group", return_value=group
    ):
        with pytest.raises(ValueError, match="draft state"):
            KVCRDirectLinker(
                None,
                params,
                components=set(),
                _nixl_probe=lambda b: {"DRAM_SEG", "VRAM_SEG"},
            )


def test_owner_thread_stats_tick_during_startup_does_not_fault(harness, caplog):
    # A stats interval shorter than core construction makes the first owner
    # tick fire before __init__ returns; it must find the adapter in place.
    import logging

    with caplog.at_level(logging.WARNING):
        h = harness(extra={"stats_log_interval_s": 0.001})
        h.wait(lambda: h.linker.snapshot_stats() is not None)
        time.sleep(0.05)
    assert h.linker._adapter.healthy
    assert not [r for r in caplog.records if "owner loop fault" in r.getMessage()]


def test_telemetry_sink_folds_kvcr_metrics_into_stats(harness):
    # With telemetry on, KVCR's counters and stage histograms land in the
    # snapshot under kvcr_ keys; off, no sink is created at all.
    h = harness(extra={"enable_telemetry": True})
    sink = h.linker._telemetry
    assert sink is not None and sink.is_empty()
    sink.increase_counter("ops", 2, ("fetch",))
    sink.increase_counter("ops", 3, ("fetch",))
    sink.set_gauge("inflight", 4)
    sink.observe_histogram("duration", 0.25, ("source_write",))
    sink.observe_histogram("duration", 0.75, ("source_write",))
    snapshot = h.linker.snapshot_stats()
    assert snapshot["kvcr_ops[fetch]"] == 5
    assert snapshot["kvcr_inflight"] == 4
    assert snapshot["kvcr_duration[source_write]_count"] == 2
    assert snapshot["kvcr_duration[source_write]_sum"] == 1.0
    assert snapshot["kvcr_duration[source_write]_max"] == 0.75
    assert not sink.is_empty()
    assert harness(extra={"enable_telemetry": False}).linker._telemetry is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


def test_multi_pool_offload_and_fetch_share_one_operation_per_chunk(harness):
    # Two pools, four pages, chunks of two: the offload issues one deposit per
    # chunk (the SWA tail rides with the KV pages of its chunk) and the
    # preparation issues one fetch per chunk carrying both pools' keys.
    h = harness(with_swa=True)
    hashes = _hashes("mp", 4)
    h.fill(0, 4, seed=25)
    h.offload(hashes, first_page=0, swa_tail=2)
    assert h.wait_offloads(1) == [True]
    assert len(h.agent.xfers) == 2
    kvcr_obj = h.linker._adapter.kvcr
    with patch.object(kvcr_obj, "fetch", wraps=kvcr_obj.fetch) as spy:
        handle = h.prepare("mp", hashes, swa_window=2)
        h.wait_ready(handle)
    assert spy.call_count == 2
    layouts = [call.kwargs["expected_layouts"] for call in spy.call_args_list]
    assert all(layouts)
    assert len(layouts[1]) == 4, "second chunk carries kv and swa keys"
    # Only the boundary whose trailing SWA window is present is restorable.
    assert h.linker.lookup("mp", h.lookup_transfers(hashes, swa_window=2)) == [4]
    h.close()
