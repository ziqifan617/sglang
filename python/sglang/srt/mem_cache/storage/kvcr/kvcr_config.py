# SPDX-License-Identifier: Apache-2.0
"""Configuration for the KVCR direct linker.

Parsed from ``--hicache-storage-backend-extra-config`` (JSON, or ``@file``),
the same channel the Mooncake and UMBP linkers use despite its name.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import msgspec

from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    MAX_TCP_PORT,
    split_control_endpoint,
)

_UNROUTABLE_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*"})

# Options the KVCR HiCache adapter (#36409) understood whose meaning changed or
# does not apply here. Each maps to the message the operator needs.
_RETIRED_OPTIONS = {
    "local_dram_bytes": (
        "local_dram_bytes is ambiguous in linker mode; set "
        "local_dram_bytes_per_worker (the total for every scheduler rank on "
        "this worker, divided among them)."
    ),
    "local_dram_slots": "local_dram_slots is derived from the physical layout.",
    "get_timeout_s": "get_timeout_s is replaced by preparation_deadline_ms.",
}


class KVCRLinkerConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Operator-facing settings for ``--unified-cache-external-linker-backend kvcr``."""

    # Total KVCR-owned DRAM for this worker; each local scheduler rank gets an
    # equal share. Required: there is no sensible default for a cache tier.
    local_dram_bytes_per_worker: int
    pin_local_dram: bool = True
    nixl_backend: str = "UCX"
    # Offload copies GPU pages into KVCR DRAM with the CUDA runtime inside
    # KVCR; False forces them through NIXL loopback (diagnostics only).
    device_copy: bool = True
    # Restore copies claimed KVCR slots into GPU pages directly from this
    # process, layer by layer, so compute starts as soon as a layer lands;
    # False routes restores through KVCR deliver and completes all layers at
    # the end.
    direct_restore: bool = True
    # Do not prefetch hinted objects into target DRAM. Read directly into
    # private HBM slots and confirm all bytes before publishing a radix hit;
    # advisory hints alone cannot safely admit progressive, unreserved reads.
    direct_remote_restore: bool = False
    # Legacy ablation switches retained for configuration compatibility.
    # The safe precommit direct path does not use progressive layer release,
    # per-layer windows, or per-layer chunk sizing (it uses fetch_chunk_pages).
    progressive_remote_restore: bool = True
    # Destination descriptors are stable for the lifetime of the registered
    # HBM pools. Cache them by physical pool and row so repeated restores do
    # not rebuild thousands of MemDescriptor objects on the critical path.
    direct_remote_descriptor_cache: bool = True
    # Legacy per-layer operation window; not used by precommit direct reads.
    direct_remote_inflight_layers: int = 0
    # Legacy per-layer chunk size; not used by precommit direct reads.
    direct_remote_chunk_pages: int = 0
    # Direct restores submit one copy batch per model layer so the forward
    # pass can start on layer 0 early; consecutive layers whose operands total
    # less than this many bytes are merged into one batch, because for models
    # with many small spans per page (DeepSeek V4: 147 spans of 1.7 KB to
    # 146 KB per page) the per-batch launch cost exceeds the copy itself.
    # 0 keeps one batch per layer.
    direct_restore_min_batch_bytes: int = 4 << 20

    # Peer control channel. control_port is a base; each rank adds its
    # engine-global attention rank so colocated ranks never collide.
    control_host: str = "0.0.0.0"
    control_port: int = 0
    control_advertise_host: Optional[str] = None
    enable_remote_hint: bool = False

    # KVCR core knobs.
    operation_timeout_ms: int = 20000
    abandon_timeout_ms: int = 60000
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100
    policy: str = "lru"
    enable_telemetry: bool = False

    # Preparation bounds. A request stops waiting at the deadline and admits
    # whatever prefix was confirmed; late completions are drained afterwards.
    preparation_deadline_ms: int = 2000
    max_inflight_prepare_requests: int = 64
    max_inflight_prepare_bytes: int = 8 << 30
    max_prepare_bytes_per_request: int = 2 << 30
    fetch_chunk_pages: int = 32
    # Pages per offload deposit; each deposit covers every pool's pages for
    # its range. Smaller chunks keep each owner-thread call short so the GIL
    # returns to the scheduler between deposits while a prefill is launched.
    offload_chunk_pages: int = 8
    # Offloads beyond this many in-flight bytes are declined; the tree retries.
    max_inflight_offload_bytes: int = 8 << 30
    # Late (abandoned) work above this stops new preparation until it drains.
    max_abandoned_bytes: int = 4 << 30
    # Owner-thread poll interval while KVCR operations are in flight; shorter
    # finishes small transfers sooner.
    poll_interval_ms: float = 0.5
    # Owner-thread wake interval with nothing in flight (deadline and stats
    # ticks only; posted commands wake it at once). Every wake costs the
    # scheduler thread a GIL hand-off, and with one owner per DP-attention
    # rank a 0.5 ms cadence added about 0.2 s to every request's TTFT on a
    # four-rank DeepSeek V4 deployment.
    idle_poll_interval_ms: float = 10.0
    stats_log_interval_s: float = 30.0
    # Interpreter switch interval to apply in the scheduler process. The owner
    # thread and the scheduler share one GIL; a shorter interval caps how long
    # either waits for the other. None leaves the interpreter default (5 ms).
    gil_switch_interval_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.local_dram_bytes_per_worker <= 0:
            raise ValueError("KVCR linker requires local_dram_bytes_per_worker > 0.")
        if self.control_port < 0 or self.control_port > MAX_TCP_PORT:
            raise ValueError(
                f"KVCR control_port ({self.control_port}) is out of range; use 0 "
                f"(OS-assigned, local-only) or 1..{MAX_TCP_PORT}."
            )
        if self.operation_timeout_ms <= 0:
            raise ValueError("KVCR operation_timeout_ms must be positive.")
        if self.abandon_timeout_ms < 2 * self.operation_timeout_ms:
            raise ValueError(
                "KVCR abandon_timeout_ms must be at least twice operation_timeout_ms."
            )
        if self.preparation_deadline_ms <= 0:
            raise ValueError("KVCR preparation_deadline_ms must be positive.")
        if self.fetch_chunk_pages <= 0:
            raise ValueError("KVCR fetch_chunk_pages must be positive.")
        if self.offload_chunk_pages <= 0:
            raise ValueError("KVCR offload_chunk_pages must be positive.")
        for name in (
            "max_inflight_prepare_requests",
            "max_inflight_prepare_bytes",
            "max_prepare_bytes_per_request",
            "max_inflight_offload_bytes",
            "max_abandoned_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"KVCR {name} must be positive.")
        if self.poll_interval_ms <= 0:
            raise ValueError("KVCR poll_interval_ms must be positive.")
        if self.idle_poll_interval_ms < self.poll_interval_ms:
            raise ValueError(
                "KVCR idle_poll_interval_ms must be at least poll_interval_ms."
            )
        if self.gil_switch_interval_ms is not None and self.gil_switch_interval_ms <= 0:
            raise ValueError("KVCR gil_switch_interval_ms must be positive.")
        if self.direct_restore_min_batch_bytes < 0:
            raise ValueError("KVCR direct_restore_min_batch_bytes must be >= 0.")
        if self.direct_remote_inflight_layers < 0:
            raise ValueError("KVCR direct_remote_inflight_layers must be >= 0.")
        if self.direct_remote_chunk_pages < 0:
            raise ValueError("KVCR direct_remote_chunk_pages must be >= 0.")
        self._validate_remote_hint_endpoint()

    def _validate_remote_hint_endpoint(self) -> None:
        """A hint source must be dialable before it binds.

        Port 0 exists only inside this process and cannot be advertised, and
        the bind host is legitimately a wildcard, so remote hints require an
        explicit advertise host and base port.
        """
        if not self.enable_remote_hint:
            return
        if self.control_port <= 0:
            raise ValueError(
                "KVCR enable_remote_hint requires an explicit control_port: an "
                "OS-assigned port cannot be registered for peers to dial."
            )
        advertise = self.control_advertise_host
        if not advertise or advertise in _UNROUTABLE_HOSTS:
            raise ValueError(
                f"KVCR enable_remote_hint cannot advertise {advertise!r}; set "
                "control_advertise_host to an address peers can dial."
            )
        if split_control_endpoint(f"tcp://{advertise}:{self.control_port}") is None:
            raise ValueError(
                f"KVCR control endpoint tcp://{advertise}:{self.control_port} "
                "is not dialable."
            )

    @classmethod
    def from_extra_config(
        cls, extra_config: Optional[Mapping[str, Any]]
    ) -> KVCRLinkerConfig:
        extra_config = dict(extra_config or {})
        for name, message in _RETIRED_OPTIONS.items():
            if name in extra_config:
                raise ValueError(f"KVCR linker config: {message}")
        known = set(cls.__struct_fields__)
        unknown = sorted(set(extra_config) - known)
        if unknown:
            raise ValueError(
                f"KVCR linker config has unknown options {unknown}; known "
                f"options: {sorted(known)}."
            )
        if "local_dram_bytes_per_worker" not in extra_config:
            raise ValueError(
                "KVCR linker config requires local_dram_bytes_per_worker in "
                "--hicache-storage-backend-extra-config."
            )
        try:
            return msgspec.convert(extra_config, cls)
        except msgspec.ValidationError as error:
            raise ValueError(f"KVCR linker config is invalid: {error}") from error
