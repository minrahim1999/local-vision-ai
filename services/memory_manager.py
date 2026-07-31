"""Shared memory manager for model lifecycle and telemetry."""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import psutil

logger = logging.getLogger(__name__)


class PipelineService(Protocol):
    """Protocol for a loadable/unloadable inference service."""

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...


@dataclass(frozen=True)
class MemorySnapshot:
    rss_mb: int
    vms_mb: int
    percent: float
    swap_used_mb: int
    timestamp: float


class MemoryManager:
    """Thread-safe memory manager ensuring only one heavy pipeline is loaded."""

    def __init__(
        self,
        memory_warn_mb: int = 12_000,
        memory_limit_mb: int = 14_000,
        idle_timeout_seconds: float = 300.0,
    ) -> None:
        self.memory_warn_mb = memory_warn_mb
        self.memory_limit_mb = memory_limit_mb
        self.idle_timeout_seconds = idle_timeout_seconds
        self._lock = threading.Lock()
        self._current_pipeline: str | None = None
        self._pipelines: dict[str, PipelineService] = {}
        self._last_active: float = time.monotonic()

    def register(self, name: str, service: PipelineService) -> None:
        with self._lock:
            self._pipelines[name] = service

    def _snapshot(self) -> MemorySnapshot:
        proc = psutil.Process()
        mem = proc.memory_info()
        swap = psutil.swap_memory()
        return MemorySnapshot(
            rss_mb=int(mem.rss // (1024 * 1024)),
            vms_mb=int(mem.vms // (1024 * 1024)),
            percent=proc.memory_percent(),
            swap_used_mb=int(swap.used // (1024 * 1024)),
            timestamp=time.monotonic(),
        )

    def get_memory_status(self) -> dict[str, Any]:
        snap = self._snapshot()
        return {
            "rss_mb": snap.rss_mb,
            "vms_mb": snap.vms_mb,
            "system_percent": snap.percent,
            "swap_used_mb": snap.swap_used_mb,
            "current_pipeline": self._current_pipeline,
            "idle_timeout_seconds": self.idle_timeout_seconds,
        }

    def _clear_backend_caches(self) -> None:
        """Attempt to clear PyTorch MPS / MLX caches."""
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _check_limits(self, snap: MemorySnapshot) -> None:
        if snap.rss_mb > self.memory_limit_mb:
            raise MemoryError(
                f"Memory limit exceeded: {snap.rss_mb} MB > {self.memory_limit_mb} MB"
            )
        if snap.rss_mb > self.memory_warn_mb:
            logger.warning("Memory warning: %s MB used", snap.rss_mb)

    def acquire(self, pipeline_name: str) -> PipelineService:
        with self._lock:
            if self._current_pipeline == pipeline_name:
                self._last_active = time.monotonic()
                return self._pipelines[pipeline_name]

            # Unload any currently loaded pipeline
            if self._current_pipeline is not None:
                logger.info("Unloading pipeline: %s", self._current_pipeline)
                try:
                    self._pipelines[self._current_pipeline].unload()
                except Exception:
                    logger.exception("Error unloading %s", self._current_pipeline)
                self._clear_backend_caches()
                self._current_pipeline = None

            service = self._pipelines[pipeline_name]
            logger.info("Loading pipeline: %s", pipeline_name)
            service.load()
            self._current_pipeline = pipeline_name
            self._last_active = time.monotonic()

            snap = self._snapshot()
            self._check_limits(snap)
            logger.info(
                "Pipeline '%s' loaded. Memory: %s MB RSS, %s%% system, %s MB swap",
                pipeline_name,
                snap.rss_mb,
                round(snap.percent, 1),
                snap.swap_used_mb,
            )
            return service

    def release_all(self) -> None:
        with self._lock:
            for name, svc in self._pipelines.items():
                if svc.is_loaded():
                    logger.info("Unloading pipeline: %s", name)
                    try:
                        svc.unload()
                    except Exception:
                        logger.exception("Error unloading %s", name)
            self._current_pipeline = None
        self._clear_backend_caches()

    def idle_unload_check(self) -> None:
        with self._lock:
            if self._current_pipeline is None:
                return
            elapsed = time.monotonic() - self._last_active
            if elapsed > self.idle_timeout_seconds:
                logger.info(
                    "Idle timeout reached for '%s' (%.0fs). Unloading.",
                    self._current_pipeline,
                    elapsed,
                )
                try:
                    self._pipelines[self._current_pipeline].unload()
                except Exception:
                    logger.exception("Idle unload error")
                self._current_pipeline = None
                self._clear_backend_caches()

    @property
    def current_pipeline(self) -> str | None:
        with self._lock:
            return self._current_pipeline
