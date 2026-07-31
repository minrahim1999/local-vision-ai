"""Tests for memory manager state transitions and thread safety."""
import threading
import time

import pytest

from services.memory_manager import MemoryManager
from services.text_to_image import TextToImageService
from services.image_to_text import ImageToTextService


class MockService:
    """Lightweight mock service for testing state transitions."""

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def is_loaded(self) -> bool:
        return self.loaded


class TestMemoryManager:
    def test_register_and_acquire(self) -> None:
        mm = MemoryManager()
        svc = MockService()
        mm.register("mock", svc)
        acquired = mm.acquire("mock")
        assert acquired is svc
        assert svc.loaded is True
        assert mm.current_pipeline == "mock"

    def test_acquire_switches_pipeline(self) -> None:
        mm = MemoryManager()
        a = MockService()
        b = MockService()
        mm.register("a", a)
        mm.register("b", b)
        mm.acquire("a")
        assert a.loaded is True
        mm.acquire("b")
        assert a.loaded is False
        assert b.loaded is True
        assert mm.current_pipeline == "b"

    def test_release_all(self) -> None:
        mm = MemoryManager()
        svc = MockService()
        mm.register("mock", svc)
        mm.acquire("mock")
        mm.release_all()
        assert svc.loaded is False
        assert mm.current_pipeline is None

    def test_thread_safety(self) -> None:
        mm = MemoryManager()
        a = MockService()
        b = MockService()
        mm.register("a", a)
        mm.register("b", b)
        errors = []

        def toggle() -> None:
            try:
                for _ in range(50):
                    mm.acquire("a")
                    time.sleep(0.001)
                    mm.acquire("b")
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert mm.current_pipeline in ("a", "b", None)

    def test_memory_status_keys(self) -> None:
        mm = MemoryManager()
        status = mm.get_memory_status()
        assert "rss_mb" in status
        assert "vms_mb" in status
        assert "current_pipeline" in status
