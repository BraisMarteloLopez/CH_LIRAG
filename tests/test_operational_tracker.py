"""Tests para shared/operational_tracker.py (R14)."""

from __future__ import annotations

import pytest

from shared.operational_tracker import (
    get_operational_stats,
    record_operational_event,
    reset_operational_stats,
)


@pytest.fixture(autouse=True)
def _reset_tracker_between_tests() -> None:
    reset_operational_stats()
    yield
    reset_operational_stats()


def test_snapshot_starts_at_zero_for_all_event_types() -> None:
    stats = get_operational_stats()
    # Los 7 tipos deben existir con valor 0, incluso sin haberse registrado.
    expected_keys = {
        "neighbor_lookup_failure",
        "chunk_keywords_vdb_error",
        "description_synthesis_error",
        "gleaning_error",
        "keywords_parse_failure",
        "retrieval_error",
        "generation_error",
    }
    assert set(stats.keys()) == expected_keys
    assert all(v == 0 for v in stats.values())


def test_record_increments_counter() -> None:
    record_operational_event("retrieval_error")
    record_operational_event("retrieval_error")
    record_operational_event("gleaning_error")

    stats = get_operational_stats()
    assert stats["retrieval_error"] == 2
    assert stats["gleaning_error"] == 1
    # Los otros tipos siguen en 0.
    assert stats["neighbor_lookup_failure"] == 0
    assert stats["keywords_parse_failure"] == 0


def test_snapshot_returns_copy_not_reference() -> None:
    record_operational_event("generation_error")
    snap1 = get_operational_stats()
    snap1["generation_error"] = 999  # Mutar el snapshot externo.

    snap2 = get_operational_stats()
    # Mutacion externa no afecta al tracker.
    assert snap2["generation_error"] == 1


def test_reset_zeros_all_counters() -> None:
    record_operational_event("chunk_keywords_vdb_error")
    record_operational_event("chunk_keywords_vdb_error")
    record_operational_event("description_synthesis_error")
    assert get_operational_stats()["chunk_keywords_vdb_error"] == 2

    reset_operational_stats()

    stats = get_operational_stats()
    assert all(v == 0 for v in stats.values())


def test_tracker_is_thread_safe_under_concurrent_increments() -> None:
    """Sanity: 100 threads × 50 increments = 5000 exactos, sin perdidas."""
    import threading

    def _worker() -> None:
        for _ in range(50):
            record_operational_event("neighbor_lookup_failure")

    threads = [threading.Thread(target=_worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert get_operational_stats()["neighbor_lookup_failure"] == 5000
