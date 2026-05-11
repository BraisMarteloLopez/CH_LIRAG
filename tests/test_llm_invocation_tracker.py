"""Tests del _LLMInvocationTracker (shared/llm.py).

Cubre: record per-fase, agregacion de percentiles, manejo de None,
ordenacion alfabetica del snapshot, reset, aislamiento entre fases.
Tests del singleton module-level (acceso via la API publica
get_llm_invocation_stats / reset_llm_invocation_stats).
"""

from __future__ import annotations

from shared.llm import (
    _LLMInvocationTracker,
    _llm_invocation_tracker,
    _percentile,
    get_llm_invocation_stats,
    reset_llm_invocation_stats,
)


def test_percentile_empty() -> None:
    assert _percentile([], 0.5) == 0.0


def test_percentile_single_value() -> None:
    assert _percentile([42.0], 0.5) == 42.0
    assert _percentile([42.0], 0.95) == 42.0


def test_percentile_ordering() -> None:
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p50 = _percentile(vals, 0.50)
    p95 = _percentile(vals, 0.95)
    assert p50 == 60.0  # index 5
    assert p95 == 100.0  # index 9 (clamped)


def test_record_increments_invocations() -> None:
    t = _LLMInvocationTracker()
    t.record("extraction", queue_ms=1.0, llm_ms=10.0)
    t.record("extraction", queue_ms=2.0, llm_ms=20.0)
    snap = t.snapshot()
    assert snap["extraction"]["invocations"] == 2
    assert snap["extraction"]["n_queue_samples"] == 2
    assert snap["extraction"]["n_llm_samples"] == 2


def test_record_none_skips_timing_but_counts_invocation() -> None:
    t = _LLMInvocationTracker()
    t.record("gleaning", queue_ms=None, llm_ms=None)
    t.record("gleaning", queue_ms=5.0, llm_ms=None)
    snap = t.snapshot()
    assert snap["gleaning"]["invocations"] == 2
    assert snap["gleaning"]["n_queue_samples"] == 1
    assert snap["gleaning"]["n_llm_samples"] == 0


def test_snapshot_percentiles() -> None:
    t = _LLMInvocationTracker()
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        t.record("extraction", queue_ms=v, llm_ms=v * 10)
    snap = t.snapshot()
    e = snap["extraction"]
    assert e["invocations"] == 5
    assert e["p50_queue_ms"] == 30.0
    assert e["max_queue_ms"] == 50.0
    assert e["p50_llm_ms"] == 300.0
    assert e["max_llm_ms"] == 500.0


def test_phases_isolated() -> None:
    t = _LLMInvocationTracker()
    t.record("extraction", queue_ms=1.0, llm_ms=100.0)
    t.record("gleaning", queue_ms=50.0, llm_ms=500.0)
    snap = t.snapshot()
    assert snap["extraction"]["p50_queue_ms"] == 1.0
    assert snap["extraction"]["p50_llm_ms"] == 100.0
    assert snap["gleaning"]["p50_queue_ms"] == 50.0
    assert snap["gleaning"]["p50_llm_ms"] == 500.0


def test_snapshot_sorted_phases() -> None:
    t = _LLMInvocationTracker()
    t.record("kg_synthesis", queue_ms=1.0, llm_ms=1.0)
    t.record("generation", queue_ms=1.0, llm_ms=1.0)
    t.record("extraction", queue_ms=1.0, llm_ms=1.0)
    snap = t.snapshot()
    assert list(snap.keys()) == ["extraction", "generation", "kg_synthesis"]


def test_snapshot_empty_when_no_records() -> None:
    t = _LLMInvocationTracker()
    assert t.snapshot() == {}


def test_reset_clears_state() -> None:
    t = _LLMInvocationTracker()
    t.record("extraction", queue_ms=1.0, llm_ms=10.0)
    assert "extraction" in t.snapshot()
    t.reset()
    assert t.snapshot() == {}


def test_unknown_bucket_used_when_phase_omitted() -> None:
    # AsyncLLMService.invoke_async defaults to phase="unknown".
    # Aqui simulamos llamando al tracker directamente con ese tag.
    t = _LLMInvocationTracker()
    t.record("unknown", queue_ms=2.0, llm_ms=20.0)
    snap = t.snapshot()
    assert snap["unknown"]["invocations"] == 1


def test_module_singleton_record_and_snapshot() -> None:
    # Verifica que la API publica (get_/reset_) opera sobre el mismo
    # tracker que la registracion implicita desde invoke_async.
    reset_llm_invocation_stats()
    assert get_llm_invocation_stats() == {}
    _llm_invocation_tracker.record("extraction", queue_ms=1.0, llm_ms=10.0)
    snap = get_llm_invocation_stats()
    assert snap["extraction"]["invocations"] == 1
    reset_llm_invocation_stats()
    assert get_llm_invocation_stats() == {}
