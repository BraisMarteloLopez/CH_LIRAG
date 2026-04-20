"""
Tests de la capa de KG synthesis en generacion.

Verifica:
  - synthesis se invoca solo con KG data + flag activa
  - flag desactivada => comportamiento identico (regression guard)
  - degradacion graceful: error LLM, respuesta vacia, timeout => fallback
  - truncacion al max_chars cuando el LLM excede budget
  - faithfulness se evalua contra structured_context, no contra narrativa
  - tracker contabiliza correctamente y se resetea entre runs
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.types import (
    DatasetType,
    GenerationResult,
    MetricType,
    NormalizedQuery,
    QueryRetrievalDetail,
    get_dataset_config,
)
from shared.metrics import MetricResult

from sandbox_mteb.generation_executor import (
    GenerationExecutor,
    get_kg_synthesis_stats,
    reset_kg_synthesis_stats,
)


# =============================================================================
# Fixtures / helpers
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_tracker():
    reset_kg_synthesis_stats()
    yield
    reset_kg_synthesis_stats()


def _make_query():
    return NormalizedQuery(
        query_id="q1",
        query_text="who founded the lab?",
        expected_answer="Alice",
        answer_type=None,
    )


def _make_retrieval(kg_meta: bool):
    meta: dict = {}
    if kg_meta:
        meta = {
            "kg_entities": [
                {"entity": "alice", "description": "founder of the lab"},
            ],
            "kg_relations": [
                {"source": "alice", "target": "lab", "relation": "founded"},
            ],
            "lightrag_mode": "hybrid",
        }
    return QueryRetrievalDetail(
        retrieved_doc_ids=["d1"],
        retrieved_contents=["Alice is the founder of the lab."],
        retrieval_scores=[0.9],
        expected_doc_ids=["d1"],
        retrieval_metadata=meta,
    )


def _make_executor(
    *,
    llm_response="answer",
    synthesis_enabled=True,
    synthesis_max_chars=0,
    synthesis_timeout_s=30.0,
):
    llm = MagicMock()
    llm.invoke_async = AsyncMock(return_value=llm_response)

    calc = MagicMock()

    async def _calc_async(mt, **kwargs):
        return MetricResult(metric_type=mt, value=0.7)

    calc.calculate_async = AsyncMock(side_effect=_calc_async)
    calc.embedding_model = None

    return GenerationExecutor(
        llm_service=llm,
        metrics_calculator=calc,
        max_context_chars=4000,
        kg_synthesis_enabled=synthesis_enabled,
        kg_synthesis_max_chars=synthesis_max_chars,
        kg_synthesis_timeout_s=synthesis_timeout_s,
    )


# =============================================================================
# Cuando se invoca / cuando no
# =============================================================================


class TestSynthesisInvocation:

    def test_invoked_when_kg_data_and_flag_on(self):
        """Con KG data + flag on, el LLM se llama 2 veces (synthesis + gen)."""
        executor = _make_executor(synthesis_enabled=True)
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), _make_retrieval(kg_meta=True),
                ds_config, "hotpotqa",
            )
        )

        # 1 synthesis call + 1 generation call
        assert executor._llm_service.invoke_async.call_count == 2
        stats = get_kg_synthesis_stats()
        assert stats["invocations"] == 1
        assert stats["successes"] == 1

    def test_skipped_when_no_kg_data(self):
        """Sin KG data, synthesis no se invoca aunque flag este on."""
        executor = _make_executor(synthesis_enabled=True)
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), _make_retrieval(kg_meta=False),
                ds_config, "hotpotqa",
            )
        )

        # Solo 1 generation call (sin synthesis)
        assert executor._llm_service.invoke_async.call_count == 1
        stats = get_kg_synthesis_stats()
        assert stats["invocations"] == 0

    def test_skipped_when_flag_off(self):
        """Con KG data pero flag off, synthesis no se invoca (regresion guard)."""
        executor = _make_executor(synthesis_enabled=False)
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), _make_retrieval(kg_meta=True),
                ds_config, "hotpotqa",
            )
        )

        # Solo 1 generation call
        assert executor._llm_service.invoke_async.call_count == 1
        stats = get_kg_synthesis_stats()
        assert stats["invocations"] == 0


# =============================================================================
# Faithfulness contra structured_context original (decision clave)
# =============================================================================


class TestFaithfulnessContext:

    def test_metrics_use_structured_context_not_synthesis(self):
        """faithfulness/secondary metrics reciben el structured_context, no la narrativa."""
        executor = _make_executor(
            synthesis_enabled=True,
            llm_response="SYNTHESIZED_NARRATIVE",
        )
        ds_config = get_dataset_config("hotpotqa")

        with patch(
            "sandbox_mteb.generation_executor.format_structured_context_with_stats"
        ) as mock_struct:
            mock_struct.return_value = ("STRUCTURED_CTX", 1)
            asyncio.run(
                executor._process_single_async(
                    _make_query(), _make_retrieval(kg_meta=True),
                    ds_config, "hotpotqa",
                )
            )

        # Inspeccionar TODOS los kwargs de las llamadas a calculate_async:
        # ninguna debe haber recibido la narrativa como context.
        for call in executor._metrics_calculator.calculate_async.call_args_list:
            ctx = call.kwargs.get("context")
            if ctx is not None:
                assert ctx == "STRUCTURED_CTX"
                assert "SYNTHESIZED_NARRATIVE" not in ctx

    def test_generation_uses_synthesis_narrative(self):
        """El LLM generador recibe la narrativa, no el structured_context."""
        executor = _make_executor(
            synthesis_enabled=True,
            llm_response="SYNTH_OUT",
        )
        ds_config = get_dataset_config("hotpotqa")

        with patch(
            "sandbox_mteb.generation_executor.format_structured_context_with_stats"
        ) as mock_struct:
            mock_struct.return_value = ("STRUCTURED_CTX", 1)
            asyncio.run(
                executor._process_single_async(
                    _make_query(), _make_retrieval(kg_meta=True),
                    ds_config, "hotpotqa",
                )
            )

        # 2 calls: la primera es synthesis (recibe el structured_context),
        # la segunda es generation (recibe la narrativa SYNTH_OUT en el
        # user_prompt formateado).
        calls = executor._llm_service.invoke_async.call_args_list
        assert len(calls) == 2

        # 2a llamada (generation): el user_prompt debe contener la narrativa
        gen_user_prompt = calls[1].args[0]
        assert "SYNTH_OUT" in gen_user_prompt


# =============================================================================
# Degradacion graceful
# =============================================================================


class TestSynthesisFallback:

    def test_llm_error_falls_back_to_structured(self):
        """Si el LLM lanza, se usa el structured_context original."""
        # Configuramos llm.invoke_async para fallar la PRIMERA llamada
        # (synthesis) y devolver respuesta normal en la SEGUNDA (generation).
        executor = _make_executor(synthesis_enabled=True)

        async def _side_effect(prompt, system_prompt=None, **kw):
            if "synthesis" in (system_prompt or "").lower() or \
               "context-synthesis" in (system_prompt or ""):
                raise RuntimeError("synthesis LLM down")
            return "generated"

        executor._llm_service.invoke_async = AsyncMock(side_effect=_side_effect)

        ds_config = get_dataset_config("hotpotqa")
        asyncio.run(
            executor._process_single_async(
                _make_query(), _make_retrieval(kg_meta=True),
                ds_config, "hotpotqa",
            )
        )

        stats = get_kg_synthesis_stats()
        assert stats["invocations"] == 1
        assert stats["errors"] == 1
        assert stats["successes"] == 0

    def test_empty_response_falls_back(self):
        """LLM devuelve solo whitespace => fallback."""
        executor = _make_executor(synthesis_enabled=True, llm_response="   \n\n  ")
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), _make_retrieval(kg_meta=True),
                ds_config, "hotpotqa",
            )
        )

        stats = get_kg_synthesis_stats()
        assert stats["empty_returns"] == 1
        assert stats["successes"] == 0

    def test_oversized_response_is_truncated(self):
        """LLM devuelve > max_chars => truncado, no fallback."""
        long_response = "A" * 5000
        executor = _make_executor(
            synthesis_enabled=True,
            synthesis_max_chars=100,
            llm_response=long_response,
        )
        ds_config = get_dataset_config("hotpotqa")

        with patch(
            "sandbox_mteb.generation_executor.format_structured_context_with_stats"
        ) as mock_struct:
            mock_struct.return_value = ("STRUCTURED_CTX", 1)
            asyncio.run(
                executor._process_single_async(
                    _make_query(), _make_retrieval(kg_meta=True),
                    ds_config, "hotpotqa",
                )
            )

        stats = get_kg_synthesis_stats()
        assert stats["truncations"] == 1
        assert stats["successes"] == 1

        # El generation prompt debe contener la version truncada (100 A),
        # no la version completa de 5000 A.
        gen_call = executor._llm_service.invoke_async.call_args_list[1]
        gen_prompt = gen_call.args[0]
        assert "A" * 100 in gen_prompt
        assert "A" * 200 not in gen_prompt

    def test_timeout_falls_back(self):
        """asyncio.wait_for timeout => contador y fallback."""
        executor = _make_executor(
            synthesis_enabled=True,
            synthesis_timeout_s=0.05,
        )

        async def _slow_then_fast(prompt, system_prompt=None, **kw):
            # Si es la llamada de synthesis, dormimos lo suficiente para timeout.
            if "context-synthesis" in (system_prompt or ""):
                await asyncio.sleep(1.0)
                return "never"
            return "generated"

        executor._llm_service.invoke_async = AsyncMock(side_effect=_slow_then_fast)
        ds_config = get_dataset_config("hotpotqa")
        asyncio.run(
            executor._process_single_async(
                _make_query(), _make_retrieval(kg_meta=True),
                ds_config, "hotpotqa",
            )
        )

        stats = get_kg_synthesis_stats()
        assert stats["timeouts"] == 1
        assert stats["successes"] == 0


# =============================================================================
# Tracker
# =============================================================================


class TestSynthesisTracker:

    def test_tracker_starts_empty(self):
        stats = get_kg_synthesis_stats()
        assert stats["invocations"] == 0
        assert stats["successes"] == 0
        assert stats["fallback_rate"] == 0.0

    def test_tracker_counts_per_event(self):
        """Multiple invocaciones se acumulan correctamente."""
        executor = _make_executor(synthesis_enabled=True)
        ds_config = get_dataset_config("hotpotqa")

        for _ in range(3):
            asyncio.run(
                executor._process_single_async(
                    _make_query(), _make_retrieval(kg_meta=True),
                    ds_config, "hotpotqa",
                )
            )

        stats = get_kg_synthesis_stats()
        assert stats["invocations"] == 3
        assert stats["successes"] == 3
        assert stats["fallback_rate"] == 0.0

    def test_fallback_rate_computed_correctly(self):
        """fallback_rate = (errors + empty + timeouts) / invocations."""
        # Inyectamos eventos directamente al tracker
        from sandbox_mteb.generation_executor import _kg_synthesis_tracker
        for _ in range(10):
            _kg_synthesis_tracker.record("invocations")
        _kg_synthesis_tracker.record("successes")
        _kg_synthesis_tracker.record("successes")
        _kg_synthesis_tracker.record("errors")
        _kg_synthesis_tracker.record("empty_returns")
        _kg_synthesis_tracker.record("timeouts")

        stats = get_kg_synthesis_stats()
        assert stats["invocations"] == 10
        # 3 fallbacks (1 error + 1 empty + 1 timeout) sobre 10
        assert abs(stats["fallback_rate"] - 0.3) < 1e-9

    def test_reset_clears_tracker(self):
        from sandbox_mteb.generation_executor import _kg_synthesis_tracker
        _kg_synthesis_tracker.record("invocations")
        _kg_synthesis_tracker.record("successes")
        assert get_kg_synthesis_stats()["invocations"] == 1

        reset_kg_synthesis_stats()
        assert get_kg_synthesis_stats()["invocations"] == 0


# =============================================================================
# Per-query metadata
# =============================================================================


class TestPerQuerySynthesisMetadata:
    """_process_single_async debe persistir outcome de synthesis en
    retrieval_metadata para que JSON/CSV lo puedan auditar per-query."""

    def test_success_writes_kg_synthesis_used_true(self):
        executor = _make_executor(synthesis_enabled=True, llm_response="narrative")
        ds_config = get_dataset_config("hotpotqa")
        retrieval = _make_retrieval(kg_meta=True)

        asyncio.run(
            executor._process_single_async(
                _make_query(), retrieval, ds_config, "hotpotqa",
            )
        )

        assert retrieval.retrieval_metadata.get("kg_synthesis_used") is True
        assert "kg_synthesis_error" not in retrieval.retrieval_metadata

    def test_timeout_writes_error_code(self):
        executor = _make_executor(
            synthesis_enabled=True, synthesis_timeout_s=0.05,
        )

        async def _slow(prompt, system_prompt=None, **kw):
            if "context-synthesis" in (system_prompt or ""):
                await asyncio.sleep(1.0)
                return "never"
            return "gen_ok"

        executor._llm_service.invoke_async = AsyncMock(side_effect=_slow)
        retrieval = _make_retrieval(kg_meta=True)
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), retrieval, ds_config, "hotpotqa",
            )
        )

        assert retrieval.retrieval_metadata.get("kg_synthesis_used") is False
        assert retrieval.retrieval_metadata.get("kg_synthesis_error") == "timeout"

    def test_empty_response_writes_error_code_empty(self):
        executor = _make_executor(synthesis_enabled=True, llm_response="   ")
        retrieval = _make_retrieval(kg_meta=True)
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), retrieval, ds_config, "hotpotqa",
            )
        )

        assert retrieval.retrieval_metadata.get("kg_synthesis_used") is False
        assert retrieval.retrieval_metadata.get("kg_synthesis_error") == "empty"

    def test_llm_exception_writes_error_code_error(self):
        executor = _make_executor(synthesis_enabled=True)

        async def _side(prompt, system_prompt=None, **kw):
            if "context-synthesis" in (system_prompt or ""):
                raise RuntimeError("synthesis LLM down")
            return "gen_ok"

        executor._llm_service.invoke_async = AsyncMock(side_effect=_side)
        retrieval = _make_retrieval(kg_meta=True)
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), retrieval, ds_config, "hotpotqa",
            )
        )

        assert retrieval.retrieval_metadata.get("kg_synthesis_used") is False
        assert retrieval.retrieval_metadata.get("kg_synthesis_error") == "error"

    def test_no_kg_data_leaves_metadata_untouched(self):
        """Sin KG data, no se escribe kg_synthesis_* en retrieval_metadata
        (queries SIMPLE_VECTOR no deben inflar el JSON)."""
        executor = _make_executor(synthesis_enabled=True)
        retrieval = _make_retrieval(kg_meta=False)
        ds_config = get_dataset_config("hotpotqa")

        asyncio.run(
            executor._process_single_async(
                _make_query(), retrieval, ds_config, "hotpotqa",
            )
        )

        assert "kg_synthesis_used" not in retrieval.retrieval_metadata
        assert "kg_synthesis_error" not in retrieval.retrieval_metadata


# =============================================================================
# Timing instrumentation
# =============================================================================


class TestSynthesisTiming:
    """Deuda #16: p50/p95/max por categoria (total/queue/llm) en snapshot."""

    def test_empty_tracker_reports_zero_timings(self):
        """Tracker sin invocaciones expone p50/p95/max=0 y n_samples=0."""
        stats = get_kg_synthesis_stats()
        for prefix in ("total", "queue", "llm"):
            assert stats[f"p50_{prefix}_ms"] == 0.0
            assert stats[f"p95_{prefix}_ms"] == 0.0
            assert stats[f"max_{prefix}_ms"] == 0.0
            assert stats[f"n_{prefix}_samples"] == 0

    def test_success_records_total_queue_llm_timings(self):
        """Synthesis exitosa => todas las listas de timing reciben una entrada."""
        async def _delayed(prompt, system_prompt=None, **kw):
            # Poblar timing_out como lo hace el AsyncLLMService real.
            out = kw.get("timing_out")
            if out is not None:
                out["queue_wait_ms"] = 1.5
                out["llm_ms"] = 25.0
            await asyncio.sleep(0.01)
            return "narrative"

        executor = _make_executor(synthesis_enabled=True)
        executor._llm_service.invoke_async = AsyncMock(side_effect=_delayed)
        ds_config = get_dataset_config("hotpotqa")

        # Ejecutamos 3 queries para tener muestras suficientes para p50/p95
        for _ in range(3):
            asyncio.run(
                executor._process_single_async(
                    _make_query(), _make_retrieval(kg_meta=True),
                    ds_config, "hotpotqa",
                )
            )

        stats = get_kg_synthesis_stats()
        assert stats["n_total_samples"] == 3
        assert stats["n_queue_samples"] == 3
        assert stats["n_llm_samples"] == 3
        # queue_ms / llm_ms vienen del timing_out que popula el servicio
        # (en este mock los inyectamos a mano) => deben reflejar los valores
        # inyectados, no el tiempo del mock.
        assert stats["p50_queue_ms"] == 1.5
        assert stats["p50_llm_ms"] == 25.0
        # total_ms es wall-clock real de `_synthesize_kg_context_async`
        # (sleep(0.01) del mock + overhead async). > 5ms como sanity check;
        # NO suma queue+llm porque esos son valores inyectados, no reales.
        assert stats["p50_total_ms"] >= 5.0

    def test_timeout_records_partial_timing(self):
        """Timeout => timing_out puede quedar parcialmente poblado segun
        donde cancelo wait_for. `n_total_samples` siempre cuenta, pero
        queue/llm pueden tener menos muestras."""
        executor = _make_executor(
            synthesis_enabled=True, synthesis_timeout_s=0.05,
        )

        async def _slow(prompt, system_prompt=None, **kw):
            # No poblamos timing_out (simulamos cancelacion antes del
            # semaphore acquire); el tracker debe seguir registrando total_ms.
            await asyncio.sleep(1.0)
            return "never"

        executor._llm_service.invoke_async = AsyncMock(side_effect=_slow)
        ds_config = get_dataset_config("hotpotqa")
        asyncio.run(
            executor._process_single_async(
                _make_query(), _make_retrieval(kg_meta=True),
                ds_config, "hotpotqa",
            )
        )

        stats = get_kg_synthesis_stats()
        assert stats["timeouts"] == 1
        # total_ms registrado (siempre se calcula al salir del except)
        assert stats["n_total_samples"] == 1
        # total_ms refleja el timeout (~50ms), no el sleep(1.0) completo
        assert 40.0 <= stats["max_total_ms"] <= 500.0
        # queue/llm no se poblaron porque el mock no los establece
        assert stats["n_queue_samples"] == 0
        assert stats["n_llm_samples"] == 0

    def test_percentile_ordering(self):
        """p50 <= p95 <= max en cualquier distribucion no-vacia."""
        from sandbox_mteb.generation_executor import _kg_synthesis_tracker

        for ms in [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
            _kg_synthesis_tracker.record_timing(ms, ms * 0.1, ms * 0.9)

        stats = get_kg_synthesis_stats()
        assert stats["p50_total_ms"] <= stats["p95_total_ms"] <= stats["max_total_ms"]
        assert stats["p50_queue_ms"] <= stats["p95_queue_ms"] <= stats["max_queue_ms"]
        assert stats["p50_llm_ms"] <= stats["p95_llm_ms"] <= stats["max_llm_ms"]

    def test_reset_clears_timings(self):
        from sandbox_mteb.generation_executor import _kg_synthesis_tracker

        _kg_synthesis_tracker.record_timing(100.0, 5.0, 80.0)
        assert get_kg_synthesis_stats()["n_total_samples"] == 1

        reset_kg_synthesis_stats()
        stats = get_kg_synthesis_stats()
        assert stats["n_total_samples"] == 0
        assert stats["p50_total_ms"] == 0.0
