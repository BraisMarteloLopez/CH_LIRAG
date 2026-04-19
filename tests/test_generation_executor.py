"""
Tests unitarios para sandbox_mteb/generation_executor.py (Audit Fase 3 — A3.5).

Cobertura:
  G1. _execute_generation_async() retorna GenerationResult
  G2. _execute_generation_async() error retorna [ERROR: ...]
  G3. _calculate_metrics_async() HYBRID label → ACCURACY
  G4. _calculate_metrics_async() HYBRID text → F1_SCORE
  G5. _calculate_metrics_async() secondary metric error no crashea
  G6. batch_generate_and_evaluate() retorna lista de resultados
  G7. Contexto estructurado se usa cuando hay KG metadata
"""

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
from sandbox_mteb.generation_executor import GenerationExecutor, GenMetricsResult


def _make_executor(llm_response="test answer"):
    llm = MagicMock()
    llm.invoke_async = AsyncMock(return_value=llm_response)

    calc = MagicMock()
    # Default: return a metric result
    async def _calc_async(mt, **kwargs):
        return MetricResult(metric_type=mt, value=0.75)
    calc.calculate_async = AsyncMock(side_effect=_calc_async)
    calc.embedding_model = None

    return GenerationExecutor(
        llm_service=llm,
        metrics_calculator=calc,
        max_context_chars=4000,
    )


def _make_query(qid="q1", answer="expected", answer_type="text"):
    return NormalizedQuery(
        query_id=qid,
        query_text="What is quantum physics?",
        expected_answer=answer,
        answer_type=answer_type,
        metadata={"question_type": "bridge"},
    )


def _make_retrieval(kg_meta=False):
    rd = QueryRetrievalDetail(
        retrieved_doc_ids=["d1"],
        retrieved_contents=["Quantum physics is a branch of science."],
        retrieval_scores=[0.9],
        expected_doc_ids=["d1"],
    )
    if kg_meta:
        rd.retrieval_metadata = {
            "kg_entities": [{"name": "quantum physics", "description": "branch of science"}],
            "kg_relations": [{"source": "quantum physics", "target": "science", "relation": "branch_of"}],
        }
    return rd


class TestExecuteGeneration:
    """G1/G2: _execute_generation_async."""

    def test_returns_generation_result(self):
        executor = _make_executor(llm_response="The answer is 42")
        result = asyncio.run(
            executor._execute_generation_async("query", "context", "hotpotqa")
        )
        assert isinstance(result, GenerationResult)
        assert result.generated_response == "The answer is 42"
        assert result.generation_time_ms > 0

    def test_error_returns_error_string(self):
        executor = _make_executor()
        executor._llm_service.invoke_async = AsyncMock(
            side_effect=RuntimeError("LLM timeout")
        )
        result = asyncio.run(
            executor._execute_generation_async("query", "context", "hotpotqa")
        )
        assert "[ERROR:" in result.generated_response


class TestCalculateMetrics:
    """G3/G4/G5: _calculate_metrics_async."""

    def test_hybrid_label_uses_accuracy(self):
        executor = _make_executor()
        ds_config = get_dataset_config("hotpotqa")

        primary, secondary = asyncio.run(
            executor._calculate_metrics_async(
                generated="yes",
                expected_answer="yes",
                answer_type="label",
                context="ctx",
                query_text="Is X true?",
                dataset_type=DatasetType.HYBRID,
                dataset_name="hotpotqa",
            )
        )
        assert primary.metric_type == MetricType.ACCURACY

    def test_hybrid_text_uses_f1(self):
        executor = _make_executor()

        primary, secondary = asyncio.run(
            executor._calculate_metrics_async(
                generated="answer",
                expected_answer="expected answer",
                answer_type="text",
                context="ctx",
                query_text="What is X?",
                dataset_type=DatasetType.HYBRID,
                dataset_name="hotpotqa",
            )
        )
        assert primary.metric_type == MetricType.F1_SCORE

    def test_secondary_error_does_not_crash(self):
        executor = _make_executor()
        call_count = 0

        async def _calc_with_error(mt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:  # First call is primary, rest are secondary
                raise ValueError("metric calc failed")
            return MetricResult(metric_type=mt, value=0.5)

        executor._metrics_calculator.calculate_async = AsyncMock(
            side_effect=_calc_with_error
        )

        primary, secondary = asyncio.run(
            executor._calculate_metrics_async(
                generated="answer",
                expected_answer="expected",
                answer_type="text",
                context="ctx",
                query_text="What?",
                dataset_type=DatasetType.HYBRID,
                dataset_name="hotpotqa",
            )
        )
        # Primary should succeed
        assert primary.value == 0.5
        # Secondary should have error entries but not crash
        for s in secondary:
            assert s.error is not None or s.value >= 0


class TestBatchGenerate:
    """G6: batch_generate_and_evaluate."""

    def test_returns_list_matching_queries(self):
        executor = _make_executor()
        queries = [_make_query("q1"), _make_query("q2")]
        retrievals = [_make_retrieval(), _make_retrieval()]
        ds_config = get_dataset_config("hotpotqa")

        results = asyncio.run(
            executor.batch_generate_and_evaluate(
                queries, retrievals, ds_config, "hotpotqa"
            )
        )
        assert len(results) == 2
        for r in results:
            if isinstance(r, BaseException):
                pytest.fail(f"Unexpected exception: {r}")
            assert isinstance(r, GenMetricsResult)


class TestStructuredContext:
    """G7: KG metadata triggers structured context."""

    def test_uses_structured_context_with_kg_meta(self):
        executor = _make_executor()
        query = _make_query()
        retrieval = _make_retrieval(kg_meta=True)
        ds_config = get_dataset_config("hotpotqa")

        with patch("sandbox_mteb.generation_executor.format_structured_context_with_stats") as mock_struct, \
             patch("sandbox_mteb.generation_executor.format_context") as mock_plain:
            mock_struct.return_value = ("structured context", 1)
            mock_plain.return_value = "plain context"

            asyncio.run(
                executor._process_single_async(query, retrieval, ds_config, "hotpotqa")
            )

            mock_struct.assert_called_once()
            mock_plain.assert_not_called()

    def test_uses_plain_context_without_kg_meta(self):
        executor = _make_executor()
        query = _make_query()
        retrieval = _make_retrieval(kg_meta=False)
        ds_config = get_dataset_config("hotpotqa")

        with patch("sandbox_mteb.generation_executor.format_structured_context_with_stats") as mock_struct, \
             patch("sandbox_mteb.generation_executor.format_context") as mock_plain:
            mock_struct.return_value = ("structured context", 1)
            mock_plain.return_value = "plain context"

            asyncio.run(
                executor._process_single_async(query, retrieval, ds_config, "hotpotqa")
            )

            mock_plain.assert_called_once()
            mock_struct.assert_not_called()

    def test_kg_budget_cap_triggered_annotated_in_metadata(self):
        """Divergencia #4+5: kg_budget_cap_triggered se anota per-query cuando hay KG data.

        Con max_context_chars=4000 (default del helper) y modo hybrid, el cap
        dispara (56000 > 2000). El observable debe aparecer en retrieval_metadata
        para que fluya a JSON/CSV via extract_retrieval_metadata_subset.
        """
        executor = _make_executor()
        query = _make_query()
        retrieval = _make_retrieval(kg_meta=True)
        retrieval.retrieval_metadata["lightrag_mode"] = "hybrid"
        ds_config = get_dataset_config("hotpotqa")

        with patch("sandbox_mteb.generation_executor.format_structured_context_with_stats") as mock_struct:
            mock_struct.return_value = ("structured context", 1)
            asyncio.run(
                executor._process_single_async(query, retrieval, ds_config, "hotpotqa")
            )

        assert retrieval.retrieval_metadata["kg_budget_cap_triggered"] is True

    def test_kg_budget_cap_not_triggered_with_wide_window(self):
        """Con ventana >= 112000 chars el cap no dispara en hybrid."""
        executor = GenerationExecutor(
            llm_service=_make_executor()._llm_service,
            metrics_calculator=_make_executor()._metrics_calculator,
            max_context_chars=200000,
        )
        query = _make_query()
        retrieval = _make_retrieval(kg_meta=True)
        retrieval.retrieval_metadata["lightrag_mode"] = "hybrid"
        ds_config = get_dataset_config("hotpotqa")

        with patch("sandbox_mteb.generation_executor.format_structured_context_with_stats") as mock_struct:
            mock_struct.return_value = ("structured context", 1)
            asyncio.run(
                executor._process_single_async(query, retrieval, ds_config, "hotpotqa")
            )

        assert retrieval.retrieval_metadata["kg_budget_cap_triggered"] is False

    def test_citation_refs_synth_populated_when_synthesis_enabled(self):
        """Divergencia #7: tras synthesis con narrativa citada, retrieval_metadata
        expone citation_refs_synth_* para auditar calidad de las citas [ref:N]."""
        # Narrativa mockeada con 3 citas validas a chunks 1-3 + una out_of_range a 99.
        narrative = "See [ref:1] and [ref:2]. Also [ref:99] inventada. And [ref:1] again."
        llm = MagicMock()
        llm.invoke_async = AsyncMock(return_value=narrative)
        calc = MagicMock()
        async def _calc_async(mt, **kwargs):
            return MetricResult(metric_type=mt, value=0.75)
        calc.calculate_async = AsyncMock(side_effect=_calc_async)
        calc.embedding_model = None
        executor = GenerationExecutor(
            llm_service=llm,
            metrics_calculator=calc,
            max_context_chars=4000,
            kg_synthesis_enabled=True,
        )

        query = _make_query()
        retrieval = _make_retrieval(kg_meta=True)
        retrieval.retrieval_metadata["lightrag_mode"] = "hybrid"
        ds_config = get_dataset_config("hotpotqa")

        # Patcheamos _with_stats para fijar n_chunks_emitted=3, asi [ref:99] cae
        # out_of_range deterministicamente. La narrativa la emite el LLM mockeado.
        with patch(
            "sandbox_mteb.generation_executor.format_structured_context_with_stats"
        ) as mock_struct:
            mock_struct.return_value = ("STRUCTURED_CTX", 3)
            asyncio.run(
                executor._process_single_async(
                    query, retrieval, ds_config, "hotpotqa",
                )
            )

        rm = retrieval.retrieval_metadata
        assert rm["citation_refs_synth_valid"] == 4  # [1], [2], [99], [1]
        assert rm["citation_refs_synth_in_range"] == 3  # [1], [2], [1]
        assert rm["citation_refs_synth_out_of_range"] == 1  # [99]
        assert rm["citation_refs_synth_distinct"] == 2  # {1, 2}
        assert rm["citation_refs_synth_malformed"] == 0

    def test_citation_refs_synth_absent_when_synthesis_disabled(self):
        """Sin synthesis, los campos citation_refs_synth_* NO se emiten."""
        executor = _make_executor()  # synthesis_enabled=False por default
        query = _make_query()
        retrieval = _make_retrieval(kg_meta=True)
        ds_config = get_dataset_config("hotpotqa")

        with patch(
            "sandbox_mteb.generation_executor.format_structured_context_with_stats"
        ) as mock_struct:
            mock_struct.return_value = ("STRUCTURED_CTX", 3)
            asyncio.run(
                executor._process_single_async(
                    query, retrieval, ds_config, "hotpotqa",
                )
            )

        assert "citation_refs_synth_valid" not in retrieval.retrieval_metadata

    def test_kg_budget_cap_not_annotated_without_kg_meta(self):
        """SIMPLE_VECTOR sin KG data NO emite kg_budget_cap_triggered."""
        executor = _make_executor()
        query = _make_query()
        retrieval = _make_retrieval(kg_meta=False)
        ds_config = get_dataset_config("hotpotqa")

        with patch("sandbox_mteb.generation_executor.format_context") as mock_plain:
            mock_plain.return_value = "plain context"
            asyncio.run(
                executor._process_single_async(query, retrieval, ds_config, "hotpotqa")
            )

        assert "kg_budget_cap_triggered" not in retrieval.retrieval_metadata
