"""Result builder: agrega metricas y construye EvaluationRun."""

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

from shared.types import (
    EvaluationRun,
    EvaluationStatus,
    QueryEvaluationResult,
    LoadedDataset,
)
from shared.metrics import JudgeMetricStats, get_judge_fallback_stats
from shared.operational_tracker import OperationalStats, get_operational_stats
from shared.retrieval import RetrievalStrategy

from .config import MTEBConfig
from .generation_executor import KGSynthesisStats, get_kg_synthesis_stats

logger = logging.getLogger(__name__)

# Etiqueta del observable `strategy_actual` en config_snapshot._runtime.
# Valores validos: nombres de RetrievalStrategy + "FALLBACK_SIMPLE_VECTOR"
# (fallback emitido cuando el retriever real divergio del configurado, ver
# `strategy_mismatches`).
StrategyActualLabel = Literal[
    "SIMPLE_VECTOR", "LIGHT_RAG", "FALLBACK_SIMPLE_VECTOR"
]


class RuntimeSnapshot(TypedDict):
    """Forma del dict `config_snapshot["_runtime"]` (R5).

    Campos derivados del run (no parte de la config estatica). Se anade
    al dict producido por `_serialize_config`; consumido por
    `EvaluationRun.to_dict()` y el JSON export. Cambios de schema aqui
    impactan dashboards y scripts post-hoc (`jq '.config_snapshot._runtime'`).
    """

    max_context_chars: int
    rerank_failures: Optional[int]
    strategy_mismatches: int
    corpus_total_available: int
    corpus_indexed: int
    gen_zero_count: int
    gen_nonzero_count: int
    strategy_actual: StrategyActualLabel
    judge_fallback_stats: Dict[str, JudgeMetricStats]
    kg_synthesis_stats: KGSynthesisStats
    operational_stats: OperationalStats


def _serialize_config(config: MTEBConfig) -> Dict[str, Any]:
    """Serializa toda la MTEBConfig a dict JSON-safe para snapshot."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, Path):
            return str(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return obj

    return {
        f.name: _convert(getattr(config, f.name))
        for f in dataclasses.fields(config)
    }


def build_run(
    config: MTEBConfig,
    run_id: str,
    dataset: LoadedDataset,
    query_results: List[QueryEvaluationResult],
    elapsed_seconds: float,
    indexed_corpus_size: int,
    max_context_chars: int,
    rerank_failures: int,
    strategy_mismatches: int,
) -> EvaluationRun:
    """Construye EvaluationRun plano a partir de resultados."""
    completed = [
        qr for qr in query_results
        if qr.status == EvaluationStatus.COMPLETED
    ]
    failed = [
        qr for qr in query_results
        if qr.status == EvaluationStatus.FAILED
    ]

    avg_hit5 = 0.0
    avg_mrr = 0.0
    recall_sums: Dict[int, float] = {}
    ndcg_sums: Dict[int, float] = {}

    if completed:
        for qr in completed:
            avg_hit5 += qr.retrieval.hit_at_k.get(5, 0.0)
            avg_mrr += qr.retrieval.mrr
            for k, v in qr.retrieval.recall_at_k.items():
                recall_sums[k] = recall_sums.get(k, 0.0) + v
            for k, v in qr.retrieval.ndcg_at_k.items():
                ndcg_sums[k] = ndcg_sums.get(k, 0.0) + v

        nc = len(completed)
        avg_hit5 /= nc
        avg_mrr /= nc
        avg_recall = {k: v / nc for k, v in recall_sums.items()}
        avg_ndcg = {k: v / nc for k, v in ndcg_sums.items()}
        complement_recall = {k: 1.0 - v for k, v in avg_recall.items()}
        avg_retrieved = sum(
            len(qr.retrieval.retrieved_doc_ids) for qr in completed
        ) / nc
        avg_expected = sum(
            len(qr.retrieval.expected_doc_ids) for qr in completed
        ) / nc
    else:
        avg_recall = {}
        avg_ndcg = {}
        complement_recall = {}
        avg_retrieved = 0.0
        avg_expected = 0.0

    # Metricas de retrieval efectivo (post-rerank)
    avg_gen_recall: Optional[float] = None
    avg_gen_hit: Optional[float] = None
    rescue_count = 0
    if completed:
        with_gen = [
            qr for qr in completed if qr.retrieval.generation_doc_ids
        ]
        if with_gen:
            retrieval_k = config.retrieval.retrieval_k
            avg_gen_recall = sum(
                qr.retrieval.generation_recall for qr in with_gen
            ) / len(with_gen)
            avg_gen_hit = sum(
                qr.retrieval.generation_hit for qr in with_gen
            ) / len(with_gen)
            rescue_count = sum(
                1 for qr in with_gen
                if qr.retrieval.generation_recall
                > qr.retrieval.recall_at_k.get(retrieval_k, 0.0)
            )

    # Generacion promedio - INCLUYE ZEROS (los fallos cuentan como 0, no se excluyen del promedio)
    avg_gen = None
    gen_zero_count = 0
    gen_nonzero_count = 0
    if config.generation_enabled and completed:
        all_gen_values = [
            qr.primary_metric_value
            for qr in completed
        ]
        gen_zero_count = sum(1 for v in all_gen_values if v == 0.0)
        gen_nonzero_count = sum(1 for v in all_gen_values if v > 0.0)
        if all_gen_values:
            avg_gen = sum(all_gen_values) / len(all_gen_values)

    # Config snapshot: serializacion completa para reproducibilidad post-hoc
    config_snapshot = _serialize_config(config)
    # Ver CLAUDE.md §Observabilidad de runs para interpretacion de cada stat.
    judge_fallback_stats = get_judge_fallback_stats()
    strategy_actual: StrategyActualLabel = (
        cast(StrategyActualLabel, config.retrieval.strategy.name)
        if strategy_mismatches == 0
        else "FALLBACK_SIMPLE_VECTOR"
    )
    runtime_snapshot: RuntimeSnapshot = {
        "max_context_chars": max_context_chars,
        "rerank_failures": rerank_failures if config.reranker.enabled else None,
        "strategy_mismatches": strategy_mismatches,
        "corpus_total_available": len(dataset.corpus),
        "corpus_indexed": indexed_corpus_size,
        "gen_zero_count": gen_zero_count,
        "gen_nonzero_count": gen_nonzero_count,
        "strategy_actual": strategy_actual,
        "judge_fallback_stats": judge_fallback_stats,
        "kg_synthesis_stats": get_kg_synthesis_stats(),
        "operational_stats": get_operational_stats(),
    }
    config_snapshot["_runtime"] = runtime_snapshot

    return EvaluationRun(
        run_id=run_id,
        dataset_name=config.dataset_name,
        embedding_model=config.infra.embedding_model_name,
        retrieval_strategy=config.retrieval.strategy.name,
        config_snapshot=config_snapshot,
        status=EvaluationStatus.COMPLETED,
        num_queries_evaluated=len(completed),
        num_queries_failed=len(failed),
        total_documents=indexed_corpus_size,
        avg_hit_rate_at_5=avg_hit5,
        avg_mrr=avg_mrr,
        avg_recall_at_k=avg_recall,
        avg_ndcg_at_k=avg_ndcg,
        retrieval_complement_recall_at_k=complement_recall,
        avg_retrieved_count=avg_retrieved,
        avg_expected_count=avg_expected,
        avg_generation_recall=avg_gen_recall,
        avg_generation_hit=avg_gen_hit,
        reranker_rescue_count=rescue_count,
        avg_generation_score=avg_gen,
        execution_time_seconds=elapsed_seconds,
        query_results=query_results,
    )


__all__ = ["RuntimeSnapshot", "StrategyActualLabel", "build_run"]
