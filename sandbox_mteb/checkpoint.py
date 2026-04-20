"""
Checkpoint / Resume para evaluaciones largas.

Serializa/deserializa QueryEvaluationResult a JSON para persistir
resultados parciales durante runs de miles de queries.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.types import (
    DatasetType,
    EvaluationStatus,
    GenerationResult,
    MetricType,
    QueryEvaluationResult,
    QueryRetrievalDetail,
)

logger = logging.getLogger(__name__)

from shared.constants import CHECKPOINT_CHUNK_SIZE  # queries per checkpoint


def checkpoint_path(results_dir: str, run_id: str) -> Path:
    return Path(results_dir) / f"{run_id}_checkpoint.json"


def save_checkpoint(
    results_dir: str,
    run_id: str,
    evaluated_query_ids: Set[str],
    results: List[QueryEvaluationResult],
) -> None:
    """Persiste resultados parciales a disco (atomic write)."""
    path = checkpoint_path(results_dir, run_id)
    data = {
        "run_id": run_id,
        "evaluated_query_ids": sorted(evaluated_query_ids),
        "num_results": len(results),
        "results": [serialize_query_result(r) for r in results],
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))
    logger.info(
        f"  Checkpoint guardado: {len(evaluated_query_ids)} queries → {path.name}"
    )


def load_checkpoint(
    results_dir: str, run_id: str
) -> Optional[Tuple[Set[str], List[QueryEvaluationResult]]]:
    """Carga checkpoint previo si existe."""
    path = checkpoint_path(results_dir, run_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        evaluated_ids = set(data["evaluated_query_ids"])
        results = [deserialize_query_result(r) for r in data["results"]]
        logger.info(
            f"  Checkpoint cargado: {len(evaluated_ids)} queries desde {path.name}"
        )
        return evaluated_ids, results
    except Exception as e:
        logger.warning(f"  Checkpoint corrupto ({path.name}): {e}. Ignorando.")
        return None


def delete_checkpoint(results_dir: str, run_id: str) -> None:
    path = checkpoint_path(results_dir, run_id)
    if path.exists():
        path.unlink()
        logger.info(f"  Checkpoint eliminado: {path.name}")


def serialize_query_result(qr: QueryEvaluationResult) -> Dict[str, Any]:
    """Serializa un QueryEvaluationResult a dict JSON-compatible."""
    r = qr.retrieval
    gen = qr.generation
    return {
        "query_id": qr.query_id,
        "query_text": qr.query_text,
        "dataset_name": qr.dataset_name,
        "dataset_type": qr.dataset_type.value,
        "status": qr.status.value,
        "error_message": qr.error_message,
        "expected_response": qr.expected_response,
        "primary_metric_type": qr.primary_metric_type.value,
        "primary_metric_value": qr.primary_metric_value,
        "secondary_metrics": qr.secondary_metrics,
        "metadata": qr.metadata,
        "retrieval": {
            "retrieved_doc_ids": r.retrieved_doc_ids,
            "retrieved_contents": r.retrieved_contents,
            "retrieval_scores": r.retrieval_scores,
            "expected_doc_ids": r.expected_doc_ids,
            "retrieval_time_ms": r.retrieval_time_ms,
            "generation_doc_ids": r.generation_doc_ids,
            "generation_contents": r.generation_contents,
            "pre_rerank_candidate_ids": r.pre_rerank_candidate_ids,
            "retrieval_metadata": r.retrieval_metadata,
        },
        "generation": {
            "generated_response": gen.generated_response,
            "generation_time_ms": gen.generation_time_ms,
        } if gen else None,
    }


def deserialize_query_result(data: Dict[str, Any]) -> QueryEvaluationResult:
    """Deserializa un dict a QueryEvaluationResult."""
    rd = data["retrieval"]
    gen_data = data.get("generation")
    return QueryEvaluationResult(
        query_id=data["query_id"],
        query_text=data["query_text"],
        dataset_name=data["dataset_name"],
        dataset_type=DatasetType(data["dataset_type"]),
        status=EvaluationStatus(data["status"]),
        error_message=data.get("error_message"),
        expected_response=data.get("expected_response"),
        primary_metric_type=MetricType(data["primary_metric_type"]),
        primary_metric_value=data.get("primary_metric_value", 0.0),
        secondary_metrics=data.get("secondary_metrics", {}),
        metadata=data.get("metadata", {}),
        retrieval=QueryRetrievalDetail(
            retrieved_doc_ids=rd["retrieved_doc_ids"],
            retrieved_contents=rd["retrieved_contents"],
            retrieval_scores=rd["retrieval_scores"],
            expected_doc_ids=rd["expected_doc_ids"],
            retrieval_time_ms=rd.get("retrieval_time_ms", 0.0),
            generation_doc_ids=rd.get("generation_doc_ids", []),
            generation_contents=rd.get("generation_contents", []),
            pre_rerank_candidate_ids=rd.get("pre_rerank_candidate_ids", []),
            retrieval_metadata=rd.get("retrieval_metadata", {}),
        ),
        generation=GenerationResult(
            generated_response=gen_data["generated_response"],
            generation_time_ms=gen_data.get("generation_time_ms", 0.0),
        ) if gen_data else None,
    )


__all__ = [
    "CHECKPOINT_CHUNK_SIZE",
    "checkpoint_path",
    "save_checkpoint",
    "load_checkpoint",
    "delete_checkpoint",
    "serialize_query_result",
    "deserialize_query_result",
]
