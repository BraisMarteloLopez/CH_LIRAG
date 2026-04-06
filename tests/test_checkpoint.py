"""
Tests unitarios para sandbox_mteb/checkpoint.py (Fase I.1).

Cobertura:
  C1. save_checkpoint atomicidad (tmp + rename)
  C2. load_checkpoint deserializacion correcta
  C3. load_checkpoint con JSON corrupto retorna None
  C4. load_checkpoint con archivo inexistente retorna None
  C5. serialize/deserialize roundtrip preserva datos
  C6. delete_checkpoint elimina archivo
  C7. save_checkpoint con set vacio funciona
  C8. deserialize con generation=None
"""

import json
from pathlib import Path

import pytest

from shared.types import (
    DatasetType,
    EvaluationStatus,
    GenerationResult,
    MetricType,
    QueryEvaluationResult,
    QueryRetrievalDetail,
)
from sandbox_mteb.checkpoint import (
    checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
    serialize_query_result,
    deserialize_query_result,
)


def _make_qr(query_id: str = "q1", with_generation: bool = True) -> QueryEvaluationResult:
    """Crea un QueryEvaluationResult minimo para tests."""
    gen = GenerationResult(
        generated_response="answer text",
        generation_time_ms=42.0,
    ) if with_generation else None

    return QueryEvaluationResult(
        query_id=query_id,
        query_text=f"What is {query_id}?",
        dataset_name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        status=EvaluationStatus.COMPLETED,
        expected_response="expected",
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=0.85,
        secondary_metrics={"exact_match": 1.0},
        metadata={"question_type": "bridge"},
        retrieval=QueryRetrievalDetail(
            retrieved_doc_ids=["d1", "d2"],
            retrieved_contents=["content 1", "content 2"],
            retrieval_scores=[0.9, 0.8],
            expected_doc_ids=["d1", "d3"],
            retrieval_time_ms=10.5,
            generation_doc_ids=["d1"],
            generation_contents=["content 1"],
            pre_rerank_candidate_ids=["d1", "d2", "d3"],
            retrieval_metadata={"graph_active": True},
        ),
        generation=gen,
    )


# =============================================================================
# C1: save_checkpoint atomicidad
# =============================================================================

def test_save_checkpoint_creates_file(tmp_path):
    """save_checkpoint crea archivo JSON con datos correctos."""
    qr = _make_qr()
    save_checkpoint(str(tmp_path), "run_001", {"q1"}, [qr])

    path = checkpoint_path(str(tmp_path), "run_001")
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_001"
    assert data["evaluated_query_ids"] == ["q1"]
    assert data["num_results"] == 1
    assert len(data["results"]) == 1


def test_save_checkpoint_no_tmp_left(tmp_path):
    """save_checkpoint no deja archivo .tmp tras escritura atomica."""
    save_checkpoint(str(tmp_path), "run_002", {"q1"}, [_make_qr()])

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_save_checkpoint_overwrites(tmp_path):
    """save_checkpoint sobreescribe checkpoint previo."""
    save_checkpoint(str(tmp_path), "run_003", {"q1"}, [_make_qr("q1")])
    save_checkpoint(str(tmp_path), "run_003", {"q1", "q2"}, [_make_qr("q1"), _make_qr("q2")])

    data = json.loads(checkpoint_path(str(tmp_path), "run_003").read_text())
    assert len(data["evaluated_query_ids"]) == 2


# =============================================================================
# C2: load_checkpoint deserializacion
# =============================================================================

def test_load_checkpoint_roundtrip(tmp_path):
    """save + load produce datos equivalentes."""
    original = _make_qr()
    save_checkpoint(str(tmp_path), "run_rt", {"q1"}, [original])

    result = load_checkpoint(str(tmp_path), "run_rt")
    assert result is not None

    ids, results = result
    assert ids == {"q1"}
    assert len(results) == 1

    loaded = results[0]
    assert loaded.query_id == "q1"
    assert loaded.primary_metric_value == 0.85
    assert loaded.retrieval.retrieved_doc_ids == ["d1", "d2"]
    assert loaded.generation is not None
    assert loaded.generation.generated_response == "answer text"


# =============================================================================
# C3: load_checkpoint JSON corrupto
# =============================================================================

def test_load_checkpoint_corrupt_json(tmp_path):
    """JSON corrupto retorna None sin crash."""
    path = checkpoint_path(str(tmp_path), "run_bad")
    path.write_text("{invalid json!!!", encoding="utf-8")

    result = load_checkpoint(str(tmp_path), "run_bad")
    assert result is None


# =============================================================================
# C4: load_checkpoint archivo inexistente
# =============================================================================

def test_load_checkpoint_nonexistent(tmp_path):
    """Checkpoint inexistente retorna None."""
    result = load_checkpoint(str(tmp_path), "nonexistent_run")
    assert result is None


# =============================================================================
# C5: serialize/deserialize roundtrip
# =============================================================================

def test_serialize_deserialize_preserves_fields():
    """Roundtrip serialize→deserialize preserva todos los campos."""
    original = _make_qr()
    data = serialize_query_result(original)
    restored = deserialize_query_result(data)

    assert restored.query_id == original.query_id
    assert restored.query_text == original.query_text
    assert restored.dataset_type == original.dataset_type
    assert restored.status == original.status
    assert restored.primary_metric_type == original.primary_metric_type
    assert restored.primary_metric_value == original.primary_metric_value
    assert restored.secondary_metrics == original.secondary_metrics
    assert restored.metadata == original.metadata
    assert restored.retrieval.retrieved_doc_ids == original.retrieval.retrieved_doc_ids
    assert restored.retrieval.retrieval_scores == original.retrieval.retrieval_scores
    assert restored.retrieval.generation_doc_ids == original.retrieval.generation_doc_ids
    assert restored.retrieval.pre_rerank_candidate_ids == original.retrieval.pre_rerank_candidate_ids


# =============================================================================
# C6: delete_checkpoint
# =============================================================================

def test_delete_checkpoint(tmp_path):
    """delete_checkpoint elimina archivo existente."""
    save_checkpoint(str(tmp_path), "run_del", {"q1"}, [_make_qr()])
    path = checkpoint_path(str(tmp_path), "run_del")
    assert path.exists()

    delete_checkpoint(str(tmp_path), "run_del")
    assert not path.exists()


def test_delete_checkpoint_nonexistent(tmp_path):
    """delete_checkpoint con archivo inexistente no falla."""
    delete_checkpoint(str(tmp_path), "nonexistent")


# =============================================================================
# C7: save con set vacio
# =============================================================================

def test_save_checkpoint_empty_set(tmp_path):
    """save_checkpoint con set vacio y lista vacia funciona."""
    save_checkpoint(str(tmp_path), "run_empty", set(), [])

    data = json.loads(checkpoint_path(str(tmp_path), "run_empty").read_text())
    assert data["evaluated_query_ids"] == []
    assert data["results"] == []


# =============================================================================
# C8: deserialize sin generation
# =============================================================================

def test_deserialize_without_generation():
    """Deserialize con generation=None produce None."""
    qr = _make_qr(with_generation=False)
    data = serialize_query_result(qr)
    assert data["generation"] is None

    restored = deserialize_query_result(data)
    assert restored.generation is None
