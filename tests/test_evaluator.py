"""
Tests unitarios para MTEBEvaluator (P8, Fase 4).

Cobertura:
  EV1. _init_components crea embedding + LLM + metrics calculator
  EV2. _init_components sin generacion -> no crea LLM service
  EV3. _load_dataset falla si conexion MinIO falla
  EV4. _load_dataset falla si load_status != "success"
  EV5. _index_documents invoca retriever.index_documents
  EV6. _cleanup libera recursos
  EV7. _assemble_results con generacion exitosa -> COMPLETED
  EV8. _assemble_results con generacion fallida -> FAILED
  EV9. _assemble_results sin generacion -> COMPLETED (retrieval-only)
"""

from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from shared.types import (
    EvaluationStatus,
    GenerationResult,
    LoadedDataset,
    NormalizedQuery,
    NormalizedDocument,
    DatasetType,
    MetricType,
    QueryRetrievalDetail,
    QueryEvaluationResult,
)
from shared.retrieval.core import RetrievalConfig, RetrievalStrategy
from sandbox_mteb.evaluator import MTEBEvaluator
from sandbox_mteb.generation_executor import GenMetricsResult


# =============================================================================
# Helpers
# =============================================================================

def _make_config(**overrides):
    """Crea MTEBConfig con defaults para testing."""
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig

    defaults = dict(
        infra=InfraConfig(
            embedding_base_url="http://fake:8000/v1",
            embedding_model_name="test-model",
            llm_base_url="http://fake:8000/v1",
            llm_model_name="test-llm",
        ),
        storage=MinIOStorageConfig(
            minio_endpoint="http://fake:9000",
            minio_access_key="test",
            minio_secret_key="test",
            minio_bucket="test",
        ),
        retrieval=RetrievalConfig(strategy=RetrievalStrategy.SIMPLE_VECTOR),
        reranker=RerankerConfig(enabled=False),
        dataset_name="hotpotqa",
        generation_enabled=True,
    )
    defaults.update(overrides)
    return MTEBConfig(**defaults)


def _make_retrieval_detail():
    return QueryRetrievalDetail(
        retrieved_doc_ids=["d1", "d2"],
        retrieved_contents=["content 1", "content 2"],
        retrieval_scores=[0.9, 0.8],
        expected_doc_ids=["d1"],
    )


def _make_gen_result():
    return GenMetricsResult(
        generation=GenerationResult(generated_response="test answer"),
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=0.8,
        secondary_metrics={"exact_match": 0.5},
    )


# =============================================================================
# EV1: _init_components con generacion
# =============================================================================

@patch("sandbox_mteb.evaluator.resolve_max_context_chars", return_value=4000)
@patch("sandbox_mteb.evaluator.load_embedding_model")
@patch("sandbox_mteb.evaluator.AsyncLLMService")
def test_init_components_with_generation(mock_llm_cls, mock_embed, mock_resolve):
    """Con generation_enabled=True, crea embedding + LLM + metrics con config correcta."""
    mock_embed.return_value = MagicMock()
    mock_llm_cls.return_value = MagicMock()

    config = _make_config(generation_enabled=True)
    ev = MTEBEvaluator(config)
    ev._init_components()

    # Verifica que se pasan los parametros de config correctos
    mock_embed.assert_called_once_with(
        base_url="http://fake:8000/v1",
        model_name="test-model",
        model_type="symmetric",
    )
    mock_llm_cls.assert_called_once_with(
        base_url="http://fake:8000/v1",
        model_name="test-llm",
        max_concurrent=config.infra.nim_max_concurrent,
        timeout_seconds=config.infra.nim_timeout,
        max_retries=config.infra.nim_max_retries,
    )
    assert ev._embedding_model is not None
    assert ev._llm_service is not None
    assert ev._metrics_calculator is not None
    assert ev._generation_executor is not None
    # Verify metrics calculator got the embedding model
    assert ev._metrics_calculator.embedding_model is ev._embedding_model


# =============================================================================
# EV2: _init_components sin generacion y sin LIGHT_RAG
# =============================================================================

@patch("sandbox_mteb.evaluator.load_embedding_model")
def test_init_components_without_generation(mock_embed):
    """Sin generation y sin LIGHT_RAG, no crea LLM service."""
    mock_embed.return_value = MagicMock()

    config = _make_config(generation_enabled=False)
    ev = MTEBEvaluator(config)
    ev._init_components()

    # Embedding se crea siempre
    mock_embed.assert_called_once_with(
        base_url="http://fake:8000/v1",
        model_name="test-model",
        model_type="symmetric",
    )
    assert ev._embedding_model is not None
    # LLM no se crea sin generacion ni LIGHT_RAG
    assert ev._llm_service is None
    assert ev._generation_executor is None
    # MetricsCalculator se crea sin LLM judge
    assert ev._metrics_calculator is not None


# =============================================================================
# EV3: _load_dataset falla en conexion
# =============================================================================

@patch("sandbox_mteb.evaluator.MinIOLoader")
def test_load_dataset_connection_failure(mock_loader_cls):
    """check_connection() False -> ConnectionError."""
    mock_loader = MagicMock()
    mock_loader.check_connection.return_value = False
    mock_loader_cls.return_value = mock_loader

    config = _make_config()
    ev = MTEBEvaluator(config)

    with pytest.raises(ConnectionError):
        ev._load_dataset()


# =============================================================================
# EV4: _load_dataset falla si status != success
# =============================================================================

@patch("sandbox_mteb.evaluator.MinIOLoader")
def test_load_dataset_load_failure(mock_loader_cls):
    """load_status != 'success' -> RuntimeError."""
    mock_loader = MagicMock()
    mock_loader.check_connection.return_value = True
    bad_dataset = LoadedDataset(
        name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )
    bad_dataset.load_status = "error"
    bad_dataset.error_message = "file not found"
    mock_loader.load_dataset.return_value = bad_dataset
    mock_loader_cls.return_value = mock_loader

    config = _make_config()
    ev = MTEBEvaluator(config)

    with pytest.raises(RuntimeError, match="file not found"):
        ev._load_dataset()


# =============================================================================
# EV5: _index_documents
# =============================================================================

def test_index_documents():
    """_index_documents invoca retriever.index_documents con docs correctos."""
    config = _make_config()
    ev = MTEBEvaluator(config)
    ev._embedding_model = MagicMock()
    ev._llm_service = None

    mock_retriever = MagicMock()
    mock_retriever.index_documents.return_value = True

    with patch("sandbox_mteb.evaluator.get_retriever", return_value=mock_retriever):
        corpus = {
            "d1": NormalizedDocument(doc_id="d1", title="Title 1", content="Content 1"),
            "d2": NormalizedDocument(doc_id="d2", title="Title 2", content="Content 2"),
        }
        ev._index_documents("hotpotqa", corpus, "run_123")

    mock_retriever.index_documents.assert_called_once()
    call_args = mock_retriever.index_documents.call_args
    docs = call_args[0][0]
    assert len(docs) == 2
    # Verificar que el contenido se serializa correctamente
    doc_ids = {d["doc_id"] for d in docs}
    assert doc_ids == {"d1", "d2"}
    for d in docs:
        assert "content" in d
        assert "title" in d
    assert ev._retriever is mock_retriever


# =============================================================================
# EV6: _cleanup
# =============================================================================

def test_cleanup():
    """_cleanup libera retriever, embedding, llm."""
    config = _make_config()
    ev = MTEBEvaluator(config)
    ev._retriever = MagicMock()
    ev._embedding_model = MagicMock()
    ev._llm_service = MagicMock()

    ev._cleanup()

    assert ev._retriever is None
    assert ev._embedding_model is None
    assert ev._llm_service is None


# =============================================================================
# EV7: _assemble_results con generacion
# =============================================================================

def test_assemble_results_with_generation():
    """Generacion exitosa -> COMPLETED con metricas."""
    config = _make_config(generation_enabled=True)
    ev = MTEBEvaluator(config)

    queries = [NormalizedQuery(
        query_id="q1", query_text="test?",
        relevant_doc_ids=["d1"], metadata={},
    )]
    retrievals = [_make_retrieval_detail()]
    gen_results = [_make_gen_result()]
    rerank_statuses = [None]
    ds_config = {"type": DatasetType.HYBRID}

    results = ev._assemble_results(
        queries, retrievals, gen_results, rerank_statuses,
        ds_config, "hotpotqa"
    )
    assert len(results) == 1
    assert results[0].status == EvaluationStatus.COMPLETED
    assert results[0].primary_metric_value == 0.8
    assert results[0].generation.generated_response == "test answer"


# =============================================================================
# EV8: _assemble_results generacion fallida
# =============================================================================

def test_assemble_results_generation_failed():
    """Generacion None con gen habilitada -> FAILED."""
    config = _make_config(generation_enabled=True)
    ev = MTEBEvaluator(config)

    queries = [NormalizedQuery(
        query_id="q1", query_text="test?",
        relevant_doc_ids=["d1"], metadata={},
    )]
    retrievals = [_make_retrieval_detail()]
    gen_results = [None]  # Failed
    rerank_statuses = [None]
    ds_config = {"type": DatasetType.HYBRID}

    results = ev._assemble_results(
        queries, retrievals, gen_results, rerank_statuses,
        ds_config, "hotpotqa"
    )
    assert results[0].status == EvaluationStatus.FAILED


# =============================================================================
# EV9: _assemble_results sin generacion
# =============================================================================

def test_assemble_results_no_generation():
    """Sin generacion -> COMPLETED (retrieval-only), sin generation field."""
    config = _make_config(generation_enabled=False)
    ev = MTEBEvaluator(config)

    queries = [NormalizedQuery(
        query_id="q1", query_text="test?",
        relevant_doc_ids=["d1"], metadata={},
    )]
    retrievals = [_make_retrieval_detail()]
    gen_results = [None]
    rerank_statuses = [None]
    ds_config = {"type": DatasetType.HYBRID}

    results = ev._assemble_results(
        queries, retrievals, gen_results, rerank_statuses,
        ds_config, "hotpotqa"
    )
    assert results[0].status == EvaluationStatus.COMPLETED
    assert results[0].generation is None
