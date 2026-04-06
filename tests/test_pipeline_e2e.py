"""
Test end-to-end mockeado del pipeline MTEBEvaluator (P8, Fase 4).

Ejecuta el flujo completo run() con:
- MinIO mockeado (datos in-memory)
- LLM mockeado (respuestas fijas)
- Embedding mockeado (vectores fijos)
- ChromaDB real (in-memory, efimero)

Verifica que EvaluationRun se construye correctamente con
queries evaluadas, metricas no-zero, y status completado.
"""

from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from dataclasses import field

import pytest

from shared.types import (
    EvaluationRun,
    EvaluationStatus,
    LoadedDataset,
    NormalizedQuery,
    NormalizedDocument,
    DatasetType,
    MetricType,
)
from shared.retrieval.core import RetrievalConfig, RetrievalStrategy
from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
from shared.config_base import InfraConfig, RerankerConfig
from sandbox_mteb.evaluator import MTEBEvaluator


# =============================================================================
# Helpers
# =============================================================================

def _make_mini_dataset():
    """Crea un dataset minimo (3 queries, 6 docs) in-memory."""
    ds = LoadedDataset(
        name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
        secondary_metrics=[MetricType.EXACT_MATCH],
    )
    ds.load_status = "success"

    # 6 documents
    for i in range(6):
        did = f"doc{i}"
        ds.corpus[did] = NormalizedDocument(
            doc_id=did,
            title=f"Document {i}",
            content=f"This is the content of document {i} about topic {i % 3}.",
        )
    ds.total_corpus = len(ds.corpus)

    # 3 queries, each relevant to 2 docs
    for i in range(3):
        ds.queries.append(NormalizedQuery(
            query_id=f"q{i}",
            query_text=f"What is topic {i}?",
            expected_answer=f"topic {i}",
            answer_type="text",
            relevant_doc_ids=[f"doc{i}", f"doc{i+3}"],
            metadata={"question_type": "bridge"},
        ))
    ds.total_queries = len(ds.queries)

    return ds


def _make_e2e_config(tmp_path):
    return MTEBConfig(
        infra=InfraConfig(
            embedding_base_url="http://fake:8000/v1",
            embedding_model_name="test-embed",
            llm_base_url="http://fake:8000/v1",
            llm_model_name="test-llm",
            nim_max_concurrent=2,
            nim_timeout=30,
        ),
        storage=MinIOStorageConfig(
            minio_endpoint="http://fake:9000",
            minio_access_key="test",
            minio_secret_key="test",
            minio_bucket="test",
            evaluation_results_dir=tmp_path / "results",
        ),
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.SIMPLE_VECTOR,
            retrieval_k=5,
        ),
        reranker=RerankerConfig(enabled=False),
        dataset_name="hotpotqa",
        generation_enabled=True,
        max_queries=3,
        max_corpus=6,
    )


# =============================================================================
# E2E: Pipeline completo mockeado
# =============================================================================

@patch("sandbox_mteb.evaluator.MinIOLoader")
@patch("sandbox_mteb.evaluator.load_embedding_model")
@patch("sandbox_mteb.evaluator.AsyncLLMService")
@patch("sandbox_mteb.evaluator.batch_embed_queries")
@patch("sandbox_mteb.evaluator.get_retriever")
def test_pipeline_e2e_mocked(
    mock_get_retriever, mock_batch_embed, mock_llm_cls, mock_embed_fn,
    mock_loader_cls, tmp_path
):
    """Pipeline completo: load -> index -> retrieve -> generate -> EvaluationRun."""
    from shared.retrieval.core import RetrievalResult

    # --- Mock MinIO Loader ---
    dataset = _make_mini_dataset()
    mock_loader = MagicMock()
    mock_loader.check_connection.return_value = True
    mock_loader.load_dataset.return_value = dataset
    mock_loader_cls.return_value = mock_loader

    # --- Mock Embedding Model ---
    mock_embed = MagicMock()
    mock_embed_fn.return_value = mock_embed

    # --- Mock LLM Service ---
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value="topic 0")
    mock_llm._max_concurrent = 2
    mock_llm_cls.return_value = mock_llm

    # --- Mock batch embed queries ---
    mock_batch_embed.return_value = [[0.5] * 64] * 3

    # --- Mock Retriever ---
    mock_retriever = MagicMock()
    mock_retriever.index_documents.return_value = True
    # Return first 2 docs for any query
    mock_retriever.retrieve.return_value = RetrievalResult(
        doc_ids=["doc0", "doc1", "doc3"],
        contents=["content 0", "content 1", "content 3"],
        scores=[0.9, 0.8, 0.7],
        vector_scores=[0.9, 0.8, 0.7],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        metadata={},
    )
    mock_retriever.retrieve_by_vector.return_value = mock_retriever.retrieve.return_value
    mock_retriever.clear_index.return_value = None
    mock_get_retriever.return_value = mock_retriever

    # --- Run ---
    config = _make_e2e_config(tmp_path)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    ev = MTEBEvaluator(config)

    result = ev.run()

    # --- Assertions ---
    assert isinstance(result, EvaluationRun)
    assert result.status == EvaluationStatus.COMPLETED
    assert result.num_queries_evaluated == 3
    assert result.dataset_name == "hotpotqa"
    assert result.embedding_model == "test-embed"
    assert result.retrieval_strategy == "SIMPLE_VECTOR"

    # Retrieval metrics: retriever returns ["doc0","doc1","doc3"] for all queries.
    # q0 relevant=["doc0","doc3"] -> MRR=1.0 (doc0@rank1), Hit@1=1, Recall@3=1.0
    # q1 relevant=["doc1","doc4"] -> MRR=0.5 (doc1@rank2), Hit@1=0, Hit@3=1.0
    # q2 relevant=["doc2","doc5"] -> MRR=0.0 (none found), Hit@3=0.0
    assert result.avg_mrr == pytest.approx(0.5, abs=0.01)  # (1.0+0.5+0.0)/3
    assert result.avg_hit_rate_at_5 == pytest.approx(2.0 / 3.0, abs=0.01)

    # Results contain query-level details
    assert len(result.query_results) == 3
    for qr in result.query_results:
        assert qr.query_id.startswith("q")
        assert qr.retrieval is not None
        assert len(qr.retrieval.retrieved_doc_ids) > 0

    # Verify individual query MRRs
    mrrs = {qr.query_id: qr.retrieval.mrr for qr in result.query_results}
    assert mrrs["q0"] == pytest.approx(1.0)
    assert mrrs["q1"] == pytest.approx(0.5)
    assert mrrs["q2"] == pytest.approx(0.0)


# =============================================================================
# E2E: LIGHT_RAG pipeline (DTm-78 / I.4)
# =============================================================================

@patch("sandbox_mteb.evaluator.MinIOLoader")
@patch("sandbox_mteb.evaluator.load_embedding_model")
@patch("sandbox_mteb.evaluator.AsyncLLMService")
@patch("sandbox_mteb.evaluator.batch_embed_queries")
@patch("sandbox_mteb.evaluator.get_retriever")
def test_pipeline_e2e_lightrag(
    mock_get_retriever, mock_batch_embed, mock_llm_cls, mock_embed_fn,
    mock_loader_cls, tmp_path
):
    """Pipeline E2E con LIGHT_RAG: verifica que KG metadata fluye y metricas se calculan."""
    from shared.retrieval.core import RetrievalResult

    # --- Mock MinIO Loader ---
    dataset = _make_mini_dataset()
    mock_loader = MagicMock()
    mock_loader.check_connection.return_value = True
    mock_loader.load_dataset.return_value = dataset
    mock_loader_cls.return_value = mock_loader

    # --- Mock Embedding Model ---
    mock_embed = MagicMock()
    mock_embed_fn.return_value = mock_embed

    # --- Mock LLM Service ---
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value="topic 0")
    mock_llm._max_concurrent = 2
    mock_llm_cls.return_value = mock_llm

    # --- Mock batch embed queries ---
    mock_batch_embed.return_value = [[0.5] * 64] * 3

    # --- Mock Retriever (simula LIGHT_RAG con metadata de grafo) ---
    mock_retriever = MagicMock()
    mock_retriever.index_documents.return_value = True
    mock_retriever.retrieve_by_vector.return_value = RetrievalResult(
        doc_ids=["doc0", "doc1", "doc3"],
        contents=["content 0", "content 1", "content 3"],
        scores=[0.9, 0.8, 0.7],
        vector_scores=[0.9, 0.8, 0.7],
        strategy_used=RetrievalStrategy.LIGHT_RAG,
        metadata={"graph_active": True},
    )
    mock_retriever.retrieve.return_value = mock_retriever.retrieve_by_vector.return_value
    mock_retriever.clear_index.return_value = None
    # Simular que NO es LightRAGRetriever real (evitar pre_extract_query_keywords)
    mock_retriever.__class__.__name__ = "MagicMock"
    mock_get_retriever.return_value = mock_retriever

    # --- Config LIGHT_RAG ---
    config = MTEBConfig(
        infra=InfraConfig(
            embedding_base_url="http://fake:8000/v1",
            embedding_model_name="test-embed",
            llm_base_url="http://fake:8000/v1",
            llm_model_name="test-llm",
            nim_max_concurrent=2,
            nim_timeout=30,
        ),
        storage=MinIOStorageConfig(
            minio_endpoint="http://fake:9000",
            minio_access_key="test",
            minio_secret_key="test",
            minio_bucket="test",
            evaluation_results_dir=tmp_path / "results",
        ),
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.LIGHT_RAG,
            retrieval_k=5,
        ),
        reranker=RerankerConfig(enabled=False),
        dataset_name="hotpotqa",
        generation_enabled=True,
        max_queries=3,
        max_corpus=6,
    )
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    ev = MTEBEvaluator(config)

    result = ev.run()

    # --- Assertions ---
    assert isinstance(result, EvaluationRun)
    assert result.status == EvaluationStatus.COMPLETED
    assert result.num_queries_evaluated == 3
    assert result.retrieval_strategy == "LIGHT_RAG"

    # Same retriever mock as SIMPLE_VECTOR -> same expected metrics
    assert result.avg_mrr == pytest.approx(0.5, abs=0.01)
    assert result.avg_hit_rate_at_5 == pytest.approx(2.0 / 3.0, abs=0.01)

    # Verify query results
    assert len(result.query_results) == 3
    for qr in result.query_results:
        assert qr.retrieval is not None
        assert len(qr.retrieval.retrieved_doc_ids) > 0
