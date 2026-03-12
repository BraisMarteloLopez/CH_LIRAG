"""
Tests DTm-38: Strategy validation guardrail.

Cobertura:
  S1. strategy_used=SIMPLE_VECTOR cuando LightRAGRetriever no tiene grafo activo.
  S2. strategy_used=LIGHT_RAG cuando LightRAGRetriever tiene grafo activo.
  S3. metadata["graph_active"] refleja estado del grafo en ambos casos.
  S4. Evaluator detecta strategy mismatch (config=LIGHT_RAG, actual=SIMPLE_VECTOR).
  S5. config_snapshot incluye strategy_actual y strategy_mismatches.
  S6. fetch_k >= retrieval_k cuando reranker activo (DTm-35).
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.retrieval.core import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
)
from shared.retrieval.knowledge_graph import KnowledgeGraph
from shared.retrieval.lightrag_retriever import LightRAGRetriever


# =============================================================================
# Helpers
# =============================================================================

def _make_lightrag(has_graph=True):
    """Crea LightRAGRetriever con dependencias mockeadas."""
    retriever = object.__new__(LightRAGRetriever)
    retriever.config = RetrievalConfig()
    retriever._graph_weight = 0.3
    retriever._vector_weight = 0.7
    retriever._kg_max_hops = 2
    retriever._GRAPH_OVERFETCH_FACTOR = 2
    retriever._kg = MagicMock(spec=KnowledgeGraph) if has_graph else None
    retriever._extractor = MagicMock() if has_graph else None
    retriever._has_graph = has_graph
    retriever._query_keywords_cache = {}
    retriever._vector_retriever = MagicMock()
    # Vector retriever returns 20 docs
    retriever._vector_retriever.retrieve.return_value = RetrievalResult(
        doc_ids=[f"d{i}" for i in range(20)],
        contents=[f"c{i}" for i in range(20)],
        scores=[1.0 - i * 0.01 for i in range(20)],
        vector_scores=[1.0 - i * 0.01 for i in range(20)],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        metadata={},
    )
    retriever._vector_retriever.retrieve_by_vector.return_value = (
        retriever._vector_retriever.retrieve.return_value
    )
    return retriever


# =============================================================================
# S1: strategy_used=SIMPLE_VECTOR sin grafo
# =============================================================================

def test_strategy_simple_vector_when_no_graph():
    """Sin grafo activo, strategy_used debe ser SIMPLE_VECTOR."""
    r = _make_lightrag(has_graph=False)
    result = r.retrieve("test query", top_k=5)
    assert result.strategy_used == RetrievalStrategy.SIMPLE_VECTOR


# =============================================================================
# S2: strategy_used=LIGHT_RAG con grafo
# =============================================================================

def test_strategy_light_rag_when_graph_active():
    """Con grafo activo, strategy_used debe ser LIGHT_RAG."""
    r = _make_lightrag(has_graph=True)
    # Mock keyword extraction y graph queries
    r._extractor.extract_query_keywords.return_value = (["alice"], ["research"])
    r._kg.query_entities.return_value = [("d0", 0.9)]
    r._kg.query_by_keywords.return_value = []

    result = r.retrieve("test query", top_k=5)
    assert result.strategy_used == RetrievalStrategy.LIGHT_RAG


# =============================================================================
# S3: metadata["graph_active"]
# =============================================================================

def test_graph_active_metadata_false():
    """metadata['graph_active']=False cuando no hay grafo."""
    r = _make_lightrag(has_graph=False)
    result = r.retrieve("test query", top_k=5)
    assert result.metadata["graph_active"] is False


def test_graph_active_metadata_true():
    """metadata['graph_active']=True cuando hay grafo."""
    r = _make_lightrag(has_graph=True)
    r._extractor.extract_query_keywords.return_value = (["x"], [])
    r._kg.query_entities.return_value = []
    r._kg.query_by_keywords.return_value = []

    result = r.retrieve("test query", top_k=5)
    assert result.metadata["graph_active"] is True


# =============================================================================
# S4: Evaluator detecta strategy mismatch
# =============================================================================

class _MockRetrieverWithStrategy:
    """Retriever que devuelve una estrategia especifica."""

    def __init__(self, strategy: RetrievalStrategy):
        self._strategy = strategy

    def retrieve(self, query_text, top_k=None):
        k = top_k or 20
        return RetrievalResult(
            doc_ids=[f"doc_{i}" for i in range(k)],
            contents=[f"content_{i}" for i in range(k)],
            scores=[1.0 - i * 0.01 for i in range(k)],
            strategy_used=self._strategy,
        )

    def retrieve_by_vector(self, query_text, query_vector, top_k=None):
        return self.retrieve(query_text, top_k)


def test_evaluator_detects_strategy_mismatch():
    """Evaluator incrementa _strategy_mismatches cuando config != actual."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.LIGHT_RAG,
            retrieval_k=20,
        ),
        reranker=RerankerConfig(enabled=False),
    )
    evaluator = MTEBEvaluator(config)
    # Retriever devuelve SIMPLE_VECTOR (simula fallback)
    evaluator._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    detail, _ = evaluator._execute_retrieval("test query", ["doc_0"])

    assert evaluator._strategy_mismatches == 1
    assert len(detail.retrieved_doc_ids) == 20


# =============================================================================
# S5: config_snapshot incluye strategy_actual y strategy_mismatches
# =============================================================================

def test_config_snapshot_strategy_fields():
    """config_snapshot registra strategy_actual y strategy_mismatches."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig
    from shared.types import LoadedDataset, NormalizedDocument

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.LIGHT_RAG,
            retrieval_k=20,
        ),
        reranker=RerankerConfig(enabled=False),
    )
    evaluator = MTEBEvaluator(config)
    evaluator._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    # Trigger mismatch
    evaluator._execute_retrieval("q1", ["doc_0"])

    dataset = LoadedDataset(
        name="test",
        corpus={"doc_0": NormalizedDocument("doc_0", "content")},
    )
    run = evaluator._build_run("test_run", dataset, [], 1.0, 1)

    assert run.config_snapshot["strategy_mismatches"] == 1
    assert run.config_snapshot["strategy_actual"] == "FALLBACK_SIMPLE_VECTOR"


def test_config_snapshot_no_mismatch():
    """Sin mismatch, strategy_actual coincide con configurada."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig
    from shared.types import LoadedDataset, NormalizedDocument

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.SIMPLE_VECTOR,
            retrieval_k=20,
        ),
        reranker=RerankerConfig(enabled=False),
    )
    evaluator = MTEBEvaluator(config)
    evaluator._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    evaluator._execute_retrieval("q1", ["doc_0"])

    dataset = LoadedDataset(
        name="test",
        corpus={"doc_0": NormalizedDocument("doc_0", "content")},
    )
    run = evaluator._build_run("test_run", dataset, [], 1.0, 1)

    assert run.config_snapshot["strategy_mismatches"] == 0
    assert run.config_snapshot["strategy_actual"] == "SIMPLE_VECTOR"


# =============================================================================
# S6: fetch_k >= retrieval_k (DTm-35)
# =============================================================================

def test_fetch_k_at_least_retrieval_k():
    """Con reranker top_n=5, fetch_k debe ser >= retrieval_k=20, no 15."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.SIMPLE_VECTOR,
            retrieval_k=20,
        ),
        reranker=RerankerConfig(enabled=True, top_n=5, fetch_k=0),
    )
    evaluator = MTEBEvaluator(config)

    # Mock reranker que retorna lo que recibe (passthrough)
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, retrieval_result, top_n: RetrievalResult(
        doc_ids=retrieval_result.doc_ids[:top_n],
        contents=retrieval_result.contents[:top_n],
        scores=retrieval_result.scores[:top_n],
        retrieval_time_ms=0.0,
        strategy_used=retrieval_result.strategy_used,
        metadata={"reranked": True},
    )
    evaluator._reranker = mock_reranker
    evaluator._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    detail, reranked_ok = evaluator._execute_retrieval("test query", ["doc_0"])

    # Pre-rerank metrics should have 20 docs, not 15
    assert len(detail.retrieved_doc_ids) == 20
    # Generation should have top_n=5
    assert len(detail.generation_doc_ids) == 5
    assert reranked_ok is True
