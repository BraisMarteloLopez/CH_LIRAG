"""
Tests DTm-38: Strategy validation guardrail.

Cobertura:
  S1. strategy_used=SIMPLE_VECTOR cuando LightRAGRetriever no tiene grafo activo.
  S2. strategy_used=LIGHT_RAG cuando LightRAGRetriever tiene grafo activo.
  S3. metadata["graph_active"] refleja estado del grafo en ambos casos.
  S4. RetrievalExecutor detecta strategy mismatch (config=LIGHT_RAG, actual=SIMPLE_VECTOR).
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
from shared.retrieval.lightrag.knowledge_graph import KnowledgeGraph
from shared.retrieval.lightrag.retriever import LightRAGRetriever
from shared.config_base import InfraConfig, RerankerConfig
from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
from sandbox_mteb.retrieval_executor import RetrievalExecutor
from sandbox_mteb.result_builder import build_run


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
    import threading
    from collections import OrderedDict
    retriever._query_keywords_cache = OrderedDict()
    retriever._cache_lock = threading.Lock()
    retriever._QUERY_CACHE_MAX_SIZE = 10_000
    retriever._kg_fusion_method = "rrf"
    retriever._kg_rrf_k = 60
    retriever._entities_vdb = None  # DAM-1: no VDB in unit tests by default
    retriever._relationships_vdb = None  # DAM-2: no VDB in unit tests by default
    retriever._lightrag_mode = "hybrid"  # F.4/DTm-79
    # DTm-62: Conditional fusion
    retriever._fusion_overlap_threshold = 0.3
    retriever._fusion_graph_only_cap = 0.2
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


def _make_executor(strategy=RetrievalStrategy.LIGHT_RAG, retrieval_k=20, reranker_enabled=False, reranker_top_n=5, reranker_fetch_k=0):
    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(
            strategy=strategy,
            retrieval_k=retrieval_k,
        ),
        reranker=RerankerConfig(enabled=reranker_enabled, top_n=reranker_top_n, fetch_k=reranker_fetch_k),
    )
    return config, RetrievalExecutor(
        retriever=None,
        reranker=None,
        config=config,
    )


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
# S4: Executor detecta strategy mismatch
# =============================================================================

def test_executor_detects_strategy_mismatch():
    """Executor incrementa strategy_mismatches cuando config != actual."""
    config, executor = _make_executor(strategy=RetrievalStrategy.LIGHT_RAG)
    executor._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    detail, _ = executor.execute("test query", ["doc_0"])

    assert executor.strategy_mismatches == 1
    assert len(detail.retrieved_doc_ids) == 20


# =============================================================================
# S5: config_snapshot incluye strategy_actual y strategy_mismatches
# =============================================================================

def test_config_snapshot_strategy_fields():
    """config_snapshot registra strategy_actual y strategy_mismatches."""
    from shared.types import LoadedDataset, NormalizedDocument

    config, executor = _make_executor(strategy=RetrievalStrategy.LIGHT_RAG)
    executor._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    # Trigger mismatch
    executor.execute("q1", ["doc_0"])

    dataset = LoadedDataset(
        name="test",
        corpus={"doc_0": NormalizedDocument("doc_0", "content")},
    )
    run = build_run(
        config=config,
        run_id="test_run",
        dataset=dataset,
        query_results=[],
        elapsed_seconds=1.0,
        indexed_corpus_size=1,
        max_context_chars=4000,
        rerank_failures=executor.rerank_failures,
        strategy_mismatches=executor.strategy_mismatches,
    )

    assert run.config_snapshot["_runtime"]["strategy_mismatches"] == 1
    assert run.config_snapshot["_runtime"]["strategy_actual"] == "FALLBACK_SIMPLE_VECTOR"


def test_config_snapshot_no_mismatch():
    """Sin mismatch, strategy_actual coincide con configurada."""
    from shared.types import LoadedDataset, NormalizedDocument

    config, executor = _make_executor(strategy=RetrievalStrategy.SIMPLE_VECTOR)
    executor._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    executor.execute("q1", ["doc_0"])

    dataset = LoadedDataset(
        name="test",
        corpus={"doc_0": NormalizedDocument("doc_0", "content")},
    )
    run = build_run(
        config=config,
        run_id="test_run",
        dataset=dataset,
        query_results=[],
        elapsed_seconds=1.0,
        indexed_corpus_size=1,
        max_context_chars=4000,
        rerank_failures=executor.rerank_failures,
        strategy_mismatches=executor.strategy_mismatches,
    )

    assert run.config_snapshot["_runtime"]["strategy_mismatches"] == 0
    assert run.config_snapshot["_runtime"]["strategy_actual"] == "SIMPLE_VECTOR"


# =============================================================================
# S6: fetch_k >= retrieval_k (DTm-35)
# =============================================================================

def test_fetch_k_at_least_retrieval_k():
    """Con reranker top_n=5, fetch_k debe ser >= retrieval_k=20, no 15."""
    config, executor = _make_executor(
        strategy=RetrievalStrategy.SIMPLE_VECTOR,
        retrieval_k=20,
        reranker_enabled=True,
        reranker_top_n=5,
        reranker_fetch_k=0,
    )

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
    executor._reranker = mock_reranker
    executor._retriever = _MockRetrieverWithStrategy(RetrievalStrategy.SIMPLE_VECTOR)

    detail, reranked_ok = executor.execute("test query", ["doc_0"])

    # Pre-rerank metrics should have 20 docs, not 15
    assert len(detail.retrieved_doc_ids) == 20
    # Generation should have top_n=5
    assert len(detail.generation_doc_ids) == 5
    assert reranked_ok is True
