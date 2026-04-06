"""
Tests unitarios para LightRAGRetriever._fuse_with_graph (DTm-23, Fase 1.3).

Cobertura:
  F1. Sin graph docs -> retorna vector_result sin cambios.
  F2. Graph-only docs se recuperan via lookup.
  F3. Normalizacion de scores a [0,1].
  F4. All graph scores 0.0 -> no division by zero.
  F5. Fusion respeta vector_weight / graph_weight.
  F6. Docs del grafo sin contenido -> excluidos del resultado.
  F7. Sin keywords extraidos -> retorna vector_result sin cambios.
  F8. Solo high-level keywords (sin low-level).
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from shared.retrieval.core import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
)
from shared.retrieval.lightrag.knowledge_graph import KnowledgeGraph
from shared.retrieval.lightrag.retriever import LightRAGRetriever


# =============================================================================
# Helpers
# =============================================================================

def _make_lightrag(
    graph_weight=0.3,
    vector_weight=0.7,
    kg=None,
    extractor=None,
    vector_retriever=None,
    fusion_method="rrf",
    rrf_k=60,
):
    """Crea LightRAGRetriever con dependencias mockeadas."""
    retriever = object.__new__(LightRAGRetriever)
    retriever.config = RetrievalConfig()
    retriever._graph_weight = graph_weight
    retriever._vector_weight = vector_weight
    retriever._kg_max_hops = 2
    retriever._GRAPH_OVERFETCH_FACTOR = 2
    retriever._kg = kg or MagicMock(spec=KnowledgeGraph)
    retriever._extractor = extractor or MagicMock()
    retriever._has_graph = True
    import threading
    from collections import OrderedDict
    retriever._query_keywords_cache = OrderedDict()
    retriever._cache_lock = threading.Lock()
    retriever._QUERY_CACHE_MAX_SIZE = 10_000
    retriever._vector_retriever = vector_retriever or MagicMock()
    retriever._kg_fusion_method = fusion_method
    retriever._kg_rrf_k = rrf_k
    retriever._entities_vdb = None  # DAM-1: no VDB in unit tests by default
    return retriever


def _make_vector_result(doc_ids, contents, scores):
    return RetrievalResult(
        doc_ids=doc_ids,
        contents=contents,
        scores=scores,
        vector_scores=scores[:],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        metadata={},
    )


# =============================================================================
# F1: Sin graph docs
# =============================================================================

def test_fuse_no_graph_docs():
    """Sin graph docs -> retorna vector_result sin cambios."""
    r = _make_lightrag()
    # KG queries return empty
    r._kg.query_entities.return_value = []
    r._kg.query_by_keywords.return_value = []
    # Keywords extraction returns something
    r._extractor.extract_query_keywords.return_value = (["alice"], ["research"])

    vr = _make_vector_result(
        ["d1", "d2"], ["c1", "c2"], [0.9, 0.8]
    )

    result = r._fuse_with_graph("query", vr, top_k=5)
    assert result.doc_ids == ["d1", "d2"]
    assert result.metadata["graph_docs_added"] == 0


# =============================================================================
# F2: Graph-only docs recovered via lookup
# =============================================================================

def test_fuse_graph_only_docs_lookup():
    """Docs solo del grafo se recuperan via get_documents_by_ids."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    # Graph returns a doc not in vector results
    r._kg.query_entities.return_value = [("d1", 1.0), ("d_graph", 0.8)]
    r._kg.query_by_keywords.return_value = []

    # Vector result only has d1
    vr = _make_vector_result(["d1"], ["content_d1"], [0.9])

    # Lookup returns d_graph content
    r._vector_retriever.get_documents_by_ids.return_value = {
        "d_graph": "content_graph"
    }

    result = r._fuse_with_graph("query", vr, top_k=5)

    # Both docs should be in result
    assert "d1" in result.doc_ids
    assert "d_graph" in result.doc_ids
    assert result.metadata["graph_resolved"] == 1


# =============================================================================
# F3: Normalizacion de scores
# =============================================================================

def test_fuse_normalization():
    """Scores se normalizan a [0,1] (linear fusion)."""
    r = _make_lightrag(graph_weight=0.5, vector_weight=0.5, fusion_method="linear")
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    # Graph: d1=2.0, d2=1.0 -> normalized: d1=1.0, d2=0.5
    r._kg.query_entities.return_value = [("d1", 2.0), ("d2", 1.0)]
    r._kg.query_by_keywords.return_value = []

    # Vector: d1=0.6, d2=0.3 -> normalized: d1=1.0, d2=0.5
    vr = _make_vector_result(["d1", "d2"], ["c1", "c2"], [0.6, 0.3])

    result = r._fuse_with_graph("query", vr, top_k=5)

    # d1: 0.5*1.0 + 0.5*1.0 = 1.0
    # d2: 0.5*0.5 + 0.5*0.5 = 0.5
    assert result.scores[0] == pytest.approx(1.0)
    assert result.scores[1] == pytest.approx(0.5)
    assert result.doc_ids[0] == "d1"


# =============================================================================
# F4: All graph scores 0.0
# =============================================================================

def test_fuse_all_graph_scores_zero():
    """Graph scores todos 0.0 -> no division by zero, vector domina."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d1", 0.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1"], ["c1"], [0.9])

    # Should not raise
    result = r._fuse_with_graph("query", vr, top_k=5)
    assert len(result.doc_ids) >= 1


# =============================================================================
# F5: Weights applied correctly
# =============================================================================

def test_fuse_weights():
    """Fusion respeta vector_weight / graph_weight (linear)."""
    r = _make_lightrag(graph_weight=0.3, vector_weight=0.7, fusion_method="linear")
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    # Only d1 in both graph and vector
    r._kg.query_entities.return_value = [("d1", 1.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1"], ["c1"], [1.0])

    result = r._fuse_with_graph("query", vr, top_k=5)

    # d1: 0.7 * 1.0 + 0.3 * 1.0 = 1.0
    assert result.scores[0] == pytest.approx(1.0)


def test_fuse_weights_graph_only_doc():
    """Doc solo del grafo: v_score=0.0, fused = graph_weight * g_score (linear)."""
    r = _make_lightrag(graph_weight=0.3, vector_weight=0.7, fusion_method="linear")
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d_graph", 1.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result([], [], [])

    r._vector_retriever.get_documents_by_ids.return_value = {
        "d_graph": "content"
    }

    result = r._fuse_with_graph("query", vr, top_k=5)

    assert "d_graph" in result.doc_ids
    idx = result.doc_ids.index("d_graph")
    # fused = 0.7 * 0.0 + 0.3 * 1.0 = 0.3
    assert result.scores[idx] == pytest.approx(0.3)


# =============================================================================
# F6: Unresolved docs excluded
# =============================================================================

def test_fuse_unresolved_docs_excluded():
    """Docs del grafo sin contenido -> excluidos del resultado final."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d_ghost", 1.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result([], [], [])

    # Lookup fails to find d_ghost
    r._vector_retriever.get_documents_by_ids.return_value = {}

    result = r._fuse_with_graph("query", vr, top_k=5)

    assert "d_ghost" not in result.doc_ids
    assert result.metadata["graph_unresolved"] == 1


# =============================================================================
# F7: Sin keywords
# =============================================================================

def test_fuse_no_keywords():
    """Sin keywords extraidos -> retorna vector_result sin cambios."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = ([], [])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])

    result = r._fuse_with_graph("query", vr, top_k=5)
    assert result.doc_ids == ["d1"]
    assert result.metadata["query_keywords"] == {"low": [], "high": []}


# =============================================================================
# F8: Solo high-level keywords
# =============================================================================

def test_fuse_high_level_only():
    """Solo high-level keywords -> query_by_keywords se invoca."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = ([], ["machine learning"])

    r._kg.query_entities.return_value = []  # no se llama con low=[]
    r._kg.query_by_keywords.return_value = [("d1", 1.0)]

    vr = _make_vector_result(["d1"], ["c1"], [0.9])

    result = r._fuse_with_graph("query", vr, top_k=5)
    assert "d1" in result.doc_ids
    r._kg.query_by_keywords.assert_called_once()


# =============================================================================
# F9a-F9c: RRF fusion
# =============================================================================

def test_fuse_rrf_default():
    """RRF fusion es el default y produce ranking valido."""
    r = _make_lightrag(graph_weight=0.5, vector_weight=0.5)
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d1", 2.0), ("d2", 1.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1", "d2"], ["c1", "c2"], [0.9, 0.8])

    result = r._fuse_with_graph("query", vr, top_k=5)
    # d1 is rank 1 in both -> highest RRF score
    assert result.doc_ids[0] == "d1"
    assert len(result.doc_ids) == 2
    assert all(s > 0 for s in result.scores)


def test_fuse_rrf_doc_in_one_ranking_only():
    """RRF handles docs that appear in only one ranking."""
    r = _make_lightrag(graph_weight=0.3, vector_weight=0.7)
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    # d_graph only in graph, not in vector
    r._kg.query_entities.return_value = [("d_graph", 1.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d_vec"], ["content_vec"], [0.9])

    r._vector_retriever.get_documents_by_ids.return_value = {
        "d_graph": "content_graph"
    }

    result = r._fuse_with_graph("query", vr, top_k=5)
    assert "d_vec" in result.doc_ids
    assert "d_graph" in result.doc_ids


def test_fuse_rrf_respects_weights():
    """RRF with higher vector_weight ranks vector-top docs higher."""
    # High vector weight
    r_vec = _make_lightrag(graph_weight=0.1, vector_weight=0.9)
    r_vec._extractor.extract_query_keywords.return_value = (["x"], [])
    r_vec._kg.query_entities.return_value = [("d2", 2.0), ("d1", 1.0)]
    r_vec._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1", "d2"], ["c1", "c2"], [0.9, 0.1])
    result_vec = r_vec._fuse_with_graph("query", vr, top_k=5)

    # High graph weight
    r_graph = _make_lightrag(graph_weight=0.9, vector_weight=0.1)
    r_graph._extractor.extract_query_keywords.return_value = (["x"], [])
    r_graph._kg.query_entities.return_value = [("d2", 2.0), ("d1", 1.0)]
    r_graph._kg.query_by_keywords.return_value = []

    vr2 = _make_vector_result(["d1", "d2"], ["c1", "c2"], [0.9, 0.1])
    result_graph = r_graph._fuse_with_graph("query", vr2, top_k=5)

    # With high vector weight: d1 (vector rank 1) should be top
    assert result_vec.doc_ids[0] == "d1"
    # With high graph weight: d2 (graph rank 1) should be top
    assert result_graph.doc_ids[0] == "d2"


# =============================================================================
# F9-F10: Corpus fingerprint y cache path (DTm-34)
# =============================================================================

def test_corpus_fingerprint_deterministic():
    """Mismo corpus produce mismo fingerprint."""
    docs = [
        {"doc_id": "d1", "content": "hello world"},
        {"doc_id": "d2", "content": "foo bar"},
    ]
    fp1 = LightRAGRetriever._corpus_fingerprint(docs)
    fp2 = LightRAGRetriever._corpus_fingerprint(docs)
    assert fp1 == fp2
    assert len(fp1) == 16


def test_corpus_fingerprint_order_independent():
    """Fingerprint no depende del orden de documentos."""
    docs_a = [
        {"doc_id": "d1", "content": "aaa"},
        {"doc_id": "d2", "content": "bbb"},
    ]
    docs_b = [
        {"doc_id": "d2", "content": "bbb"},
        {"doc_id": "d1", "content": "aaa"},
    ]
    assert LightRAGRetriever._corpus_fingerprint(docs_a) == \
           LightRAGRetriever._corpus_fingerprint(docs_b)


def test_corpus_fingerprint_changes_with_content():
    """Fingerprint cambia si contenido del corpus cambia."""
    docs_v1 = [{"doc_id": "d1", "content": "version 1"}]
    docs_v2 = [{"doc_id": "d1", "content": "version 2"}]
    assert LightRAGRetriever._corpus_fingerprint(docs_v1) != \
           LightRAGRetriever._corpus_fingerprint(docs_v2)


def test_corpus_fingerprint_changes_with_max_text_chars():
    """Fingerprint cambia si max_text_chars difiere (DTm-34 caveat)."""
    docs = [{"doc_id": "d1", "content": "some content"}]
    fp_default = LightRAGRetriever._corpus_fingerprint(docs)
    fp_3000 = LightRAGRetriever._corpus_fingerprint(docs, max_text_chars=3000)
    fp_5000 = LightRAGRetriever._corpus_fingerprint(docs, max_text_chars=5000)
    # Con max_text_chars=0 (default) vs con valor explícito deben diferir
    assert fp_default != fp_3000
    # Distintos valores deben producir fingerprints distintos
    assert fp_3000 != fp_5000


# =============================================================================
# Entity VDB (DAM-1)
# =============================================================================

def test_fuse_with_entity_vdb_resolves_via_similarity():
    """Con entity VDB activo, low-level usa _resolve_entities_via_vdb."""
    mock_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "barack obama", "entity_type": "PERSON"}
    mock_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]

    r = _make_lightrag()
    r._entities_vdb = mock_vdb

    # KG returns docs for the resolved entity
    r._kg.query_entities.return_value = [("doc_1", 1.0)]
    r._kg.query_by_keywords.return_value = []
    r._extractor.extract_query_keywords.return_value = (["Obama"], ["politics"])
    r._vector_retriever.get_documents_by_ids.return_value = {"doc_1": "content1"}

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._fuse_with_graph("Who is Obama?", vr, top_k=5)

    # VDB was called with the low-level keyword
    mock_vdb.similarity_search_with_score.assert_called_once_with("Obama", k=10)
    # query_entities was called with pre_resolved
    r._kg.query_entities.assert_called_once()
    call_kwargs = r._kg.query_entities.call_args
    assert call_kwargs[1].get("pre_resolved") == ["barack obama"]


def test_fuse_without_entity_vdb_falls_back_to_string_matching():
    """Sin entity VDB, low-level usa _resolve_entity_names (string matching)."""
    r = _make_lightrag()
    assert r._entities_vdb is None

    r._kg.query_entities.return_value = [("doc_1", 0.8)]
    r._kg.query_by_keywords.return_value = []
    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    r._fuse_with_graph("alice", vr, top_k=5)

    # query_entities called without pre_resolved (None)
    call_kwargs = r._kg.query_entities.call_args
    assert call_kwargs[1].get("pre_resolved") is None


def test_resolve_entities_via_vdb_deduplicates():
    """_resolve_entities_via_vdb no retorna entidades duplicadas."""
    mock_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "entity_a"}
    # Same entity returned for both keywords
    mock_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]

    r = _make_lightrag()
    r._entities_vdb = mock_vdb

    result = r._resolve_entities_via_vdb(["keyword1", "keyword2"], top_k=5)
    assert result == ["entity_a"]  # deduplicated


def test_resolve_entities_via_vdb_empty_keywords():
    """_resolve_entities_via_vdb con keywords vacios retorna []."""
    r = _make_lightrag()
    r._entities_vdb = MagicMock()
    assert r._resolve_entities_via_vdb([]) == []
    assert r._resolve_entities_via_vdb(["", "  "]) == []
