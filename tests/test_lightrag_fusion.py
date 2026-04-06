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
    retriever._relationships_vdb = None  # DAM-2: no VDB in unit tests by default
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


# =============================================================================
# Relationship VDB (DAM-2)
# =============================================================================

def test_fuse_with_relationship_vdb_resolves_via_similarity():
    """Con relationship VDB activo, high-level usa _resolve_relationships_via_vdb."""
    mock_rel_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"doc_id": "doc_1", "weight": 3}
    mock_rel_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.2)]

    r = _make_lightrag()
    r._relationships_vdb = mock_rel_vdb

    # No low-level keywords, only high-level
    r._kg.query_entities.return_value = []
    r._extractor.extract_query_keywords.return_value = ([], ["military attack"])
    r._vector_retriever.get_documents_by_ids.return_value = {"doc_1": "content1"}

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._fuse_with_graph("military attack", vr, top_k=5)

    # Relationship VDB was called
    mock_rel_vdb.similarity_search_with_score.assert_called_once_with(
        "military attack", k=20,
    )
    # Result should include doc_1 from relationship VDB
    assert "doc_1" in result.doc_ids or len(result.doc_ids) > 0


def test_fuse_without_relationship_vdb_falls_back():
    """Sin relationship VDB, high-level usa query_by_keywords."""
    r = _make_lightrag()
    assert r._relationships_vdb is None

    r._kg.query_entities.return_value = []
    r._kg.query_by_keywords.return_value = [("doc_1", 0.5)]
    r._extractor.extract_query_keywords.return_value = ([], ["theme"])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    r._fuse_with_graph("theme", vr, top_k=5)

    # Fallback: query_by_keywords was called
    r._kg.query_by_keywords.assert_called_once()


def test_resolve_relationships_via_vdb_weights_by_edge():
    """_resolve_relationships_via_vdb pondera por edge weight (DAM-5)."""
    mock_vdb = MagicMock()
    mock_doc1 = MagicMock()
    mock_doc1.metadata = {"doc_id": "doc_a", "weight": 5}
    mock_doc2 = MagicMock()
    mock_doc2.metadata = {"doc_id": "doc_b", "weight": 1}
    # Same distance but different weights
    mock_vdb.similarity_search_with_score.return_value = [
        (mock_doc1, 0.2), (mock_doc2, 0.2),
    ]

    r = _make_lightrag()
    r._relationships_vdb = mock_vdb

    results = r._resolve_relationships_via_vdb(["keyword"])
    result_dict = dict(results)

    # doc_a with weight=5 should score higher than doc_b with weight=1
    assert result_dict.get("doc_a", 0) > result_dict.get("doc_b", 0)


def test_resolve_relationships_via_vdb_empty():
    """_resolve_relationships_via_vdb con keywords vacios retorna []."""
    r = _make_lightrag()
    r._relationships_vdb = MagicMock()
    assert r._resolve_relationships_via_vdb([]) == []


# =============================================================================
# F.1: _select_chunks_from_graph (DTm-76)
# =============================================================================


def test_select_chunks_combines_entity_and_relationship_sources():
    """Docs de entity y relationship se combinan, scores se acumulan."""
    entity_results = [("doc_a", 1.0), ("doc_b", 0.5)]
    rel_results = [("doc_b", 0.3), ("doc_c", 0.8)]

    doc_ids, scores = LightRAGRetriever._select_chunks_from_graph(
        entity_results, rel_results, top_k=10,
    )

    assert set(doc_ids) == {"doc_a", "doc_b", "doc_c"}
    score_map = dict(zip(doc_ids, scores))
    # doc_b aparece en ambos canales: 0.5 + 0.3 = 0.8
    assert score_map["doc_b"] == pytest.approx(0.8)
    assert score_map["doc_a"] == pytest.approx(1.0)
    assert score_map["doc_c"] == pytest.approx(0.8)


def test_select_chunks_deduplicates_across_channels():
    """Si un doc aparece en ambos canales, su score es la suma."""
    entity_results = [("doc_x", 0.6)]
    rel_results = [("doc_x", 0.4)]

    doc_ids, scores = LightRAGRetriever._select_chunks_from_graph(
        entity_results, rel_results, top_k=10,
    )

    assert doc_ids == ["doc_x"]
    assert scores == [pytest.approx(1.0)]


def test_select_chunks_respects_top_k():
    """Solo retorna top_k docs por score."""
    entity_results = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]
    rel_results = [("d4", 0.5)]

    doc_ids, scores = LightRAGRetriever._select_chunks_from_graph(
        entity_results, rel_results, top_k=2,
    )

    assert len(doc_ids) == 2
    assert doc_ids[0] == "d1"  # highest score
    assert doc_ids[1] == "d2"


def test_select_chunks_empty_inputs():
    """Sin resultados de entity ni relationship retorna listas vacias."""
    doc_ids, scores = LightRAGRetriever._select_chunks_from_graph([], [], top_k=5)
    assert doc_ids == []
    assert scores == []


def test_select_chunks_ordered_by_score_descending():
    """Resultado ordenado por score descendente."""
    entity_results = [("low", 0.1)]
    rel_results = [("high", 0.9), ("mid", 0.5)]

    doc_ids, scores = LightRAGRetriever._select_chunks_from_graph(
        entity_results, rel_results, top_k=10,
    )

    assert doc_ids == ["high", "mid", "low"]
    assert scores[0] > scores[1] > scores[2]
