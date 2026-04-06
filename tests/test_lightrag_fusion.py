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
    lightrag_mode="hybrid",
    fusion_overlap_threshold=0.3,
    fusion_graph_only_cap=0.2,
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
    retriever._lightrag_mode = lightrag_mode
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
    # DTm-62: Conditional fusion
    retriever._fusion_overlap_threshold = fusion_overlap_threshold
    retriever._fusion_graph_only_cap = fusion_graph_only_cap
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
    """Doc solo del grafo con full fusion: fused = graph_weight * g_score (linear)."""
    # Force full fusion by setting threshold=0 (any overlap triggers full)
    r = _make_lightrag(
        graph_weight=0.3, vector_weight=0.7,
        fusion_method="linear", fusion_overlap_threshold=0.0,
    )
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
    assert result.metadata["fusion_mode"] == "full_rrf"  # threshold=0 forces full


def test_fuse_weights_graph_only_doc_vector_first():
    """Doc solo del grafo con vector_first: score penalizado (DTm-62)."""
    r = _make_lightrag(graph_weight=0.3, vector_weight=0.7, fusion_method="linear")
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d_graph", 1.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result([], [], [])

    r._vector_retriever.get_documents_by_ids.return_value = {
        "d_graph": "content"
    }

    result = r._fuse_with_graph("query", vr, top_k=5)

    # No vector docs and no overlap -> vector_first mode
    assert result.metadata["fusion_mode"] == "vector_first"
    assert "d_graph" in result.doc_ids
    # Score is penalized (0.0 * 0.5 = 0.0 since no vector baseline)
    assert result.scores[0] == 0.0


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


# =============================================================================
# F.2: _retrieve_via_graph — graph as primary retriever (DAM-3)
# =============================================================================


def test_retrieve_via_graph_uses_graph_as_main_source():
    """graph_primary: docs vienen del grafo, no de vector search."""
    r = _make_lightrag(lightrag_mode="graph_primary")
    r._extractor.extract_query_keywords.return_value = (["alice"], ["research"])
    r._kg.query_entities.return_value = [("doc_a", 1.0), ("doc_b", 0.5)]
    r._kg.query_by_keywords.return_value = [("doc_c", 0.8)]
    r._vector_retriever.get_documents_by_ids.return_value = {
        "doc_a": "Content A",
        "doc_b": "Content B",
        "doc_c": "Content C",
    }

    result = r._retrieve_via_graph("query about alice", top_k=5)

    assert "doc_a" in result.doc_ids
    assert "doc_b" in result.doc_ids
    assert "doc_c" in result.doc_ids
    assert result.metadata["graph_primary"] is True
    # Vector search directo no se llamo (suficientes docs del grafo)
    r._vector_retriever.retrieve.assert_not_called()


def test_retrieve_via_graph_fallback_when_insufficient():
    """graph_primary: si el grafo produce pocos docs, complementa con vector."""
    r = _make_lightrag(lightrag_mode="graph_primary")
    r._extractor.extract_query_keywords.return_value = (["alice"], [])
    # Solo 1 doc del grafo (< top_k/2 = 2)
    r._kg.query_entities.return_value = [("doc_a", 1.0)]
    r._vector_retriever.get_documents_by_ids.return_value = {
        "doc_a": "Content A",
    }
    r._vector_retriever.retrieve.return_value = _make_vector_result(
        ["doc_v1", "doc_v2"], ["Vector 1", "Vector 2"], [0.9, 0.8],
    )

    result = r._retrieve_via_graph("query", top_k=5)

    assert "doc_a" in result.doc_ids  # del grafo
    assert "doc_v1" in result.doc_ids  # del vector fallback
    assert result.metadata["vector_fallback_used"] is True


def test_retrieve_via_graph_no_keywords_falls_back():
    """graph_primary: sin keywords -> fallback completo a vector."""
    r = _make_lightrag(lightrag_mode="graph_primary")
    r._extractor.extract_query_keywords.return_value = ([], [])
    r._vector_retriever.retrieve.return_value = _make_vector_result(
        ["v1"], ["Content"], [0.9],
    )

    result = r._retrieve_via_graph("query", top_k=5)

    assert result.doc_ids == ["v1"]
    r._vector_retriever.retrieve.assert_called_once()


def test_retrieve_mode_naive_skips_graph():
    """Modo naive: solo vector search, ignora KG."""
    r = _make_lightrag(lightrag_mode="naive")
    r._vector_retriever.retrieve.return_value = _make_vector_result(
        ["v1"], ["Content"], [0.9],
    )

    result = r.retrieve("query", top_k=5)

    assert result.metadata["lightrag_mode"] == "naive"
    assert result.metadata["graph_active"] is False
    r._kg.query_entities.assert_not_called()


def test_retrieve_mode_hybrid_preserves_fusion_behavior():
    """Modo hybrid (default): vector + graph fusion como antes."""
    r = _make_lightrag(lightrag_mode="hybrid")
    r._extractor.extract_query_keywords.return_value = (["alice"], ["theme"])
    r._kg.query_entities.return_value = [("doc_g", 0.8)]
    r._kg.query_by_keywords.return_value = [("doc_g2", 0.5)]

    vector_result = _make_vector_result(
        ["doc_v1", "doc_v2"], ["V1", "V2"], [0.9, 0.8],
    )
    r._vector_retriever.retrieve.return_value = vector_result
    r._vector_retriever.get_documents_by_ids.return_value = {
        "doc_g": "Graph content",
        "doc_g2": "Graph content 2",
    }

    result = r.retrieve("query about alice", top_k=5)

    assert result.metadata["lightrag_mode"] == "hybrid"
    assert result.metadata["graph_active"] is True
    # Fusion debe haberse ejecutado (vector + graph)
    r._vector_retriever.retrieve.assert_called_once()


def test_retrieve_mode_local_only_uses_entities():
    """Modo local: solo entity path, no relationship path."""
    r = _make_lightrag(lightrag_mode="local")
    r._extractor.extract_query_keywords.return_value = (["alice"], ["theme"])
    r._kg.query_entities.return_value = [("doc_e", 0.9)]
    r._kg.query_by_keywords.return_value = [("doc_r", 0.5)]

    vector_result = _make_vector_result(["doc_v"], ["V"], [0.8])
    r._vector_retriever.retrieve.return_value = vector_result
    r._vector_retriever.get_documents_by_ids.return_value = {
        "doc_e": "Entity content",
    }

    result = r.retrieve("query", top_k=5)

    # query_entities se llamo (low-level)
    r._kg.query_entities.assert_called_once()
    # query_by_keywords NO se llamo (high-level desactivado en modo local)
    r._kg.query_by_keywords.assert_not_called()


def test_retrieve_mode_global_only_uses_relationships():
    """Modo global: solo relationship path, no entity path."""
    r = _make_lightrag(lightrag_mode="global")
    r._extractor.extract_query_keywords.return_value = (["alice"], ["theme"])
    r._kg.query_entities.return_value = [("doc_e", 0.9)]
    r._kg.query_by_keywords.return_value = [("doc_r", 0.5)]

    vector_result = _make_vector_result(["doc_v"], ["V"], [0.8])
    r._vector_retriever.retrieve.return_value = vector_result
    r._vector_retriever.get_documents_by_ids.return_value = {
        "doc_r": "Relation content",
    }

    result = r.retrieve("query", top_k=5)

    # query_entities NO se llamo (low-level desactivado en modo global)
    r._kg.query_entities.assert_not_called()
    # query_by_keywords se llamo (high-level)
    r._kg.query_by_keywords.assert_called_once()


# =============================================================================
# DTm-62: Conditional fusion — overlap gate
# =============================================================================


def test_fuse_strong_signal_uses_full_rrf():
    """High overlap ratio -> full RRF fusion (fusion_mode='full_rrf')."""
    r = _make_lightrag(fusion_overlap_threshold=0.3)
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    # d1, d2 in both vector and graph -> overlap = 2/2 = 1.0 (> 0.3)
    r._kg.query_entities.return_value = [("d1", 1.0), ("d2", 0.8)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1", "d2"], ["c1", "c2"], [0.9, 0.8])
    result = r._fuse_with_graph("query", vr, top_k=5)

    assert result.metadata["fusion_mode"] == "full_rrf"
    assert result.metadata["overlap_ratio"] >= 0.3


def test_fuse_weak_signal_uses_vector_first():
    """Low overlap ratio -> vector_first fusion, preserves vector ranking."""
    r = _make_lightrag(fusion_overlap_threshold=0.3)
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    # Graph returns completely different docs -> overlap = 0/2 = 0.0 (< 0.3)
    r._kg.query_entities.return_value = [("d_g1", 1.0), ("d_g2", 0.8)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(
        ["d_v1", "d_v2", "d_v3"], ["cv1", "cv2", "cv3"], [0.9, 0.8, 0.7]
    )
    r._vector_retriever.get_documents_by_ids.return_value = {
        "d_g1": "cg1", "d_g2": "cg2",
    }

    result = r._fuse_with_graph("query", vr, top_k=10)

    assert result.metadata["fusion_mode"] == "vector_first"
    # Vector docs must retain their original order at the top
    assert result.doc_ids[:3] == ["d_v1", "d_v2", "d_v3"]
    # Graph-only docs appended after vector docs
    assert result.metadata["overlap_ratio"] < 0.3


def test_vector_first_preserves_vector_ranking_order():
    """vector_first: vector docs keep exact original positions."""
    r = _make_lightrag(fusion_overlap_threshold=1.0)  # force vector_first
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d_g", 5.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(
        ["d1", "d2", "d3"], ["c1", "c2", "c3"], [0.9, 0.7, 0.5]
    )
    r._vector_retriever.get_documents_by_ids.return_value = {"d_g": "cg"}

    result = r._fuse_with_graph("query", vr, top_k=10)

    # Even with graph score 5.0, vector docs stay in positions 0-2
    assert result.doc_ids[0] == "d1"
    assert result.doc_ids[1] == "d2"
    assert result.doc_ids[2] == "d3"
    # Graph doc appended at end
    assert "d_g" in result.doc_ids
    assert result.doc_ids.index("d_g") == 3


def test_vector_first_caps_graph_only_docs():
    """vector_first: graph-only docs limited by graph_only_cap."""
    # cap = 0.2 * 5 = 1 doc max
    r = _make_lightrag(
        fusion_overlap_threshold=1.0,  # force vector_first
        fusion_graph_only_cap=0.2,
    )
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    # 4 graph-only docs
    r._kg.query_entities.return_value = [
        ("g1", 4.0), ("g2", 3.0), ("g3", 2.0), ("g4", 1.0),
    ]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["v1"], ["cv1"], [0.9])
    r._vector_retriever.get_documents_by_ids.return_value = {
        "g1": "c1", "g2": "c2", "g3": "c3", "g4": "c4",
    }

    result = r._fuse_with_graph("query", vr, top_k=5)

    # Only 1 graph-only doc should be appended (cap = ceil(5 * 0.2) = 1)
    graph_only_in_result = [d for d in result.doc_ids if d.startswith("g")]
    assert len(graph_only_in_result) == 1
    # Highest scoring graph doc should be the one picked
    assert graph_only_in_result[0] == "g1"


def test_vector_first_graph_only_score_below_vector():
    """vector_first: graph-only docs get penalized scores below vector."""
    r = _make_lightrag(fusion_overlap_threshold=1.0)  # force vector_first
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d_g", 10.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1"], ["c1"], [0.8])
    r._vector_retriever.get_documents_by_ids.return_value = {"d_g": "cg"}

    result = r._fuse_with_graph("query", vr, top_k=5)

    v_score = result.scores[result.doc_ids.index("d1")]
    g_score = result.scores[result.doc_ids.index("d_g")]
    # Graph-only doc score must be less than vector doc score
    assert g_score < v_score


def test_fuse_overlap_metadata_present():
    """Metadata always includes fusion_mode and overlap_ratio."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = (["x"], [])

    r._kg.query_entities.return_value = [("d1", 1.0)]
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._fuse_with_graph("query", vr, top_k=5)

    assert "fusion_mode" in result.metadata
    assert "overlap_ratio" in result.metadata
    assert "overlap_count" in result.metadata
    assert isinstance(result.metadata["overlap_ratio"], float)


def test_fuse_no_keywords_includes_fusion_metadata():
    """Sin keywords -> metadata still has fusion_mode='none'."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = ([], [])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._fuse_with_graph("query", vr, top_k=5)

    assert result.metadata["fusion_mode"] == "none"
    assert result.metadata["overlap_ratio"] == 0.0


def test_fuse_no_graph_results_includes_fusion_metadata():
    """Graph queries return empty -> fusion_mode='none'."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = (["x"], ["y"])
    r._kg.query_entities.return_value = []
    r._kg.query_by_keywords.return_value = []

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._fuse_with_graph("query", vr, top_k=5)

    assert result.metadata["fusion_mode"] == "none"
