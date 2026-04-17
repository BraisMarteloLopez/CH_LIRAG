"""
Tests unitarios para LightRAGRetriever._enrich_with_graph y funciones auxiliares.

Cobertura:
  E1. Sin keywords extraidos -> retorna vector_result con metadata KG vacia.
  E2. Modo hybrid: recopila entidades + relaciones en metadata.
  E3. Modo local: solo entidades, no relaciones.
  E4. Modo global: solo relaciones, no entidades.
  E5. Modo naive: no enriquece con KG.
  E6. Entity VDB: resolucion semantica + dedup.
  E7. Relationship VDB: resolucion a descripciones.
  E8. Corpus fingerprint: determinismo, orden, cambios.
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from shared.retrieval.core import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
)
from shared.retrieval.lightrag.knowledge_graph import KnowledgeGraph, KGEntity
from shared.retrieval.lightrag.retriever import LightRAGRetriever


# =============================================================================
# Helpers
# =============================================================================

def _make_lightrag(
    kg=None,
    extractor=None,
    vector_retriever=None,
    lightrag_mode="hybrid",
):
    """Crea LightRAGRetriever con dependencias mockeadas."""
    retriever = object.__new__(LightRAGRetriever)
    retriever.config = RetrievalConfig()
    retriever._kg_max_hops = 2
    mock_kg = kg or MagicMock(spec=KnowledgeGraph)
    if not kg:
        mock_kg.get_neighbors_ranked.return_value = []
    retriever._kg = mock_kg
    retriever._extractor = extractor or MagicMock()
    retriever._has_graph = True
    retriever._lightrag_mode = lightrag_mode
    retriever._max_neighbors_per_entity = 5
    import threading
    from collections import OrderedDict
    retriever._query_keywords_cache = OrderedDict()
    retriever._cache_lock = threading.Lock()
    retriever._QUERY_CACHE_MAX_SIZE = 10_000
    retriever._vector_retriever = vector_retriever or MagicMock()
    retriever._entities_vdb = None
    retriever._relationships_vdb = None
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


def _make_entity(name, entity_type="PERSON", description=""):
    entity = object.__new__(KGEntity)
    entity.name = name
    entity.entity_type = entity_type
    entity.description = description
    entity.source_doc_ids = set()
    entity._descriptions = []
    return entity


# =============================================================================
# E1-E4: _enrich_with_graph — core behavior
# =============================================================================

def test_enrich_no_keywords():
    """Sin keywords extraidos, retorna vector_result con metadata KG vacia."""
    r = _make_lightrag()
    r._extractor.extract_query_keywords.return_value = ([], [])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._enrich_with_graph("query", vr, top_k=5)

    assert result.doc_ids == ["d1"]
    assert result.metadata["kg_entities"] == []
    assert result.metadata["kg_relations"] == []
    assert result.metadata["query_keywords"] == {"low": [], "high": []}


def test_enrich_hybrid_collects_entities_and_relations():
    """Modo hybrid: recopila entidades + relaciones en metadata."""
    r = _make_lightrag(lightrag_mode="hybrid")

    # Setup entity VDB
    mock_entity_vdb = MagicMock()
    mock_entity_doc = MagicMock()
    mock_entity_doc.metadata = {"entity_name": "alice"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_entity_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    # Setup KG entities
    alice = _make_entity("alice", "PERSON", "A researcher")
    r._kg.get_all_entities.return_value = {"alice": alice}

    # Setup relationship VDB
    mock_rel_vdb = MagicMock()
    mock_rel_doc = MagicMock()
    mock_rel_doc.metadata = {
        "source_entity": "alice", "target_entity": "bob",
        "relation": "works_with",
    }
    mock_rel_doc.page_content = "alice -> works_with -> bob: collaboration"
    mock_rel_vdb.similarity_search_with_score.return_value = [(mock_rel_doc, 0.2)]
    r._relationships_vdb = mock_rel_vdb

    r._extractor.extract_query_keywords.return_value = (["alice"], ["collaboration"])

    vr = _make_vector_result(["d1", "d2"], ["c1", "c2"], [0.9, 0.8])
    result = r._enrich_with_graph("query about alice", vr, top_k=5)

    # Vector ranking unchanged
    assert result.doc_ids == ["d1", "d2"]
    assert result.scores == [0.9, 0.8]

    # KG data in metadata
    assert len(result.metadata["kg_entities"]) == 1
    assert result.metadata["kg_entities"][0]["entity"] == "alice"
    assert result.metadata["kg_entities"][0]["description"] == "A researcher"

    assert len(result.metadata["kg_relations"]) == 1
    assert result.metadata["kg_relations"][0]["source"] == "alice"
    assert result.metadata["kg_relations"][0]["target"] == "bob"


def test_enrich_local_only_entities():
    """Modo local: solo recopila entidades, no relaciones."""
    r = _make_lightrag(lightrag_mode="local")

    mock_entity_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "entity_a"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    entity_a = _make_entity("entity_a", "ORG", "An organization")
    r._kg.get_all_entities.return_value = {"entity_a": entity_a}

    mock_rel_vdb = MagicMock()
    r._relationships_vdb = mock_rel_vdb

    r._extractor.extract_query_keywords.return_value = (["entity_a"], ["theme"])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._enrich_with_graph("query", vr, top_k=5)

    assert len(result.metadata["kg_entities"]) == 1
    assert result.metadata["kg_relations"] == []
    # Relationship VDB should NOT be called in local mode
    mock_rel_vdb.similarity_search_with_score.assert_not_called()


def test_enrich_global_only_relations():
    """Modo global: solo recopila relaciones, no entidades."""
    r = _make_lightrag(lightrag_mode="global")

    mock_entity_vdb = MagicMock()
    r._entities_vdb = mock_entity_vdb

    mock_rel_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "source_entity": "x", "target_entity": "y", "relation": "rel",
    }
    mock_doc.page_content = "x -> rel -> y"
    mock_rel_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.2)]
    r._relationships_vdb = mock_rel_vdb

    r._extractor.extract_query_keywords.return_value = (["entity"], ["theme"])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._enrich_with_graph("query", vr, top_k=5)

    assert result.metadata["kg_entities"] == []
    assert len(result.metadata["kg_relations"]) == 1
    # Entity VDB should NOT be called in global mode
    mock_entity_vdb.similarity_search_with_score.assert_not_called()


def test_enrich_preserves_vector_ranking():
    """Enrichment never changes the vector ranking or scores."""
    r = _make_lightrag()

    mock_entity_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "e1"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    r._kg.get_all_entities.return_value = {
        "e1": _make_entity("e1", "THING", "desc"),
    }
    r._extractor.extract_query_keywords.return_value = (["e1"], [])

    original_ids = ["d1", "d2", "d3"]
    original_scores = [0.9, 0.7, 0.5]
    vr = _make_vector_result(original_ids, ["c1", "c2", "c3"], original_scores)
    result = r._enrich_with_graph("query", vr, top_k=5)

    assert result.doc_ids == original_ids
    assert result.scores == original_scores


# =============================================================================
# E5: Modo naive
# =============================================================================

def test_retrieve_mode_naive_skips_graph():
    """Modo naive: solo vector search, ignora KG."""
    r = _make_lightrag(lightrag_mode="naive")
    r._vector_retriever.retrieve.return_value = _make_vector_result(
        ["v1"], ["Content"], [0.9],
    )

    result = r.retrieve("query", top_k=5)

    assert result.metadata["lightrag_mode"] == "naive"
    assert result.metadata["graph_active"] is False


# =============================================================================
# E6: Entity VDB resolution
# =============================================================================

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


def test_resolve_entities_via_vdb_filters_by_distance():
    """_resolve_entities_via_vdb filtra por threshold de distancia."""
    mock_vdb = MagicMock()
    close_doc = MagicMock()
    close_doc.metadata = {"entity_name": "close_entity"}
    far_doc = MagicMock()
    far_doc.metadata = {"entity_name": "far_entity"}
    # One within threshold (0.1), one beyond (0.9)
    mock_vdb.similarity_search_with_score.return_value = [
        (close_doc, 0.1), (far_doc, 0.9),
    ]

    r = _make_lightrag()
    r._entities_vdb = mock_vdb

    result = r._resolve_entities_via_vdb(["keyword"], top_k=5)
    assert result == ["close_entity"]


# =============================================================================
# E7: Relationship VDB — _resolve_relations_for_context
# =============================================================================

def test_resolve_relations_for_context_returns_descriptions():
    """_resolve_relations_for_context returns relation descriptions."""
    mock_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "source_entity": "alice",
        "target_entity": "bob",
        "relation": "mentors",
    }
    mock_doc.page_content = "alice -> mentors -> bob: academic mentorship"
    mock_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.2)]

    r = _make_lightrag()
    r._relationships_vdb = mock_vdb

    result = r._resolve_relations_for_context(["mentorship"])

    assert len(result) == 1
    assert result[0]["source"] == "alice"
    assert result[0]["target"] == "bob"
    assert result[0]["relation"] == "mentors"
    assert "mentorship" in result[0]["description"]


def test_resolve_relations_for_context_deduplicates():
    """_resolve_relations_for_context deduplicates by (source, target, relation)."""
    mock_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "source_entity": "a", "target_entity": "b", "relation": "r",
    }
    mock_doc.page_content = "description"
    # Same relation returned for both keywords
    mock_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]

    r = _make_lightrag()
    r._relationships_vdb = mock_vdb

    result = r._resolve_relations_for_context(["kw1", "kw2"])
    assert len(result) == 1  # deduplicated


def test_resolve_relations_for_context_empty():
    """_resolve_relations_for_context con keywords vacios retorna []."""
    r = _make_lightrag()
    r._relationships_vdb = MagicMock()
    assert r._resolve_relations_for_context([]) == []


def test_resolve_relations_for_context_filters_by_distance():
    """_resolve_relations_for_context filters by distance threshold."""
    mock_vdb = MagicMock()
    close_doc = MagicMock()
    close_doc.metadata = {
        "source_entity": "a", "target_entity": "b", "relation": "r1",
    }
    close_doc.page_content = "close"
    far_doc = MagicMock()
    far_doc.metadata = {
        "source_entity": "c", "target_entity": "d", "relation": "r2",
    }
    far_doc.page_content = "far"
    mock_vdb.similarity_search_with_score.return_value = [
        (close_doc, 0.1), (far_doc, 0.9),
    ]

    r = _make_lightrag()
    r._relationships_vdb = mock_vdb

    result = r._resolve_relations_for_context(["keyword"])
    assert len(result) == 1
    assert result[0]["relation"] == "r1"


# =============================================================================
# E8: Corpus fingerprint
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
    """Fingerprint cambia si max_text_chars difiere."""
    docs = [{"doc_id": "d1", "content": "some content"}]
    fp_default = LightRAGRetriever._corpus_fingerprint(docs)
    fp_3000 = LightRAGRetriever._corpus_fingerprint(docs, max_text_chars=3000)
    fp_5000 = LightRAGRetriever._corpus_fingerprint(docs, max_text_chars=5000)
    assert fp_default != fp_3000
    assert fp_3000 != fp_5000


# =============================================================================
# Divergencia #9: 1-hop neighbor expansion en _enrich_with_graph
# =============================================================================


def test_enrich_entity_with_neighbors():
    """Entidades resueltas incluyen vecinos 1-hop cuando get_neighbors_ranked retorna datos."""
    r = _make_lightrag(lightrag_mode="hybrid")

    mock_entity_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "alice"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    alice = _make_entity("alice", "PERSON", "A researcher")
    r._kg.get_all_entities.return_value = {"alice": alice}
    r._kg.get_neighbors_ranked.return_value = [
        {"entity": "bob", "type": "PERSON", "description": "Engineer", "relation": "works_with", "score": 3.5},
        {"entity": "carol", "type": "PERSON", "description": "PI", "relation": "supervises", "score": 2.1},
    ]

    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._enrich_with_graph("query about alice", vr, top_k=5)

    entity = result.metadata["kg_entities"][0]
    assert entity["entity"] == "alice"
    assert "neighbors" in entity
    assert len(entity["neighbors"]) == 2
    assert entity["neighbors"][0]["entity"] == "bob"
    assert entity["neighbors"][1]["entity"] == "carol"


def test_enrich_entity_no_neighbors():
    """Sin vecinos, el dict de entidad no tiene clave 'neighbors'."""
    r = _make_lightrag(lightrag_mode="local")

    mock_entity_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "alice"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    alice = _make_entity("alice", "PERSON", "A researcher")
    r._kg.get_all_entities.return_value = {"alice": alice}
    r._kg.get_neighbors_ranked.return_value = []

    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._enrich_with_graph("query", vr, top_k=5)

    entity = result.metadata["kg_entities"][0]
    assert entity["entity"] == "alice"
    assert "neighbors" not in entity


def test_enrich_entity_neighbor_fallback():
    """Si get_neighbors_ranked lanza excepcion, la entidad aparece sin vecinos."""
    r = _make_lightrag(lightrag_mode="hybrid")

    mock_entity_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "alice"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    alice = _make_entity("alice", "PERSON", "A researcher")
    r._kg.get_all_entities.return_value = {"alice": alice}
    r._kg.get_neighbors_ranked.side_effect = RuntimeError("graph error")

    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    vr = _make_vector_result(["d1"], ["c1"], [0.9])
    result = r._enrich_with_graph("query", vr, top_k=5)

    entity = result.metadata["kg_entities"][0]
    assert entity["entity"] == "alice"
    assert "neighbors" not in entity


def test_enrich_preserves_vector_ranking_with_neighbors():
    """Neighbor expansion no altera el ranking vectorial."""
    r = _make_lightrag(lightrag_mode="hybrid")

    mock_entity_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "alice"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    alice = _make_entity("alice", "PERSON", "A researcher")
    r._kg.get_all_entities.return_value = {"alice": alice}
    r._kg.get_neighbors_ranked.return_value = [
        {"entity": "bob", "type": "PERSON", "description": "Engineer", "score": 3.5},
    ]

    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    vr = _make_vector_result(["d1", "d2", "d3"], ["c1", "c2", "c3"], [0.9, 0.7, 0.5])
    result = r._enrich_with_graph("query", vr, top_k=5)

    assert result.doc_ids == ["d1", "d2", "d3"]
    assert result.scores == [0.9, 0.7, 0.5]
