"""
Tests unitarios para LightRAGRetriever._retrieve_via_kg y funciones auxiliares.

Cobertura:
  E1. Sin keywords extraidos -> fallback a vector search.
  E2. Modo hybrid: chunks via KG + entidades + relaciones en metadata.
  E3. Modo local: chunks via entidades, no relaciones.
  E4. Modo global: chunks via relaciones, no entidades.
  E5. Modo naive: no usa KG.
  E6. Entity VDB: resolucion semantica + dedup.
  E7. Relationship VDB: resolucion a descripciones.
  E8. Corpus fingerprint: determinismo, orden, cambios.
  E9. Reference-count scoring: docs referenciados por mas entidades puntuan mas.
  E10. Fallback: KG no produce doc_ids -> vector search.
  E11. Fallback: doc_ids del KG no encontrados en vector store -> vector search.
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

from tests.helpers import make_lightrag


# =============================================================================
# Helpers
# =============================================================================


def _make_vector_result(doc_ids, contents, scores):
    return RetrievalResult(
        doc_ids=doc_ids,
        contents=contents,
        scores=scores,
        vector_scores=scores[:],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        metadata={},
    )


def _make_entity(name, entity_type="PERSON", description="", source_doc_ids=None):
    entity = object.__new__(KGEntity)
    entity.name = name
    entity.entity_type = entity_type
    entity.description = description
    entity.source_doc_ids = set(source_doc_ids) if source_doc_ids else set()
    entity._descriptions = []
    return entity


def _setup_kg_retrieval(r, entities_map, relations=None, store_contents=None):
    """Helper to set up the full KG retrieval pipeline for a test.

    Args:
        r: LightRAGRetriever mock.
        entities_map: dict of entity_name -> KGEntity (with source_doc_ids).
        relations: list of relation dicts for _resolve_relations_for_context.
        store_contents: dict of doc_id -> content for get_documents_by_ids.
    """
    # Entity VDB: return all entity names from entities_map
    mock_entity_vdb = MagicMock()
    entity_docs = []
    for name in entities_map:
        doc = MagicMock()
        doc.metadata = {"entity_name": name}
        entity_docs.append((doc, 0.1))
    mock_entity_vdb.similarity_search_with_score.return_value = entity_docs
    r._entities_vdb = mock_entity_vdb

    # KG entities lookup
    r._kg.get_all_entities.return_value = entities_map
    r._kg.get_entity.side_effect = lambda n: entities_map.get(n)

    # Vector store: return contents for doc_ids
    if store_contents:
        r._vector_retriever._vector_store.get_documents_by_ids.return_value = (
            store_contents
        )
    else:
        r._vector_retriever._vector_store.get_documents_by_ids.return_value = {}

    # Relationship VDB if relations provided
    if relations:
        mock_rel_vdb = MagicMock()
        rel_docs = []
        for rel in relations:
            doc = MagicMock()
            doc.metadata = {
                "source_entity": rel["source"],
                "target_entity": rel["target"],
                "relation": rel["relation"],
            }
            doc.page_content = rel.get("description", "")
            rel_docs.append((doc, 0.2))
        mock_rel_vdb.similarity_search_with_score.return_value = rel_docs
        r._relationships_vdb = mock_rel_vdb


# =============================================================================
# E1: No keywords -> fallback to vector search
# =============================================================================

def test_retrieve_via_kg_no_keywords_fallback():
    """Sin keywords extraidos, fallback a vector search."""
    r = make_lightrag()
    r._extractor.extract_query_keywords.return_value = ([], [])
    r._vector_retriever.retrieve.return_value = _make_vector_result(
        ["d1"], ["c1"], [0.9],
    )

    result = r._retrieve_via_kg("query", top_k=5)

    assert result.doc_ids == ["d1"]
    assert result.metadata["kg_fallback"] == "no_keywords"
    assert result.metadata["kg_entities"] == []
    assert result.metadata["kg_relations"] == []
    r._vector_retriever.retrieve.assert_called_once()


# =============================================================================
# E2: Hybrid mode — chunks via KG entities + relations
# =============================================================================

def test_retrieve_via_kg_hybrid():
    """Modo hybrid: chunks obtenidos via source_doc_ids del KG."""
    r = make_lightrag(lightrag_mode="hybrid")

    alice = _make_entity("alice", "PERSON", "A researcher", source_doc_ids=["d1", "d2"])
    bob = _make_entity("bob", "PERSON", "An engineer", source_doc_ids=["d2", "d3"])

    _setup_kg_retrieval(
        r,
        entities_map={"alice": alice, "bob": bob},
        relations=[{
            "source": "alice", "target": "bob",
            "relation": "works_with", "description": "collaboration",
        }],
        store_contents={"d1": "content1", "d2": "content2", "d3": "content3"},
    )

    r._extractor.extract_query_keywords.return_value = (
        ["alice", "bob"], ["collaboration"],
    )

    result = r._retrieve_via_kg("query about alice and bob", top_k=5)

    # Chunks come from KG, not vector search
    assert "d2" in result.doc_ids  # referenced by both entities -> highest score
    assert "d1" in result.doc_ids
    assert "d3" in result.doc_ids
    assert result.metadata["kg_fallback"] is None
    assert len(result.metadata["kg_entities"]) >= 1
    r._vector_retriever.retrieve.assert_not_called()


# =============================================================================
# E3: Local mode — only entities, no relations
# =============================================================================

def test_retrieve_via_kg_local_only_entities():
    """Modo local: chunks via entidades, relaciones no consultadas."""
    r = make_lightrag(lightrag_mode="local")

    entity_a = _make_entity("entity_a", "ORG", "An org", source_doc_ids=["d1", "d2"])

    _setup_kg_retrieval(
        r,
        entities_map={"entity_a": entity_a},
        store_contents={"d1": "content1", "d2": "content2"},
    )

    mock_rel_vdb = MagicMock()
    r._relationships_vdb = mock_rel_vdb

    r._extractor.extract_query_keywords.return_value = (["entity_a"], ["theme"])

    result = r._retrieve_via_kg("query", top_k=5)

    assert set(result.doc_ids) == {"d1", "d2"}
    assert result.metadata["kg_relations"] == []
    mock_rel_vdb.similarity_search_with_score.assert_not_called()


# =============================================================================
# E4: Global mode — only relations, no entities
# =============================================================================

def test_retrieve_via_kg_global_only_relations():
    """Modo global: chunks via relaciones, entidades no consultadas."""
    r = make_lightrag(lightrag_mode="global")

    # Entities exist in KG (needed for source_doc_ids via relations)
    x = _make_entity("x", "CONCEPT", "concept x", source_doc_ids=["d1"])
    y = _make_entity("y", "CONCEPT", "concept y", source_doc_ids=["d2"])

    r._kg.get_entity.side_effect = lambda n: {"x": x, "y": y}.get(n)
    r._kg.get_all_entities.return_value = {"x": x, "y": y}

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

    r._vector_retriever._vector_store.get_documents_by_ids.return_value = {
        "d1": "content1", "d2": "content2",
    }

    r._extractor.extract_query_keywords.return_value = (["entity"], ["theme"])

    result = r._retrieve_via_kg("query", top_k=5)

    assert set(result.doc_ids) == {"d1", "d2"}
    assert result.metadata["kg_entities"] == []
    mock_entity_vdb.similarity_search_with_score.assert_not_called()


# =============================================================================
# E5: Naive mode
# =============================================================================

def test_retrieve_mode_naive_skips_graph():
    """Modo naive: solo vector search, ignora KG."""
    r = make_lightrag(lightrag_mode="naive")
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

    r = make_lightrag()
    r._entities_vdb = mock_vdb

    result = r._resolve_entities_via_vdb(["keyword1", "keyword2"], top_k=5)
    assert result == ["entity_a"]  # deduplicated


def test_resolve_entities_via_vdb_empty_keywords():
    """_resolve_entities_via_vdb con keywords vacios retorna []."""
    r = make_lightrag()
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

    r = make_lightrag()
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

    r = make_lightrag()
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

    r = make_lightrag()
    r._relationships_vdb = mock_vdb

    result = r._resolve_relations_for_context(["kw1", "kw2"])
    assert len(result) == 1  # deduplicated


def test_resolve_relations_for_context_empty():
    """_resolve_relations_for_context con keywords vacios retorna []."""
    r = make_lightrag()
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

    r = make_lightrag()
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
# E9: Reference-count scoring
# =============================================================================

def test_reference_count_scoring_ranks_by_entity_overlap():
    """Docs referenciados por mas entidades obtienen mayor score."""
    r = make_lightrag(lightrag_mode="local")

    # d2 is referenced by both entities -> should rank first
    alice = _make_entity("alice", "PERSON", "A", source_doc_ids=["d1", "d2"])
    bob = _make_entity("bob", "PERSON", "B", source_doc_ids=["d2", "d3"])

    _setup_kg_retrieval(
        r,
        entities_map={"alice": alice, "bob": bob},
        store_contents={"d1": "c1", "d2": "c2", "d3": "c3"},
    )
    r._extractor.extract_query_keywords.return_value = (["alice", "bob"], [])

    result = r._retrieve_via_kg("query", top_k=5)

    # d2 has score 2.0 (from alice + bob), d1 and d3 have 1.0 each
    assert result.doc_ids[0] == "d2"
    assert result.scores[0] > result.scores[1]


def test_reference_count_relations_contribute_half_score():
    """Relations contribute 0.5 per doc_id to the scoring."""
    r = make_lightrag(lightrag_mode="hybrid")

    alice = _make_entity("alice", "PERSON", "A", source_doc_ids=["d1"])
    bob = _make_entity("bob", "PERSON", "B", source_doc_ids=["d2"])

    _setup_kg_retrieval(
        r,
        entities_map={"alice": alice, "bob": bob},
        relations=[{
            "source": "alice", "target": "bob",
            "relation": "works_with", "description": "collab",
        }],
        store_contents={"d1": "c1", "d2": "c2"},
    )
    r._extractor.extract_query_keywords.return_value = (["alice"], ["collab"])

    result = r._retrieve_via_kg("query", top_k=5)

    # d1: 1.0 (entity alice) + 0.5 (relation endpoint alice) = 1.5
    # d2: 0.5 (relation endpoint bob)
    assert "d1" in result.doc_ids
    assert "d2" in result.doc_ids
    d1_idx = result.doc_ids.index("d1")
    d2_idx = result.doc_ids.index("d2")
    assert result.scores[d1_idx] > result.scores[d2_idx]


# =============================================================================
# E10-E11: Fallback scenarios
# =============================================================================

def test_fallback_no_doc_ids_from_kg():
    """KG resuelve entidades pero sin source_doc_ids -> fallback a vector search."""
    r = make_lightrag(lightrag_mode="local")

    # Entity exists but has no source_doc_ids
    alice = _make_entity("alice", "PERSON", "A researcher", source_doc_ids=[])

    mock_entity_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"entity_name": "alice"}
    mock_entity_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    r._entities_vdb = mock_entity_vdb

    r._kg.get_all_entities.return_value = {"alice": alice}
    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    r._vector_retriever.retrieve.return_value = _make_vector_result(
        ["v1"], ["vector content"], [0.9],
    )

    result = r._retrieve_via_kg("query", top_k=5)

    assert result.metadata["kg_fallback"] == "no_doc_ids"
    assert result.doc_ids == ["v1"]
    r._vector_retriever.retrieve.assert_called_once()


def test_fallback_docs_not_in_store():
    """KG produce doc_ids pero no se encuentran en vector store -> fallback."""
    r = make_lightrag(lightrag_mode="local")

    alice = _make_entity("alice", "PERSON", "A", source_doc_ids=["orphan_doc"])

    _setup_kg_retrieval(
        r,
        entities_map={"alice": alice},
        store_contents={},  # doc not found
    )
    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    r._vector_retriever.retrieve.return_value = _make_vector_result(
        ["v1"], ["fallback content"], [0.9],
    )

    result = r._retrieve_via_kg("query", top_k=5)

    assert result.metadata["kg_fallback"] == "docs_not_in_store"
    assert result.doc_ids == ["v1"]
    r._vector_retriever.retrieve.assert_called_once()


# =============================================================================
# Divergencia #9: 1-hop neighbor expansion in _retrieve_via_kg
# =============================================================================


def test_retrieve_entity_with_neighbors():
    """Entidades resueltas incluyen vecinos 1-hop en metadata."""
    r = make_lightrag(lightrag_mode="local")

    alice = _make_entity("alice", "PERSON", "A researcher", source_doc_ids=["d1"])

    _setup_kg_retrieval(
        r,
        entities_map={"alice": alice},
        store_contents={"d1": "content1"},
    )
    r._kg.get_neighbors_ranked.return_value = [
        {"entity": "bob", "type": "PERSON", "description": "Engineer",
         "relation": "works_with", "score": 3.5},
    ]
    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    result = r._retrieve_via_kg("query about alice", top_k=5)

    entity = result.metadata["kg_entities"][0]
    assert entity["entity"] == "alice"
    assert "neighbors" in entity
    assert entity["neighbors"][0]["entity"] == "bob"


def test_retrieve_entity_no_neighbors():
    """Sin vecinos, el dict de entidad no tiene clave 'neighbors'."""
    r = make_lightrag(lightrag_mode="local")

    alice = _make_entity("alice", "PERSON", "A researcher", source_doc_ids=["d1"])

    _setup_kg_retrieval(
        r,
        entities_map={"alice": alice},
        store_contents={"d1": "content1"},
    )
    r._kg.get_neighbors_ranked.return_value = []
    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    result = r._retrieve_via_kg("query", top_k=5)

    entity = result.metadata["kg_entities"][0]
    assert entity["entity"] == "alice"
    assert "neighbors" not in entity


def test_retrieve_entity_neighbor_fallback():
    """Si get_neighbors_ranked lanza excepcion, la entidad aparece sin vecinos."""
    r = make_lightrag(lightrag_mode="local")

    alice = _make_entity("alice", "PERSON", "A researcher", source_doc_ids=["d1"])

    _setup_kg_retrieval(
        r,
        entities_map={"alice": alice},
        store_contents={"d1": "content1"},
    )
    r._kg.get_neighbors_ranked.side_effect = RuntimeError("graph error")
    r._extractor.extract_query_keywords.return_value = (["alice"], [])

    result = r._retrieve_via_kg("query", top_k=5)

    entity = result.metadata["kg_entities"][0]
    assert entity["entity"] == "alice"
    assert "neighbors" not in entity


# =============================================================================
# Divergencia #9: Relation endpoint enrichment
# =============================================================================


def test_resolve_relations_endpoint_enrichment():
    """Relaciones incluyen description y type de sus entidades endpoint."""
    mock_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "source_entity": "alice",
        "target_entity": "bob",
        "relation": "mentors",
    }
    mock_doc.page_content = "alice -> mentors -> bob"
    mock_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.2)]

    r = make_lightrag()
    r._relationships_vdb = mock_vdb
    alice = _make_entity("alice", "PERSON", "A researcher at MIT")
    bob = _make_entity("bob", "PERSON", "A data scientist")
    r._kg.get_all_entities.return_value = {"alice": alice, "bob": bob}

    result = r._resolve_relations_for_context(["mentorship"])

    assert len(result) == 1
    assert result[0]["source"] == "alice"
    assert result[0]["target"] == "bob"
    assert result[0]["source_description"] == "A researcher at MIT"
    assert result[0]["source_type"] == "PERSON"
    assert result[0]["target_description"] == "A data scientist"
    assert result[0]["target_type"] == "PERSON"


def test_resolve_relations_missing_endpoint():
    """Endpoint no encontrado en KG: campos de description/type ausentes."""
    mock_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "source_entity": "alice",
        "target_entity": "unknown_entity",
        "relation": "works_with",
    }
    mock_doc.page_content = "alice -> works_with -> unknown_entity"
    mock_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.2)]

    r = make_lightrag()
    r._relationships_vdb = mock_vdb
    alice = _make_entity("alice", "PERSON", "A researcher")
    r._kg.get_all_entities.return_value = {"alice": alice}

    result = r._resolve_relations_for_context(["collaboration"])

    assert len(result) == 1
    assert result[0]["source_description"] == "A researcher"
    assert result[0]["source_type"] == "PERSON"
    assert "target_description" not in result[0]
    assert "target_type" not in result[0]


def test_resolve_relations_no_kg():
    """Sin KG disponible, relaciones se resuelven sin enrichment."""
    mock_vdb = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "source_entity": "a",
        "target_entity": "b",
        "relation": "r",
    }
    mock_doc.page_content = "a -> r -> b"
    mock_vdb.similarity_search_with_score.return_value = [(mock_doc, 0.1)]

    r = make_lightrag()
    r._kg = None
    r._relationships_vdb = mock_vdb

    result = r._resolve_relations_for_context(["keyword"])

    assert len(result) == 1
    assert result[0]["source"] == "a"
    assert "source_description" not in result[0]
