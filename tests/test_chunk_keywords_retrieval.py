"""Tests unitarios para el canal de chunk keywords en `_retrieve_via_kg`.

Divergencia #10 (paper HKUDS/LightRAG): durante indexacion el LLM extrae
high-level keywords por chunk; durante retrieval el path high-level
(modos `global`/`hybrid`) consulta una Chunk Keywords VDB dedicada y
acumula scoring `1/(1+rank) * similarity` por chunk matched. Estos tests
ejercitan el canal con mocks, sin infraestructura real.

Cobertura:
  CK1. hybrid/global: el canal contribuye a doc_scores.
  CK2. local: el canal NO se activa (solo paths que usan high_level).
  CK3. naive: no aplica (no pasa por _retrieve_via_kg).
  CK4. kg_chunk_keywords_enabled=False: canal deshabilitado aunque
        el VDB exista y haya high_level keywords.
  CK5. Sin high_level keywords en la query: sin llamadas al VDB.
  CK6. Sin Chunk Keywords VDB construida: canal inerte.
  CK7. Threshold de distance: matches por encima se filtran.
  CK8. Dedup doc_id con mejor distancia y orden de aparicion.
  CK9. metadata.kg_chunk_keyword_matches presente en happy path.
  CK10. metadata.kg_chunk_keyword_matches presente en fallback no_doc_ids.
  CK11. Canal rescata chunks que el canal de relaciones no aporta.
"""

from unittest.mock import MagicMock

import pytest

from shared.retrieval.core import RetrievalResult, RetrievalStrategy
from shared.retrieval.lightrag.knowledge_graph import KGEntity
from shared.retrieval.lightrag.retriever import LightRAGRetriever

from tests.helpers import make_lightrag


# =============================================================================
# Helpers
# =============================================================================

def _make_entity(name, source_doc_ids=None):
    e = object.__new__(KGEntity)
    e.name = name
    e.entity_type = "PERSON"
    e.description = ""
    e.source_doc_ids = set(source_doc_ids or [])
    e._descriptions = []
    return e


def _make_vector_result(doc_ids):
    return RetrievalResult(
        doc_ids=doc_ids,
        contents=[f"c_{did}" for did in doc_ids],
        scores=[1.0 - i * 0.01 for i in range(len(doc_ids))],
        vector_scores=[1.0 - i * 0.01 for i in range(len(doc_ids))],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        metadata={},
    )


def _setup_chunk_keywords_vdb(retriever, matches):
    """Configura `_chunk_keywords_vdb` para devolver matches por keyword.

    Args:
        matches: list of (doc_id, distance) por cada keyword. Todas las
            keywords retornan los mismos matches (simplificacion del mock).
    """
    vdb = MagicMock()
    results = []
    for doc_id, distance in matches:
        doc = MagicMock()
        doc.metadata = {"doc_id": doc_id}
        results.append((doc, distance))
    vdb.similarity_search_with_score.return_value = results
    retriever._chunk_keywords_vdb = vdb
    return vdb


def _prime_retriever_for_hybrid(
    retriever,
    store_contents,
    entities_map=None,
    low_level=None,
    high_level=None,
):
    """Setup minimo para que _retrieve_via_kg no caiga en fallback por
    razones no relacionadas con el canal que queremos testear.
    """
    retriever._extractor.extract_query_keywords.return_value = (
        list(low_level or []), list(high_level or []),
    )
    retriever._kg.get_all_entities.return_value = entities_map or {}
    retriever._kg.get_entity.side_effect = lambda n: (entities_map or {}).get(n)
    retriever._vector_retriever.get_documents_by_ids.return_value = store_contents

    # Entity VDB vacio por defecto (tests que no lo necesitan)
    mock_entity_vdb = MagicMock()
    mock_entity_vdb.similarity_search_with_score.return_value = []
    retriever._entities_vdb = mock_entity_vdb

    # Relationship VDB vacio por defecto
    mock_rel_vdb = MagicMock()
    mock_rel_vdb.similarity_search_with_score.return_value = []
    retriever._relationships_vdb = mock_rel_vdb


# =============================================================================
# CK1: El canal contribuye a doc_scores en hybrid
# =============================================================================

def test_chunk_keywords_channel_contributes_hybrid():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True
    r.config.kg_chunk_keywords_top_k = 5

    _prime_retriever_for_hybrid(
        r,
        store_contents={"d1": "c1", "d2": "c2"},
        high_level=["quantum mechanics"],
    )
    _setup_chunk_keywords_vdb(r, matches=[("d1", 0.1), ("d2", 0.3)])

    result = r._retrieve_via_kg("query", top_k=5)

    # El canal produjo doc_scores, asi que no hay fallback
    assert result.metadata["kg_fallback"] is None
    assert set(result.doc_ids) == {"d1", "d2"}
    assert result.metadata["kg_chunk_keyword_matches"] == 2


# =============================================================================
# CK2: En mode=local el canal no se activa (no hay high_level path)
# =============================================================================

def test_chunk_keywords_channel_inactive_in_local_mode():
    r = make_lightrag(lightrag_mode="local")
    r.config.kg_chunk_keywords_enabled = True

    # Un entity resuelve d1 (para que el canal de entidades produzca algo).
    alice = _make_entity("alice", source_doc_ids=["d1"])
    _prime_retriever_for_hybrid(
        r,
        store_contents={"d1": "c1"},
        entities_map={"alice": alice},
        low_level=["alice"],
        high_level=["quantum mechanics"],  # irrelevante en local
    )
    # Entity VDB resuelve alice
    r._entities_vdb.similarity_search_with_score.return_value = [
        (MagicMock(metadata={"entity_name": "alice"}), 0.1),
    ]
    vdb = _setup_chunk_keywords_vdb(r, matches=[("d2", 0.1)])

    result = r._retrieve_via_kg("query", top_k=5)

    # d2 NO aparece — canal de chunk keywords no se activa en local
    assert result.doc_ids == ["d1"]
    # VDB no recibio consulta
    vdb.similarity_search_with_score.assert_not_called()
    assert result.metadata["kg_chunk_keyword_matches"] == 0


# =============================================================================
# CK3: global lo activa pero solo si hay high_level
# =============================================================================

def test_chunk_keywords_channel_active_in_global_mode():
    r = make_lightrag(lightrag_mode="global")
    r.config.kg_chunk_keywords_enabled = True

    _prime_retriever_for_hybrid(
        r,
        store_contents={"d5": "c5"},
        high_level=["methodology"],
    )
    _setup_chunk_keywords_vdb(r, matches=[("d5", 0.05)])

    result = r._retrieve_via_kg("q", top_k=5)

    assert result.doc_ids == ["d5"]
    assert result.metadata["kg_chunk_keyword_matches"] == 1


# =============================================================================
# CK4: flag disabled -> canal inerte aunque VDB exista y haya high_level
# =============================================================================

def test_chunk_keywords_disabled_via_flag():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = False

    _prime_retriever_for_hybrid(
        r,
        store_contents={"d1": "c1"},
        high_level=["methodology"],
    )
    r._vector_retriever.retrieve.return_value = _make_vector_result(["v1"])
    vdb = _setup_chunk_keywords_vdb(r, matches=[("d1", 0.1)])

    result = r._retrieve_via_kg("q", top_k=5)

    # Canal no activado → sin doc_scores de chunk keywords; cae a fallback
    # (los otros canales tambien estan vacios).
    assert result.metadata["kg_fallback"] == "no_doc_ids"
    vdb.similarity_search_with_score.assert_not_called()
    assert result.metadata["kg_chunk_keyword_matches"] == 0


# =============================================================================
# CK5: sin high_level en la query -> sin llamadas al VDB
# =============================================================================

def test_chunk_keywords_no_high_level_no_vdb_calls():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True

    alice = _make_entity("alice", source_doc_ids=["d1"])
    _prime_retriever_for_hybrid(
        r,
        store_contents={"d1": "c1"},
        entities_map={"alice": alice},
        low_level=["alice"],
        high_level=[],  # vacio
    )
    r._entities_vdb.similarity_search_with_score.return_value = [
        (MagicMock(metadata={"entity_name": "alice"}), 0.1),
    ]
    vdb = _setup_chunk_keywords_vdb(r, matches=[("d9", 0.1)])

    result = r._retrieve_via_kg("q", top_k=5)

    vdb.similarity_search_with_score.assert_not_called()
    assert "d9" not in result.doc_ids
    assert result.metadata["kg_chunk_keyword_matches"] == 0


# =============================================================================
# CK6: sin VDB construida -> canal inerte, no explota
# =============================================================================

def test_chunk_keywords_no_vdb_graceful():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True
    r._chunk_keywords_vdb = None  # explicito

    _prime_retriever_for_hybrid(
        r,
        store_contents={},
        high_level=["methodology"],
    )
    r._vector_retriever.retrieve.return_value = _make_vector_result(["v1"])

    # No debe crashear; cae a fallback por no_doc_ids
    result = r._retrieve_via_kg("q", top_k=5)
    assert result.metadata["kg_fallback"] == "no_doc_ids"
    assert result.metadata["kg_chunk_keyword_matches"] == 0


# =============================================================================
# CK7: threshold de distance filtra matches por encima
# =============================================================================

def test_chunk_keywords_distance_threshold_filters():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True

    _prime_retriever_for_hybrid(
        r,
        store_contents={"close": "c1"},
        high_level=["methodology"],
    )
    # 0.9 > _CHUNK_KEYWORDS_VDB_MAX_DISTANCE (0.8) → filtrado.
    # 0.2 ≤ 0.8 → aceptado.
    _setup_chunk_keywords_vdb(
        r, matches=[("close", 0.2), ("far", 0.9)],
    )
    r._vector_retriever.get_documents_by_ids.return_value = {"close": "c1"}

    result = r._retrieve_via_kg("q", top_k=5)

    assert result.doc_ids == ["close"]
    assert "far" not in result.doc_ids
    assert result.metadata["kg_chunk_keyword_matches"] == 1


# =============================================================================
# CK8: dedup con mejor distancia cuando aparece en varias keywords
# =============================================================================

def test_chunk_keywords_dedup_keeps_best_distance():
    """Si un mismo doc_id aparece en varias keywords, se queda la mejor
    distancia y su orden de aparicion."""
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True

    _prime_retriever_for_hybrid(
        r,
        store_contents={"d1": "c1"},
        high_level=["kw1", "kw2"],
    )
    # El mock devuelve los mismos matches para ambas keywords:
    # primera llamada: d1 con distance=0.3
    # segunda llamada: d1 con distance=0.1 (mejor)
    vdb = MagicMock()
    d1_doc = MagicMock()
    d1_doc.metadata = {"doc_id": "d1"}
    vdb.similarity_search_with_score.side_effect = [
        [(d1_doc, 0.3)],
        [(d1_doc, 0.1)],
    ]
    r._chunk_keywords_vdb = vdb

    result = r._retrieve_via_kg("q", top_k=5)

    assert result.doc_ids == ["d1"]
    # kg_chunk_keyword_matches = docs unicos matched
    assert result.metadata["kg_chunk_keyword_matches"] == 1


# =============================================================================
# CK9/CK10: metadata.kg_chunk_keyword_matches siempre presente
# =============================================================================

def test_metadata_key_present_on_happy_path():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True

    _prime_retriever_for_hybrid(
        r,
        store_contents={"d1": "c1"},
        high_level=["theme"],
    )
    _setup_chunk_keywords_vdb(r, matches=[("d1", 0.1)])

    result = r._retrieve_via_kg("q", top_k=5)
    assert "kg_chunk_keyword_matches" in result.metadata


def test_metadata_key_present_on_no_doc_ids_fallback():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True

    _prime_retriever_for_hybrid(
        r,
        store_contents={},
        high_level=["theme"],
    )
    r._vector_retriever.retrieve.return_value = _make_vector_result(["v1"])
    # VDB vacio → no matches → sin doc_scores → fallback
    _setup_chunk_keywords_vdb(r, matches=[])

    result = r._retrieve_via_kg("q", top_k=5)
    assert result.metadata["kg_fallback"] == "no_doc_ids"
    assert result.metadata["kg_chunk_keyword_matches"] == 0


def test_metadata_key_present_on_no_keywords_fallback():
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True
    r._extractor.extract_query_keywords.return_value = ([], [])
    r._vector_retriever.retrieve.return_value = _make_vector_result(["v1"])

    result = r._retrieve_via_kg("q", top_k=5)
    assert result.metadata["kg_fallback"] == "no_keywords"
    assert result.metadata["kg_chunk_keyword_matches"] == 0


# =============================================================================
# CK11: canal aporta chunks que relaciones no cubren
# =============================================================================

def test_chunk_keywords_rescues_docs_not_in_relations():
    """Verifica que el canal aporta doc_ids que el canal de relaciones no
    tiene (tematicas no expresadas como relaciones entre entidades)."""
    r = make_lightrag(lightrag_mode="hybrid")
    r.config.kg_chunk_keywords_enabled = True

    # Sin entidades/relaciones resueltas, solo chunk keywords
    _prime_retriever_for_hybrid(
        r,
        store_contents={"d_theme": "content about methodology"},
        high_level=["methodology"],
    )
    _setup_chunk_keywords_vdb(r, matches=[("d_theme", 0.05)])

    result = r._retrieve_via_kg("q", top_k=5)

    assert result.doc_ids == ["d_theme"]
    assert result.metadata["kg_fallback"] is None
    # kg_entities vacio, kg_relations vacio, chunk_keyword_matches=1
    assert result.metadata["kg_entities"] == []
    assert result.metadata["kg_relations"] == []
    assert result.metadata["kg_chunk_keyword_matches"] == 1
