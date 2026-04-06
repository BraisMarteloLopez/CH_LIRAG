"""
Tests unitarios para KnowledgeGraph (DTm-23, Fase 1.1).

Cobertura:
  KG1. add_triplets basico — entidades y relaciones se anaden al grafo.
  KG2. add_triplets dedup — misma entidad desde 2 docs -> 1 nodo, 2 source_doc_ids.
  KG3. add_triplets con nombre vacio — tripleta con src/tgt vacio se ignora.
  KG4. add_triplets self-loop — source == target funciona sin error.
  KG5. add_triplets relacion duplicada en arista — se acumula (2 entries).
  KG6. query_entities BFS — devuelve docs conectados con scoring 1/(1+depth).
  KG7. query_entities max_hops — docs a distancia > max_hops no aparecen.
  KG8. query_entities entidad desconocida — retorna lista vacia.
  KG9. query_by_keywords — substring match en nombres de entidad.
  KG10. query_by_keywords vacia — sin keywords retorna lista vacia.
  KG11. query_by_keywords en relaciones — match en descripcion de relacion.
  KG12. grafo vacio — todas las queries retornan vacio.
  KG14. get_stats — estadisticas correctas.
  KG15. add_entity_metadata — actualiza type y description.
"""

import json
from pathlib import Path

import pytest

from shared.retrieval.lightrag.knowledge_graph import KGEntity, KGRelation, KnowledgeGraph


# =============================================================================
# Helpers
# =============================================================================

def _rel(src: str, tgt: str, relation: str = "related to",
         desc: str = "", doc_id: str = "doc1") -> KGRelation:
    return KGRelation(
        source=src, target=tgt, relation=relation,
        description=desc, source_doc_id=doc_id,
    )


def _build_chain_graph() -> KnowledgeGraph:
    """Grafo lineal: A -- B -- C -- D, todos desde doc1."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("A", "B", "knows"),
        _rel("B", "C", "works with"),
        _rel("C", "D", "manages"),
    ])
    kg.build_keyword_indices()
    return kg


# =============================================================================
# KG1: add_triplets basico
# =============================================================================

def test_add_triplets_basic():
    """Entidades y relaciones se anaden al grafo."""
    kg = KnowledgeGraph()
    added = kg.add_triplets("doc1", [
        _rel("Alice", "Bob", "knows"),
        _rel("Bob", "Acme Corp", "works at"),
    ])

    assert added == 2
    assert kg.num_entities == 3  # alice, bob, acme corp
    assert kg.num_relations == 2
    assert kg.num_docs == 1


# =============================================================================
# KG2: dedup de entidades
# =============================================================================

def test_add_triplets_dedup_entities():
    """Misma entidad desde 2 docs -> 1 nodo, 2 source_doc_ids."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows", doc_id="doc1")])
    kg.add_triplets("doc2", [_rel("Alice", "Charlie", "manages", doc_id="doc2")])

    assert kg.num_entities == 3  # alice, bob, charlie
    # Alice aparece en doc1 y doc2
    alice = kg._entities["alice"]
    assert alice.source_doc_ids == {"doc1", "doc2"}


# =============================================================================
# KG3: nombre vacio
# =============================================================================

def test_add_triplets_empty_name_skipped():
    """Tripleta con source o target vacio se ignora."""
    kg = KnowledgeGraph()
    added = kg.add_triplets("doc1", [
        _rel("", "Bob", "knows"),
        _rel("Alice", "", "knows"),
        _rel("  ", "Bob", "knows"),  # solo whitespace
    ])
    assert added == 0
    assert kg.num_entities == 0


# =============================================================================
# KG4: self-loop
# =============================================================================

def test_add_triplets_self_loop():
    """source == target funciona (self-loop en NetworkX)."""
    kg = KnowledgeGraph()
    added = kg.add_triplets("doc1", [_rel("Alice", "Alice", "self-reference")])
    assert added == 1
    assert kg.num_entities == 1


# =============================================================================
# KG5: relacion duplicada en arista
# =============================================================================

def test_add_triplets_duplicate_relation_accumulates():
    """Misma arista desde 2 docs -> 2 entries en relations list."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows", doc_id="doc1")])
    kg.add_triplets("doc2", [_rel("Alice", "Bob", "collaborates", doc_id="doc2")])

    edge_data = kg._get_edge_data("alice", "bob")
    assert edge_data is not None
    assert len(edge_data["relations"]) == 2
    assert edge_data["relations"][0]["relation"] == "knows"
    assert edge_data["relations"][1]["relation"] == "collaborates"
    # Solo 1 arista (misma pair de nodos)
    assert kg.num_relations == 1


# =============================================================================
# KG6: query_entities BFS scoring
# =============================================================================

def test_query_entities_bfs_scoring():
    """BFS devuelve docs con scoring 1/(1+depth)."""
    kg = KnowledgeGraph()
    # A(doc1) -- B(doc2) -- C(doc3)
    kg.add_triplets("doc1", [_rel("A", "B", "r", doc_id="doc1")])
    kg.add_triplets("doc2", [_rel("B", "C", "r", doc_id="doc2")])
    # doc1 tiene A (hop 0 from A), doc2 tiene B (hop 1 from A) y C,
    # doc2 tiene C y B
    # A -> entity_to_docs: A={doc1}, B={doc1,doc2}, C={doc2}

    results = kg.query_entities(["A"], max_hops=2, max_docs=10)
    result_dict = dict(results)

    # doc1: A at hop 0 -> 1.0, B at hop 1 -> 0.5 => doc1 gets 1.0+0.5=1.5
    # doc2: B at hop 1 -> 0.5, C at hop 2 -> 0.333 => doc2 gets 0.5+0.333=0.833
    assert "doc1" in result_dict
    assert "doc2" in result_dict
    assert result_dict["doc1"] > result_dict["doc2"]


# =============================================================================
# KG7: max_hops limita profundidad
# =============================================================================

def test_query_entities_max_hops():
    """Docs a distancia > max_hops no aparecen."""
    kg = _build_chain_graph()
    # A--B--C--D, all from doc1
    # Con max_hops=0: solo entidad directa
    # Pero doc1 esta asociado a A (hop 0), asi que aparece
    results = kg.query_entities(["A"], max_hops=0, max_docs=10)
    result_dict = dict(results)
    assert "doc1" in result_dict

    # Crear grafo donde diferentes docs estan a diferentes distancias
    kg2 = KnowledgeGraph()
    kg2.add_triplets("doc_near", [_rel("X", "Y", "r", doc_id="doc_near")])
    kg2.add_triplets("doc_far", [_rel("Y", "Z", "r", doc_id="doc_far")])
    kg2.add_triplets("doc_very_far", [_rel("Z", "W", "r", doc_id="doc_very_far")])

    # max_hops=1: X(hop0), Y(hop1) -> doc_near, doc_far
    # Z esta a hop2, W a hop3 -> doc_very_far excluido
    results = kg2.query_entities(["X"], max_hops=1, max_docs=10)
    result_dict = dict(results)
    assert "doc_near" in result_dict
    assert "doc_far" in result_dict
    assert "doc_very_far" not in result_dict


# =============================================================================
# KG8: entidad desconocida
# =============================================================================

def test_query_entities_unknown():
    """Entidad inexistente -> lista vacia."""
    kg = _build_chain_graph()
    results = kg.query_entities(["NonExistent"], max_hops=2)
    assert results == []


# =============================================================================
# KG9: query_by_keywords
# =============================================================================

def test_query_by_keywords():
    """Substring match en nombres de entidad."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Machine Learning", "Neural Networks", "r")])
    kg.add_triplets("doc2", [_rel("Deep Learning", "CNN", "r")])
    kg.build_keyword_indices()

    results = kg.query_by_keywords(["learning"], max_docs=10)
    result_dict = dict(results)
    assert "doc1" in result_dict
    assert "doc2" in result_dict


# =============================================================================
# KG10: query_by_keywords vacia
# =============================================================================

def test_query_by_keywords_empty():
    """Sin keywords retorna lista vacia."""
    kg = _build_chain_graph()
    assert kg.query_by_keywords([], max_docs=10) == []
    assert kg.query_by_keywords(["   "], max_docs=10) == []


# =============================================================================
# KG11: query_by_keywords en descripciones de relaciones
# =============================================================================

def test_query_by_keywords_in_relations():
    """Match en descripcion de relacion contribuye 0.5 al score."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("X", "Y", "funds", desc="provides financial support", doc_id="doc1"),
    ])
    kg.build_keyword_indices()

    results = kg.query_by_keywords(["financial"], max_docs=10)
    result_dict = dict(results)
    assert "doc1" in result_dict


# =============================================================================
# KG12b: stemming en keyword search
# =============================================================================

def test_query_by_keywords_stemming():
    """Variantes morfologicas matchean via stemming: 'mechanical' -> 'mechanics'."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("quantum mechanics", "physics", "branch of",
             desc="study of subatomic particles", doc_id="doc1"),
    ])
    kg.build_keyword_indices()
    # "mechanical" stems to same root as "mechanics"
    results = kg.query_by_keywords(["mechanical"], max_docs=10)
    result_dict = dict(results)
    assert "doc1" in result_dict, (
        "Stemming deberia matchear 'mechanical' con entidad 'quantum mechanics'"
    )

    # "studying" stems to same root as "study"
    results2 = kg.query_by_keywords(["studying"], max_docs=10)
    result_dict2 = dict(results2)
    assert "doc1" in result_dict2, (
        "Stemming deberia matchear 'studying' con descripcion 'study of...'"
    )


# =============================================================================
# KG13: grafo vacio
# =============================================================================

def test_empty_graph_queries():
    """Todas las queries sobre grafo vacio -> resultados vacios."""
    kg = KnowledgeGraph()
    assert kg.num_entities == 0
    assert kg.num_relations == 0
    assert kg.num_docs == 0
    assert kg.query_entities(["A"]) == []
    assert kg.query_by_keywords(["algo"]) == []


# =============================================================================
# KG14: get_stats
# =============================================================================

def test_get_stats():
    """Estadisticas correctas."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    kg.add_triplets("doc2", [_rel("C", "D", "r")])

    stats = kg.get_stats()
    assert stats["num_entities"] == 4
    assert stats["num_relations"] == 2
    assert stats["num_docs_with_entities"] == 2
    assert stats["avg_entities_per_doc"] == 2.0
    assert stats["graph_connected_components"] == 2


def test_get_stats_empty():
    """Stats de grafo vacio no lanza errores."""
    kg = KnowledgeGraph()
    stats = kg.get_stats()
    assert stats["num_entities"] == 0
    assert stats["avg_entities_per_doc"] == 0.0
    assert stats["graph_connected_components"] == 0


# =============================================================================
# KG15: add_entity_metadata
# =============================================================================

def test_add_entity_metadata():
    """Actualiza type y description de entidad existente."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows")])

    kg.add_entity_metadata("Alice", "PERSON", "A researcher")
    alice = kg._entities["alice"]
    assert alice.entity_type == "PERSON"
    assert alice.description == "A researcher"


def test_add_entity_metadata_unknown_entity():
    """Metadata para entidad inexistente no lanza error."""
    kg = KnowledgeGraph()
    kg.add_entity_metadata("Ghost", "PERSON", "Does not exist")
    assert kg.num_entities == 0  # sin cambios


# =============================================================================
# KG16: max_entities cap (DTm-21)
# =============================================================================

def test_max_entities_cap():
    """Entidades nuevas se rechazan al alcanzar el cap."""
    kg = KnowledgeGraph(max_entities=3)
    # A, B -> 2 entidades
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    assert kg.num_entities == 2

    # C, D -> C se acepta (3ra), D se rechaza (4ta > cap)
    # La tripleta entera se salta porque D no puede crearse
    added = kg.add_triplets("doc2", [_rel("C", "D", "r")])
    assert kg.num_entities == 3
    assert added == 0  # tripleta descartada porque D no cabe
    assert kg._entities_dropped > 0


def test_max_entities_existing_entities_still_updated():
    """Entidades existentes se actualizan aunque el cap este lleno."""
    kg = KnowledgeGraph(max_entities=2)
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    assert kg.num_entities == 2

    # A y B ya existen, se actualizan sin problemas
    added = kg.add_triplets("doc2", [_rel("A", "B", "collaborates")])
    assert added == 1
    assert kg.num_entities == 2  # sin entidades nuevas


# =============================================================================
# KG17: dedup de relaciones en aristas (DTm-21)
# =============================================================================

def test_relation_dedup_same_doc():
    """Misma relacion del mismo doc no se duplica en la arista."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("A", "B", "knows", doc_id="doc1"),
        _rel("A", "B", "knows", doc_id="doc1"),  # duplicado exacto
    ])

    edge_data = kg._get_edge_data("a", "b")
    assert len(edge_data["relations"]) == 1


def test_relation_dedup_different_doc():
    """Misma relacion de diferente doc SI se anade."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("A", "B", "knows", doc_id="doc1")])
    kg.add_triplets("doc2", [_rel("A", "B", "knows", doc_id="doc2")])

    edge_data = kg._get_edge_data("a", "b")
    assert len(edge_data["relations"]) == 2


def test_relation_dedup_different_relation():
    """Diferente tipo de relacion del mismo doc SI se anade."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("A", "B", "knows", doc_id="doc1"),
        _rel("A", "B", "works with", doc_id="doc1"),
    ])

    edge_data = kg._get_edge_data("a", "b")
    assert len(edge_data["relations"]) == 2


# =============================================================================
# KG18: get_stats incluye memoria y cap info (DTm-21)
# =============================================================================

def test_get_stats_includes_memory_and_cap():
    """Stats incluyen approx_memory_mb, entities_dropped, max_entities."""
    kg = KnowledgeGraph(max_entities=100)
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    stats = kg.get_stats()

    assert "approx_memory_mb" in stats
    assert stats["approx_memory_mb"] >= 0
    assert stats["entities_dropped"] == 0
    assert stats["max_entities"] == 100


# =============================================================================
# KG19-KG23: Persistencia (DTm-34)
# =============================================================================

def test_to_dict_from_dict_roundtrip():
    """to_dict -> from_dict preserva estado completo."""
    kg = KnowledgeGraph(max_entities=1000)
    kg.add_triplets("doc1", [
        _rel("Alice", "Bob", "knows", desc="friends", doc_id="doc1"),
        _rel("Bob", "Acme", "works at", desc="employee", doc_id="doc1"),
    ])
    kg.add_triplets("doc2", [
        _rel("Alice", "Charlie", "manages", doc_id="doc2"),
    ])
    kg.add_entity_metadata("Alice", "PERSON", "A researcher")
    kg.add_entity_metadata("Acme", "ORG", "A company")

    data = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(data)

    assert kg2.num_entities == kg.num_entities
    assert kg2.num_relations == kg.num_relations
    assert kg2.num_docs == kg.num_docs
    assert kg2._max_entities == 1000

    # Entidades preservadas con metadata
    alice = kg2._entities["alice"]
    assert alice.entity_type == "PERSON"
    assert alice.description == "A researcher"
    assert "doc1" in alice.source_doc_ids
    assert "doc2" in alice.source_doc_ids

    # Indices invertidos preservados
    assert "doc1" in kg2._entity_to_docs["alice"]
    assert "alice" in kg2._doc_to_entities["doc1"]

    # Relaciones por doc preservadas
    assert len(kg2._doc_to_relations["doc1"]) == 2
    assert len(kg2._doc_to_relations["doc2"]) == 1

    # Grafo funcional — queries producen mismos resultados
    orig_results = kg.query_entities(["Alice"], max_hops=2)
    loaded_results = kg2.query_entities(["Alice"], max_hops=2)
    assert dict(orig_results) == dict(loaded_results)


def test_to_dict_version_field():
    """to_dict incluye campo version."""
    kg = KnowledgeGraph()
    data = kg.to_dict()
    assert data["version"] == 2


def test_save_load_file(tmp_path):
    """save() y load() persisten y restauran desde archivo JSON."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("X", "Y", "related", desc="test relation", doc_id="doc1"),
    ])
    kg.add_entity_metadata("X", "CONCEPT", "concept X")

    path = tmp_path / "subdir" / "kg.json"
    kg.save(path)

    assert path.exists()
    # Verificar que es JSON valido
    with open(path) as f:
        data = json.load(f)
    assert data["version"] == 2

    # Load y verificar
    kg2 = KnowledgeGraph.load(path)
    assert kg2.num_entities == 2
    assert kg2.num_relations == 1
    assert kg2._entities["x"].entity_type == "CONCEPT"


def test_from_dict_empty_graph():
    """from_dict con datos minimos produce grafo vacio funcional."""
    kg = KnowledgeGraph.from_dict({"version": 1})
    assert kg.num_entities == 0
    assert kg.num_relations == 0
    assert kg.query_entities(["anything"]) == []


def test_roundtrip_preserves_edge_relations():
    """Roundtrip preserva relaciones en aristas del grafo NetworkX."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("A", "B", "knows", doc_id="doc1"),
        _rel("A", "B", "works with", doc_id="doc1"),
    ])

    kg2 = KnowledgeGraph.from_dict(kg.to_dict())

    edge = kg2._get_edge_data("a", "b")
    assert len(edge["relations"]) == 2
    rel_types = {r["relation"] for r in edge["relations"]}
    assert rel_types == {"knows", "works with"}


# =============================================================================
# KG24-KG28: Indice invertido para query_by_keywords (DTm-30)
# =============================================================================


def test_keyword_index_populated_on_add():
    """KG24: _kw_entity_index se puebla tras build_keyword_indices (DTm-30/69)."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Albert Einstein", "Physics", "studied")])
    kg.build_keyword_indices()
    # Tokens de entity names deben estar en el indice (stemmed)
    assert "albert" in kg._kw_entity_index
    assert "einstein" in kg._kw_entity_index
    # "physics" se stemiza a "physic"
    stemmed_physics = KnowledgeGraph._tokenize("physics")[0]
    assert stemmed_physics in kg._kw_entity_index
    # Deben apuntar a entidades correctas
    assert "albert einstein" in kg._kw_entity_index["albert"]
    assert "physics" in kg._kw_entity_index[stemmed_physics]


def test_keyword_index_relations():
    """KG25: _kw_relation_index se puebla tras build_keyword_indices (DTm-30/69)."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="MIT", relation="works at",
                   description="employed since 2020", source_doc_id="doc1"),
    ])
    kg.build_keyword_indices()
    stemmed_works = KnowledgeGraph._tokenize("works")[0]
    stemmed_employed = KnowledgeGraph._tokenize("employed")[0]
    assert stemmed_works in kg._kw_relation_index
    assert stemmed_employed in kg._kw_relation_index
    # La entrada debe contener el doc_id
    entries = kg._kw_relation_index[stemmed_works]
    assert any(doc_id == "doc1" for _, _, doc_id in entries)


def test_keyword_index_metadata_update():
    """KG26: Descripcion de metadata se indexa tras build_keyword_indices (DTm-30/69)."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Quantum", "Physics", "branch of")])
    # Actualizar con descripcion que contiene "mechanics"
    kg.add_entity_metadata("Quantum", "CONCEPT", "quantum mechanics theory")
    # Antes de build, indices vacios
    stemmed_mechanics = KnowledgeGraph._tokenize("mechanics")[0]
    assert stemmed_mechanics not in kg._kw_entity_index
    # Despues de build, descripcion indexada
    kg.build_keyword_indices()
    assert stemmed_mechanics in kg._kw_entity_index
    assert "quantum" in kg._kw_entity_index[stemmed_mechanics]


def test_keyword_index_from_dict_roundtrip():
    """KG27: Indices se reconstruyen tras from_dict (DTm-30)."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Newton", "Gravity", "discovered")])
    kg.add_entity_metadata("Newton", "PERSON", "physicist and mathematician")

    kg2 = KnowledgeGraph.from_dict(kg.to_dict())

    # El indice debe estar reconstruido (stemmed tokens)
    assert KnowledgeGraph._tokenize("newton")[0] in kg2._kw_entity_index
    assert KnowledgeGraph._tokenize("gravity")[0] in kg2._kw_entity_index
    assert KnowledgeGraph._tokenize("mathematician")[0] in kg2._kw_entity_index
    # query_by_keywords debe funcionar
    results = kg2.query_by_keywords(["gravity"])
    assert len(results) > 0
    assert results[0][0] == "doc1"


def test_keyword_query_uses_index():
    """KG28: query_by_keywords produce mismos resultados con indice (DTm-30)."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Einstein", target="Relativity",
                   relation="developed theory",
                   description="general relativity", source_doc_id="doc1"),
    ])
    kg.add_triplets("doc2", [
        KGRelation(source="Newton", target="Gravity",
                   relation="discovered law",
                   description="gravitational force", source_doc_id="doc2"),
    ])
    kg.build_keyword_indices()

    # Keyword "relativity" debe encontrar doc1
    results = kg.query_by_keywords(["relativity"])
    doc_ids = [d for d, _ in results]
    assert "doc1" in doc_ids

    # Keyword "gravity" debe encontrar doc2
    results = kg.query_by_keywords(["gravity"])
    doc_ids = [d for d, _ in results]
    assert "doc2" in doc_ids


# =============================================================================
# DTm-70: Fuzzy entity matching in query_entities
# =============================================================================

def test_resolve_entity_names_exact_match():
    """DTm-70: exact match devuelve confidence 1.0."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Albert Einstein", "Physics", "field")])
    kg.build_keyword_indices()

    resolved = kg._resolve_entity_names(["Albert Einstein"])
    assert len(resolved) == 1
    name, confidence = resolved[0]
    assert name == kg._normalize_name("Albert Einstein")
    assert confidence == 1.0


def test_resolve_entity_names_fuzzy_partial_name():
    """DTm-70: nombre parcial matchea via token overlap."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Vlatko Gilić", "Serbia", "nationality")])
    kg.build_keyword_indices()

    # "Vlatko" no existe como entidad exacta, pero matchea via token
    resolved = kg._resolve_entity_names(["Vlatko"])
    names = [n for n, _ in resolved]
    assert kg._normalize_name("Vlatko Gilić") in names


def test_resolve_entity_names_no_match():
    """DTm-70: nombre sin overlap no produce resultados."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Albert Einstein", "Physics", "field")])
    kg.build_keyword_indices()

    resolved = kg._resolve_entity_names(["Completely Unknown Entity"])
    assert len(resolved) == 0


def test_resolve_entity_names_dedup():
    """DTm-70: misma entidad resuelta desde dos keywords no se duplica."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Albert Einstein", "Physics", "field")])
    kg.build_keyword_indices()

    resolved = kg._resolve_entity_names(["Albert Einstein", "Albert Einstein"])
    assert len(resolved) == 1


def test_query_entities_fuzzy_does_bfs():
    """DTm-70: fuzzy match activa BFS y descubre docs via graph traversal."""
    kg = KnowledgeGraph()
    # Doc A: The Surveyor -> Vlatko Gilić
    kg.add_triplets("docA", [_rel("The Surveyor", "Vlatko Gilić", "directed by")])
    # Doc B: Vlatko Gilić -> Serbian (entity merge: Vlatko Gilić in both docs)
    kg.add_triplets("docB", [_rel("Vlatko Gilić", "Serbian", "nationality")])
    kg.build_keyword_indices()

    # Query con nombre parcial "Surveyor" (sin "The", que se stripea)
    # Exact match funciona porque _normalize_name quita "The"
    results = kg.query_entities(["The Surveyor"], max_hops=2, max_docs=10)
    result_dict = dict(results)
    # BFS debe descubrir docB via Vlatko Gilić
    assert "docA" in result_dict
    assert "docB" in result_dict

    # Ahora con un nombre que SOLO funciona por fuzzy match
    results_fuzzy = kg.query_entities(["Gilić"], max_hops=1, max_docs=10)
    result_dict_fuzzy = dict(results_fuzzy)
    # "Gilić" no es entidad exacta, pero fuzzy matchea "vlatko gilić"
    # BFS desde "vlatko gilić": docA (hop 0) + docB (hop 0) + neighbours
    assert "docA" in result_dict_fuzzy or "docB" in result_dict_fuzzy


def test_query_entities_fuzzy_confidence_scales_score():
    """DTm-70: fuzzy match con confidence < 1.0 reduce hop scores."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Vlatko Gilić", "Film", "works in")])
    kg.build_keyword_indices()

    # Exact match: confidence=1.0, hop_score at depth 0 = 1.0
    results_exact = kg.query_entities(["Vlatko Gilić"], max_hops=0, max_docs=10)

    # Fuzzy match: confidence<1.0, hop_score at depth 0 = confidence
    results_fuzzy = kg.query_entities(["Gilić"], max_hops=0, max_docs=10)

    if results_exact and results_fuzzy:
        exact_score = dict(results_exact).get("doc1", 0)
        fuzzy_score = dict(results_fuzzy).get("doc1", 0)
        assert fuzzy_score <= exact_score


# =============================================================================
# query_entities con pre_resolved (DAM-1)
# =============================================================================

def test_query_entities_pre_resolved_skips_string_matching():
    """Con pre_resolved, query_entities salta _resolve_entity_names."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="friendship", source_doc_id="doc1"),
    ])
    kg.build_keyword_indices()

    # Pre-resolved con nombre normalizado que existe
    results = kg.query_entities(
        ["irrelevant"], max_hops=1, max_docs=10,
        pre_resolved=["alice"],
    )
    doc_ids = [d for d, _ in results]
    assert "doc1" in doc_ids


def test_query_entities_pre_resolved_filters_unknown_entities():
    """pre_resolved filtra entidades que no existen en el KG."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="friendship", source_doc_id="doc1"),
    ])
    kg.build_keyword_indices()

    results = kg.query_entities(
        [], max_hops=1, max_docs=10,
        pre_resolved=["nonexistent_entity"],
    )
    assert results == []


def test_get_all_entities_returns_dict():
    """get_all_entities retorna dict de entidades."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="test", source_doc_id="doc1"),
    ])
    entities = kg.get_all_entities()
    assert "alice" in entities
    assert "bob" in entities
    assert entities["alice"].name == "alice"


def test_get_all_relations_returns_list():
    """get_all_relations retorna lista de dicts con weight."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="friendship", source_doc_id="doc1"),
    ])
    kg.add_triplets("doc2", [
        KGRelation(source="Alice", target="Bob", relation="works_with",
                   description="colleagues", source_doc_id="doc2"),
    ])
    relations = kg.get_all_relations()
    assert len(relations) >= 2
    # All relations have required keys
    for rel in relations:
        assert "source" in rel
        assert "target" in rel
        assert "relation" in rel
        assert "doc_id" in rel
        assert "weight" in rel
    # Weight = number of relations on the edge (2: knows + works_with)
    assert any(r["weight"] == 2 for r in relations)


# =============================================================================
# merge_entity_descriptions (DAM-4)
# =============================================================================

def test_merge_entity_descriptions_single_doc():
    """Entidad con 1 descripcion no se modifica."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="test", source_doc_id="doc1"),
    ])
    kg.add_entity_metadata("Alice", "PERSON", "Alice is a researcher")
    merged = kg.merge_entity_descriptions()
    assert merged == 0
    assert kg.get_all_entities()["alice"].description == "Alice is a researcher"


def test_merge_entity_descriptions_multi_doc():
    """Entidad con N descripciones distintas se concatenan."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="test", source_doc_id="doc1"),
    ])
    kg.add_entity_metadata("Alice", "PERSON", "Alice is a researcher")
    kg.add_entity_metadata("Alice", "PERSON", "Alice works at MIT")
    merged = kg.merge_entity_descriptions()
    assert merged == 1
    desc = kg.get_all_entities()["alice"].description
    assert "researcher" in desc
    assert "MIT" in desc


def test_merge_entity_descriptions_dedup():
    """Descripciones duplicadas se deduplican."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="test", source_doc_id="doc1"),
    ])
    kg.add_entity_metadata("Alice", "PERSON", "Alice is a researcher")
    kg.add_entity_metadata("Alice", "PERSON", "Alice is a researcher")
    kg.add_entity_metadata("Alice", "PERSON", "Alice works at MIT")
    merged = kg.merge_entity_descriptions()
    assert merged == 1
    desc = kg.get_all_entities()["alice"].description
    # Should have 2 unique descriptions, not 3
    assert desc.count("|") == 1


def test_merge_entity_descriptions_serialization_roundtrip():
    """_descriptions sobreviven a serialization/deserialization."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        KGRelation(source="Alice", target="Bob", relation="knows",
                   description="test", source_doc_id="doc1"),
    ])
    kg.add_entity_metadata("Alice", "PERSON", "desc1")
    kg.add_entity_metadata("Alice", "PERSON", "desc2")

    data = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(data)
    assert kg2.get_all_entities()["alice"]._descriptions == ["desc1", "desc2"]


# =============================================================================
# F.3: get_entities_for_docs / get_relations_for_docs (DAM-8)
# =============================================================================


def test_get_entities_for_docs_returns_linked_entities():
    """get_entities_for_docs retorna entidades vinculadas a doc_ids."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows")])
    kg.add_triplets("doc2", [_rel("Charlie", "David", "works_with")])

    entities = kg.get_entities_for_docs(["doc1"])
    names = {e["entity"] for e in entities}
    assert "alice" in names
    assert "bob" in names
    assert "charlie" not in names


def test_get_entities_for_docs_deduplicates():
    """Entidades que aparecen en multiples docs no se duplican."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows")])
    kg.add_triplets("doc2", [_rel("Alice", "Charlie", "knows")])

    entities = kg.get_entities_for_docs(["doc1", "doc2"])
    names = [e["entity"] for e in entities]
    assert names.count("alice") == 1


def test_get_entities_for_docs_empty():
    """Doc sin entidades retorna lista vacia."""
    kg = KnowledgeGraph()
    assert kg.get_entities_for_docs(["nonexistent"]) == []


def test_get_relations_for_docs_returns_linked_relations():
    """get_relations_for_docs retorna relaciones vinculadas a doc_ids."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows")])
    kg.add_triplets("doc2", [_rel("Charlie", "David", "works_with")])

    rels = kg.get_relations_for_docs(["doc1"])
    assert len(rels) == 1
    # KGRelation preserva case original del constructor
    assert rels[0]["source"] == "Alice"
    assert rels[0]["target"] == "Bob"
    assert rels[0]["relation"] == "knows"


def test_get_relations_for_docs_deduplicates():
    """Misma relacion en multiples docs no se duplica."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows")])
    kg.add_triplets("doc2", [_rel("Alice", "Bob", "knows")])

    rels = kg.get_relations_for_docs(["doc1", "doc2"])
    # KGRelation.source/target preserva case original
    keys = [(r["source"], r["target"], r["relation"]) for r in rels]
    assert keys.count(("Alice", "Bob", "knows")) == 1


def test_get_relations_for_docs_empty():
    """Doc sin relaciones retorna lista vacia."""
    kg = KnowledgeGraph()
    assert kg.get_relations_for_docs(["nonexistent"]) == []
