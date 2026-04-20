"""
Tests unitarios para KnowledgeGraph.

Cobertura:
  KG1. add_triplets basico — entidades y relaciones se anaden al grafo.
  KG2. add_triplets dedup — misma entidad desde 2 docs -> 1 nodo, 2 source_doc_ids.
  KG3. add_triplets con nombre vacio — tripleta con src/tgt vacio se ignora.
  KG4. add_triplets self-loop — source == target funciona sin error.
  KG5. add_triplets relacion duplicada en arista — se acumula (2 entries).
  KG12. grafo vacio — graph reports zero entities/relations/docs.
  KG14. get_stats — estadisticas correctas.
  KG15. add_entity_metadata — actualiza type y description.
  KG16. max_entities cap con eviction.
  KG17. dedup de relaciones en aristas.
  KG18. get_stats incluye memoria y cap info.
  KG19-KG23. Persistencia: to_dict/from_dict, save/load.
  Co-occurrence bridging.
  merge_entity_descriptions.
  get_entity / get_neighbors_ranked.
  get_all_entities / get_all_relations.

Note: tests for query_entities, query_by_keywords, get_entities_for_docs,
  get_relations_for_docs, _resolve_entity_names, build_keyword_indices,
  and _tokenize were removed — those functions no longer exist in production.
"""

import json

import pytest

igraph = pytest.importorskip("igraph", reason="igraph required for KnowledgeGraph tests")

from shared.retrieval.lightrag.knowledge_graph import KGRelation, KnowledgeGraph


# =============================================================================
# Helpers
# =============================================================================

def _rel(src: str, tgt: str, relation: str = "related to",
         desc: str = "", doc_id: str = "doc1") -> KGRelation:
    return KGRelation(
        source=src, target=tgt, relation=relation,
        description=desc, source_doc_id=doc_id,
    )


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
    alice = kg.get_entity("Alice")
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
#
# White-box intencional: _get_edge_data es un helper privado que inspecciona
# la metadata interna de una arista (dict con "relations" list). No tiene
# equivalente publico porque get_all_relations() devuelve un formato plano.
# Los tests lo usan para asertar acumulacion de relaciones duplicadas sobre
# la misma arista, que es comportamiento interno de add_triplets.
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
# KG13: grafo vacio
# =============================================================================

def test_empty_graph_queries():
    """Grafo vacio reporta ceros en todas las metricas."""
    kg = KnowledgeGraph()
    assert kg.num_entities == 0
    assert kg.num_relations == 0
    assert kg.num_docs == 0
    assert kg.get_all_entities() == {}
    assert kg.get_all_relations() == []


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
    alice = kg.get_entity("Alice")
    assert alice.entity_type == "PERSON"
    assert alice.description == "A researcher"


def test_add_entity_metadata_unknown_entity():
    """Metadata para entidad inexistente no lanza error."""
    kg = KnowledgeGraph()
    kg.add_entity_metadata("Ghost", "PERSON", "Does not exist")
    assert kg.num_entities == 0  # sin cambios


# =============================================================================
# KG16: max_entities cap
# =============================================================================

def test_max_entities_cap_with_eviction():
    """Al alcanzar el cap, entidades de baja importancia se evictan."""
    kg = KnowledgeGraph(max_entities=3)
    # A, B -> 2 entidades
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    assert kg.num_entities == 2

    # C, D -> cap=3, C se acepta (3ra). D necesita evictar: A o B son
    # candidatos (1 doc, degree 1). Uno se evicta, D entra.
    added = kg.add_triplets("doc2", [_rel("C", "D", "r")])
    assert kg.num_entities == 3  # cap respetado
    assert added == 1  # tripleta anadida gracias a eviction
    assert kg.get_stats()["entities_evicted"] >= 1
    # C y D deben estar en el grafo
    entities = kg.get_all_entities()
    assert "c" in entities  # normalized
    assert "d" in entities


def test_max_entities_cap_no_eviction_possible():
    """Sin candidatos evictables, la tripleta se descarta como antes."""
    kg = KnowledgeGraph(max_entities=2)
    # A, B -> 2 entidades, ambos en doc1 y doc2 (multi-doc = no evictable)
    kg.add_triplets("doc1", [_rel("A", "B", "r1")])
    kg.add_triplets("doc2", [_rel("A", "B", "r2")])
    assert kg.num_entities == 2
    # Ahora A y B tienen source_doc_ids={doc1, doc2} -> no evictables

    # C, D -> no caben, no hay candidato evictable -> drop
    added = kg.add_triplets("doc3", [_rel("C", "D", "r")])
    assert kg.num_entities == 2
    assert added == 0
    assert kg.get_stats()["entities_dropped"] > 0


def test_max_entities_existing_entities_still_updated():
    """Entidades existentes se actualizan aunque el cap este lleno."""
    kg = KnowledgeGraph(max_entities=2)
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    assert kg.num_entities == 2

    # A y B ya existen, se actualizan sin problemas
    added = kg.add_triplets("doc2", [_rel("A", "B", "collaborates")])
    assert added == 1
    assert kg.num_entities == 2  # sin entidades nuevas


def test_eviction_prefers_low_doc_count():
    """Eviction elige entidades con menos source_doc_ids."""
    kg = KnowledgeGraph(max_entities=4)
    # A, B en doc1 y doc2 — 2 docs cada uno (no evictables)
    kg.add_triplets("doc1", [_rel("A", "B", "r1")])
    kg.add_triplets("doc2", [_rel("A", "B", "r2")])
    # C, D en doc3 — 1 doc cada uno, degree 1 (evictables)
    kg.add_triplets("doc3", [_rel("C", "D", "r3")])
    assert kg.num_entities == 4

    # E necesita 1 slot nuevo -> C o D evicta (1 doc, degree 1)
    # A necesita A existente (ya existe, no consume slot)
    added = kg.add_triplets("doc4", [_rel("A", "E", "r4")])
    assert added == 1
    assert kg.num_entities == 4  # cap respetado
    entities = kg.get_all_entities()
    assert "a" in entities  # A sobrevive (2 docs)
    assert "b" in entities  # B sobrevive (2 docs)
    assert "e" in entities  # E entro
    # C o D fue evicta (la que el algoritmo eligio)
    evicted = {"c", "d"} - set(entities.keys())
    assert len(evicted) == 1


def test_eviction_cleans_indices():
    """Eviction limpia la entidad del catalogo: tras eviction,
    la entidad evicta deja de aparecer en get_all_entities()."""
    kg = KnowledgeGraph(max_entities=2)
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    assert "a" in kg.get_all_entities()

    # C, D necesitan evictar A o B
    kg.add_triplets("doc2", [_rel("C", "D", "r")])
    # La entidad evicta no debe estar en el catalogo
    entities_all = kg.get_all_entities()
    evicted = {"a", "b"} - set(entities_all.keys())
    assert len(evicted) >= 1
    evicted_name = evicted.pop()
    assert evicted_name not in entities_all


def test_eviction_serialization_roundtrip():
    """entities_evicted se preserva en to_dict/from_dict."""
    kg = KnowledgeGraph(max_entities=2)
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    kg.add_triplets("doc2", [_rel("C", "D", "r")])  # triggers eviction

    data = kg.to_dict()
    assert "entities_evicted" in data
    assert data["entities_evicted"] >= 1

    kg2 = KnowledgeGraph.from_dict(data)
    assert kg2.get_stats()["entities_evicted"] == kg.get_stats()["entities_evicted"]


# =============================================================================
# KG17: dedup de relaciones en aristas
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
# KG18: get_stats incluye memoria y cap info
# =============================================================================

def test_get_stats_includes_memory_and_cap():
    """Stats incluyen approx_memory_mb, entities_dropped, max_entities."""
    kg = KnowledgeGraph(max_entities=100)
    kg.add_triplets("doc1", [_rel("A", "B", "r")])
    stats = kg.get_stats()

    assert "approx_memory_mb" in stats
    assert stats["approx_memory_mb"] >= 0
    assert stats["entities_dropped"] == 0
    assert stats["entities_evicted"] == 0
    assert stats["max_entities"] == 100


# =============================================================================
# KG19-KG23: Persistencia
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

    # Grafo funcional — entidades y stats coinciden
    assert set(kg2.get_all_entities().keys()) == set(kg.get_all_entities().keys())
    assert kg2.get_stats()["num_entities"] == kg.get_stats()["num_entities"]
    assert kg2.get_stats()["num_relations"] == kg.get_stats()["num_relations"]


def test_to_dict_version_field():
    """to_dict incluye campo version. Bumped a 3 por divergencia #10."""
    kg = KnowledgeGraph()
    data = kg.to_dict()
    assert data["version"] == 3


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
    assert data["version"] == 3

    # Load y verificar
    kg2 = KnowledgeGraph.load(path)
    assert kg2.num_entities == 2
    assert kg2.num_relations == 1
    assert kg2._entities["x"].entity_type == "CONCEPT"


def test_from_dict_empty_graph():
    """from_dict con datos minimos produce grafo vacio funcional (v3+ tras #10)."""
    kg = KnowledgeGraph.from_dict({"version": 3})
    assert kg.num_entities == 0
    assert kg.num_relations == 0
    assert kg.get_all_entities() == {}


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
# merge_entity_descriptions
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
# Co-occurrence bridging
# =============================================================================


def test_co_occurrence_bridges_disconnected_components():
    """Entidades del mismo doc se conectan, reduciendo componentes."""
    kg = KnowledgeGraph()
    # Dos componentes desconectados
    kg.add_triplets("doc1", [_rel("A", "B", "r1")])
    kg.add_triplets("doc1", [_rel("C", "D", "r2")])
    # A-B y C-D son 2 componentes (A,B no conectan con C,D via triplet)
    components_before = kg.get_stats()["graph_connected_components"]
    assert components_before == 2

    edges_added = kg.build_co_occurrence_edges()

    # doc1 tiene A, B, C, D -> co-occurrence los conecta
    assert edges_added > 0
    components_after = kg.get_stats()["graph_connected_components"]
    assert components_after < components_before


def test_co_occurrence_no_duplicate_edges():
    """No crea aristas donde ya existe una relacion LLM."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("A", "B", "knows")])
    edges_before = kg.num_relations

    edges_added = kg.build_co_occurrence_edges()

    # A y B ya estan conectados -> no se crea arista nueva
    assert edges_added == 0
    assert kg.num_relations == edges_before


def test_co_occurrence_single_entity_doc_no_edges():
    """Doc con una sola entidad no genera aristas co-occurrence."""
    kg = KnowledgeGraph()
    # Un doc con self-loop entities (solo 1 entidad unica)
    kg.add_triplets("doc1", [_rel("A", "A", "self")])

    edges_added = kg.build_co_occurrence_edges()
    assert edges_added == 0


def test_co_occurrence_caps_pairs_per_doc():
    """Maximo de pares por doc se respeta."""
    kg = KnowledgeGraph()
    # Crear muchas entidades en un solo doc (N entidades -> N*(N-1)/2 pares)
    # Con 10 entidades -> 45 pares posibles
    triplets = []
    for i in range(10):
        name_a = f"E{i}"
        name_b = f"E{i+10}"
        triplets.append(_rel(name_a, name_b, f"r{i}"))
    kg.add_triplets("doc1", triplets)

    edges_added = kg.build_co_occurrence_edges()
    # Cap = 10 pares por doc, y ya hay 10 aristas LLM
    # Pares sin arista: 20 entidades -> muchos pares posibles, capped at 10
    assert edges_added <= kg._MAX_COOCCURRENCE_PAIRS_PER_DOC


# =============================================================================
# get_entity / get_neighbors_ranked (divergencia #9)
# =============================================================================


def test_get_entity_returns_entity():
    """get_entity retorna KGEntity cuando existe."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows")])
    kg.add_entity_metadata("Alice", "PERSON", "A researcher")
    entity = kg.get_entity("Alice")
    assert entity is not None
    assert entity.name == "alice"
    assert entity.entity_type == "PERSON"


def test_get_entity_returns_none_for_missing():
    """get_entity retorna None para entidad inexistente."""
    kg = KnowledgeGraph()
    assert kg.get_entity("nonexistent") is None


def test_get_neighbors_ranked_basic():
    """get_neighbors_ranked retorna vecinos con score y metadata."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("A", "B", "knows", desc="A knows B"),
        _rel("A", "C", "works_with", desc="A works with C"),
    ])
    kg.add_entity_metadata("B", "PERSON", "Engineer")
    kg.add_entity_metadata("C", "PERSON", "Scientist")

    neighbors = kg.get_neighbors_ranked("a", max_neighbors=5)
    assert len(neighbors) == 2
    names = {n["entity"] for n in neighbors}
    assert "b" in names
    assert "c" in names
    for n in neighbors:
        assert "score" in n
        assert n["score"] > 0
        assert "type" in n
        assert "description" in n


def test_get_neighbors_ranked_sort_order():
    """Vecinos con mayor edge_weight + degree se rankean primero."""
    kg = KnowledgeGraph()
    # B tiene degree=1 (solo A-B), C tiene degree=2 (A-C y C-D)
    kg.add_triplets("doc1", [
        _rel("A", "B", "knows"),
        _rel("A", "C", "works_with"),
        _rel("C", "D", "manages"),
    ])

    neighbors = kg.get_neighbors_ranked("a", max_neighbors=5)
    assert len(neighbors) == 2
    # C should rank higher: same edge_weight (1 doc each) but degree(C)=2 > degree(B)=1
    assert neighbors[0]["entity"] == "c"
    assert neighbors[1]["entity"] == "b"
    assert neighbors[0]["score"] > neighbors[1]["score"]


def test_get_neighbors_ranked_max_neighbors():
    """max_neighbors limita el numero de resultados."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [
        _rel("Hub", "N1", "r1"),
        _rel("Hub", "N2", "r2"),
        _rel("Hub", "N3", "r3"),
        _rel("Hub", "N4", "r4"),
        _rel("Hub", "N5", "r5"),
    ])

    neighbors = kg.get_neighbors_ranked("hub", max_neighbors=3)
    assert len(neighbors) == 3


def test_get_neighbors_ranked_unknown_entity():
    """Entidad desconocida retorna lista vacia."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("A", "B", "knows")])
    assert kg.get_neighbors_ranked("nonexistent") == []


def test_get_neighbors_ranked_includes_relation_label():
    """Cada vecino incluye la etiqueta de relacion del edge."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("A", "B", "mentors")])

    neighbors = kg.get_neighbors_ranked("a", max_neighbors=5)
    assert len(neighbors) == 1
    assert neighbors[0]["relation"] == "mentors"


def test_get_neighbors_ranked_edge_weight_from_multiple_docs():
    """Edges con mas docs producen mayor edge_weight y score."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("A", "B", "knows", doc_id="doc1")])
    kg.add_triplets("doc2", [_rel("A", "B", "knows", doc_id="doc2")])
    kg.add_triplets("doc3", [_rel("A", "B", "knows", doc_id="doc3")])
    kg.add_triplets("doc1", [_rel("A", "C", "works_with", doc_id="doc1")])

    neighbors = kg.get_neighbors_ranked("a", max_neighbors=5)
    scores = {n["entity"]: n["score"] for n in neighbors}
    # B has 3 docs on edge -> higher edge_weight than C with 1 doc
    assert scores["b"] > scores["c"]


# =============================================================================
# Divergencia #10: chunk high-level keywords por doc
# =============================================================================

def test_chunk_keywords_add_and_get():
    """#10: add_doc_keywords + get_doc_keywords round-trip."""
    kg = KnowledgeGraph()
    kg.add_doc_keywords("doc1", ["theme one", "theme two"])
    assert kg.get_doc_keywords("doc1") == ["theme one", "theme two"]
    # Doc sin keywords -> lista vacia, no KeyError
    assert kg.get_doc_keywords("doc_missing") == []


def test_chunk_keywords_get_all():
    """#10: get_all_doc_keywords devuelve copia de todos los mappings."""
    kg = KnowledgeGraph()
    kg.add_doc_keywords("d1", ["a"])
    kg.add_doc_keywords("d2", ["b", "c"])

    all_kws = kg.get_all_doc_keywords()
    assert all_kws == {"d1": ["a"], "d2": ["b", "c"]}
    # Es copia: mutar no afecta al KG
    all_kws["d1"].append("mutated")
    assert kg.get_doc_keywords("d1") == ["a"]


def test_chunk_keywords_overwrite_on_reindex():
    """#10: re-indexar un doc sobreescribe sus keywords (ultima extraccion gana)."""
    kg = KnowledgeGraph()
    kg.add_doc_keywords("d1", ["old"])
    kg.add_doc_keywords("d1", ["new1", "new2"])
    assert kg.get_doc_keywords("d1") == ["new1", "new2"]


def test_chunk_keywords_empty_inputs_noop():
    """#10: doc_id o keywords vacios son no-op (no persisten)."""
    kg = KnowledgeGraph()
    kg.add_doc_keywords("", ["theme"])
    kg.add_doc_keywords("d1", [])
    assert kg.get_all_doc_keywords() == {}
    assert kg.num_docs_with_keywords == 0


def test_chunk_keywords_stats_reported():
    """#10: get_stats reporta num_docs_with_keywords y total_chunk_keywords."""
    kg = KnowledgeGraph()
    kg.add_doc_keywords("d1", ["a", "b"])
    kg.add_doc_keywords("d2", ["c"])

    stats = kg.get_stats()
    assert stats["num_docs_with_keywords"] == 2
    assert stats["total_chunk_keywords"] == 3


# =============================================================================
# Serializacion v3 (divergencia #10)
# =============================================================================

def test_cache_version_is_v3():
    """#10: to_dict emite version=3 (gate pre-P0)."""
    kg = KnowledgeGraph()
    data = kg.to_dict()
    assert data["version"] == 3


def test_cache_roundtrip_with_keywords():
    """#10: to_dict/from_dict preserva doc_to_keywords."""
    kg = KnowledgeGraph()
    kg.add_triplets("d1", [_rel("A", "B", "knows", doc_id="d1")])
    kg.add_doc_keywords("d1", ["theme one", "theme two"])
    kg.add_doc_keywords("d2", ["theme three"])

    data = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(data)

    assert kg2.get_doc_keywords("d1") == ["theme one", "theme two"]
    assert kg2.get_doc_keywords("d2") == ["theme three"]
    assert kg2.num_docs_with_keywords == 2


def test_cache_rejects_v2_and_earlier():
    """#10: from_dict rechaza caches v<3 (pre-#10) con mensaje explicito."""
    # Cache v2 minimo pero valido pre-#10
    v2_cache = {
        "version": 2,
        "max_entities": 100_000,
        "entities": {},
        "doc_to_relations": {},
        "entity_to_docs": {},
        "doc_to_entities": {},
        "graph": {"nodes": [], "edges": []},
    }
    with pytest.raises(ValueError, match="KG cache version 2"):
        KnowledgeGraph.from_dict(v2_cache)


def test_cache_rejects_v1():
    """#10: from_dict rechaza caches v1 (legacy NetworkX)."""
    v1_cache = {
        "version": 1,
        "entities": {},
        "graph": {"nodes": [], "links": []},
    }
    with pytest.raises(ValueError, match="KG cache version 1"):
        KnowledgeGraph.from_dict(v1_cache)


def test_cache_save_load_roundtrip(tmp_path):
    """#10: save/load persiste y recupera doc_to_keywords via JSON."""
    kg = KnowledgeGraph()
    kg.add_triplets("d1", [_rel("A", "B", doc_id="d1")])
    kg.add_doc_keywords("d1", ["theme one"])

    cache_path = tmp_path / "kg.json"
    kg.save(cache_path)

    kg2 = KnowledgeGraph.load(cache_path)
    assert kg2.get_doc_keywords("d1") == ["theme one"]
