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
  KG12. get_subgraph_context — genera texto con relaciones.
  KG13. grafo vacio — todas las queries retornan vacio.
  KG14. get_stats — estadisticas correctas.
  KG15. add_entity_metadata — actualiza type y description.
"""

import pytest

from shared.retrieval.knowledge_graph import KGEntity, KGRelation, KnowledgeGraph


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

    edge_data = kg._graph["alice"]["bob"]
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

    results = kg.query_by_keywords(["financial"], max_docs=10)
    result_dict = dict(results)
    assert "doc1" in result_dict


# =============================================================================
# KG12: get_subgraph_context
# =============================================================================

def test_get_subgraph_context():
    """Genera texto con relaciones del subgrafo."""
    kg = KnowledgeGraph()
    kg.add_triplets("doc1", [_rel("Alice", "Bob", "knows")])

    ctx = kg.get_subgraph_context(["Alice"], max_hops=1)
    assert "knows" in ctx
    assert ctx.endswith(".")


def test_get_subgraph_context_unknown_entity():
    """Entidad desconocida -> string vacio."""
    kg = _build_chain_graph()
    assert kg.get_subgraph_context(["NoExiste"]) == ""


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
    assert kg.get_subgraph_context(["A"]) == ""


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
