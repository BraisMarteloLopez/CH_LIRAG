"""
Modulo: Knowledge Graph
Descripcion: Grafo de conocimiento in-memory basado en NetworkX.
             Almacena entidades y relaciones extraidas por LLM,
             soporta traversal multi-hop y busqueda por entidades.

Ubicacion: shared/retrieval/knowledge_graph.py

Uso: LIGHT_RAG construye el grafo durante indexacion y lo consulta
durante retrieval para complementar busqueda vectorial.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTACION CONDICIONAL — NetworkX
# =============================================================================

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None  # type: ignore[assignment]


# =============================================================================
# TIPOS
# =============================================================================

@dataclass
class KGEntity:
    """Entidad extraida por LLM."""
    name: str                       # nombre normalizado
    entity_type: str                # PERSON, ORG, CONCEPT, etc.
    description: str = ""           # descripcion del LLM
    source_doc_ids: Set[str] = field(default_factory=set)


@dataclass
class KGRelation:
    """Relacion (tripleta) entre dos entidades."""
    source: str                     # nombre entidad origen (normalizado)
    target: str                     # nombre entidad destino (normalizado)
    relation: str                   # tipo de relacion
    description: str = ""           # descripcion del LLM
    source_doc_id: str = ""         # doc de donde se extrajo


# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

class KnowledgeGraph:
    """Knowledge graph in-memory basado en NetworkX.

    Operaciones principales:
      - add_triplets(): construir grafo incrementalmente
      - query_entities(): traversal por entidades especificas (low-level)
      - query_by_keywords(): busqueda por keywords en descripciones (high-level)
      - get_subgraph_context(): generar texto de contexto del subgrafo
    """

    # Cap por defecto de entidades. 0 = sin limite.
    _DEFAULT_MAX_ENTITIES = 50_000

    def __init__(self, max_entities: int = 0) -> None:
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx no instalado. Instalar: pip install networkx"
            )
        self._graph: nx.Graph = nx.Graph()
        self._max_entities = max_entities or self._DEFAULT_MAX_ENTITIES
        # Indices invertidos
        self._entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self._doc_to_entities: Dict[str, Set[str]] = defaultdict(set)
        # Entidades con metadata
        self._entities: Dict[str, KGEntity] = {}
        # Relaciones por doc
        self._doc_to_relations: Dict[str, List[KGRelation]] = defaultdict(list)
        # Contador de entidades nuevas rechazadas por cap
        self._entities_dropped = 0
        # Indices invertidos para query_by_keywords (DTm-30)
        # token -> set de entity names que contienen ese token
        self._kw_entity_index: Dict[str, Set[str]] = defaultdict(set)
        # token -> set de (src, tgt, doc_id) para relaciones
        self._kw_relation_index: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)

    @property
    def num_entities(self) -> int:
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        return self._graph.number_of_edges()

    @property
    def num_docs(self) -> int:
        return len(self._doc_to_entities)

    def _normalize_name(self, name: str) -> str:
        """Normalizacion simple de nombres de entidad."""
        return name.lower().strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokeniza texto en palabras lowercase para el indice invertido (DTm-30)."""
        return text.lower().split()

    def _index_entity_tokens(self, entity_name: str) -> None:
        """Indexa tokens de nombre y descripcion de una entidad (DTm-30)."""
        entity = self._entities.get(entity_name)
        if not entity:
            return
        tokens = self._tokenize(entity_name)
        if entity.description:
            tokens.extend(self._tokenize(entity.description))
        for token in tokens:
            self._kw_entity_index[token].add(entity_name)

    def _index_relation_tokens(
        self, src: str, tgt: str, rel_info: Dict[str, str]
    ) -> None:
        """Indexa tokens de una relacion (DTm-30)."""
        rel_text = (rel_info.get("relation", "") + " " + rel_info.get("description", ""))
        doc_id = rel_info.get("doc_id", "")
        entry = (src, tgt, doc_id)
        for token in self._tokenize(rel_text):
            self._kw_relation_index[token].add(entry)

    def add_triplets(self, doc_id: str, triplets: List[KGRelation]) -> int:
        """Anade tripletas al grafo desde un documento.

        Deduplica entidades por nombre normalizado. Multiples docs pueden
        contribuir a la misma entidad (merge por union de source_doc_ids).

        Args:
            doc_id: ID del documento fuente.
            triplets: Lista de relaciones extraidas por LLM.

        Returns:
            Numero de tripletas anadidas efectivamente.
        """
        added = 0
        for triplet in triplets:
            src_name = self._normalize_name(triplet.source)
            tgt_name = self._normalize_name(triplet.target)

            if not src_name or not tgt_name:
                continue

            # Registrar/actualizar entidades (con cap DTm-21)
            skip_triplet = False
            for name in (src_name, tgt_name):
                if name not in self._entities:
                    if len(self._entities) >= self._max_entities:
                        self._entities_dropped += 1
                        skip_triplet = True
                        break
                    self._entities[name] = KGEntity(
                        name=name,
                        entity_type="UNKNOWN",
                        source_doc_ids={doc_id},
                    )
                else:
                    self._entities[name].source_doc_ids.add(doc_id)
                self._entity_to_docs[name].add(doc_id)
                self._doc_to_entities[doc_id].add(name)

            if skip_triplet:
                continue

            # Indexar tokens de entidades (DTm-30)
            self._index_entity_tokens(src_name)
            self._index_entity_tokens(tgt_name)

            # Anadir nodos y arista al grafo
            self._graph.add_node(src_name, entity_type=self._entities[src_name].entity_type)
            self._graph.add_node(tgt_name, entity_type=self._entities[tgt_name].entity_type)

            # Si ya existe arista, acumular relaciones (con dedup DTm-21)
            new_rel = {
                "relation": triplet.relation,
                "description": triplet.description,
                "doc_id": doc_id,
            }
            if self._graph.has_edge(src_name, tgt_name):
                existing = self._graph[src_name][tgt_name].get("relations", [])
                # Dedup: skip si misma relacion del mismo doc ya existe
                is_dup = any(
                    r.get("relation") == triplet.relation
                    and r.get("doc_id") == doc_id
                    for r in existing
                )
                if not is_dup:
                    existing.append(new_rel)
                    self._graph[src_name][tgt_name]["relations"] = existing
            else:
                self._graph.add_edge(src_name, tgt_name, relations=[new_rel])

            # Indexar tokens de relacion (DTm-30)
            self._index_relation_tokens(src_name, tgt_name, new_rel)

            self._doc_to_relations[doc_id].append(triplet)
            added += 1

        if self._entities_dropped > 0:
            logger.warning(
                f"KnowledgeGraph: {self._entities_dropped} entidades nuevas "
                f"rechazadas (cap={self._max_entities})"
            )

        return added

    def add_entity_metadata(
        self, name: str, entity_type: str, description: str
    ) -> None:
        """Actualiza metadata de una entidad (tipo, descripcion del LLM)."""
        norm = self._normalize_name(name)
        if norm in self._entities:
            if entity_type and entity_type != "UNKNOWN":
                self._entities[norm].entity_type = entity_type
            if description:
                self._entities[norm].description = description
            # Actualizar atributo del nodo
            if self._graph.has_node(norm):
                self._graph.nodes[norm]["entity_type"] = self._entities[norm].entity_type
            # Re-indexar tokens con nueva descripcion (DTm-30)
            self._index_entity_tokens(norm)

    def query_entities(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        max_docs: int = 20,
    ) -> List[Tuple[str, float]]:
        """Low-level retrieval: busca doc_ids conectados a entidades dadas.

        Traversal BFS hasta max_hops desde cada entidad query.
        Scoring: docs mas cercanos (menos hops) reciben score mas alto.

        Args:
            entity_names: Nombres de entidades a buscar.
            max_hops: Profundidad maxima de traversal.
            max_docs: Numero maximo de doc_ids a devolver.

        Returns:
            Lista de (doc_id, score) ordenada por score descendente.
        """
        doc_scores: Counter = Counter()

        for name in entity_names:
            norm = self._normalize_name(name)
            if norm not in self._entities or not self._graph.has_node(norm):
                continue

            # BFS con distancia (DTm-29: deque.popleft() en lugar de list.pop(0))
            visited: Set[str] = set()
            queue: deque[Tuple[str, int]] = deque([(norm, 0)])
            visited.add(norm)

            while queue:
                current, depth = queue.popleft()
                if depth > max_hops:
                    continue

                # Score inversamente proporcional a la distancia
                hop_score = 1.0 / (1.0 + depth)

                # Documentos asociados a esta entidad
                for doc_id in self._entity_to_docs.get(current, set()):
                    doc_scores[doc_id] += hop_score

                # Expandir vecinos
                if depth < max_hops:
                    for neighbor in self._graph.neighbors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))

        # Ordenar por score y limitar
        ranked = doc_scores.most_common(max_docs)
        return ranked

    def query_by_keywords(
        self,
        keywords: List[str],
        max_docs: int = 20,
    ) -> List[Tuple[str, float]]:
        """High-level retrieval: busca docs por keywords en nombres de entidad
        y descripciones de relaciones.

        Usa indice invertido por token (DTm-30) en lugar de scan completo.
        Complejidad: O(tokens_keyword × matches) en vez de O(entidades × keywords).

        Args:
            keywords: Temas/keywords de alto nivel.
            max_docs: Numero maximo de doc_ids a devolver.

        Returns:
            Lista de (doc_id, score) ordenada por score descendente.
        """
        doc_scores: Counter = Counter()
        keywords_lower = [k.lower().strip() for k in keywords if k.strip()]

        if not keywords_lower:
            return []

        # Buscar entidades via indice invertido (DTm-30)
        matched_entities: Set[str] = set()
        for kw in keywords_lower:
            kw_tokens = self._tokenize(kw)
            for token in kw_tokens:
                matched_entities.update(self._kw_entity_index.get(token, set()))

        for entity_name in matched_entities:
            entity = self._entities.get(entity_name)
            if not entity:
                continue
            for doc_id in entity.source_doc_ids:
                doc_scores[doc_id] += 1.0

        # Buscar relaciones via indice invertido (DTm-30)
        matched_relations: Set[Tuple[str, str, str]] = set()
        for kw in keywords_lower:
            kw_tokens = self._tokenize(kw)
            for token in kw_tokens:
                matched_relations.update(self._kw_relation_index.get(token, set()))

        for _src, _tgt, doc_id in matched_relations:
            if doc_id:
                doc_scores[doc_id] += 0.5

        return doc_scores.most_common(max_docs)

    def get_subgraph_context(
        self,
        entity_names: List[str],
        max_hops: int = 1,
        max_relations: int = 20,
    ) -> str:
        """Genera texto de contexto del subgrafo para enriquecer la query.

        Recorre el subgrafo alrededor de las entidades dadas y genera
        texto con las relaciones encontradas.

        Returns:
            Texto descriptivo de relaciones relevantes.
        """
        lines: List[str] = []
        seen_edges: Set[Tuple[str, str]] = set()

        for name in entity_names:
            norm = self._normalize_name(name)
            if not self._graph.has_node(norm):
                continue

            # Subgrafo local
            ego = nx.ego_graph(self._graph, norm, radius=max_hops)
            for src, tgt, data in ego.edges(data=True):
                edge_key = (min(src, tgt), max(src, tgt))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                for rel_info in data.get("relations", []):
                    rel = rel_info.get("relation", "related to")
                    lines.append(f"{src} {rel} {tgt}")

                    if len(lines) >= max_relations:
                        break
                if len(lines) >= max_relations:
                    break
            if len(lines) >= max_relations:
                break

        return ". ".join(lines) + "." if lines else ""

    # =================================================================
    # SERIALIZACION (DTm-34)
    # =================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el estado completo del KG a un dict JSON-compatible."""
        entities_ser = {
            name: {
                "name": e.name,
                "entity_type": e.entity_type,
                "description": e.description,
                "source_doc_ids": sorted(e.source_doc_ids),
            }
            for name, e in self._entities.items()
        }

        relations_ser: Dict[str, List[Dict[str, str]]] = {
            doc_id: [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "description": r.description,
                    "source_doc_id": r.source_doc_id,
                }
                for r in rels
            ]
            for doc_id, rels in self._doc_to_relations.items()
        }

        graph_data = nx.node_link_data(self._graph)

        return {
            "version": 1,
            "max_entities": self._max_entities,
            "entities_dropped": self._entities_dropped,
            "entities": entities_ser,
            "doc_to_relations": relations_ser,
            "entity_to_docs": {
                k: sorted(v) for k, v in self._entity_to_docs.items()
            },
            "doc_to_entities": {
                k: sorted(v) for k, v in self._doc_to_entities.items()
            },
            "graph": graph_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Reconstruye un KnowledgeGraph desde un dict serializado."""
        kg = cls(max_entities=data.get("max_entities", 0))
        kg._entities_dropped = data.get("entities_dropped", 0)

        # Reconstruir entidades
        for name, e_data in data.get("entities", {}).items():
            kg._entities[name] = KGEntity(
                name=e_data["name"],
                entity_type=e_data["entity_type"],
                description=e_data.get("description", ""),
                source_doc_ids=set(e_data.get("source_doc_ids", [])),
            )

        # Reconstruir indices
        for name, doc_ids in data.get("entity_to_docs", {}).items():
            kg._entity_to_docs[name] = set(doc_ids)
        for doc_id, entity_names in data.get("doc_to_entities", {}).items():
            kg._doc_to_entities[doc_id] = set(entity_names)

        # Reconstruir relaciones por doc
        for doc_id, rels_data in data.get("doc_to_relations", {}).items():
            kg._doc_to_relations[doc_id] = [
                KGRelation(
                    source=r["source"],
                    target=r["target"],
                    relation=r["relation"],
                    description=r.get("description", ""),
                    source_doc_id=r.get("source_doc_id", ""),
                )
                for r in rels_data
            ]

        # Reconstruir grafo NetworkX
        graph_data = data.get("graph")
        if graph_data:
            kg._graph = nx.node_link_graph(graph_data)

        # Reconstruir indices invertidos de keywords (DTm-30)
        kg._rebuild_keyword_indices()

        return kg

    def _rebuild_keyword_indices(self) -> None:
        """Reconstruye _kw_entity_index y _kw_relation_index desde el estado actual (DTm-30)."""
        self._kw_entity_index = defaultdict(set)
        self._kw_relation_index = defaultdict(set)

        for entity_name in self._entities:
            self._index_entity_tokens(entity_name)

        for src, tgt, edge_data in self._graph.edges(data=True):
            for rel_info in edge_data.get("relations", []):
                self._index_relation_tokens(src, tgt, rel_info)

    def save(self, path: Path) -> None:
        """Persiste el KG a un archivo JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(
            f"KnowledgeGraph persistido en {path} "
            f"({self.num_entities} entidades, {self.num_relations} relaciones)"
        )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        """Carga un KG desde un archivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kg = cls.from_dict(data)
        logger.info(
            f"KnowledgeGraph cargado desde {path} "
            f"({kg.num_entities} entidades, {kg.num_relations} relaciones)"
        )
        return kg

    def _estimate_memory_bytes(self) -> int:
        """Estimacion aproximada del uso de memoria del grafo."""
        size = sys.getsizeof(self._graph)
        size += sys.getsizeof(self._entities)
        for e in self._entities.values():
            size += sys.getsizeof(e.name) + sys.getsizeof(e.description)
            size += sys.getsizeof(e.source_doc_ids)
        size += sys.getsizeof(self._entity_to_docs)
        size += sys.getsizeof(self._doc_to_entities)
        size += sys.getsizeof(self._doc_to_relations)
        return size

    def get_stats(self) -> Dict[str, Any]:
        """Estadisticas del grafo para logging."""
        mem_bytes = self._estimate_memory_bytes()
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_docs_with_entities": self.num_docs,
            "avg_entities_per_doc": (
                sum(len(e) for e in self._doc_to_entities.values()) / self.num_docs
                if self.num_docs > 0
                else 0.0
            ),
            "graph_connected_components": (
                nx.number_connected_components(self._graph)
                if self._graph.number_of_nodes() > 0
                else 0
            ),
            "entities_dropped": self._entities_dropped,
            "max_entities": self._max_entities,
            "approx_memory_mb": round(mem_bytes / (1024 * 1024), 2),
        }


__all__ = [
    "HAS_NETWORKX",
    "KGEntity",
    "KGRelation",
    "KnowledgeGraph",
]
