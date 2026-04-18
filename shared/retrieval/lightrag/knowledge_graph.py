"""
Modulo: Knowledge Graph
Descripcion: Grafo de conocimiento in-memory basado en igraph (C-backed).
             Almacena entidades y relaciones extraidas por LLM.

Ubicacion: shared/retrieval/knowledge_graph.py

Uso: LIGHT_RAG construye el grafo durante indexacion. Durante retrieval,
las entidades resueltas via Entity VDB proveen source_doc_ids para
obtener chunks (paper-aligned, divergencia #8 resuelta opcion A).
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.constants import KG_DEFAULT_MAX_ENTITIES

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTACION CONDICIONAL — igraph
# =============================================================================

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    ig = None  # type: ignore[assignment,unused-ignore]



# =============================================================================
# TIPOS
# =============================================================================

@dataclass
class KGEntity:
    """Entidad extraida por LLM."""
    name: str                       # nombre normalizado
    entity_type: str                # PERSON, ORG, CONCEPT, etc.
    description: str = ""           # descripcion consolidada (DAM-4)
    source_doc_ids: Set[str] = field(default_factory=set)
    _descriptions: List[str] = field(default_factory=list)  # DAM-4: all descriptions


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
    """Knowledge graph in-memory basado en igraph (C-backed).

    Operaciones principales:
      - add_triplets(): construir grafo incrementalmente
      - get_entity() / get_all_entities(): lookup de entidades con source_doc_ids
      - get_neighbors_ranked(): vecinos 1-hop rankeados por peso
    """

    _DEFAULT_MAX_ENTITIES = KG_DEFAULT_MAX_ENTITIES

    def __init__(self, max_entities: int = 0) -> None:
        if not HAS_IGRAPH:
            raise ImportError(
                "igraph no instalado. Instalar: pip install python-igraph"
            )
        self._graph: ig.Graph = ig.Graph()
        # name -> vertex index mapping
        self._name_to_vid: Dict[str, int] = {}
        self._max_entities = max_entities or self._DEFAULT_MAX_ENTITIES
        # Indices invertidos
        self._entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self._doc_to_entities: Dict[str, Set[str]] = defaultdict(set)
        # Entidades con metadata
        self._entities: Dict[str, KGEntity] = {}
        # Relaciones por doc
        self._doc_to_relations: Dict[str, List[KGRelation]] = defaultdict(list)
        # Divergencia #10: chunk high-level keywords por doc_id. Populadas
        # durante indexacion por `_build_knowledge_graph` y consumidas por
        # `_build_chunk_keywords_vdb` para construir la tercera VDB
        # (analoga a entities_vdb y relationships_vdb). Texto libre, sin
        # normalizacion porque el Chunk Keywords VDB hace resolucion
        # semantica via embeddings.
        self._doc_to_keywords: Dict[str, List[str]] = {}
        # Contador de entidades nuevas rechazadas por cap
        self._entities_dropped = 0
        # DTm-49: contador de tripletas completas descartadas por entity cap
        self._triplets_dropped_by_cap = 0
        # DTm-63: contador de entidades evictas por importancia
        self._entities_evicted = 0

    @property
    def num_entities(self) -> int:
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        return int(self._graph.ecount())

    @property
    def num_docs(self) -> int:
        return len(self._doc_to_entities)

    def get_all_entities(self) -> Dict[str, "KGEntity"]:
        """Retorna dict de entity_name -> KGEntity para todas las entidades."""
        return self._entities

    def get_entity(self, name: str) -> Optional["KGEntity"]:
        """Retorna una entidad por nombre, o None si no existe.

        Normaliza el nombre igual que `add_triplets` (lowercase + strip)
        para que el caller pueda pasar la forma original sin conocer
        las reglas internas de normalizacion.
        """
        return self._entities.get(self._normalize_name(name))

    # =================================================================
    # CHUNK KEYWORDS (Divergencia #10)
    # =================================================================

    def add_doc_keywords(self, doc_id: str, keywords: List[str]) -> None:
        """Persiste chunk high-level keywords para un doc.

        Divergencia #10: keywords tematicas por chunk extraidas durante
        indexacion (piggyback en triplet extraction prompt). Usadas
        en retrieval high-level path via Chunk Keywords VDB.

        La validacion (longitud, dedup) se hace upstream en
        `TripletExtractor._build_entities_relations`. Aqui solo persistimos.
        Si un doc_id recibe keywords varias veces (gleaning, re-indexacion)
        se sobreescribe — refleja la ultima extraccion autoritaria.
        """
        if not doc_id or not keywords:
            return
        self._doc_to_keywords[doc_id] = list(keywords)

    def get_doc_keywords(self, doc_id: str) -> List[str]:
        """Retorna la lista de chunk keywords de un doc, o [] si no hay."""
        return list(self._doc_to_keywords.get(doc_id, []))

    def get_all_doc_keywords(self) -> Dict[str, List[str]]:
        """Retorna copia del mapping doc_id -> keywords para todos los docs."""
        return {did: list(kws) for did, kws in self._doc_to_keywords.items()}

    @property
    def num_docs_with_keywords(self) -> int:
        """Numero de docs con al menos una chunk keyword."""
        return len(self._doc_to_keywords)

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """Retorna lista de relaciones unicas con metadata.

        Cada dict contiene: source, target, relation, description,
        doc_id, weight (DAM-5: numero de docs que mencionan esta arista).
        """
        relations = []
        for eid in range(self._graph.ecount()):
            edge = self._graph.es[eid]
            src_name = self._graph.vs[edge.source]["name"]
            tgt_name = self._graph.vs[edge.target]["name"]
            edge_rels = edge["relations"]
            # V.3/DAM-5: weight = numero de docs unicos que mencionan esta arista
            unique_docs = {r.get("doc_id", "") for r in edge_rels if r.get("doc_id")}
            weight = max(len(unique_docs), 1)
            for rel_info in edge_rels:
                relations.append({
                    "source": src_name,
                    "target": tgt_name,
                    "relation": rel_info.get("relation", ""),
                    "description": rel_info.get("description", ""),
                    "doc_id": rel_info.get("doc_id", ""),
                    "weight": weight,
                })
        return relations

    # Articulos iniciales en ingles (DTm-18).
    _LEADING_ARTICLES = ("the ", "a ", "an ")
    # Patron para eliminar puntuacion excepto guiones y apostrofes internos (DTm-18, G.7/DTm-57).
    _RE_NON_ALNUM = re.compile(r"[^\w\s\-']")

    def _normalize_name(self, name: str) -> str:
        """Normalizacion de nombres de entidad (DTm-18)."""
        result = name.lower().strip()
        if not result:
            return ""
        for article in self._LEADING_ARTICLES:
            if result.startswith(article):
                result = result[len(article):]
                break
        result = self._RE_NON_ALNUM.sub("", result)
        result = " ".join(result.split())
        return result.strip()

    # -----------------------------------------------------------------
    # Graph primitive helpers (igraph)
    # -----------------------------------------------------------------

    def _ensure_node(self, name: str, entity_type: str = "UNKNOWN") -> int:
        """Ensure a vertex exists for the given name, return its vertex id."""
        vid = self._name_to_vid.get(name)
        if vid is not None:
            return vid
        vid = self._graph.vcount()
        self._graph.add_vertex(name=name, entity_type=entity_type)
        self._name_to_vid[name] = vid
        return int(vid)

    def _has_node(self, name: str) -> bool:
        return name in self._name_to_vid

    def _get_edge_id(self, src: str, tgt: str) -> Optional[int]:
        """Return edge id between src and tgt, or None."""
        src_vid = self._name_to_vid.get(src)
        tgt_vid = self._name_to_vid.get(tgt)
        if src_vid is None or tgt_vid is None:
            return None
        eid = self._graph.get_eid(src_vid, tgt_vid, error=False)
        return eid if eid >= 0 else None

    def _get_edge_data(self, src: str, tgt: str) -> Optional[Dict[str, Any]]:
        """Return edge attribute dict between src and tgt, or None."""
        eid = self._get_edge_id(src, tgt)
        if eid is None:
            return None
        return {"relations": self._graph.es[eid]["relations"]}

    def _get_neighbors(self, name: str) -> List[str]:
        """Return list of neighbor names for the given node."""
        vid = self._name_to_vid.get(name)
        if vid is None:
            return []
        neighbor_vids = self._graph.neighbors(vid)
        return [self._graph.vs[nv]["name"] for nv in neighbor_vids]

    def _get_neighbors_weighted(self, name: str) -> List[Tuple[str, float]]:
        """Return neighbors with edge weight factor (DTm-72).

        Weight = log(1 + unique_docs_on_edge). Co-occurrence edges (1 doc)
        get ~0.69, strong LLM edges (10 docs) get ~2.40.

        Returns:
            List of (neighbor_name, weight_factor) tuples.
        """
        vid = self._name_to_vid.get(name)
        if vid is None:
            return []
        result: List[Tuple[str, float]] = []
        for eid in self._graph.incident(vid):
            edge = self._graph.es[eid]
            neighbor_vid = edge.target if edge.source == vid else edge.source
            neighbor_name = self._graph.vs[neighbor_vid]["name"]
            # Edge weight = unique docs mentioning this edge
            relations = edge["relations"]
            unique_docs = len({r.get("doc_id", "") for r in relations if r.get("doc_id")})
            weight_factor = math.log1p(max(unique_docs, 1))
            result.append((neighbor_name, weight_factor))
        return result

    def get_neighbors_ranked(
        self, name: str, max_neighbors: int = 5,
    ) -> List[Dict[str, Any]]:
        """1-hop neighbors ranked by edge_weight + degree_centrality.

        Paper-aligned scoring: neighbors with stronger edges AND higher
        connectivity in the graph rank first. Returns metadata for each
        neighbor (name, type, description, relation label, score).
        """
        vid = self._name_to_vid.get(name)
        if vid is None:
            return []

        scored: List[Tuple[float, str, str]] = []
        for eid in self._graph.incident(vid):
            edge = self._graph.es[eid]
            neighbor_vid = edge.target if edge.source == vid else edge.source
            neighbor_name = self._graph.vs[neighbor_vid]["name"]

            relations = edge["relations"]
            unique_docs = len(
                {r.get("doc_id", "") for r in relations if r.get("doc_id")}
            )
            edge_weight = math.log1p(max(unique_docs, 1))
            degree = self._graph.degree(neighbor_vid)
            score = edge_weight + degree

            relation_label = ""
            if relations:
                relation_label = relations[0].get("relation", "")

            scored.append((score, neighbor_name, relation_label))

        scored.sort(key=lambda x: x[0], reverse=True)

        result: List[Dict[str, Any]] = []
        for score, nb_name, rel_label in scored[:max_neighbors]:
            entry: Dict[str, Any] = {
                "entity": nb_name,
                "score": round(score, 3),
            }
            if rel_label:
                entry["relation"] = rel_label
            entity = self._entities.get(nb_name)
            if entity:
                entry["type"] = entity.entity_type
                entry["description"] = entity.description
            result.append(entry)

        return result

    # -----------------------------------------------------------------
    # DTm-63: Eviction por importancia
    # -----------------------------------------------------------------

    def _find_eviction_candidate(
        self, exclude: Optional[set] = None,
    ) -> Optional[str]:
        """Encuentra la entidad menos importante para evictar (A5.3).

        Criterio: score compuesto = n_docs * (degree + 1) * desc_factor.
        Entidades con menor score son evictadas primero.

        Prioridad de eviccion:
          1. Single-doc, leaf (degree 0-1): score minimo
          2. Single-doc, low-degree: score bajo
          3. Multi-doc nunca se evictan (demasiado valiosas)

        Args:
            exclude: Nombres de entidades a excluir (ej: las del triplet actual).

        Returns:
            Nombre de la entidad a evictar, o None si no hay candidato.
        """
        best_name: Optional[str] = None
        best_score = float("inf")
        _exclude = exclude or set()

        for name, entity in self._entities.items():
            if name in _exclude:
                continue
            n_docs = len(entity.source_doc_ids)
            if n_docs > 1:
                continue  # multi-doc entities son demasiado valiosas
            vid = self._name_to_vid.get(name)
            degree = self._graph.degree(vid) if vid is not None else 0
            # Factor de descripcion: entidades con descripcion informativa
            # son mas valiosas (1.0 si vacia, 1 + log(len) si tiene contenido)
            desc_len = len(entity.description)
            desc_factor = 1.0 if desc_len == 0 else (1.0 + (desc_len / 100.0))
            score = n_docs * (degree + 1) * desc_factor
            if score < best_score:
                best_score = score
                best_name = name

        return best_name

    def _evict_entity(self, name: str) -> None:
        """Remueve una entidad del KG (DTm-63).

        Elimina de _entities, _entity_to_docs, _doc_to_entities,
        y las aristas del grafo asociadas. El vertice igraph queda
        huerfano (sin aristas) para evitar reindexacion O(V).
        """
        # Remover aristas del grafo
        vid = self._name_to_vid.get(name)
        if vid is not None:
            # Eliminar todas las aristas conectadas a este nodo
            edge_ids = self._graph.incident(vid)
            if edge_ids:
                self._graph.delete_edges(edge_ids)

        # Limpiar indices
        doc_ids = self._entity_to_docs.pop(name, set())
        for doc_id in doc_ids:
            discard_set = self._doc_to_entities.get(doc_id)
            if discard_set is not None:
                discard_set.discard(name)

        # Remover relaciones asociadas del indice _doc_to_relations
        for doc_id in doc_ids:
            rels = self._doc_to_relations.get(doc_id)
            if rels is not None:
                self._doc_to_relations[doc_id] = [
                    r for r in rels
                    if self._normalize_name(r.source) != name
                    and self._normalize_name(r.target) != name
                ]

        # Remover de _entities (libera el slot para el cap)
        self._entities.pop(name, None)
        self._entities_evicted += 1

    # -----------------------------------------------------------------

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
        dropped_this_call = 0
        for triplet in triplets:
            src_name = self._normalize_name(triplet.source)
            tgt_name = self._normalize_name(triplet.target)

            if not src_name or not tgt_name:
                dropped_this_call += 1
                logger.debug(
                    "Triplet dropped: empty name after normalization "
                    f"(src={triplet.source!r} -> {src_name!r}, "
                    f"tgt={triplet.target!r} -> {tgt_name!r})"
                )
                continue

            # Registrar/actualizar entidades (con cap DTm-21, eviction DTm-63)
            skip_triplet = False
            triplet_names = {src_name, tgt_name}
            for name in (src_name, tgt_name):
                if name not in self._entities:
                    if len(self._entities) >= self._max_entities:
                        # DTm-63: intentar evictar entidad de baja importancia
                        candidate = self._find_eviction_candidate(
                            exclude=triplet_names,
                        )
                        if candidate is not None:
                            self._evict_entity(candidate)
                        else:
                            # Sin candidato evictable — drop como antes
                            self._entities_dropped += 1
                            self._triplets_dropped_by_cap += 1
                            dropped_this_call += 1
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

            # Anadir nodos y arista al grafo
            src_vid = self._ensure_node(src_name, self._entities[src_name].entity_type)
            tgt_vid = self._ensure_node(tgt_name, self._entities[tgt_name].entity_type)

            # Si ya existe arista, acumular relaciones (con dedup DTm-21)
            new_rel = {
                "relation": triplet.relation,
                "description": triplet.description,
                "doc_id": doc_id,
            }
            eid = self._get_edge_id(src_name, tgt_name)
            if eid is not None:
                existing = self._graph.es[eid]["relations"]
                # Dedup: skip si misma relacion del mismo doc ya existe
                is_dup = any(
                    r.get("relation") == triplet.relation
                    and r.get("doc_id") == doc_id
                    for r in existing
                )
                if not is_dup:
                    existing.append(new_rel)
            else:
                self._graph.add_edge(src_vid, tgt_vid, relations=[new_rel])

            self._doc_to_relations[doc_id].append(triplet)
            added += 1

        if dropped_this_call > 0:
            logger.debug(
                f"KnowledgeGraph: {dropped_this_call} tripletas descartadas "
                f"por entity cap en doc '{doc_id}' "
                f"(total acumulado: {self._triplets_dropped_by_cap})"
            )

        return added

    def log_entity_cap_summary(self) -> None:
        """Emite un WARNING resumen si hubo entidades descartadas por cap.

        Debe llamarse una vez al final de la construccion del KG,
        no por cada doc (DTm-49).
        """
        if self._entities_evicted > 0:
            logger.info(
                f"KnowledgeGraph: {self._entities_evicted} entidades evictas "
                f"por baja importancia (DTm-63)"
            )
        if self._triplets_dropped_by_cap > 0:
            logger.warning(
                f"KnowledgeGraph: {self._entities_dropped} entidades nuevas "
                f"rechazadas (cap={self._max_entities}), "
                f"{self._triplets_dropped_by_cap} tripletas descartadas"
            )

    def add_entity_metadata(
        self, name: str, entity_type: str, description: str
    ) -> None:
        """Actualiza metadata de una entidad (tipo, descripcion del LLM).

        DAM-4: Acumula descripciones de multiples docs en _descriptions.
        La descripcion consolidada se genera en merge_entity_descriptions().
        """
        norm = self._normalize_name(name)
        if norm in self._entities:
            if entity_type and entity_type != "UNKNOWN":
                self._entities[norm].entity_type = entity_type
            if description:
                # DAM-4: acumular en vez de sobrescribir
                self._entities[norm]._descriptions.append(description)
                self._entities[norm].description = description  # last-write para compat
            # Actualizar atributo del nodo
            vid = self._name_to_vid.get(norm)
            if vid is not None:
                self._graph.vs[vid]["entity_type"] = self._entities[norm].entity_type

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
                "_descriptions": e._descriptions,  # DAM-4: persist raw descriptions
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

        # Serialize graph as edge list with attributes
        graph_edges = []
        for eid in range(self._graph.ecount()):
            edge = self._graph.es[eid]
            src_name = self._graph.vs[edge.source]["name"]
            tgt_name = self._graph.vs[edge.target]["name"]
            graph_edges.append({
                "source": src_name,
                "target": tgt_name,
                "relations": edge["relations"],
            })

        graph_nodes = []
        for vid in range(self._graph.vcount()):
            v = self._graph.vs[vid]
            graph_nodes.append({
                "name": v["name"],
                "entity_type": v["entity_type"],
            })

        return {
            # v3: divergencia #10 cerrada. Se anade `doc_to_keywords`.
            # Caches v<3 son incompatibles (falta la tercera VDB de chunks);
            # permitir cargarlos ocultaria una parte de la arquitectura.
            "version": 3,
            "max_entities": self._max_entities,
            "entities_dropped": self._entities_dropped,
            "triplets_dropped_by_cap": self._triplets_dropped_by_cap,
            "entities_evicted": self._entities_evicted,
            "entities": entities_ser,
            "doc_to_relations": relations_ser,
            "entity_to_docs": {
                k: sorted(v) for k, v in self._entity_to_docs.items()
            },
            "doc_to_entities": {
                k: sorted(v) for k, v in self._doc_to_entities.items()
            },
            "doc_to_keywords": {
                doc_id: list(kws)
                for doc_id, kws in self._doc_to_keywords.items()
            },
            "graph": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        }

    # Version minima del cache que incluye todos los canales paper-aligned
    # (divergencia #10 cerrada). Caches anteriores carecen de
    # `doc_to_keywords` y por tanto representan una variante arquitecturalmente
    # incompleta del KG — se rechazan explicitamente.
    _MIN_CACHE_VERSION = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Reconstruye un KnowledgeGraph desde un dict serializado.

        Divergencia #10: rechaza caches con `version < _MIN_CACHE_VERSION`
        porque carecen del canal de chunk keywords (representarian una
        arquitectura incompleta). Invalidar cache y re-extraer es la unica
        forma de recuperar el canal.
        """
        version = data.get("version", 1)
        if version < cls._MIN_CACHE_VERSION:
            raise ValueError(
                f"KG cache version {version} < {cls._MIN_CACHE_VERSION}. "
                f"El cache es pre-#10 y no contiene chunk keywords; cargarlo "
                f"produciria un KG arquitecturalmente incompleto. Borra el "
                f"cache o cambia KG_CACHE_DIR para forzar re-extraccion."
            )
        kg = cls(max_entities=data.get("max_entities", 0))
        kg._entities_dropped = data.get("entities_dropped", 0)
        kg._triplets_dropped_by_cap = data.get("triplets_dropped_by_cap", 0)
        kg._entities_evicted = data.get("entities_evicted", 0)

        # Reconstruir entidades
        for name, e_data in data.get("entities", {}).items():
            kg._entities[name] = KGEntity(
                name=e_data["name"],
                entity_type=e_data["entity_type"],
                description=e_data.get("description", ""),
                source_doc_ids=set(e_data.get("source_doc_ids", [])),
                _descriptions=e_data.get("_descriptions", []),  # DAM-4
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

        # Divergencia #10: reconstruir chunk keywords por doc (cache v3+).
        for doc_id, kws in data.get("doc_to_keywords", {}).items():
            if isinstance(kws, list) and kws:
                kg._doc_to_keywords[doc_id] = [str(k) for k in kws if isinstance(k, str)]

        # Reconstruir grafo igraph
        graph_data = data.get("graph")
        if graph_data:
            version = data.get("version", 1)
            if version >= 2:
                # v2 format: explicit nodes + edges lists
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                for node in nodes:
                    kg._ensure_node(node["name"], node.get("entity_type", "UNKNOWN"))
                for edge_info in edges:
                    src = edge_info["source"]
                    tgt = edge_info["target"]
                    src_vid = kg._name_to_vid.get(src)
                    tgt_vid = kg._name_to_vid.get(tgt)
                    if src_vid is not None and tgt_vid is not None:
                        kg._graph.add_edge(src_vid, tgt_vid, relations=edge_info.get("relations", []))
            else:
                # v1 format (legacy NetworkX node_link_data): reconstruct from nodes/links
                for node in graph_data.get("nodes", []):
                    node_name = node.get("id", node.get("name", ""))
                    if node_name:
                        kg._ensure_node(node_name, node.get("entity_type", "UNKNOWN"))
                for link in graph_data.get("links", []):
                    src = link.get("source", "")
                    tgt = link.get("target", "")
                    src_vid = kg._name_to_vid.get(src)
                    tgt_vid = kg._name_to_vid.get(tgt)
                    if src_vid is not None and tgt_vid is not None:
                        kg._graph.add_edge(src_vid, tgt_vid, relations=link.get("relations", []))

        return kg


    # -----------------------------------------------------------------
    # DTm-73: Co-occurrence bridging
    # -----------------------------------------------------------------

    # Maximo de pares co-occurrence por documento para evitar O(N^2)
    _MAX_COOCCURRENCE_PAIRS_PER_DOC = 10

    def build_co_occurrence_edges(self) -> int:
        """Crea aristas entre entidades que co-ocurren en un mismo documento.

        Reduce fragmentacion del grafo (DTm-73) conectando entidades que
        aparecen en el mismo doc pero no tienen relacion explicita del LLM.
        Las aristas se crean con relacion "co-occurs" y peso bajo.

        Debe llamarse despues de add_triplets().

        Returns:
            Numero de aristas co-occurrence creadas.
        """
        edges_added = 0

        for doc_id, entity_names in self._doc_to_entities.items():
            names = [n for n in entity_names if n in self._entities and self._has_node(n)]
            if len(names) < 2:
                continue

            pairs_this_doc = 0
            for i in range(len(names)):
                if pairs_this_doc >= self._MAX_COOCCURRENCE_PAIRS_PER_DOC:
                    break
                for j in range(i + 1, len(names)):
                    if pairs_this_doc >= self._MAX_COOCCURRENCE_PAIRS_PER_DOC:
                        break
                    src, tgt = names[i], names[j]
                    # Solo crear si no existe arista (no sobreescribir relaciones LLM)
                    if self._get_edge_id(src, tgt) is not None:
                        continue

                    src_vid = self._name_to_vid[src]
                    tgt_vid = self._name_to_vid[tgt]
                    self._graph.add_edge(
                        src_vid, tgt_vid,
                        relations=[{
                            "relation": "co-occurs",
                            "description": f"co-occurrence in {doc_id}",
                            "doc_id": doc_id,
                        }],
                    )
                    edges_added += 1
                    pairs_this_doc += 1

        if edges_added > 0:
            components_after = 0
            if self._graph.vcount() > 0:
                components_after = len(self._graph.connected_components())
            logger.info(
                f"KnowledgeGraph: {edges_added} co-occurrence edges added (DTm-73), "
                f"{components_after} connected components"
            )

        return edges_added

    # V.4: max chars para descripcion mergeada. Embedding models truncan
    # a ~512 tokens (~2000 chars). Con "name: description" el budget
    # efectivo es ~500 chars para la parte de descripcion.
    _MAX_MERGED_DESCRIPTION_CHARS = 500
    _MAX_DESCRIPTIONS_TO_MERGE = 5

    def merge_entity_descriptions(self) -> int:
        """Consolida descripciones multi-doc por entidad (DAM-4).

        Para cada entidad con >1 descripcion, deduplica, selecciona las
        mas informativas (por longitud), y concatena con " | " hasta
        _MAX_MERGED_DESCRIPTION_CHARS.

        Referencia: _merge_nodes_then_upsert() en HKUDS/LightRAG.
        El original usa LLM para sintetizar; aqui usamos concatenacion
        con dedup como primer paso (LLM synthesis se puede anadir despues).

        Returns:
            Numero de entidades con descripciones mergeadas.
        """
        merged_count = 0
        for entity in self._entities.values():
            descs = entity._descriptions
            if len(descs) <= 1:
                continue
            # Dedup por contenido (case-insensitive, stripped)
            seen: set = set()
            unique: list = []
            for d in descs:
                key = d.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    unique.append(d.strip())
            if len(unique) <= 1:
                if unique:
                    entity.description = unique[0]
                continue
            # V.4: seleccionar las mas informativas (por longitud) y truncar
            unique.sort(key=len, reverse=True)
            selected = unique[:self._MAX_DESCRIPTIONS_TO_MERGE]
            merged = " | ".join(selected)
            if len(merged) > self._MAX_MERGED_DESCRIPTION_CHARS:
                truncated = merged[:self._MAX_MERGED_DESCRIPTION_CHARS]
                parts = truncated.rsplit(" | ", 1)
                # Si rsplit encontro delimitador, cortar en el ultimo limpio;
                # si no, truncar directamente para no devolver el string completo.
                merged = parts[0] if len(parts) > 1 else truncated
            entity.description = merged
            merged_count += 1
        if merged_count > 0:
            logger.info(
                f"KnowledgeGraph: descripciones mergeadas para "
                f"{merged_count} entidades (multi-doc)"
            )
        return merged_count

    def save(self, path: Path) -> None:
        """Persiste el KG a un archivo JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(
            f"KnowledgeGraph persistido en {path} "
            f"({self.num_entities} entidades, {self.num_relations} relaciones, "
            f"{size_mb:.1f} MB)"
        )
        if size_mb > 100:
            logger.warning(
                f"KG cache file grande: {size_mb:.1f} MB. "
                f"Considerar reducir kg_max_entities o el corpus."
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
        components = 0
        if self._graph.vcount() > 0:
            components = len(self._graph.connected_components())
        total_chunk_keywords = sum(
            len(kws) for kws in self._doc_to_keywords.values()
        )
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_docs_with_entities": self.num_docs,
            "avg_entities_per_doc": (
                sum(len(e) for e in self._doc_to_entities.values()) / self.num_docs
                if self.num_docs > 0
                else 0.0
            ),
            "graph_connected_components": components,
            "entities_dropped": self._entities_dropped,
            "entities_evicted": self._entities_evicted,
            "triplets_dropped_by_cap": self._triplets_dropped_by_cap,
            "max_entities": self._max_entities,
            "approx_memory_mb": round(mem_bytes / (1024 * 1024), 2),
            # Divergencia #10
            "num_docs_with_keywords": self.num_docs_with_keywords,
            "total_chunk_keywords": total_chunk_keywords,
        }


__all__ = [
    "HAS_IGRAPH",
    "KGEntity",
    "KGRelation",
    "KnowledgeGraph",
]
