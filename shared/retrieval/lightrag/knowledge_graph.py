"""
Modulo: Knowledge Graph
Descripcion: Grafo de conocimiento in-memory basado en igraph (C-backed).
             Almacena entidades y relaciones extraidas por LLM,
             soporta traversal multi-hop y busqueda por entidades.

Ubicacion: shared/retrieval/knowledge_graph.py

Uso: LIGHT_RAG construye el grafo durante indexacion y lo consulta
durante retrieval para complementar busqueda vectorial.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTACION CONDICIONAL — igraph
# =============================================================================

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    ig = None  # type: ignore[assignment]

# =============================================================================
# IMPORTACION CONDICIONAL — Snowball Stemmer
# =============================================================================

try:
    import snowballstemmer
    _STEMMER = snowballstemmer.stemmer("english")
    HAS_STEMMER = True
except ImportError:
    _STEMMER = None
    HAS_STEMMER = False


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
      - query_entities(): traversal por entidades especificas (low-level)
      - query_by_keywords(): busqueda por keywords en descripciones (high-level)
    """

    # Cap por defecto de entidades. 0 = sin limite.
    _DEFAULT_MAX_ENTITIES = 100_000  # DTm-63: subido de 50K a 100K

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
        # Contador de entidades nuevas rechazadas por cap
        self._entities_dropped = 0
        # DTm-49: contador de tripletas completas descartadas por entity cap
        self._triplets_dropped_by_cap = 0
        # DTm-63: contador de entidades evictas por importancia
        self._entities_evicted = 0
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
        return self._graph.ecount()

    @property
    def num_docs(self) -> int:
        return len(self._doc_to_entities)

    def get_all_entities(self) -> Dict[str, "KGEntity"]:
        """Retorna dict de entity_name -> KGEntity para todas las entidades."""
        return self._entities

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

    # -----------------------------------------------------------------
    # F.3: Acceso a entidades/relaciones por doc_ids (DAM-8)
    # -----------------------------------------------------------------

    def get_entities_for_docs(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retorna entidades asociadas a una lista de doc_ids.

        Recorre _doc_to_entities para encontrar entidades vinculadas
        a los documentos indicados. Deduplica por nombre.

        Returns:
            Lista de dicts con entity_name, entity_type, description.
        """
        seen: Set[str] = set()
        result: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            for name in self._doc_to_entities.get(doc_id, set()):
                if name in seen:
                    continue
                seen.add(name)
                entity = self._entities.get(name)
                if entity:
                    result.append({
                        "entity": entity.name,
                        "type": entity.entity_type,
                        "description": entity.description,
                    })
        return result

    def get_relations_for_docs(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retorna relaciones cuyos source_doc_id esta en doc_ids.

        Recorre _doc_to_relations para encontrar relaciones vinculadas
        a los documentos indicados. Deduplica por (source, target, relation).

        Returns:
            Lista de dicts con source, target, relation, description.
        """
        seen: Set[Tuple[str, str, str]] = set()
        result: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            for rel in self._doc_to_relations.get(doc_id, []):
                key = (rel.source, rel.target, rel.relation)
                if key in seen:
                    continue
                seen.add(key)
                result.append({
                    "source": rel.source,
                    "target": rel.target,
                    "relation": rel.relation,
                    "description": rel.description,
                })
        return result

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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokeniza texto en palabras lowercase con stemming para el indice invertido."""
        tokens = text.lower().split()
        if _STEMMER is not None:
            try:
                tokens = _STEMMER.stemWords(tokens)
            except Exception as e:
                logger.debug("Stemmer fallo, usando tokens sin stemming: %s", e)
        return tokens

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
        return vid

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

    # -----------------------------------------------------------------
    # DTm-63: Eviction por importancia
    # -----------------------------------------------------------------

    def _find_eviction_candidate(
        self, exclude: Optional[set] = None,
    ) -> Optional[str]:
        """Encuentra la entidad menos importante para evictar.

        Criterio: menor len(source_doc_ids), desempate por menor degree
        en el grafo. Solo evicta entidades con source_doc_ids == 1 y
        degree <= 1 (nodos hoja de baja importancia).

        Args:
            exclude: Nombres de entidades a excluir (ej: las del triplet actual).

        Returns:
            Nombre de la entidad a evictar, o None si no hay candidato.
        """
        best_name: Optional[str] = None
        best_docs = float("inf")
        best_degree = float("inf")
        _exclude = exclude or set()

        for name, entity in self._entities.items():
            if name in _exclude:
                continue
            n_docs = len(entity.source_doc_ids)
            if n_docs > 1:
                continue  # solo considerar entidades de un solo doc
            vid = self._name_to_vid.get(name)
            degree = self._graph.degree(vid) if vid is not None else 0
            if degree > 1:
                continue  # solo considerar nodos hoja o quasi-hoja
            if n_docs < best_docs or (n_docs == best_docs and degree < best_degree):
                best_docs = n_docs
                best_degree = degree
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
            # DTm-69: token indexing diferido a build_keyword_indices()

    def _resolve_entity_names(
        self, entity_names: List[str],
    ) -> List[Tuple[str, float]]:
        """Resuelve nombres de entidad a entidades del KG (DTm-70).

        Intenta exact match primero. Si falla, usa fuzzy matching via
        token overlap con _kw_entity_index. Devuelve lista de
        (entity_name_in_kg, confidence) donde confidence es:
          - 1.0 para exact match
          - tokens_matched/tokens_query para fuzzy match

        Fuzzy match requiere que TODOS los tokens del query name
        aparezcan en el nombre/descripcion de la entidad candidata.
        Esto evita falsos positivos (e.g. "John" matcheando "John Smith"
        Y "John Wayne") a costa de no matchear parciales.
        """
        resolved: List[Tuple[str, float]] = []
        seen: Set[str] = set()

        for name in entity_names:
            norm = self._normalize_name(name)
            if not norm:
                continue

            # Fast path: exact match
            if norm in self._entities and self._has_node(norm):
                if norm not in seen:
                    resolved.append((norm, 1.0))
                    seen.add(norm)
                continue

            # Fuzzy match: buscar entidades via token overlap en el indice
            query_tokens = self._tokenize(norm)
            if not query_tokens:
                continue

            # Recopilar candidatos: entidades que contienen al menos 1 token
            candidate_hits: Counter = Counter()
            for token in query_tokens:
                for entity_name in self._kw_entity_index.get(token, set()):
                    candidate_hits[entity_name] += 1

            # Filtrar: candidatos que matchean TODOS los tokens del query
            n_tokens = len(query_tokens)
            for entity_name, hits in candidate_hits.items():
                if hits >= n_tokens and entity_name not in seen:
                    confidence = hits / max(
                        n_tokens, len(self._tokenize(entity_name))
                    )
                    resolved.append((entity_name, confidence))
                    seen.add(entity_name)

        if resolved:
            n_exact = sum(1 for _, c in resolved if c == 1.0)
            n_fuzzy = len(resolved) - n_exact
            if n_fuzzy > 0:
                logger.debug(
                    f"query_entities: resolved {len(entity_names)} names -> "
                    f"{n_exact} exact + {n_fuzzy} fuzzy matches"
                )

        return resolved

    def query_entities(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        max_docs: int = 20,
        pre_resolved: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Low-level retrieval: busca doc_ids conectados a entidades dadas.

        Traversal BFS hasta max_hops desde cada entidad query.
        Scoring: docs mas cercanos (menos hops) reciben score mas alto.

        Args:
            entity_names: Keywords de entidad de la query.
            max_hops: Profundidad maxima de BFS.
            max_docs: Maximo de docs a retornar.
            pre_resolved: Si proporcionado, entity names ya resueltos por
                VDB similarity search (DAM-1). Salta _resolve_entity_names.
        """
        doc_scores: Counter = Counter()

        if pre_resolved is not None:
            # DAM-1: entidades resueltas externamente via entity VDB
            resolved = [
                (name, 1.0) for name in pre_resolved
                if name in self._entities and self._has_node(name)
            ]
        else:
            resolved = self._resolve_entity_names(entity_names)

        for norm, confidence in resolved:
            # BFS con distancia y peso de arista (DTm-72)
            # Queue: (entity_name, depth, accumulated_weight_factor)
            visited: Set[str] = set()
            queue: deque[Tuple[str, int, float]] = deque([(norm, 0, 1.0)])
            visited.add(norm)

            while queue:
                current, depth, weight_acc = queue.popleft()
                if depth > max_hops:
                    continue

                # Score: distancia inversa * confidence * peso acumulado
                hop_score = (confidence * weight_acc) / (1.0 + depth)

                # Documentos asociados a esta entidad
                for doc_id in self._entity_to_docs.get(current, set()):
                    doc_scores[doc_id] += hop_score

                # Expandir vecinos con peso de arista (DTm-72)
                if depth < max_hops:
                    for neighbor, edge_weight in self._get_neighbors_weighted(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1, edge_weight))

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
            "version": 2,
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
            "graph": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Reconstruye un KnowledgeGraph desde un dict serializado."""
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

        # Reconstruir indices invertidos de keywords (DTm-30)
        kg._rebuild_keyword_indices()

        return kg

    def build_keyword_indices(self) -> None:
        """Construye indices invertidos de keywords para query_by_keywords (DTm-30/69).

        Debe llamarse una vez despues de que todos los add_triplets() hayan
        terminado (fase post-build). Llamar durante add_triplets() por cada
        tripleta era O(n*stemming) entrelazado con I/O — diferirlo reduce
        el tiempo de build del KG.

        Tambien se llama desde from_dict() al cargar desde cache.
        """
        self._rebuild_keyword_indices()

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

        Debe llamarse despues de add_triplets() y antes de build_keyword_indices().

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

    def _rebuild_keyword_indices(self) -> None:
        """Reconstruye _kw_entity_index y _kw_relation_index desde el estado actual."""
        self._kw_entity_index = defaultdict(set)
        self._kw_relation_index = defaultdict(set)

        for entity_name in self._entities:
            self._index_entity_tokens(entity_name)

        for eid in range(self._graph.ecount()):
            edge = self._graph.es[eid]
            src_name = self._graph.vs[edge.source]["name"]
            tgt_name = self._graph.vs[edge.target]["name"]
            for rel_info in edge["relations"]:
                self._index_relation_tokens(src_name, tgt_name, rel_info)

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
                merged = merged[:self._MAX_MERGED_DESCRIPTION_CHARS].rsplit(" | ", 1)[0]
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
        }


__all__ = [
    "HAS_IGRAPH",
    "KGEntity",
    "KGRelation",
    "KnowledgeGraph",
]
