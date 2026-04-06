"""
Modulo: LightRAG Retriever
Descripcion: Retriever basado en LightRAG (EMNLP 2025).
             Combina busqueda vectorial con knowledge graph.
             Sin BM25 — el grafo reemplaza la funcion de bridging lexical.

Ubicacion: shared/retrieval/lightrag_retriever.py

Flujo:
    Indexacion:
      1. Extraer tripletas de cada doc via LLM (TripletExtractor)
      2. Construir KnowledgeGraph con entidades y relaciones
      3. Indexar contenido original en ChromaDB (vector search)

    Retrieval:
      1. Vector search (ChromaDB) -> top_k candidatos
      2. Query analysis via LLM -> low_level + high_level keywords
      3. Graph traversal (low-level: entity BFS, high-level: keyword match)
      4. Fusion de resultados vector + graph
      5. Contenido original para generacion
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.llm import AsyncLLMService, run_sync
from shared.types import EmbeddingModelProtocol

from ..core import (
    BaseRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
    SimpleVectorRetriever,
    reciprocal_rank_fusion,
)
from .knowledge_graph import KnowledgeGraph, KGRelation, HAS_IGRAPH
from .triplet_extractor import TripletExtractor

logger = logging.getLogger(__name__)


class LightRAGRetriever(BaseRetriever):
    """Retriever LightRAG: Vector + Knowledge Graph dual-level.

    Requiere:
      - AsyncLLMService para extraccion de tripletas y query analysis
      - EmbeddingModel para busqueda vectorial (ChromaDB)
      - igraph para knowledge graph

    Sin LLM service: fallback a SimpleVectorRetriever puro.
    Sin igraph: fallback a SimpleVectorRetriever puro.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        llm_service: Optional[AsyncLLMService] = None,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
        kg_max_hops: int = 2,
        kg_max_text_chars: int = 3000,
        kg_max_entities: int = 0,
        graph_weight: float = 0.3,
        vector_weight: float = 0.7,
        kg_cache_dir: str = "",
    ):
        super().__init__(config)
        self._llm_service = llm_service
        self._kg_max_hops = kg_max_hops
        self._kg_max_text_chars = kg_max_text_chars
        self._kg_max_entities = kg_max_entities
        self._graph_weight = graph_weight
        self._vector_weight = vector_weight
        self._kg_cache_dir = Path(kg_cache_dir) if kg_cache_dir else None
        self._kg_fusion_method = config.kg_fusion_method
        self._kg_rrf_k = config.kg_rrf_k
        self._kg_keyword_max_tokens = config.kg_keyword_max_tokens
        self._kg_extraction_max_tokens = config.kg_extraction_max_tokens
        self._kg_batch_docs_per_call = config.kg_batch_docs_per_call
        self._GRAPH_OVERFETCH_FACTOR = config.kg_graph_overfetch_factor
        self._kg_gleaning_rounds = config.kg_gleaning_rounds
        self._lightrag_mode = config.lightrag_mode

        # Vector retriever (siempre disponible)
        self._vector_retriever = SimpleVectorRetriever(
            config, embedding_model, collection_name,
            embedding_batch_size=embedding_batch_size,
        )

        # Knowledge graph + triplet extractor (requieren LLM + igraph)
        self._kg: Optional[KnowledgeGraph] = None
        self._extractor: Optional[TripletExtractor] = None
        self._has_graph = False

        if llm_service and HAS_IGRAPH:
            self._kg = KnowledgeGraph(max_entities=kg_max_entities)
            self._extractor = TripletExtractor(
                llm_service,
                max_text_chars=kg_max_text_chars,
                keyword_max_tokens=self._kg_keyword_max_tokens,
                extraction_max_tokens=self._kg_extraction_max_tokens,
            )
        else:
            reasons = []
            if not llm_service:
                reasons.append("LLM service no proporcionado")
            if not HAS_IGRAPH:
                reasons.append("igraph no instalado")
            logger.warning(
                f"LIGHT_RAG: sin knowledge graph ({', '.join(reasons)}). "
                f"Fallback a SimpleVector."
            )

        # Entity VDB (DAM-1) y Relationship VDB (DAM-2): ChromaDB collections
        # para resolver entidades/relaciones por embedding similarity.
        self._entities_vdb: Any = None  # ChromaVectorStore, lazy init
        self._relationships_vdb: Any = None  # ChromaVectorStore, lazy init
        self._embedding_model = embedding_model

        # Cache de keywords de queries con limite (DTm-47) y lock (DTm-51)
        self._QUERY_CACHE_MAX_SIZE = 10_000
        self._query_keywords_cache: OrderedDict[str, Tuple[List[str], List[str]]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        """Indexa documentos: vector index + knowledge graph.

        1. Indexa en ChromaDB (contenido original, embeddings limpios)
        2. Extrae tripletas via LLM (batch async)
        3. Construye knowledge graph
        """
        if not documents:
            logger.warning("index_documents llamado con lista vacia")
            return False

        start_time = time.perf_counter()
        logger.info(
            f"LightRAGRetriever: indexando {len(documents)} documentos..."
        )

        # Paso 1: Vector index (siempre)
        vector_ok = self._vector_retriever.index_documents(
            documents, collection_name
        )
        if not vector_ok:
            logger.error("LightRAGRetriever: fallo indexacion vectorial")
            return False

        # Paso 2: Knowledge graph (si disponible)
        if self._kg and self._extractor:
            # G.1/DTm-55: snapshot stats antes de build para restaurar si falla
            stats_snapshot = self._extractor.get_stats()
            try:
                cache_path = self._resolve_cache_path(documents)
                if cache_path and cache_path.exists():
                    self._kg = KnowledgeGraph.load(cache_path)
                    logger.info(
                        f"LightRAGRetriever: KG cargado desde cache "
                        f"({self._kg.num_entities} entidades)"
                    )
                    # DAM-1/DAM-2: rebuild VDBs from cached KG
                    self._build_entities_vdb()
                    self._build_relationships_vdb()
                else:
                    self._build_knowledge_graph(documents)
                    if cache_path:
                        self._kg.save(cache_path)
                self._has_graph = True
            except Exception as e:
                logger.error(
                    f"LightRAGRetriever: error construyendo KG: {e}. "
                    f"Continuando solo con vector search."
                )
                self._has_graph = False
                # G.1/DTm-55: restaurar stats previos al fallo
                if self._extractor:
                    self._extractor.reset_stats()
                    for k, v in stats_snapshot.items():
                        self._extractor._stats[k] = v

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._is_indexed = True

        kg_stats = self._kg.get_stats() if self._kg and self._has_graph else {}
        logger.info(
            f"LightRAGRetriever: indexacion {elapsed_ms:.0f}ms. "
            f"Graph: {self._has_graph}. KG stats: {kg_stats}"
        )

        # DTm-26: alerta si el cap de entidades descarto un porcentaje significativo
        if kg_stats:
            dropped = kg_stats.get("entities_dropped", 0)
            kept = kg_stats.get("num_entities", 0)
            total_seen = kept + dropped
            if dropped > 0 and total_seen > 0:
                pct = dropped / total_seen * 100
                msg = (
                    f"KG entity cap activo: {dropped}/{total_seen} entidades "
                    f"descartadas ({pct:.1f}%). cap={kg_stats.get('max_entities')}. "
                    f"Documentos indexados tarde tendran KG incompleto."
                )
                if pct > 10:
                    logger.error(f"DTm-26 ALERT: {msg}")
                else:
                    logger.warning(f"DTm-26: {msg}")
        return True

    def _build_knowledge_graph(
        self, documents: List[Dict[str, Any]]
    ) -> None:
        """Construye el knowledge graph extrayendo tripletas via LLM."""
        assert self._extractor is not None
        assert self._kg is not None

        t0 = time.perf_counter()

        # Extraccion batch (async, controlada por semaphore del LLM)
        extraction_results = self._extractor.extract_batch(
            documents, batch_docs_per_call=self._kg_batch_docs_per_call,
        )

        # Construir grafo
        total_triplets = 0
        for doc_id, (entities, relations) in extraction_results.items():
            # Anadir tripletas
            added = self._kg.add_triplets(doc_id, relations)
            total_triplets += added

            # Actualizar metadata de entidades
            for entity in entities:
                self._kg.add_entity_metadata(
                    entity.name, entity.entity_type, entity.description
                )

        # DAM-6: gleaning — re-extraccion para capturar entidades perdidas
        if self._kg_gleaning_rounds > 0 and self._extractor:
            for gleaning_round in range(self._kg_gleaning_rounds):
                t_glean = time.perf_counter()
                gleaning_count = 0
                for doc in documents:
                    doc_id = doc.get("doc_id", "")
                    text = doc.get("content", "")
                    prev_entities = extraction_results.get(doc_id, ([], []))[0]
                    if not prev_entities:
                        continue
                    new_entities, new_relations = run_sync(
                        self._extractor.glean_from_doc_async(
                            doc_id, text, prev_entities,
                        )
                    )
                    if new_relations:
                        added = self._kg.add_triplets(doc_id, new_relations)
                        total_triplets += added
                        gleaning_count += added
                    for entity in new_entities:
                        self._kg.add_entity_metadata(
                            entity.name, entity.entity_type, entity.description
                        )
                glean_ms = (time.perf_counter() - t_glean) * 1000
                logger.info(
                    f"LightRAGRetriever: gleaning round {gleaning_round + 1} "
                    f"en {glean_ms:.0f}ms, +{gleaning_count} tripletas"
                )

        # DTm-49: emitir resumen una sola vez al final
        self._kg.log_entity_cap_summary()

        # DTm-69: construir indices invertidos de keywords en fase post-build
        # (antes se hacia por cada tripleta, entrelazado con I/O del grafo)
        t_idx = time.perf_counter()
        self._kg.build_keyword_indices()
        idx_ms = (time.perf_counter() - t_idx) * 1000
        logger.info(f"LightRAGRetriever: keyword indices construidos en {idx_ms:.0f}ms")

        # DAM-4: merge descripciones multi-doc antes de construir VDBs
        merged = self._kg.merge_entity_descriptions()
        if merged:
            logger.info(f"LightRAGRetriever: {merged} entidades con descripciones consolidadas")

        # DAM-1: construir entity VDB (ChromaDB con embeddings de entidades)
        self._build_entities_vdb()
        # DAM-2: construir relationship VDB (ChromaDB con embeddings de relaciones)
        self._build_relationships_vdb()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"LightRAGRetriever: KG construido en {elapsed_ms:.0f}ms. "
            f"{total_triplets} tripletas de {len(documents)} docs."
        )

    def _build_entities_vdb(self) -> None:
        """Construye entity VDB: ChromaDB collection con embeddings de entidades.

        Cada entidad se indexa como "entity_name: description" para que
        similarity search encuentre entidades semanticamente relevantes
        a la query, sin depender de string matching (DAM-1).

        Referencia: entities_vdb.query() en HKUDS/LightRAG operate.py.
        """
        if self._kg is None:
            return

        entities = self._kg.get_all_entities()
        if not entities:
            logger.warning("LightRAGRetriever: KG sin entidades, entity VDB no construido")
            return

        t0 = time.perf_counter()

        from shared.vector_store import ChromaVectorStore
        from langchain_core.documents import Document

        # Collection name: {base}_entities para no colisionar con docs
        base_name = self._vector_retriever._vector_store.collection_name
        entity_collection_name = f"{base_name}_entities"

        # Limpiar VDB anterior si existe
        if self._entities_vdb is not None:
            try:
                self._entities_vdb.delete_all_documents()
            except Exception as e:
                logger.debug("Error limpiando entity VDB (no fatal): %s", e)

        self._entities_vdb = ChromaVectorStore(
            config={
                "CHROMA_COLLECTION_NAME": entity_collection_name,
                "EMBEDDING_BATCH_SIZE": 50,
                "HNSW_SPACE": "cosine",  # V.1: cosine distance for semantic matching
            },
            embedding_model=self._embedding_model,
        )

        # Indexar entidades como "name: description"
        lc_docs = []
        for entity_name, entity in entities.items():
            desc = entity.description or ""
            text = f"{entity.name}: {desc}" if desc else entity.name
            lc_docs.append(Document(
                page_content=text,
                metadata={
                    "entity_name": entity_name,
                    "entity_type": entity.entity_type,
                },
            ))

        self._entities_vdb.add_documents(lc_docs)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"LightRAGRetriever: entity VDB construido en {elapsed_ms:.0f}ms "
            f"({len(lc_docs)} entidades indexadas)"
        )

    # V.2: Max cosine distance to consider a match.
    # Cosine distance range: [0, 2]. 0 = identical, 1 = orthogonal, 2 = opposite.
    # 0.8 is permissive: allows semantically related but lexically different matches.
    _ENTITY_VDB_MAX_DISTANCE = 0.8

    def _resolve_entities_via_vdb(
        self,
        keywords: List[str],
        top_k: int = 10,
    ) -> List[str]:
        """Resuelve keywords de query a entity names via embedding similarity.

        Reemplaza _resolve_entity_names (string matching) con semantic search
        contra entities_vdb (DAM-1).

        V.2: Filtra resultados por threshold de cosine distance para evitar
        matches irrelevantes.

        Returns:
            Lista de entity names (normalizados) encontrados en el KG.
        """
        if not self._entities_vdb or not keywords:
            return []

        resolved_names: List[str] = []
        seen: set = set()

        for keyword in keywords:
            if not keyword.strip():
                continue
            results = self._entities_vdb.similarity_search_with_score(
                keyword, k=top_k,
            )
            for doc, distance in results:
                # V.2: filter by cosine distance threshold
                if distance > self._ENTITY_VDB_MAX_DISTANCE:
                    continue
                entity_name = doc.metadata.get("entity_name", "")
                if entity_name and entity_name not in seen:
                    seen.add(entity_name)
                    resolved_names.append(entity_name)

        return resolved_names

    def _build_relationships_vdb(self) -> None:
        """Construye relationship VDB: ChromaDB collection con embeddings de relaciones.

        Cada relacion se indexa como "source -> relation -> target: description"
        para que similarity search encuentre relaciones semanticamente relevantes
        a la query (DAM-2). Edge weight se almacena en metadata.

        Referencia: relationships_vdb.query() en HKUDS/LightRAG operate.py.
        """
        if self._kg is None:
            return

        relations = self._kg.get_all_relations()
        if not relations:
            logger.warning("LightRAGRetriever: KG sin relaciones, relationship VDB no construido")
            return

        t0 = time.perf_counter()

        from shared.vector_store import ChromaVectorStore
        from langchain_core.documents import Document

        base_name = self._vector_retriever._vector_store.collection_name
        rel_collection_name = f"{base_name}_relationships"

        if self._relationships_vdb is not None:
            try:
                self._relationships_vdb.delete_all_documents()
            except Exception as e:
                logger.debug("Error limpiando relationship VDB (no fatal): %s", e)

        self._relationships_vdb = ChromaVectorStore(
            config={
                "CHROMA_COLLECTION_NAME": rel_collection_name,
                "EMBEDDING_BATCH_SIZE": 50,
                "HNSW_SPACE": "cosine",  # V.1: cosine distance for semantic matching
            },
            embedding_model=self._embedding_model,
        )

        # Indexar relaciones como "source -> relation -> target: description"
        lc_docs = []
        for rel in relations:
            desc = rel.get("description", "")
            relation_type = rel.get("relation", "")
            source = rel.get("source", "")
            target = rel.get("target", "")
            text = f"{source} -> {relation_type} -> {target}"
            if desc:
                text += f": {desc}"
            lc_docs.append(Document(
                page_content=text,
                metadata={
                    "source_entity": source,
                    "target_entity": target,
                    "relation": relation_type,
                    "doc_id": rel.get("doc_id", ""),
                    "weight": rel.get("weight", 1),
                },
            ))

        self._relationships_vdb.add_documents(lc_docs)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"LightRAGRetriever: relationship VDB construido en {elapsed_ms:.0f}ms "
            f"({len(lc_docs)} relaciones indexadas)"
        )

    # V.6: Max cosine distance for relationship matches (same scale as entity VDB).
    _RELATIONSHIP_VDB_MAX_DISTANCE = 0.8

    def _resolve_relationships_via_vdb(
        self,
        keywords: List[str],
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """Resuelve high-level keywords a doc_ids via relationship VDB.

        Busca relaciones semanticamente similares a los keywords y retorna
        los doc_ids asociados con scores ponderados por edge weight (DAM-2, DAM-5).

        V.6: Cosine distance [0, 2] → similarity [1.0, 0.0] via (1 - d/2).
        Filtra por threshold antes de acumular scores.

        Returns:
            Lista de (doc_id, score) ordenada por score desc.
        """
        if not self._relationships_vdb or not keywords:
            return []

        from collections import Counter
        doc_scores: Counter = Counter()

        for keyword in keywords:
            if not keyword.strip():
                continue
            results = self._relationships_vdb.similarity_search_with_score(
                keyword, k=top_k,
            )
            for doc, distance in results:
                # V.6: filter by cosine distance threshold
                if distance > self._RELATIONSHIP_VDB_MAX_DISTANCE:
                    continue
                doc_id = doc.metadata.get("doc_id", "")
                weight = doc.metadata.get("weight", 1)
                if doc_id:
                    # V.6: cosine distance [0,2] -> similarity [1.0, 0.0]
                    similarity = max(0.0, 1.0 - distance / 2.0)
                    doc_scores[doc_id] += similarity * weight

        return doc_scores.most_common(top_k)

    # -----------------------------------------------------------------
    # F.1: Chunk selection desde grafo (DTm-76)
    # -----------------------------------------------------------------

    @staticmethod
    def _select_chunks_from_graph(
        entity_results: List[Tuple[str, float]],
        relationship_results: List[Tuple[str, float]],
        top_k: int,
    ) -> Tuple[List[str], List[float]]:
        """Selecciona y rankea doc_ids combinando entity + relationship results.

        Combina scores de ambos canales (low-level entities y high-level
        relationships). Docs que aparecen en ambos canales acumulan score.
        Retorna los top_k doc_ids ordenados por score acumulado.

        Args:
            entity_results: (doc_id, score) de query_entities / entity VDB.
            relationship_results: (doc_id, score) de relationship VDB / query_by_keywords.
            top_k: Maximo de docs a retornar.

        Returns:
            (doc_ids, scores) ordenados por score descendente.
        """
        doc_scores: Dict[str, float] = {}
        for doc_id, score in entity_results:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
        for doc_id, score in relationship_results:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        if not doc_scores:
            return [], []

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]
        doc_ids = [doc_id for doc_id, _ in ranked]
        scores = [score for _, score in ranked]
        return doc_ids, scores

    @staticmethod
    def _corpus_fingerprint(
        documents: List[Dict[str, Any]],
        max_text_chars: int = 0,
        kg_max_entities: int = 0,
    ) -> str:
        """Hash determinista del corpus + config para invalidar cache si cambia.

        Incluye max_text_chars y kg_max_entities en el hash para que
        cambios en config KG invaliden el cache (G.2/DTm-56).
        """
        h = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d.get("doc_id", "")):
            h.update(doc.get("doc_id", "").encode())
            content = doc.get("content", "")
            h.update(content.encode())
        h.update(f"n={len(documents)}".encode())
        # Config que afecta el KG resultante (G.2/DTm-56)
        if max_text_chars:
            h.update(f"mtc={max_text_chars}".encode())
        if kg_max_entities:
            h.update(f"me={kg_max_entities}".encode())
        return h.hexdigest()[:16]

    def _resolve_cache_path(
        self, documents: List[Dict[str, Any]],
    ) -> Optional[Path]:
        """Determina la ruta del cache del KG, o None si no hay cache dir."""
        if not self._kg_cache_dir:
            return None
        fingerprint = self._corpus_fingerprint(
            documents,
            max_text_chars=self._kg_max_text_chars,
            kg_max_entities=self._kg_max_entities,
        )
        return self._kg_cache_dir / f"kg_cache_{fingerprint}.json"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieval con modo configurable (F.4/DTm-79).

        Modos:
          naive         — solo vector search (sin KG)
          graph_primary — grafo como primary retriever, vector fallback (DAM-3)
          local         — entity VDB + BFS + vector fusion
          global        — relationship VDB + vector fusion
          hybrid        — local + global + vector fusion (default)
        """
        k = top_k or self.config.retrieval_k
        start_time = time.perf_counter()

        mode = self._lightrag_mode
        graph_available = self._has_graph and self._kg and self._extractor

        if mode == "naive" or not graph_available:
            result = self._vector_retriever.retrieve(query, top_k=k)
        elif mode == "graph_primary":
            result = self._retrieve_via_graph(query, k)
        else:
            # local, global, hybrid: vector search + graph fusion
            vector_result = self._vector_retriever.retrieve(query, top_k=k)
            result = self._fuse_with_graph(query, vector_result, k)

        result.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        result.strategy_used = (
            RetrievalStrategy.LIGHT_RAG if graph_available and mode != "naive"
            else RetrievalStrategy.SIMPLE_VECTOR
        )
        result.metadata["graph_active"] = graph_available and mode != "naive"
        result.metadata["lightrag_mode"] = mode
        return result

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieval con vector pre-computado + graph traversal."""
        k = top_k or self.config.retrieval_k
        start_time = time.perf_counter()

        mode = self._lightrag_mode
        graph_available = self._has_graph and self._kg and self._extractor

        if mode == "naive" or not graph_available:
            result = self._vector_retriever.retrieve_by_vector(
                query_text, query_vector, top_k=k
            )
        elif mode == "graph_primary":
            # graph_primary usa query text, no vector pre-computado
            result = self._retrieve_via_graph(query_text, k)
        else:
            vector_result = self._vector_retriever.retrieve_by_vector(
                query_text, query_vector, top_k=k
            )
            result = self._fuse_with_graph(query_text, vector_result, k)

        result.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        result.strategy_used = (
            RetrievalStrategy.LIGHT_RAG if graph_available and mode != "naive"
            else RetrievalStrategy.SIMPLE_VECTOR
        )
        result.metadata["graph_active"] = graph_available and mode != "naive"
        result.metadata["lightrag_mode"] = mode
        return result

    # -----------------------------------------------------------------
    # F.2: Graph as primary retriever (DAM-3)
    # -----------------------------------------------------------------

    def _retrieve_via_graph(
        self,
        query: str,
        top_k: int,
    ) -> RetrievalResult:
        """Retrieval con grafo como mecanismo primario (DAM-3).

        Flujo:
        1. Extraer keywords (low-level + high-level)
        2. Entity VDB → query_entities() → doc_ids (low-level)
        3. Relationship VDB → doc_ids (high-level)
        4. _select_chunks_from_graph() → top_k doc_ids
        5. Recuperar contenido de ChromaDB
        6. Si insuficientes resultados, complementar con vector search
        """
        if self._kg is None or self._extractor is None:
            return self._vector_retriever.retrieve(query, top_k=top_k)

        low_level, high_level = self._get_query_keywords(query)

        if not low_level and not high_level:
            logger.debug("graph_primary: sin keywords, fallback a vector")
            return self._vector_retriever.retrieve(query, top_k=top_k)

        # Low-level: entidades
        entity_results: List[Tuple[str, float]] = []
        if low_level:
            pre_resolved = None
            if self._entities_vdb is not None:
                pre_resolved = self._resolve_entities_via_vdb(low_level)
            entity_results = self._kg.query_entities(
                low_level,
                max_hops=self._kg_max_hops,
                max_docs=top_k * self._GRAPH_OVERFETCH_FACTOR,
                pre_resolved=pre_resolved,
            )

        # High-level: relaciones
        relationship_results: List[Tuple[str, float]] = []
        if high_level:
            if self._relationships_vdb is not None:
                relationship_results = self._resolve_relationships_via_vdb(
                    high_level
                )
            else:
                relationship_results = self._kg.query_by_keywords(
                    high_level,
                    max_docs=top_k * self._GRAPH_OVERFETCH_FACTOR,
                )

        # F.1: Seleccionar chunks desde el grafo
        graph_doc_ids, graph_scores = self._select_chunks_from_graph(
            entity_results, relationship_results, top_k,
        )

        # Recuperar contenido desde ChromaDB
        final_ids: List[str] = []
        final_contents: List[str] = []
        final_scores: List[float] = []

        if graph_doc_ids:
            contents_map = self._vector_retriever.get_documents_by_ids(
                graph_doc_ids
            )
            for doc_id, score in zip(graph_doc_ids, graph_scores):
                if doc_id in contents_map:
                    final_ids.append(doc_id)
                    final_contents.append(contents_map[doc_id])
                    final_scores.append(score)

        graph_resolved = len(final_ids)
        graph_unresolved = len(graph_doc_ids) - graph_resolved

        if graph_unresolved > 0:
            logger.warning(
                f"graph_primary: {graph_unresolved}/{len(graph_doc_ids)} "
                f"docs del grafo sin contenido en ChromaDB"
            )

        # Fallback: si el grafo produjo menos de top_k/2, complementar con vector
        vector_fallback_used = False
        if len(final_ids) < max(top_k // 2, 1):
            vector_fallback_used = True
            logger.debug(
                f"graph_primary: {len(final_ids)} docs insuficientes "
                f"(< {top_k // 2}), complementando con vector search"
            )
            vector_result = self._vector_retriever.retrieve(
                query, top_k=top_k,
            )
            seen = set(final_ids)
            for doc_id, content, score in zip(
                vector_result.doc_ids,
                vector_result.contents,
                vector_result.scores,
            ):
                if doc_id not in seen:
                    final_ids.append(doc_id)
                    final_contents.append(content)
                    # Penalizar docs de fallback para que el grafo domine
                    final_scores.append(score * 0.5)
                    seen.add(doc_id)
                if len(final_ids) >= top_k:
                    break

        # F.3/DAM-8: Recopilar entidades y relaciones para contexto estructurado
        _MAX_CONTEXT_ENTITIES = 30
        _MAX_CONTEXT_RELATIONS = 30
        kg_entities = self._kg.get_entities_for_docs(
            final_ids[:top_k]
        )[:_MAX_CONTEXT_ENTITIES]
        kg_relations = self._kg.get_relations_for_docs(
            final_ids[:top_k]
        )[:_MAX_CONTEXT_RELATIONS]

        return RetrievalResult(
            doc_ids=final_ids[:top_k],
            contents=final_contents[:top_k],
            scores=final_scores[:top_k],
            vector_scores=[0.0] * min(len(final_ids), top_k),
            metadata={
                "graph_primary": True,
                "graph_docs_requested": len(graph_doc_ids),
                "graph_resolved": graph_resolved,
                "graph_unresolved": graph_unresolved,
                "vector_fallback_used": vector_fallback_used,
                "query_keywords": {
                    "low": low_level,
                    "high": high_level,
                },
                "kg_entities": kg_entities,
                "kg_relations": kg_relations,
            },
        )

    def _fuse_with_graph(
        self,
        query: str,
        vector_result: RetrievalResult,
        top_k: int,
    ) -> RetrievalResult:
        """Fusiona resultados de vector search con graph traversal.

        1. Extrae keywords de la query (low-level + high-level)
        2. Graph traversal por entidades (low-level)
        3. Graph traversal por temas (high-level)
        4. Fusion via RRF (default) o linear weighted sum
        """
        if self._kg is None:
            raise RuntimeError("_fuse_with_graph llamado sin KnowledgeGraph")
        if self._extractor is None:
            raise RuntimeError("_fuse_with_graph llamado sin TripletExtractor")

        # Paso 1: Query analysis
        low_level, high_level = self._get_query_keywords(query)

        if not low_level and not high_level:
            # Sin keywords, no hay graph traversal posible
            vector_result.metadata["graph_docs_added"] = 0
            vector_result.metadata["query_keywords"] = {"low": [], "high": []}
            return vector_result

        # Paso 2: Graph traversal (respeta lightrag_mode: local/global/hybrid)
        use_local = self._lightrag_mode in ("local", "hybrid")
        use_global = self._lightrag_mode in ("global", "hybrid")
        graph_docs: Dict[str, float] = {}

        # Low-level: entidades especificas
        if use_local and low_level:
            # DAM-1: resolver entidades via VDB (semantic) si disponible,
            # fallback a string matching si no hay VDB.
            pre_resolved = None
            if self._entities_vdb is not None:
                pre_resolved = self._resolve_entities_via_vdb(low_level)
            entity_results = self._kg.query_entities(
                low_level,
                max_hops=self._kg_max_hops,
                max_docs=top_k * self._GRAPH_OVERFETCH_FACTOR,
                pre_resolved=pre_resolved,
            )
            for doc_id, score in entity_results:
                graph_docs[doc_id] = max(graph_docs.get(doc_id, 0.0), score)

        # High-level: temas abstractos
        if use_global and high_level:
            # DAM-2: resolver relaciones via VDB (semantic) si disponible,
            # fallback a token matching si no hay VDB.
            if self._relationships_vdb is not None:
                theme_results = self._resolve_relationships_via_vdb(high_level)
            else:
                theme_results = self._kg.query_by_keywords(
                    high_level,
                    max_docs=top_k * self._GRAPH_OVERFETCH_FACTOR,
                )
            for doc_id, score in theme_results:
                graph_docs[doc_id] = max(graph_docs.get(doc_id, 0.0), score)

        if not graph_docs:
            vector_result.metadata["graph_docs_added"] = 0
            vector_result.metadata["query_keywords"] = {
                "low": low_level, "high": high_level,
            }
            return vector_result

        # Paso 3: Build vector ranking and contents map
        vector_docs: Dict[str, float] = {}
        vector_contents: Dict[str, str] = {}
        for doc_id, content, score in zip(
            vector_result.doc_ids, vector_result.contents, vector_result.scores
        ):
            vector_docs[doc_id] = score
            vector_contents[doc_id] = content

        # Paso 4: Recuperar contenido de graph-only docs desde ChromaDB.
        graph_only_ids = [
            did for did in graph_docs if did not in vector_contents
        ]
        graph_resolved = 0
        if graph_only_ids:
            looked_up = self._vector_retriever.get_documents_by_ids(
                graph_only_ids
            )
            for did, content in looked_up.items():
                vector_contents[did] = content
                graph_resolved += 1
        graph_unresolved = len(graph_only_ids) - graph_resolved

        # Paso 5: Fusion
        use_rrf = (self._kg_fusion_method != "linear")

        if not use_rrf:
            # Linear fusion: normalize + weighted sum
            max_graph_score = max(graph_docs.values()) if graph_docs else 1.0
            if max_graph_score > 0:
                graph_docs = {k: v / max_graph_score for k, v in graph_docs.items()}
            max_vector_score = max(vector_docs.values()) if vector_docs else 1.0
            if max_vector_score > 0:
                vector_docs = {k: v / max_vector_score for k, v in vector_docs.items()}

            # DTm-53: si ambos rankings son all-zero, fallback a RRF
            if max_graph_score == 0 and max_vector_score == 0:
                logger.warning(
                    "Linear fusion: scores all-zero en ambos rankings, "
                    "usando RRF como fallback"
                )
                use_rrf = True

        if not use_rrf:
            all_doc_ids = set(vector_docs.keys()) | set(graph_docs.keys())
            fused_scores: List[Tuple[str, float]] = []
            for doc_id in all_doc_ids:
                v_score = vector_docs.get(doc_id, 0.0)
                g_score = graph_docs.get(doc_id, 0.0)
                fused = self._vector_weight * v_score + self._graph_weight * g_score
                fused_scores.append((doc_id, fused))
            fused_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            # RRF fusion (default): rank-based, robust to score scale differences
            vector_ranking = sorted(
                vector_docs.items(), key=lambda x: x[1], reverse=True
            )
            graph_ranking = sorted(
                graph_docs.items(), key=lambda x: x[1], reverse=True
            )
            fused_scores = reciprocal_rank_fusion(
                rankings=[vector_ranking, graph_ranking],
                weights=[self._vector_weight, self._graph_weight],
                k=self._kg_rrf_k,
                top_n=top_k,
            )

        # Paso 7: Construir resultado (solo docs con contenido)
        final_ids = []
        final_contents = []
        final_scores = []

        for doc_id, score in fused_scores:
            if doc_id in vector_contents:
                final_ids.append(doc_id)
                final_contents.append(vector_contents[doc_id])
                final_scores.append(score)

            if len(final_ids) >= top_k:
                break

        result = RetrievalResult(
            doc_ids=final_ids,
            contents=final_contents,
            scores=final_scores,
            # 0.0 para docs que provienen solo del grafo (sin similarity score).
            vector_scores=[vector_docs.get(d, 0.0) for d in final_ids],
            retrieval_time_ms=0.0,  # Se actualiza en el caller
            strategy_used=RetrievalStrategy.LIGHT_RAG,
            metadata={
                "graph_only_candidates": len(graph_only_ids),
                "graph_resolved": graph_resolved,
                "graph_unresolved": graph_unresolved,
                "query_keywords": {
                    "low": low_level,
                    "high": high_level,
                },
                "graph_candidates": len(graph_docs),
                "vector_candidates": len(vector_docs),
            },
        )

        if graph_unresolved > 0:
            # DTm-52: warn (not just debug) when graph docs can't be resolved
            logger.warning(
                f"LightRAG fusion: {graph_unresolved}/{len(graph_only_ids)} "
                f"graph-only docs sin contenido en ChromaDB (excluidos)"
            )
        elif graph_only_ids:
            logger.debug(
                f"LightRAG fusion: {len(graph_only_ids)} graph-only docs, "
                f"todos recuperados via lookup. "
                f"Retornando {len(final_ids)} docs."
            )

        return result

    def _get_query_keywords(
        self, query: str,
    ) -> Tuple[List[str], List[str]]:
        """Extrae keywords de query con cache thread-safe y LRU eviction (DTm-47/51)."""
        with self._cache_lock:
            if query in self._query_keywords_cache:
                self._query_keywords_cache.move_to_end(query)
                return self._query_keywords_cache[query]

        if self._extractor is None:
            raise RuntimeError(
                "LightRAGRetriever._get_query_keywords llamado sin extractor"
            )
        low, high = self._extractor.extract_query_keywords(query)

        with self._cache_lock:
            self._query_keywords_cache[query] = (low, high)
            if len(self._query_keywords_cache) > self._QUERY_CACHE_MAX_SIZE:
                self._query_keywords_cache.popitem(last=False)
        return low, high

    def pre_extract_query_keywords(
        self, queries: List[str],
    ) -> None:
        """Pre-extrae keywords de multiples queries en batch.

        Analogo a pre-embed de vectores: reduce latencia durante retrieval.
        """
        if not self._extractor:
            return

        # Filtrar queries ya cacheadas (DTm-51: thread-safe)
        with self._cache_lock:
            uncached = [q for q in queries if q not in self._query_keywords_cache]
        if not uncached:
            return

        logger.info(
            f"LightRAGRetriever: pre-extrayendo keywords de "
            f"{len(uncached)} queries..."
        )
        results = self._extractor.extract_query_keywords_batch(uncached)
        with self._cache_lock:
            for query, (low, high) in zip(uncached, results):
                self._query_keywords_cache[query] = (low, high)
                if len(self._query_keywords_cache) > self._QUERY_CACHE_MAX_SIZE:
                    self._query_keywords_cache.popitem(last=False)

    def clear_index(self) -> None:
        self._vector_retriever.clear_index()
        if self._kg:
            self._kg = KnowledgeGraph(max_entities=self._kg_max_entities)
        for vdb_attr in ("_entities_vdb", "_relationships_vdb"):
            vdb = getattr(self, vdb_attr, None)
            if vdb is not None:
                try:
                    vdb.delete_all_documents()
                except Exception as e:
                    logger.debug("Error limpiando %s (no fatal): %s", vdb_attr, e)
                setattr(self, vdb_attr, None)
        self._has_graph = False
        with self._cache_lock:
            self._query_keywords_cache.clear()
        self._is_indexed = False
        logger.debug("LightRAGRetriever: indice, grafo y entity VDB limpiados")


__all__ = [
    "LightRAGRetriever",
]
