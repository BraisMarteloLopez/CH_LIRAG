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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.llm import AsyncLLMService, run_sync
from shared.types import EmbeddingModelProtocol

from .core import (
    BaseRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
    SimpleVectorRetriever,
)
from .knowledge_graph import KnowledgeGraph, KGRelation, HAS_NETWORKX
from .triplet_extractor import TripletExtractor

logger = logging.getLogger(__name__)


class LightRAGRetriever(BaseRetriever):
    """Retriever LightRAG: Vector + Knowledge Graph dual-level.

    Requiere:
      - AsyncLLMService para extraccion de tripletas y query analysis
      - EmbeddingModel para busqueda vectorial (ChromaDB)
      - igraph para knowledge graph

    Sin LLM service: fallback a SimpleVectorRetriever puro.
    Sin NetworkX: fallback a SimpleVectorRetriever puro.
    """

    # Factor de over-fetch para graph traversal: pedimos N * top_k
    # candidatos al grafo para tener margen tras la fusion con vector.
    _GRAPH_OVERFETCH_FACTOR = 2

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

        # Vector retriever (siempre disponible)
        self._vector_retriever = SimpleVectorRetriever(
            config, embedding_model, collection_name,
            embedding_batch_size=embedding_batch_size,
        )

        # Knowledge graph + triplet extractor (requieren LLM + networkx)
        self._kg: Optional[KnowledgeGraph] = None
        self._extractor: Optional[TripletExtractor] = None
        self._has_graph = False

        if llm_service and HAS_NETWORKX:
            self._kg = KnowledgeGraph(max_entities=kg_max_entities)
            self._extractor = TripletExtractor(
                llm_service, max_text_chars=kg_max_text_chars,
            )
        else:
            reasons = []
            if not llm_service:
                reasons.append("LLM service no proporcionado")
            if not HAS_NETWORKX:
                reasons.append("networkx no instalado")
            logger.warning(
                f"LIGHT_RAG: sin knowledge graph ({', '.join(reasons)}). "
                f"Fallback a SimpleVector."
            )

        # Cache de keywords de queries (para evitar re-extraccion)
        self._query_keywords_cache: Dict[str, Tuple[List[str], List[str]]] = {}

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
            try:
                cache_path = self._resolve_cache_path(documents)
                if cache_path and cache_path.exists():
                    self._kg = KnowledgeGraph.load(cache_path)
                    logger.info(
                        f"LightRAGRetriever: KG cargado desde cache "
                        f"({self._kg.num_entities} entidades)"
                    )
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
        extraction_results = self._extractor.extract_batch(documents)

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

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"LightRAGRetriever: KG construido en {elapsed_ms:.0f}ms. "
            f"{total_triplets} tripletas de {len(documents)} docs."
        )

    @staticmethod
    def _corpus_fingerprint(
        documents: List[Dict[str, Any]],
        max_text_chars: int = 0,
    ) -> str:
        """Hash determinista del corpus para invalidar cache si cambia.

        Incluye max_text_chars en el hash para que cambios en
        KG_MAX_TEXT_CHARS invaliden el cache (el KG resultante difiere).
        """
        h = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d.get("doc_id", "")):
            h.update(doc.get("doc_id", "").encode())
            content = doc.get("content", "")
            h.update(content.encode())
        h.update(str(len(documents)).encode())
        # max_text_chars afecta cuanto texto ve el LLM para extraccion
        if max_text_chars:
            h.update(f"mtc={max_text_chars}".encode())
        return h.hexdigest()[:16]

    def _resolve_cache_path(
        self, documents: List[Dict[str, Any]],
    ) -> Optional[Path]:
        """Determina la ruta del cache del KG, o None si no hay cache dir."""
        if not self._kg_cache_dir:
            return None
        fingerprint = self._corpus_fingerprint(
            documents, max_text_chars=self._kg_max_text_chars,
        )
        return self._kg_cache_dir / f"kg_cache_{fingerprint}.json"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieval: vector search + graph traversal + fusion."""
        k = top_k or self.config.retrieval_k
        start_time = time.perf_counter()

        # Paso 1: Vector search
        vector_result = self._vector_retriever.retrieve(query, top_k=k)

        # Paso 2: Graph expansion (si disponible)
        if self._has_graph and self._kg and self._extractor:
            result = self._fuse_with_graph(query, vector_result, k)
        else:
            result = vector_result

        result.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        result.strategy_used = (
            RetrievalStrategy.LIGHT_RAG if self._has_graph
            else RetrievalStrategy.SIMPLE_VECTOR
        )
        result.metadata["graph_active"] = self._has_graph
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

        # Paso 1: Vector search con embedding pre-computado
        vector_result = self._vector_retriever.retrieve_by_vector(
            query_text, query_vector, top_k=k
        )

        # Paso 2: Graph expansion (si disponible)
        if self._has_graph and self._kg and self._extractor:
            result = self._fuse_with_graph(query_text, vector_result, k)
        else:
            result = vector_result

        result.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        result.strategy_used = (
            RetrievalStrategy.LIGHT_RAG if self._has_graph
            else RetrievalStrategy.SIMPLE_VECTOR
        )
        result.metadata["graph_active"] = self._has_graph
        return result

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
        4. Merge scores: vector_weight * vector_score + graph_weight * graph_score
        """
        assert self._kg is not None
        assert self._extractor is not None

        # Paso 1: Query analysis
        low_level, high_level = self._get_query_keywords(query)

        if not low_level and not high_level:
            # Sin keywords, no hay graph traversal posible
            vector_result.metadata["graph_docs_added"] = 0
            vector_result.metadata["query_keywords"] = {"low": [], "high": []}
            return vector_result

        # Paso 2: Graph traversal
        graph_docs: Dict[str, float] = {}

        # Low-level: entidades especificas
        if low_level:
            entity_results = self._kg.query_entities(
                low_level,
                max_hops=self._kg_max_hops,
                max_docs=top_k * self._GRAPH_OVERFETCH_FACTOR,
            )
            for doc_id, score in entity_results:
                graph_docs[doc_id] = max(graph_docs.get(doc_id, 0.0), score)

        # High-level: temas abstractos
        if high_level:
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

        # Paso 3: Normalizar scores del grafo a [0, 1]
        max_graph_score = max(graph_docs.values()) if graph_docs else 1.0
        if max_graph_score > 0:
            graph_docs = {
                k: v / max_graph_score for k, v in graph_docs.items()
            }

        # Paso 4: Normalizar scores de vector a [0, 1]
        vector_docs: Dict[str, float] = {}
        vector_contents: Dict[str, str] = {}
        max_vector_score = max(vector_result.scores) if vector_result.scores else 1.0
        if max_vector_score > 0:
            for doc_id, content, score in zip(
                vector_result.doc_ids, vector_result.contents, vector_result.scores
            ):
                vector_docs[doc_id] = score / max_vector_score
                vector_contents[doc_id] = content

        # Paso 5: Recuperar contenido de graph-only docs desde ChromaDB.
        # El grafo puede descubrir docs que el vector search no retorno.
        # En vez de descartarlos, hacemos lookup de su contenido real.
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

        # Paso 6: Fusion — merge todos los doc_ids que tienen contenido
        all_doc_ids = set(vector_docs.keys()) | set(graph_docs.keys())
        fused_scores: List[Tuple[str, float]] = []

        for doc_id in all_doc_ids:
            v_score = vector_docs.get(doc_id, 0.0)
            g_score = graph_docs.get(doc_id, 0.0)
            fused = self._vector_weight * v_score + self._graph_weight * g_score
            fused_scores.append((doc_id, fused))

        fused_scores.sort(key=lambda x: x[1], reverse=True)

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

        if graph_only_ids:
            logger.debug(
                f"LightRAG fusion: {len(graph_only_ids)} graph-only docs, "
                f"{graph_resolved} recuperados via lookup, "
                f"{graph_unresolved} sin contenido (excluidos). "
                f"Retornando {len(final_ids)} docs."
            )

        return result

    def _get_query_keywords(
        self, query: str,
    ) -> Tuple[List[str], List[str]]:
        """Extrae keywords de query con cache."""
        if query in self._query_keywords_cache:
            return self._query_keywords_cache[query]

        assert self._extractor is not None
        low, high = self._extractor.extract_query_keywords(query)
        self._query_keywords_cache[query] = (low, high)
        return low, high

    def pre_extract_query_keywords(
        self, queries: List[str],
    ) -> None:
        """Pre-extrae keywords de multiples queries en batch.

        Analogo a pre-embed de vectores: reduce latencia durante retrieval.
        """
        if not self._extractor:
            return

        # Filtrar queries ya cacheadas
        uncached = [q for q in queries if q not in self._query_keywords_cache]
        if not uncached:
            return

        logger.info(
            f"LightRAGRetriever: pre-extrayendo keywords de "
            f"{len(uncached)} queries..."
        )
        results = self._extractor.extract_query_keywords_batch(uncached)
        for query, (low, high) in zip(uncached, results):
            self._query_keywords_cache[query] = (low, high)

    def clear_index(self) -> None:
        self._vector_retriever.clear_index()
        if self._kg:
            self._kg = KnowledgeGraph(max_entities=self._kg_max_entities)
        self._has_graph = False
        self._query_keywords_cache.clear()
        self._is_indexed = False
        logger.debug("LightRAGRetriever: indice y grafo limpiados")


__all__ = [
    "LightRAGRetriever",
]
