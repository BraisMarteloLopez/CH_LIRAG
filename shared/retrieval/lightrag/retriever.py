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
        kg_cache_dir: str = "",
    ):
        super().__init__(config)
        self._llm_service = llm_service
        self._kg_max_hops = kg_max_hops
        self._kg_max_text_chars = kg_max_text_chars
        self._kg_max_entities = kg_max_entities
        self._kg_cache_dir = Path(kg_cache_dir) if kg_cache_dir else None
        self._kg_keyword_max_tokens = config.kg_keyword_max_tokens
        self._kg_extraction_max_tokens = config.kg_extraction_max_tokens
        self._kg_batch_docs_per_call = config.kg_batch_docs_per_call
        self._kg_gleaning_rounds = config.kg_gleaning_rounds
        self._lightrag_mode = config.lightrag_mode
        self._max_neighbors_per_entity = config.kg_max_neighbors_per_entity

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

        # DTm-73: co-occurrence bridging para reducir fragmentacion del grafo
        co_edges = self._kg.build_co_occurrence_edges()
        if co_edges:
            logger.info(f"LightRAGRetriever: {co_edges} co-occurrence edges creadas")

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

        # A5.1/DAM-4 completo: LLM synthesis para descripciones largas
        if self.config.kg_description_synthesis and self._llm_service:
            synthesized = self._synthesize_descriptions()
            if synthesized:
                logger.info(
                    f"LightRAGRetriever: {synthesized} entidades sintetizadas via LLM"
                )

        # DAM-1: construir entity VDB (ChromaDB con embeddings de entidades)
        self._build_entities_vdb()
        # DAM-2: construir relationship VDB (ChromaDB con embeddings de relaciones)
        self._build_relationships_vdb()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"LightRAGRetriever: KG construido en {elapsed_ms:.0f}ms. "
            f"{total_triplets} tripletas de {len(documents)} docs."
        )

    # A5.1: Prompt para LLM synthesis de descripciones (DAM-4 completo)
    _SYNTHESIS_SYSTEM = (
        "You are a knowledge graph curator. Your task is to synthesize "
        "multiple descriptions of the same entity into a single, concise, "
        "and informative description. Remove redundancy, keep key facts, "
        "and produce a coherent summary."
    )
    _SYNTHESIS_PROMPT = (
        "Entity: {entity_name}\n\n"
        "Descriptions from different sources:\n{descriptions}\n\n"
        "Synthesize these into a single concise description (max 2 sentences). "
        "Keep only the most important facts. Respond with ONLY the synthesized description."
    )

    def _synthesize_descriptions(self) -> int:
        """LLM synthesis para entidades con descripciones concatenadas largas (A5.1/DAM-4).

        Solo aplica a entidades cuya descripcion concatenada excede el
        threshold configurado. Usa el LLM para sintetizar en una descripcion
        coherente, como en el paper original (map-reduce).

        Returns:
            Numero de entidades sintetizadas.
        """
        assert self._kg is not None
        assert self._llm_service is not None

        threshold = self.config.kg_synthesis_char_threshold
        candidates = [
            (name, entity)
            for name, entity in self._kg._entities.items()
            if len(entity._descriptions) > 1
            and len(entity.description) > threshold
        ]

        if not candidates:
            return 0

        logger.info(
            f"LightRAGRetriever: sintetizando {len(candidates)} entidades "
            f"con descripciones > {threshold} chars"
        )

        synthesized = 0
        for name, entity in candidates:
            # Formatear descripciones como lista numerada
            desc_list = "\n".join(
                f"  {i+1}. {d}" for i, d in enumerate(entity._descriptions[:10])
            )
            prompt = self._SYNTHESIS_PROMPT.format(
                entity_name=name,
                descriptions=desc_list,
            )
            try:
                result = self._llm_service.invoke(
                    prompt,
                    system_prompt=self._SYNTHESIS_SYSTEM,
                    max_tokens=256,
                )
                if result and len(result.strip()) > 5:
                    entity.description = result.strip()
                    synthesized += 1
            except Exception as e:
                logger.debug(
                    f"LLM synthesis failed for entity '{name}': {e}. "
                    "Keeping concatenated description."
                )

        return synthesized

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

        ranked: List[Tuple[str, float]] = [
            (doc_id, float(score)) for doc_id, score in doc_scores.most_common(top_k)
        ]
        return ranked

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

        Modos (paper original):
          naive         — solo vector search (sin KG)
          local         — entidades relevantes + vector chunks
          global        — relaciones relevantes + vector chunks
          hybrid        — entidades + relaciones + vector chunks (default)

        En local/global/hybrid, el vector search produce el ranking de chunks
        y el KG aporta contexto complementario (entidades/relaciones) como
        secciones separadas en metadata. No hay fusion RRF.
        """
        k = top_k or self.config.retrieval_k
        start_time = time.perf_counter()

        mode = self._lightrag_mode
        graph_available = self._has_graph and self._kg and self._extractor

        if mode == "naive" or not graph_available:
            result = self._vector_retriever.retrieve(query, top_k=k)
        else:
            # local, global, hybrid: vector search + KG enrichment
            vector_result = self._vector_retriever.retrieve(query, top_k=k)
            result = self._enrich_with_graph(query, vector_result, k)

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
        """Retrieval con vector pre-computado + KG enrichment."""
        k = top_k or self.config.retrieval_k
        start_time = time.perf_counter()

        mode = self._lightrag_mode
        graph_available = self._has_graph and self._kg and self._extractor

        if mode == "naive" or not graph_available:
            result = self._vector_retriever.retrieve_by_vector(
                query_text, query_vector, top_k=k
            )
        else:
            vector_result = self._vector_retriever.retrieve_by_vector(
                query_text, query_vector, top_k=k
            )
            result = self._enrich_with_graph(query_text, vector_result, k)

        result.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        result.strategy_used = (
            RetrievalStrategy.LIGHT_RAG if graph_available and mode != "naive"
            else RetrievalStrategy.SIMPLE_VECTOR
        )
        result.metadata["graph_active"] = graph_available and mode != "naive"
        result.metadata["lightrag_mode"] = mode
        return result

    # -----------------------------------------------------------------
    # KG Enrichment: entidades/relaciones como contexto complementario
    # -----------------------------------------------------------------

    _MAX_CONTEXT_ENTITIES = 30
    _MAX_CONTEXT_RELATIONS = 30

    def _enrich_with_graph(
        self,
        query: str,
        vector_result: RetrievalResult,
        top_k: int,
    ) -> RetrievalResult:
        """Enriquece resultados vectoriales con datos del KG (paper-aligned).

        No fusiona rankings. El vector search produce el ranking de chunks
        (para metricas de retrieval). El KG aporta entidades y relaciones
        relevantes a la query como contexto complementario en metadata.

        Cada modo determina que canales KG se consultan:
          local  — entidades (low-level keywords)
          global — relaciones (high-level keywords)
          hybrid — entidades + relaciones
        """
        if self._kg is None:
            raise RuntimeError("_enrich_with_graph llamado sin KnowledgeGraph")
        if self._extractor is None:
            raise RuntimeError("_enrich_with_graph llamado sin TripletExtractor")

        # Paso 1: Query analysis
        low_level, high_level = self._get_query_keywords(query)

        if not low_level and not high_level:
            vector_result.metadata["query_keywords"] = {"low": [], "high": []}
            vector_result.metadata["kg_entities"] = []
            vector_result.metadata["kg_relations"] = []
            return vector_result

        use_local = self._lightrag_mode in ("local", "hybrid")
        use_global = self._lightrag_mode in ("global", "hybrid")

        # Paso 2: Recopilar entidades relevantes a la query (low-level)
        # Divergencia #9: cada entidad incluye vecinos 1-hop ranked por
        # edge_weight + degree_centrality (paper-aligned).
        kg_entities: List[Dict[str, Any]] = []
        if use_local and low_level:
            resolved_names = self._resolve_entities_via_vdb(low_level)
            if resolved_names:
                entities = self._kg.get_all_entities()
                for name in resolved_names:
                    entity = entities.get(name)
                    if entity:
                        entry: Dict[str, Any] = {
                            "entity": entity.name,
                            "type": entity.entity_type,
                            "description": entity.description,
                        }
                        try:
                            neighbors = self._kg.get_neighbors_ranked(
                                name,
                                max_neighbors=self._max_neighbors_per_entity,
                            )
                            if neighbors:
                                entry["neighbors"] = neighbors
                        except Exception:
                            logger.debug(
                                f"Neighbor lookup failed for {name}, "
                                f"continuing without"
                            )
                        kg_entities.append(entry)
                    if len(kg_entities) >= self._MAX_CONTEXT_ENTITIES:
                        break

        # Paso 3: Recopilar relaciones relevantes a la query (high-level)
        kg_relations: List[Dict[str, Any]] = []
        if use_global and high_level:
            kg_relations = self._resolve_relations_for_context(high_level)

        logger.debug(
            f"KG enrichment: {len(kg_entities)} entities, "
            f"{len(kg_relations)} relations for query '{query[:60]}...'"
        )

        # Paso 4: Retornar resultado vectorial con KG data en metadata
        vector_result.metadata["query_keywords"] = {
            "low": low_level, "high": high_level,
        }
        vector_result.metadata["kg_entities"] = kg_entities
        vector_result.metadata["kg_relations"] = kg_relations
        return vector_result

    def _resolve_relations_for_context(
        self,
        keywords: List[str],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Resuelve high-level keywords a relaciones con sus descripciones.

        Divergencia #9: enriquece cada relacion con las descripciones y tipos
        de sus entidades endpoint (source/target) desde el KG.
        """
        if not self._relationships_vdb or not keywords:
            return []

        # Fetch entities dict once for endpoint enrichment
        all_entities = self._kg.get_all_entities() if self._kg else {}

        seen: set = set()
        relations: List[Dict[str, Any]] = []

        for keyword in keywords:
            if not keyword.strip():
                continue
            results = self._relationships_vdb.similarity_search_with_score(
                keyword, k=top_k,
            )
            for doc, distance in results:
                if distance > self._RELATIONSHIP_VDB_MAX_DISTANCE:
                    continue
                source = doc.metadata.get("source_entity", "")
                target = doc.metadata.get("target_entity", "")
                relation = doc.metadata.get("relation", "")
                key = (source, target, relation)
                if key in seen:
                    continue
                seen.add(key)
                entry: Dict[str, Any] = {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "description": doc.page_content,
                }
                src_entity = all_entities.get(source)
                if src_entity:
                    entry["source_description"] = src_entity.description
                    entry["source_type"] = src_entity.entity_type
                tgt_entity = all_entities.get(target)
                if tgt_entity:
                    entry["target_description"] = tgt_entity.description
                    entry["target_type"] = tgt_entity.entity_type
                relations.append(entry)
                if len(relations) >= self._MAX_CONTEXT_RELATIONS:
                    return relations

        return relations

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
