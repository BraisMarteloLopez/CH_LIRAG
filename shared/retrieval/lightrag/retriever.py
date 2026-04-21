"""
Modulo: LightRAG Retriever
Descripcion: Retriever basado en LightRAG (EMNLP 2025). Chunks obtenidos a
             traves del KG con tres canales agregados al mismo doc_scores.

Flujo:
    Indexacion:
      1. Extraer tripletas + high_level_keywords de cada doc via LLM
         (TripletExtractor; piggyback de keywords en la misma llamada)
      2. Construir KnowledgeGraph con entidades y relaciones (igraph)
      3. Indexar contenido original en ChromaDB (vector store principal)
      4. Construir 3 VDBs: Entity, Relationship, Chunk Keywords

    Retrieval (modos local/global/hybrid):
      1. Query analysis via LLM -> low_level + high_level keywords
      2. Resolver keywords contra Entity / Relationship / Chunk Keywords VDBs
      3. Obtener source_doc_ids de los tres canales
      4. Scoring agregado al mismo doc_scores:
         `1/(1+rank) * similarity [* edge_weight]` por canal
      5. Fetch contenido desde vector store via get_documents_by_ids
      6. Fallback a vector search si el KG no produce doc_ids
         (anotado en retrieval_metadata.kg_fallback)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.llm import AsyncLLMService, run_sync
from shared.operational_tracker import record_operational_event
from shared.types import EmbeddingModelProtocol

from ..core import (
    BaseRetriever,
    LightRAGMode,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
    SimpleVectorRetriever,
)
from .knowledge_graph import KnowledgeGraph, KGRelation, HAS_IGRAPH
from .triplet_extractor import TripletExtractor

logger = logging.getLogger(__name__)


def _neighbor_coverage_stats(
    kg_entities: List[Dict[str, Any]],
) -> Tuple[int, float]:
    """Cobertura del enriquecimiento 1-hop por entidad.

    Retorna `(entities_with_neighbors, mean_neighbors_per_entity)`.
    `mean` se calcula sobre el total de entidades resueltas (no solo las
    que tienen vecinos) para que un mean bajo discrimine "muchas entidades
    sin vecinos" vs "pocas entidades con muchos vecinos". Si la lista es
    vacia, retorna `(0, 0.0)`.
    """
    if not kg_entities:
        return 0, 0.0
    with_nb = sum(1 for e in kg_entities if e.get("neighbors"))
    total = sum(len(e.get("neighbors", [])) for e in kg_entities)
    return with_nb, round(total / len(kg_entities), 3)


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
        self._lightrag_mode: LightRAGMode = config.lightrag_mode
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

        # Entity VDB y Relationship VDB: ChromaDB collections que resuelven
        # entidades/relaciones por embedding similarity.
        self._entities_vdb: Any = None  # ChromaVectorStore, lazy init
        self._relationships_vdb: Any = None  # ChromaVectorStore, lazy init
        # Chunk Keywords VDB (divergencia #10): tercer canal del path
        # high-level. Indexa, por doc, las high-level keywords extraidas
        # durante indexacion. En retrieval resuelve query.high_level contra
        # la VDB y aporta doc_ids al scoring agregado de _retrieve_via_kg.
        self._chunk_keywords_vdb: Any = None  # ChromaVectorStore, lazy init
        self._embedding_model = embedding_model

        # Cache de keywords de queries con LRU eviction y lock thread-safe.
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

        vector_ok = self._vector_retriever.index_documents(
            documents, collection_name
        )
        if not vector_ok:
            logger.error("LightRAGRetriever: fallo indexacion vectorial")
            return False

        if self._kg and self._extractor:
            # Snapshot de stats del extractor antes del build para restaurarlos
            # ante fallo (evita stats corruptas si el build aborta a la mitad).
            stats_snapshot = self._extractor.get_stats()
            try:
                cache_path = self._resolve_cache_path(documents)
                if cache_path and cache_path.exists():
                    self._kg = KnowledgeGraph.load(cache_path)
                    logger.info(
                        f"LightRAGRetriever: KG cargado desde cache "
                        f"({self._kg.num_entities} entidades, "
                        f"{self._kg.num_docs_with_keywords} docs con keywords)"
                    )
                    # Rebuild de las 3 VDBs desde el KG cacheado.
                    self._build_entities_vdb()
                    self._build_relationships_vdb()
                    self._build_chunk_keywords_vdb()
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
                record_operational_event("retrieval_error")
                self._has_graph = False
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

        # Alerta si el cap de entidades descarto un porcentaje significativo:
        # documentos indexados tarde verian su KG incompleto.
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
                    logger.error(f"KG entity cap ALERT: {msg}")
                else:
                    logger.warning(msg)
        return True

    def _build_knowledge_graph(
        self, documents: List[Dict[str, Any]]
    ) -> None:
        """Construye el knowledge graph extrayendo tripletas via LLM."""
        assert self._extractor is not None
        assert self._kg is not None

        t0 = time.perf_counter()

        # Extraccion batch (async, controlada por semaphore del LLM).
        # Divergencia #10: cada doc ademas de (entities, relations) devuelve
        # `chunk_keywords` (high-level themes) que luego se indexan en
        # Chunk Keywords VDB para el path high-level de retrieval.
        extraction_results = self._extractor.extract_batch(
            documents, batch_docs_per_call=self._kg_batch_docs_per_call,
        )

        # Construir grafo
        total_triplets = 0
        for doc_id, (entities, relations, chunk_keywords) in extraction_results.items():
            added = self._kg.add_triplets(doc_id, relations)
            total_triplets += added

            for entity in entities:
                self._kg.add_entity_metadata(
                    entity.name, entity.entity_type, entity.description
                )

            # Divergencia #10: persistir chunk keywords en el KG (si hay)
            if chunk_keywords:
                self._kg.add_doc_keywords(doc_id, chunk_keywords)

        # Gleaning: re-extraccion opcional para capturar entidades perdidas
        # (no re-extrae high_level_keywords; ver KG_GLEANING_ROUNDS en .env).
        if self._kg_gleaning_rounds > 0 and self._extractor:
            for gleaning_round in range(self._kg_gleaning_rounds):
                t_glean = time.perf_counter()
                gleaning_count = 0
                for doc in documents:
                    doc_id = doc.get("doc_id", "")
                    text = doc.get("content", "")
                    prev_entities = extraction_results.get(doc_id, ([], [], []))[0]
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

        self._kg.log_entity_cap_summary()

        # Co-occurrence bridging: anade edges entre entidades co-ocurrentes
        # para reducir fragmentacion del grafo.
        co_edges = self._kg.build_co_occurrence_edges()
        if co_edges:
            logger.info(f"LightRAGRetriever: {co_edges} co-occurrence edges creadas")

        # Merge de descripciones multi-doc antes de construir VDBs.
        merged = self._kg.merge_entity_descriptions()
        if merged:
            logger.info(f"LightRAGRetriever: {merged} entidades con descripciones consolidadas")

        # LLM synthesis opcional para entidades con descripciones largas.
        if self.config.kg_description_synthesis and self._llm_service:
            synthesized = self._synthesize_descriptions()
            if synthesized:
                logger.info(
                    f"LightRAGRetriever: {synthesized} entidades sintetizadas via LLM"
                )

        # Construir las 3 VDBs (ChromaDB con embeddings de entidades,
        # relaciones y chunk keywords — esta ultima es la divergencia #10).
        self._build_entities_vdb()
        self._build_relationships_vdb()
        self._build_chunk_keywords_vdb()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"LightRAGRetriever: KG construido en {elapsed_ms:.0f}ms. "
            f"{total_triplets} tripletas de {len(documents)} docs."
        )

    # Prompt para LLM synthesis de descripciones de entidades multi-doc.
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
        """LLM synthesis para entidades con descripciones concatenadas largas.

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
                record_operational_event("description_synthesis_error")

        return synthesized

    def _build_entities_vdb(self) -> None:
        """Construye entity VDB: ChromaDB collection con embeddings de entidades.

        Cada entidad se indexa como "entity_name: description" para que
        similarity search encuentre entidades semanticamente relevantes
        a la query, sin depender de string matching.

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
                "HNSW_SPACE": "cosine",  # cosine distance for semantic matching
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

    # Max cosine distance para considerar un match valido.
    # Rango: [0, 2]. 0 = identical, 1 = orthogonal, 2 = opposite.
    # 0.8 es permisivo: admite matches semanticamente relacionados aunque
    # difieran lexicamente.
    _ENTITY_VDB_MAX_DISTANCE = 0.8

    def _resolve_entities_via_vdb(
        self,
        keywords: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Resuelve keywords de query a entity names via embedding similarity.

        Semantic search contra `entities_vdb`. Filtra resultados por threshold
        de cosine distance (`_ENTITY_VDB_MAX_DISTANCE`) para evitar matches
        irrelevantes.

        Returns:
            Lista de (entity_name, vdb_distance) ordenada por aparicion.
            distance es cosine distance [0, 2]: 0=identical, 2=opposite.
        """
        if not self._entities_vdb or not keywords:
            return []

        resolved: List[Tuple[str, float]] = []
        seen: set = set()

        for keyword in keywords:
            if not keyword.strip():
                continue
            results = self._entities_vdb.similarity_search_with_score(
                keyword, k=top_k,
            )
            for doc, distance in results:
                if distance > self._ENTITY_VDB_MAX_DISTANCE:
                    continue
                entity_name = doc.metadata.get("entity_name", "")
                if entity_name and entity_name not in seen:
                    seen.add(entity_name)
                    resolved.append((entity_name, float(distance)))

        return resolved

    def _build_relationships_vdb(self) -> None:
        """Construye relationship VDB: ChromaDB collection con embeddings de relaciones.

        Cada relacion se indexa como "source -> relation -> target: description"
        para que similarity search encuentre relaciones semanticamente relevantes
        a la query. Edge weight se almacena en metadata.

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
                "HNSW_SPACE": "cosine",  # cosine distance for semantic matching
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

    # Max cosine distance para matches de relacion (misma escala que entity VDB).
    _RELATIONSHIP_VDB_MAX_DISTANCE = 0.8

    # Divergencia #10: max cosine distance para matches en Chunk Keywords VDB.
    # Simetrico con Entity/Relationship VDB; relajar solo si se observa
    # demasiado corte en queries con high_level keywords legitimos.
    _CHUNK_KEYWORDS_VDB_MAX_DISTANCE = 0.8

    def _build_chunk_keywords_vdb(self) -> None:
        """Construye la Chunk Keywords VDB (divergencia #10, paper-aligned).

        Una entrada por doc_id cuyo `page_content` es la concatenacion con
        comas de las high-level keywords extraidas durante indexacion. La
        metadata `doc_id` permite recuperar el chunk asociado en el path
        de scoring.

        Se salta si `KG_CHUNK_KEYWORDS_ENABLED=false` o si no hay keywords
        en el KG. No falla silenciosamente — emite WARNING si activado pero
        sin datos, porque indica que la extraccion upstream no produjo
        keywords (posible prompt o modelo roto).

        Referencia: chunks_vdb en HKUDS/LightRAG operate.py (path high-level).
        """
        if not self.config.kg_chunk_keywords_enabled:
            logger.debug(
                "LightRAGRetriever: chunk keywords VDB desactivada "
                "(KG_CHUNK_KEYWORDS_ENABLED=false)"
            )
            return
        if self._kg is None:
            return

        all_keywords = self._kg.get_all_doc_keywords()
        if not all_keywords:
            logger.warning(
                "LightRAGRetriever: KG_CHUNK_KEYWORDS_ENABLED=true pero el "
                "KG no tiene chunk keywords (divergencia #10). Revisar logs "
                "de TripletExtractor: stats.docs_with_keywords."
            )
            return

        t0 = time.perf_counter()

        from shared.vector_store import ChromaVectorStore
        from langchain_core.documents import Document

        base_name = self._vector_retriever._vector_store.collection_name
        ck_collection_name = f"{base_name}_chunk_keywords"

        if self._chunk_keywords_vdb is not None:
            try:
                self._chunk_keywords_vdb.delete_all_documents()
            except Exception as e:
                logger.debug(
                    "Error limpiando chunk keywords VDB (no fatal): %s", e
                )

        self._chunk_keywords_vdb = ChromaVectorStore(
            config={
                "CHROMA_COLLECTION_NAME": ck_collection_name,
                "EMBEDDING_BATCH_SIZE": 50,
                "HNSW_SPACE": "cosine",
            },
            embedding_model=self._embedding_model,
        )

        lc_docs = []
        for doc_id, keywords in all_keywords.items():
            # Texto del embedding: keywords separadas por coma. Mantener el
            # casing original (las embeddings manejan casing). Sin prefijo
            # ni sufijo artificial — el paper no los usa.
            text = ", ".join(keywords)
            lc_docs.append(Document(
                page_content=text,
                metadata={"doc_id": doc_id},
            ))

        self._chunk_keywords_vdb.add_documents(lc_docs)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"LightRAGRetriever: chunk keywords VDB construida en "
            f"{elapsed_ms:.0f}ms ({len(lc_docs)} docs indexados)"
        )

    def _resolve_chunks_via_keywords_vdb(
        self,
        keywords: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Resuelve query high_level keywords contra Chunk Keywords VDB
        (divergencia #10: canal adicional del path high-level).

        Para cada keyword de la query hace `similarity_search_with_score`
        contra la Chunk Keywords VDB y acumula los doc_ids resueltos con
        su mejor distancia (menor cosine distance = mas similar).

        Returns:
            Lista de (doc_id, best_distance) ordenada por orden de
            aparicion (primera keyword de la query domina); cada doc_id
            aparece una sola vez con la mejor distancia observada.
        """
        if not self._chunk_keywords_vdb or not keywords:
            return []

        best_distance: Dict[str, float] = {}
        first_seen: Dict[str, int] = {}
        seq = 0
        for keyword in keywords:
            kw = keyword.strip()
            if not kw:
                continue
            try:
                results = self._chunk_keywords_vdb.similarity_search_with_score(
                    kw, k=top_k,
                )
            except Exception as e:
                logger.debug(
                    "Chunk keywords VDB search failed for '%s': %s", kw, e
                )
                record_operational_event("chunk_keywords_vdb_error")
                continue
            for doc, distance in results:
                if distance > self._CHUNK_KEYWORDS_VDB_MAX_DISTANCE:
                    continue
                doc_id = doc.metadata.get("doc_id", "")
                if not doc_id:
                    continue
                d = float(distance)
                prev = best_distance.get(doc_id)
                if prev is None or d < prev:
                    best_distance[doc_id] = d
                    if doc_id not in first_seen:
                        first_seen[doc_id] = seq
                        seq += 1

        # Orden de aparicion (mantiene la semantica de rank de la query)
        return sorted(
            best_distance.items(),
            key=lambda kv: first_seen.get(kv[0], 1 << 30),
        )

    @staticmethod
    def _corpus_fingerprint(
        documents: List[Dict[str, Any]],
        max_text_chars: int = 0,
        kg_max_entities: int = 0,
    ) -> str:
        """Hash determinista del corpus + config para invalidar cache si cambia.

        Incluye max_text_chars y kg_max_entities en el hash para que
        cambios en config KG invaliden el cache.
        """
        h = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d.get("doc_id", "")):
            h.update(doc.get("doc_id", "").encode())
            content = doc.get("content", "")
            h.update(content.encode())
        h.update(f"n={len(documents)}".encode())
        # Config que afecta el KG resultante.
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
        """Retrieval con modo configurable (paper-aligned).

        Modos (paper original):
          naive         — solo vector search (sin KG)
          local         — chunks via entidades del KG + entidades en metadata
          global        — chunks via relaciones del KG + relaciones en metadata
          hybrid        — chunks via entidades + relaciones (default)

        En local/global/hybrid, los chunks se obtienen a traves del KG:
        query keywords → VDB resolution → source_doc_ids → scoring agregado
        `1/(1+rank) × similarity [× edge_weight]`. Fallback a vector search
        cuando el KG no produce resultados.
        """
        k = top_k or self.config.retrieval_k
        start_time = time.perf_counter()

        mode = self._lightrag_mode
        graph_available = self._has_graph and self._kg and self._extractor

        if mode == "naive" or not graph_available:
            result = self._vector_retriever.retrieve(query, top_k=k)
        else:
            result = self._retrieve_via_kg(query, k)

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
        """Retrieval con vector pre-computado (paper-aligned).

        Para modos local/global/hybrid usa KG-based retrieval (el vector
        pre-computado no se usa porque los chunks vienen del KG).
        Para naive, usa el vector pre-computado directamente.
        """
        k = top_k or self.config.retrieval_k
        start_time = time.perf_counter()

        mode = self._lightrag_mode
        graph_available = self._has_graph and self._kg and self._extractor

        if mode == "naive" or not graph_available:
            result = self._vector_retriever.retrieve_by_vector(
                query_text, query_vector, top_k=k
            )
        else:
            result = self._retrieve_via_kg(query_text, k)

        result.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        result.strategy_used = (
            RetrievalStrategy.LIGHT_RAG if graph_available and mode != "naive"
            else RetrievalStrategy.SIMPLE_VECTOR
        )
        result.metadata["graph_active"] = graph_available and mode != "naive"
        result.metadata["lightrag_mode"] = mode
        return result

    # -----------------------------------------------------------------
    # KG-based retrieval (paper-aligned)
    # -----------------------------------------------------------------

    _MAX_CONTEXT_ENTITIES = 30
    _MAX_CONTEXT_RELATIONS = 30

    def _retrieve_via_kg(
        self,
        query: str,
        top_k: int,
    ) -> RetrievalResult:
        """Retrieval paper-aligned: chunks obtenidos a traves del KG.

        Flujo:
          1. Extraer query keywords (low-level + high-level)
          2. Resolver keywords contra Entity VDB, Relationship VDB y
             Chunk Keywords VDB (divergencia #10)
          3. Obtener source_doc_ids de entidades/relaciones + doc_ids
             directos del canal de chunk keywords
          4. Scoring: cada doc_id acumula score de los tres canales
          5. Fetch contenido de chunks desde el vector store
          6. Fallback a vector search si el KG no produce resultados

        Scoring formula (simetrica entre canales):
          entidades:       sum(1/(1+rank) * similarity)
          relaciones:      sum(1/(1+rank) * similarity * edge_weight)
          chunk keywords:  sum(1/(1+rank) * similarity)   [divergencia #10]

        Donde similarity = max(0, 1 - cosine_distance/2), rank es la
        posicion en los resultados del VDB (0 = mas relevante), y
        edge_weight es el numero de docs que mencionan la relacion. El
        canal de chunk keywords no pondera por weight — no hay equivalente
        de edge_weight para chunks (cada chunk es una unidad).

        Diferencias con el paper (HKUDS/LightRAG operate.py):
        - El paper usa un contador descendiente `order = N - i` para
          decay por posicion; aqui usamos `1/(1+rank)` (inverse-rank).
          Ambos producen el mismo efecto (posiciones altas dominan)
          pero con curva de decay distinta: inverse-rank decae mas
          rapido (1.0, 0.5, 0.33...) que lineal (N, N-1, N-2...).
        - Las entidades no ponderan por weight porque el Entity VDB
          no almacena un peso equivalente al edge_weight de relaciones.
          len(entity.source_doc_ids) seria un proxy razonable (mas docs
          = entidad mas relevante) pero el paper no lo hace
          explicitamente para el canal de entidades.
        """
        if self._kg is None:
            raise RuntimeError("_retrieve_via_kg llamado sin KnowledgeGraph")
        if self._extractor is None:
            raise RuntimeError("_retrieve_via_kg llamado sin TripletExtractor")

        low_level, high_level = self._get_query_keywords(query)

        if not low_level and not high_level:
            logger.debug(
                "KG retrieval: no keywords extracted, falling back to vector search"
            )
            result = self._vector_retriever.retrieve(query, top_k=top_k)
            result.metadata["kg_fallback"] = "no_keywords"
            result.metadata["query_keywords"] = {"low": [], "high": []}
            result.metadata["kg_entities"] = []
            result.metadata["kg_relations"] = []
            result.metadata["kg_chunk_keyword_matches"] = 0
            result.metadata["kg_entities_with_neighbors"] = 0
            result.metadata["kg_mean_neighbors_per_entity"] = 0.0
            return result

        use_local = self._lightrag_mode in ("local", "hybrid")
        use_global = self._lightrag_mode in ("global", "hybrid")

        resolved_entities: List[Tuple[str, float]] = []
        kg_entities: List[Dict[str, Any]] = []
        if use_local and low_level:
            resolved_entities = self._resolve_entities_via_vdb(low_level)
            if resolved_entities:
                entities = self._kg.get_all_entities()
                for name, _ in resolved_entities:
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
                                "Neighbor lookup failed for %s, continuing without",
                                name,
                            )
                            record_operational_event("neighbor_lookup_failure")
                        kg_entities.append(entry)
                    if len(kg_entities) >= self._MAX_CONTEXT_ENTITIES:
                        break

        resolved_relations: List[Dict[str, Any]] = []
        if use_global and high_level:
            resolved_relations = self._resolve_relations_for_context(high_level)

        # Divergencia #10: canal de chunk keywords (solo paths que usan
        # high-level: global/hybrid). Resuelve query.high_level contra la
        # Chunk Keywords VDB. Si el flag esta off o la VDB no existe,
        # resolved_chunk_matches queda vacio y no contribuye al scoring.
        resolved_chunk_matches: List[Tuple[str, float]] = []
        if (
            use_global
            and high_level
            and self.config.kg_chunk_keywords_enabled
            and self._chunk_keywords_vdb is not None
        ):
            resolved_chunk_matches = self._resolve_chunks_via_keywords_vdb(
                high_level,
                top_k=self.config.kg_chunk_keywords_top_k,
            )

        doc_scores: Dict[str, float] = {}

        if resolved_entities:
            all_entities = self._kg.get_all_entities()
            for rank, (name, distance) in enumerate(resolved_entities):
                entity = all_entities.get(name)
                if entity:
                    similarity = max(0.0, 1.0 - distance / 2.0)
                    order_score = 1.0 / (1 + rank)
                    ent_score = order_score * similarity
                    for doc_id in entity.source_doc_ids:
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + ent_score

        if resolved_relations:
            for rank, rel in enumerate(resolved_relations):
                source = rel.get("source", "")
                target = rel.get("target", "")
                distance = rel.get("vdb_distance", 0.0)
                weight = rel.get("weight", 1)
                similarity = max(0.0, 1.0 - distance / 2.0)
                order_score = 1.0 / (1 + rank)
                rel_score = order_score * similarity * weight
                rel_doc_ids: set = set()
                for ent_name in (source, target):
                    entity = self._kg.get_entity(ent_name)
                    if entity:
                        rel_doc_ids.update(entity.source_doc_ids)
                for doc_id in rel_doc_ids:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rel_score

        # Divergencia #10: contribucion del canal de chunk keywords
        if resolved_chunk_matches:
            for rank, (doc_id, distance) in enumerate(resolved_chunk_matches):
                similarity = max(0.0, 1.0 - distance / 2.0)
                order_score = 1.0 / (1 + rank)
                ck_score = order_score * similarity
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + ck_score

        if not doc_scores:
            logger.debug(
                "KG retrieval: no doc_ids from KG, falling back to vector search"
            )
            result = self._vector_retriever.retrieve(query, top_k=top_k)
            result.metadata["kg_fallback"] = "no_doc_ids"
            result.metadata["query_keywords"] = {
                "low": low_level, "high": high_level,
            }
            result.metadata["kg_entities"] = kg_entities
            result.metadata["kg_relations"] = resolved_relations
            result.metadata["kg_chunk_keyword_matches"] = len(resolved_chunk_matches)
            with_nb, mean_nb = _neighbor_coverage_stats(kg_entities)
            result.metadata["kg_entities_with_neighbors"] = with_nb
            result.metadata["kg_mean_neighbors_per_entity"] = mean_nb
            return result

        ranked_doc_ids = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True,
        )[:top_k]

        target_ids = [did for did, _ in ranked_doc_ids]
        contents_map = self._vector_retriever.get_documents_by_ids(target_ids)

        doc_ids: List[str] = []
        contents: List[str] = []
        scores: List[float] = []
        for did, score in ranked_doc_ids:
            content = contents_map.get(did, "")
            if content:
                doc_ids.append(did)
                contents.append(content)
                scores.append(score)

        if not doc_ids:
            logger.debug(
                "KG retrieval: doc_ids from KG not found in vector store, "
                "falling back to vector search"
            )
            result = self._vector_retriever.retrieve(query, top_k=top_k)
            result.metadata["kg_fallback"] = "docs_not_in_store"
            result.metadata["query_keywords"] = {
                "low": low_level, "high": high_level,
            }
            result.metadata["kg_entities"] = kg_entities
            result.metadata["kg_relations"] = resolved_relations
            result.metadata["kg_chunk_keyword_matches"] = len(resolved_chunk_matches)
            with_nb, mean_nb = _neighbor_coverage_stats(kg_entities)
            result.metadata["kg_entities_with_neighbors"] = with_nb
            result.metadata["kg_mean_neighbors_per_entity"] = mean_nb
            return result

        logger.debug(
            "KG retrieval: %d chunks via KG (%d entities, %d relations, "
            "%d chunk keyword matches) for query '%s...'",
            len(doc_ids), len(kg_entities), len(resolved_relations),
            len(resolved_chunk_matches), query[:60],
        )

        with_nb, mean_nb = _neighbor_coverage_stats(kg_entities)
        return RetrievalResult(
            doc_ids=doc_ids,
            contents=contents,
            scores=scores,
            metadata={
                "query_keywords": {"low": low_level, "high": high_level},
                "kg_entities": kg_entities,
                "kg_relations": resolved_relations,
                "kg_chunk_keyword_matches": len(resolved_chunk_matches),
                "kg_entities_with_neighbors": with_nb,
                "kg_mean_neighbors_per_entity": mean_nb,
                "kg_doc_scores": {did: s for did, s in ranked_doc_ids},
                "kg_fallback": None,
            },
        )

    def _resolve_relations_for_context(
        self,
        keywords: List[str],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Resuelve high-level keywords a relaciones con sus descripciones.

        Divergencia #9: enriquece cada relacion con las descripciones y tipos
        de sus entidades endpoint (source/target) desde el KG.

        Cada relacion incluye vdb_distance y weight para que el scoring
        en _retrieve_via_kg pueda calcular order × similarity × weight
        (paper-aligned: _find_most_related_text_unit_from_relationships).
        """
        if not self._relationships_vdb or not keywords:
            return []

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
                    "vdb_distance": float(distance),
                    "weight": doc.metadata.get("weight", 1),
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
        """Extrae keywords de query con cache thread-safe y LRU eviction."""
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

        # Filtrar queries ya cacheadas (acceso al cache bajo lock).
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
