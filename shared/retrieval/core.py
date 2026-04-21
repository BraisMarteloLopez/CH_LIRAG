"""Tipos base, configuracion, y SimpleVectorRetriever."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional, Tuple, cast, get_args

# Tipos cerrados (R2): documentan y acotan el dominio de valores validos.
LightRAGMode = Literal["naive", "local", "global", "hybrid"]
# Razones posibles por las que un retrieval LIGHT_RAG cae al fallback
# vector directo. Anotadas en `RetrievalResult.metadata["kg_fallback"]`;
# None cuando no hubo fallback.
KGFallbackReason = Literal["no_keywords", "no_doc_ids", "docs_not_in_store"]


def _parse_lightrag_mode(raw: str) -> LightRAGMode:
    """Valida LIGHTRAG_MODE y lo estrecha al tipo Literal."""
    valid = get_args(LightRAGMode)
    if raw not in valid:
        raise ValueError(
            f"LIGHTRAG_MODE='{raw}' invalido; validos: {valid}"
        )
    return cast(LightRAGMode, raw)

from shared.types import EmbeddingModelProtocol

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Estrategias de retrieval disponibles."""
    SIMPLE_VECTOR = auto()
    LIGHT_RAG = auto()


@dataclass
class RetrievalConfig:
    """
    Configuracion para una estrategia de retrieval.
    Todos los parametros operativos viajan en este objeto.
    """
    strategy: RetrievalStrategy = RetrievalStrategy.SIMPLE_VECTOR
    retrieval_k: int = 20

    # HNSW (ChromaDB): num_threads=1 reduce no-determinismo del grafo
    # (elimina variabilidad de threading). No garantiza reproducibilidad
    # total: ChromaDB no expone hnsw:random_seed (deuda #3 en CLAUDE.md).
    hnsw_num_threads: int = 1

    # Knowledge graph (LIGHT_RAG)
    kg_max_hops: int = 1  # 1-hop como el paper original
    kg_max_text_chars: int = 3000
    kg_max_entities: int = 0
    kg_cache_dir: str = ""
    kg_keyword_max_tokens: int = 1024  # max_tokens para keyword extraction LLM call
    kg_extraction_max_tokens: int = 4096  # max_tokens para extraction LLM call
    kg_batch_docs_per_call: int = 5  # docs por LLM call en batch extraction
    kg_gleaning_rounds: int = 0  # rounds de re-extraccion (0 = desactivado)
    lightrag_mode: LightRAGMode = "hybrid"  # Modos del paper (ver LightRAGMode).
    kg_max_neighbors_per_entity: int = 5  # 1-hop neighbors por entidad resuelta

    # Chunks enviados al LLM para generacion (post KG-scoring).
    # Paper: CHUNK_TOP_K=20 (con GPT-4o-mini). El KG ya rankea los chunks
    # por scoring agregado `1/(1+rank) × similarity [× edge_weight]`; este
    # parametro selecciona los top-N para generacion, analogo a
    # reranker.top_n en SimpleVector.
    # 0 = usar retrieval_k (todos los docs van a generacion).
    lightrag_generation_top_n: int = 0

    # LLM synthesis para merge de descripciones de entidades multi-doc.
    kg_description_synthesis: bool = False
    kg_synthesis_char_threshold: int = 200  # chars minimos para trigger LLM synthesis

    # Divergencia #10: chunk high-level keywords VDB (tercer canal del path
    # high-level). Cuando esta activado, `global`/`hybrid` resuelven query
    # high_level keywords contra la Chunk Keywords VDB ademas de contra
    # el Relationship VDB, y los chunks matched contribuyen al doc_scores.
    kg_chunk_keywords_enabled: bool = True
    kg_chunk_keywords_top_k: int = 20  # top-K devueltos por keyword al consultar la VDB

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        from shared.config_base import _env, _env_int, _env_float
        return cls(
            strategy=RetrievalStrategy[_env("RETRIEVAL_STRATEGY", "SIMPLE_VECTOR")],
            retrieval_k=_env_int("RETRIEVAL_K", 20),
            hnsw_num_threads=_env_int("HNSW_NUM_THREADS", 1),
            kg_max_hops=_env_int("KG_MAX_HOPS", 1),
            kg_max_text_chars=_env_int("KG_MAX_TEXT_CHARS", 3000),
            kg_max_entities=_env_int("KG_MAX_ENTITIES", 0),
            kg_cache_dir=_env("KG_CACHE_DIR", ""),
            kg_keyword_max_tokens=_env_int("KG_KEYWORD_MAX_TOKENS", 1024),
            kg_extraction_max_tokens=_env_int("KG_EXTRACTION_MAX_TOKENS", 4096),
            kg_batch_docs_per_call=_env_int("KG_BATCH_DOCS_PER_CALL", 5),
            kg_gleaning_rounds=_env_int("KG_GLEANING_ROUNDS", 0),
            lightrag_mode=_parse_lightrag_mode(_env("LIGHTRAG_MODE", "hybrid")),
            kg_max_neighbors_per_entity=_env_int("KG_MAX_NEIGHBORS_PER_ENTITY", 5),
            lightrag_generation_top_n=_env_int("LIGHTRAG_GENERATION_TOP_N", 0),
            kg_description_synthesis=_env("KG_DESCRIPTION_SYNTHESIS", "false").lower() == "true",
            kg_synthesis_char_threshold=_env_int("KG_SYNTHESIS_CHAR_THRESHOLD", 200),
            kg_chunk_keywords_enabled=_env("KG_CHUNK_KEYWORDS_ENABLED", "true").lower() == "true",
            kg_chunk_keywords_top_k=_env_int("KG_CHUNK_KEYWORDS_TOP_K", 20),
        )


@dataclass
class RetrievalResult:
    """Resultado de una operacion de retrieval."""
    doc_ids: List[str]
    contents: List[str]
    scores: List[float]

    vector_scores: Optional[List[float]] = None

    retrieval_time_ms: float = 0.0
    strategy_used: RetrievalStrategy = RetrievalStrategy.SIMPLE_VECTOR
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Clase base abstracta para implementaciones de retrieval."""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._is_indexed = False

    @abstractmethod
    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievalResult:
        pass

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieval usando vector pre-computado.

        Evita la llamada REST al NIM de embeddings por query.
        Default: fallback a retrieve() (subclases pueden optimizar).
        """
        return self.retrieve(query_text, top_k)

    @abstractmethod
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        pass

    def clear_index(self) -> None:
        self._is_indexed = False

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed


class SimpleVectorRetriever(BaseRetriever):
    """Retriever usando solo busqueda vectorial (ChromaDB)."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embedding_batch_size = embedding_batch_size
        self._vector_store: Any = None  # ChromaVectorStore, lazy import

    @property
    def active_collection_name(self) -> Optional[str]:
        """Nombre real de la coleccion ChromaDB activa, o None si no indexado.

        Consumidores externos (p.ej. LightRAGRetriever construyendo VDBs
        auxiliares) deben usar esta property en vez de acceder a
        `_vector_store.collection_name` directamente.
        """
        if self._vector_store is not None:
            return self._vector_store.collection_name
        return self.collection_name

    def _init_vector_store(self, collection_name: str) -> None:
        from shared.vector_store import ChromaStoreConfig, ChromaVectorStore

        store_config: ChromaStoreConfig = {
            "CHROMA_COLLECTION_NAME": collection_name,
            "EMBEDDING_BATCH_SIZE": self.embedding_batch_size,
            "HNSW_NUM_THREADS": self.config.hnsw_num_threads,
        }
        self._vector_store = ChromaVectorStore(
            store_config, self.embedding_model
        )
        logger.debug(f"Vector store inicializado: {collection_name}")

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        try:
            from langchain_core.documents import Document

            name = (
                collection_name or self.collection_name or "default_collection"
            )
            self._init_vector_store(name)

            lc_docs = [
                Document(
                    page_content=doc.get("content", ""),
                    metadata={
                        "doc_id": doc.get("doc_id", ""),
                        "title": doc.get("title", ""),
                    },
                )
                for doc in documents
            ]

            self._vector_store.add_documents(lc_docs)
            self._is_indexed = True
            logger.info(f"Indexados {len(lc_docs)} documentos en '{name}'")
            return True

        except Exception as e:
            logger.error(f"Error indexando documentos: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        k = top_k or self.config.retrieval_k

        if not self._vector_store:
            logger.warning("Vector store no inicializado")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        start_time = time.perf_counter()

        try:
            results = self._vector_store.similarity_search_with_score(
                query, k=k
            )

            doc_ids = []
            contents = []
            scores = []

            for doc, score in results:
                doc_ids.append(doc.metadata.get("doc_id", ""))
                contents.append(doc.page_content)
                scores.append(float(score))

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return RetrievalResult(
                doc_ids=doc_ids,
                contents=contents,
                scores=scores,
                vector_scores=scores,
                retrieval_time_ms=elapsed_ms,
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        except Exception as e:
            logger.error(f"Error en retrieval: {e}")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
                metadata={"error": str(e)},
            )

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Busqueda vectorial usando embedding pre-computado."""
        k = top_k or self.config.retrieval_k

        if not self._vector_store:
            logger.warning("Vector store no inicializado")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        start_time = time.perf_counter()

        try:
            results = self._vector_store.similarity_search_by_vector_with_score(
                query_vector, k=k
            )

            doc_ids = []
            contents = []
            scores = []

            for doc, score in results:
                doc_ids.append(doc.metadata.get("doc_id", ""))
                contents.append(doc.page_content)
                scores.append(float(score))

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return RetrievalResult(
                doc_ids=doc_ids,
                contents=contents,
                scores=scores,
                vector_scores=scores,
                retrieval_time_ms=elapsed_ms,
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        except Exception as e:
            logger.error(f"Error en retrieval por vector: {e}")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
                metadata={"error": str(e)},
            )

    def get_documents_by_ids(self, doc_ids: List[str]) -> Dict[str, str]:
        """Recupera contenido de docs por doc_id desde el vector store.

        Retorna {doc_id: content}. Vacio si store no inicializado.
        """
        if not self._vector_store or not doc_ids:
            return {}
        result: Dict[str, str] = self._vector_store.get_documents_by_ids(doc_ids)
        return result

    def clear_index(self) -> None:
        if self._vector_store:
            self._vector_store.delete_all_documents()
        self._is_indexed = False
        logger.debug("Indice limpiado")


__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "BaseRetriever",
    "SimpleVectorRetriever",
]
