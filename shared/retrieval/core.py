"""
Modulo: Retrieval Core
Descripcion: Tipos base, configuracion, y SimpleVectorRetriever.

Ubicacion: shared/retrieval/core.py

Consolida base.py + simple_retriever.py en un unico archivo.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from shared.types import EmbeddingModelProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERACIONES
# =============================================================================

class RetrievalStrategy(Enum):
    """Estrategias de retrieval disponibles."""
    SIMPLE_VECTOR = auto()
    LIGHT_RAG = auto()


# =============================================================================
# CONFIGURACION
# =============================================================================

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
    # total: ChromaDB no soporta hnsw:random_seed. Ver DTm-13.
    hnsw_num_threads: int = 1

    # Graph expansion cap (LIGHT_RAG). 0 = sin limite.
    max_graph_expansion: int = 30

    # Knowledge graph (LIGHT_RAG)
    kg_max_hops: int = 1  # DAM-7: 1-hop como el original, configurable via KG_MAX_HOPS
    kg_max_text_chars: int = 3000
    kg_max_entities: int = 0
    kg_graph_weight: float = 0.3
    kg_vector_weight: float = 0.7
    kg_cache_dir: str = ""  # Directorio para persistir KG entre runs (DTm-34)
    kg_fusion_method: str = "rrf"  # "rrf" (default) o "linear"
    kg_rrf_k: int = 60  # Constante k para RRF
    kg_keyword_max_tokens: int = 1024  # max_tokens para keyword extraction LLM call
    kg_extraction_max_tokens: int = 4096  # max_tokens para extraction LLM call (DTm-66)
    kg_batch_docs_per_call: int = 5  # docs por LLM call en batch extraction (DTm-67)
    kg_graph_overfetch_factor: int = 2  # graph traversal pide N * top_k candidatos
    kg_gleaning_rounds: int = 0  # DAM-6: rounds de re-extraccion (0 = desactivado)
    lightrag_mode: str = "hybrid"  # F.4/DTm-79: "hybrid" (default), "graph_primary", "local", "global", "naive"

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        from shared.config_base import _env, _env_int, _env_float
        return cls(
            strategy=RetrievalStrategy[_env("RETRIEVAL_STRATEGY", "SIMPLE_VECTOR")],
            retrieval_k=_env_int("RETRIEVAL_K", 20),
            hnsw_num_threads=_env_int("HNSW_NUM_THREADS", 1),
            max_graph_expansion=_env_int("MAX_GRAPH_EXPANSION", 30),
            kg_max_hops=_env_int("KG_MAX_HOPS", 1),
            kg_max_text_chars=_env_int("KG_MAX_TEXT_CHARS", 3000),
            kg_max_entities=_env_int("KG_MAX_ENTITIES", 0),
            kg_graph_weight=_env_float("KG_GRAPH_WEIGHT", 0.3),
            kg_vector_weight=_env_float("KG_VECTOR_WEIGHT", 0.7),
            kg_cache_dir=_env("KG_CACHE_DIR", ""),
            kg_fusion_method=_env("KG_FUSION_METHOD", "rrf"),
            kg_rrf_k=_env_int("KG_RRF_K", 60),
            kg_keyword_max_tokens=_env_int("KG_KEYWORD_MAX_TOKENS", 1024),
            kg_extraction_max_tokens=_env_int("KG_EXTRACTION_MAX_TOKENS", 4096),
            kg_batch_docs_per_call=_env_int("KG_BATCH_DOCS_PER_CALL", 5),
            kg_graph_overfetch_factor=_env_int("KG_GRAPH_OVERFETCH_FACTOR", 2),
            kg_gleaning_rounds=_env_int("KG_GLEANING_ROUNDS", 0),
            lightrag_mode=_env("LIGHTRAG_MODE", "hybrid"),
        )


# =============================================================================
# RESULTADO
# =============================================================================

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


# =============================================================================
# CLASE BASE ABSTRACTA
# =============================================================================

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


# =============================================================================
# SIMPLE VECTOR RETRIEVER
# =============================================================================

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

    def _init_vector_store(self, collection_name: str) -> None:
        from shared.vector_store import ChromaVectorStore

        store_config = {
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
        return self._vector_store.get_documents_by_ids(doc_ids)

    def clear_index(self) -> None:
        if self._vector_store:
            self._vector_store.delete_all_documents()
        self._is_indexed = False
        logger.debug("Indice limpiado")


# =============================================================================
# RECIPROCAL RANK FUSION (RRF)
# =============================================================================

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Fusiona multiples rankings usando RRF.
    RRF_score(d) = SUM weight_i / (k + rank_i(d))
    """
    if not rankings:
        return []

    num_rankings = len(rankings)

    if weights is None or len(weights) != num_rankings:
        if weights is not None and len(weights) != num_rankings:
            logger.warning(
                f"weights ({len(weights)}) != rankings ({num_rankings}). "
                "Pesos iguales."
            )
        weights = [1.0 / num_rankings] * num_rankings

    rrf_scores: Dict[str, float] = {}

    for ranking_idx, ranking in enumerate(rankings):
        weight = weights[ranking_idx]
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            rrf_contribution = weight / (k + rank)
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0) + rrf_contribution
            )

    sorted_results = sorted(
        rrf_scores.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_results[:top_n]


__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "BaseRetriever",
    "SimpleVectorRetriever",
    "reciprocal_rank_fusion",
]
