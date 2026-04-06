"""
Retrieval strategies for RAG evaluation.

Estrategias soportadas:
  - SIMPLE_VECTOR: embedding search puro via ChromaDB
  - LIGHT_RAG: Vector + Knowledge Graph dual-level (LLM triplet extraction)
"""

import logging
from typing import Optional

from shared.types import EmbeddingModelProtocol

from .core import (
    RetrievalStrategy,
    RetrievalConfig,
    RetrievalResult,
    BaseRetriever,
    SimpleVectorRetriever,
)
from .lightrag import LightRAGRetriever, HAS_IGRAPH

logger = logging.getLogger(__name__)


def get_retriever(
    config: RetrievalConfig,
    embedding_model: EmbeddingModelProtocol,
    collection_name: Optional[str] = None,
    embedding_batch_size: int = 0,
    llm_service: Optional[object] = None,
) -> BaseRetriever:
    """
    Factory para obtener un retriever segun la estrategia en config.

    Args:
        config: Configuracion de retrieval con estrategia seleccionada.
        embedding_model: Modelo de embeddings (NVIDIAEmbeddings o compatible).
        collection_name: Nombre de la coleccion ChromaDB.
        embedding_batch_size: Batch size para embeddings (0 = default).
        llm_service: AsyncLLMService (requerido para LIGHT_RAG).

    Returns:
        BaseRetriever configurado segun la estrategia.
    """
    strategy = config.strategy
    logger.info(f"Factory: creando retriever {strategy.name}")

    if strategy == RetrievalStrategy.SIMPLE_VECTOR:
        return SimpleVectorRetriever(
            config, embedding_model, collection_name,
            embedding_batch_size=embedding_batch_size,
        )

    if strategy == RetrievalStrategy.LIGHT_RAG:
        return LightRAGRetriever(
            config=config,
            embedding_model=embedding_model,
            llm_service=llm_service,
            collection_name=collection_name,
            embedding_batch_size=embedding_batch_size,
            kg_max_hops=config.kg_max_hops,
            kg_max_text_chars=config.kg_max_text_chars,
            kg_max_entities=config.kg_max_entities,
            graph_weight=config.kg_graph_weight,
            vector_weight=config.kg_vector_weight,
            kg_cache_dir=config.kg_cache_dir,
        )

    raise ValueError(f"Estrategia no soportada: {strategy}")


__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "BaseRetriever",
    "SimpleVectorRetriever",
    "LightRAGRetriever",
    "HAS_IGRAPH",
    "get_retriever",
]
