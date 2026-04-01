"""
Subpaquete: Hybrid Retrieval
Estrategias: HYBRID_PLUS (BM25 + Vector + RRF + entity cross-linking)

Re-exporta las clases publicas para mantener compatibilidad con
shared.retrieval.__init__.py.
"""

from .retriever import HybridRetriever, reciprocal_rank_fusion, HAS_BM25, HAS_TANTIVY
from .plus_retriever import HybridPlusRetriever
from .entity_linker import HAS_SPACY
from .tantivy_index import TantivyIndex

__all__ = [
    "HybridRetriever",
    "HybridPlusRetriever",
    "HAS_BM25",
    "HAS_TANTIVY",
    "HAS_SPACY",
    "TantivyIndex",
    "reciprocal_rank_fusion",
]
