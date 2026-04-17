"""
Factories compartidas para tests que usan object.__new__().

Centraliza la creacion de objetos sin pasar por __init__ (que conecta
a infra real: ChromaDB, NVIDIARerank, etc.). Un solo punto de verdad
por clase: si el __init__ de produccion anade un atributo, se actualiza
aqui y todos los tests lo heredan.

Referencia: TESTS.md seccion "Patron object.__new__()".
"""

import threading
from collections import OrderedDict
from unittest.mock import MagicMock

from shared.retrieval.core import RetrievalConfig
from shared.retrieval.lightrag.knowledge_graph import KnowledgeGraph
from shared.retrieval.lightrag.retriever import LightRAGRetriever


def make_lightrag(
    kg=None,
    extractor=None,
    vector_retriever=None,
    lightrag_mode="hybrid",
    has_graph=True,
    max_neighbors_per_entity=5,
):
    """Crea LightRAGRetriever con dependencias mockeadas.

    Args:
        kg: KnowledgeGraph mock. Si None y has_graph=True, crea uno con
            defaults seguros (get_neighbors_ranked=[], get_all_entities={}).
        extractor: TripletExtractor mock. Si None y has_graph=True, crea MagicMock.
        vector_retriever: SimpleVectorRetriever mock. Si None, crea MagicMock.
        lightrag_mode: "hybrid", "local", "global", "naive".
        has_graph: Si False, kg y extractor se fijan a None.
        max_neighbors_per_entity: default 5 (paper-aligned).
    """
    retriever = object.__new__(LightRAGRetriever)
    retriever.config = RetrievalConfig()
    retriever._kg_max_hops = 2

    if has_graph:
        mock_kg = kg or MagicMock(spec=KnowledgeGraph)
        if not kg:
            mock_kg.get_neighbors_ranked.return_value = []
            mock_kg.get_all_entities.return_value = {}
        retriever._kg = mock_kg
        retriever._extractor = extractor or MagicMock()
    else:
        retriever._kg = None
        retriever._extractor = None

    retriever._has_graph = has_graph
    retriever._lightrag_mode = lightrag_mode
    retriever._max_neighbors_per_entity = max_neighbors_per_entity
    retriever._query_keywords_cache = OrderedDict()
    retriever._cache_lock = threading.Lock()
    retriever._QUERY_CACHE_MAX_SIZE = 10_000
    retriever._vector_retriever = vector_retriever or MagicMock()
    retriever._entities_vdb = None
    retriever._relationships_vdb = None
    return retriever
