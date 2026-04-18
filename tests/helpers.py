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

from pathlib import Path

from shared.retrieval.core import RetrievalConfig, SimpleVectorRetriever
from shared.retrieval.lightrag.knowledge_graph import KnowledgeGraph
from shared.retrieval.lightrag.retriever import LightRAGRetriever
from shared.retrieval.lightrag.triplet_extractor import TripletExtractor
from shared.retrieval.reranker import CrossEncoderReranker
from shared.vector_store import ChromaVectorStore


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
    # Divergencia #10: Chunk Keywords VDB (tercer canal high-level).
    # None por defecto; tests que quieran ejercitar el canal lo setean.
    retriever._chunk_keywords_vdb = None
    return retriever


def make_extractor(mock_llm=None, max_text_chars=3000, batch_size=64):
    """Crea TripletExtractor con LLM mockeado.

    Args:
        mock_llm: AsyncLLMService mock. Si None, crea MagicMock.
        max_text_chars: limite de chars por doc para extraccion.
        batch_size: tamano de batch de coroutines.
    """
    ext = object.__new__(TripletExtractor)
    ext._llm = mock_llm or MagicMock()
    ext._max_text_chars = max_text_chars
    ext._keyword_max_tokens = 1024
    ext._extraction_max_tokens = 4096
    ext._batch_size = batch_size
    ext._stats = {
        "docs_processed": 0, "docs_success": 0, "docs_failed": 0,
        "docs_empty_input": 0, "docs_empty_result": 0,
        "docs_json_recovered": 0,
        "total_entities": 0, "total_relations": 0,
        # Divergencia #10
        "docs_with_keywords": 0, "total_chunk_keywords": 0,
    }
    return ext


def make_reranker():
    """Crea CrossEncoderReranker con _reranker mockeado.

    El caller configura compress_documents.return_value o .side_effect
    segun el escenario del test.
    """
    reranker = object.__new__(CrossEncoderReranker)
    reranker.base_url = "mock"
    reranker.model_name = "mock"
    reranker._reranker = MagicMock()
    return reranker


def make_retriever(with_store=False):
    """Crea SimpleVectorRetriever sin ChromaDB real.

    Args:
        with_store: Si True, _vector_store es MagicMock (ready to configure).
                    Si False, _vector_store es None (uninitialized).
    """
    r = object.__new__(SimpleVectorRetriever)
    r.config = RetrievalConfig()
    r._vector_store = MagicMock() if with_store else None
    r._is_indexed = False
    r.embedding_model = MagicMock()
    r.collection_name = "test_col"
    r.embedding_batch_size = 50
    return r


def make_vector_store(batch_size=0):
    """Crea ChromaVectorStore sin ChromaDB real."""
    store = object.__new__(ChromaVectorStore)
    store.collection_name = "test_collection"
    store.persist_directory = None
    store.embedding_model = MagicMock()
    store.batch_size = batch_size
    store._hnsw_num_threads = 1
    store._hnsw_space = None
    store._collection_metadata = {"hnsw:num_threads": 1}
    store._store = MagicMock()
    store._client = MagicMock()
    store._document_count = 0
    store._CHROMA_IN_BATCH_SIZE = 100
    return store


def make_loader(mock_client=None):
    """Crea MinIOLoader sin boto3 real.

    Args:
        mock_client: boto3 client mock. Si None, crea MagicMock.
    """
    from sandbox_mteb.loader import MinIOLoader
    loader = object.__new__(MinIOLoader)
    loader.endpoint = "http://fake:9000"
    loader.bucket = "test-bucket"
    loader.prefix = "datasets/eval"
    loader.cache_dir = Path("/tmp/test_cache")
    loader.client = mock_client or MagicMock()
    loader._manifest = None
    return loader
