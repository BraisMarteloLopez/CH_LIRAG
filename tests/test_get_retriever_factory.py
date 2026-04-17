"""
Tests para shared/retrieval/__init__.py::get_retriever (factory).

Cobertura:
  F1. strategy=SIMPLE_VECTOR -> devuelve SimpleVectorRetriever
  F2. strategy=LIGHT_RAG -> devuelve LightRAGRetriever
  F3. strategy invalida -> ValueError
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.retrieval.core import RetrievalConfig, RetrievalStrategy


@patch("shared.retrieval.SimpleVectorRetriever")
def test_simple_vector_strategy(mock_svr_cls):
    """F1: SIMPLE_VECTOR -> instancia SimpleVectorRetriever."""
    from shared.retrieval import get_retriever

    config = RetrievalConfig(strategy=RetrievalStrategy.SIMPLE_VECTOR)
    embedding = MagicMock()
    mock_svr_cls.return_value = MagicMock()

    result = get_retriever(config, embedding, collection_name="test")

    mock_svr_cls.assert_called_once_with(
        config, embedding, "test", embedding_batch_size=0,
    )
    assert result is mock_svr_cls.return_value


@patch("shared.retrieval.LightRAGRetriever")
def test_lightrag_strategy(mock_lr_cls):
    """F2: LIGHT_RAG -> instancia LightRAGRetriever con params del config."""
    from shared.retrieval import get_retriever

    config = RetrievalConfig(strategy=RetrievalStrategy.LIGHT_RAG)
    embedding = MagicMock()
    llm = MagicMock()
    mock_lr_cls.return_value = MagicMock()

    result = get_retriever(
        config, embedding, collection_name="test",
        llm_service=llm, embedding_batch_size=32,
    )

    mock_lr_cls.assert_called_once_with(
        config=config,
        embedding_model=embedding,
        llm_service=llm,
        collection_name="test",
        embedding_batch_size=32,
        kg_max_hops=config.kg_max_hops,
        kg_max_text_chars=config.kg_max_text_chars,
        kg_max_entities=config.kg_max_entities,
        kg_cache_dir=config.kg_cache_dir,
    )
    assert result is mock_lr_cls.return_value


def test_invalid_strategy_raises():
    """F3: estrategia no soportada -> ValueError."""
    from shared.retrieval import get_retriever

    config = RetrievalConfig()
    fake_strategy = MagicMock()
    fake_strategy.name = "UNSUPPORTED"
    config.strategy = fake_strategy
    embedding = MagicMock()

    with pytest.raises(ValueError):
        get_retriever(config, embedding)
