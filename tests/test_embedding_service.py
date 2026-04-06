"""
Tests unitarios para sandbox_mteb/embedding_service.py (Fase I.3).

Cobertura:
  E1. query_model_context_window con respuesta valida
  E2. query_model_context_window con respuesta sin max_model_len
  E3. query_model_context_window con error de red
  E4. resolve_max_context_chars con override manual
  E5. resolve_max_context_chars con auto-deteccion
  E6. resolve_max_context_chars fallback 4000
  E7. batch_embed_queries retorna vectores correctos
  E8. batch_embed_queries retorna [] en fallo
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sandbox_mteb.embedding_service import (
    query_model_context_window,
    resolve_max_context_chars,
    batch_embed_queries,
)
from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
from shared.config_base import InfraConfig, RerankerConfig
from shared.retrieval.core import RetrievalConfig


def _make_config(**overrides):
    defaults = dict(
        infra=InfraConfig(
            embedding_base_url="http://fake:8000/v1",
            embedding_model_name="test-embed",
            embedding_model_type="asymmetric",
            embedding_batch_size=2,
            llm_base_url="http://fake-llm:8000/v1",
            llm_model_name="test-llm",
        ),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(),
        reranker=RerankerConfig(),
        generation_enabled=True,
        generation_max_context_chars=0,
    )
    defaults.update(overrides)
    return MTEBConfig(**defaults)


# =============================================================================
# E1-E3: query_model_context_window
# =============================================================================

@patch("sandbox_mteb.embedding_service.urllib.request.urlopen")
def test_query_context_window_valid(mock_urlopen):
    """Respuesta valida retorna max_model_len."""
    response_data = json.dumps({
        "data": [{"id": "test-model", "max_model_len": 8192}]
    }).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_data
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    result = query_model_context_window("http://fake:8000/v1")
    assert result == 8192


@patch("sandbox_mteb.embedding_service.urllib.request.urlopen")
def test_query_context_window_no_max_model_len(mock_urlopen):
    """Respuesta sin max_model_len retorna None."""
    response_data = json.dumps({
        "data": [{"id": "test-model"}]
    }).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_data
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    result = query_model_context_window("http://fake:8000/v1")
    assert result is None


@patch("sandbox_mteb.embedding_service.urllib.request.urlopen")
def test_query_context_window_network_error(mock_urlopen):
    """Error de red retorna None."""
    mock_urlopen.side_effect = ConnectionError("timeout")

    result = query_model_context_window("http://fake:8000/v1")
    assert result is None


# =============================================================================
# E4-E6: resolve_max_context_chars
# =============================================================================

def test_resolve_manual_override():
    """Override manual tiene prioridad."""
    config = _make_config(generation_max_context_chars=5000)
    assert resolve_max_context_chars(config) == 5000


@patch("sandbox_mteb.embedding_service.query_model_context_window")
def test_resolve_auto_detection(mock_query):
    """Auto-deteccion del modelo: (max_model_len - 1024) * 4.0."""
    mock_query.return_value = 8192
    config = _make_config(generation_max_context_chars=0)

    result = resolve_max_context_chars(config)
    expected = int((8192 - 1024) * 4.0)  # 28672
    assert result == expected


@patch("sandbox_mteb.embedding_service.query_model_context_window")
def test_resolve_fallback(mock_query):
    """Sin override ni auto-deteccion, usa fallback 4000."""
    mock_query.return_value = None
    config = _make_config(generation_max_context_chars=0)

    result = resolve_max_context_chars(config)
    assert result == 4000


# =============================================================================
# E7-E8: batch_embed_queries
# =============================================================================

@patch("sandbox_mteb.embedding_service.urllib.request.urlopen")
def test_batch_embed_success(mock_urlopen):
    """batch_embed retorna vectores en orden."""
    response_data = json.dumps({
        "data": [
            {"index": 1, "embedding": [0.2, 0.3]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    }).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_data
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    config = _make_config()
    result = batch_embed_queries(["q1", "q2"], config)

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]  # index 0 first
    assert result[1] == [0.2, 0.3]  # index 1 second


@patch("sandbox_mteb.embedding_service.urllib.request.urlopen")
def test_batch_embed_failure_returns_empty(mock_urlopen):
    """Fallo en batch retorna lista vacia tras retries."""
    mock_urlopen.side_effect = ConnectionError("timeout")

    config = _make_config()
    result = batch_embed_queries(["q1"], config)
    assert result == []
