"""
Tests unitarios para MinIOLoader (P8, Fase 4).

Cobertura:
  LO1. check_connection exitosa
  LO2. check_connection falla
  LO3. _populate_from_dataframes crea queries y corpus correctamente
  LO4. _populate_from_dataframes maneja qrels
  LO5. _populate_from_dataframes con qrels None
  LO6. load_dataset con error retorna LoadedDataset con status error
"""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from shared.types import (
    LoadedDataset,
    NormalizedQuery,
    NormalizedDocument,
    DatasetType,
    MetricType,
)
from sandbox_mteb.config import MinIOStorageConfig


# =============================================================================
# Helpers
# =============================================================================

class _FakeClientError(Exception):
    """Excepcion real que simula botocore.exceptions.ClientError.

    Necesaria porque conftest.py mockea botocore como MagicMock cuando
    no esta instalado, y `except MagicMock` no funciona en Python.
    """
    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"{operation_name}: {error_response}")


def _make_storage_config():
    return MinIOStorageConfig(
        minio_endpoint="http://fake:9000",
        minio_access_key="test",
        minio_secret_key="test",
        minio_bucket="test-bucket",
        s3_datasets_prefix="datasets/eval",
        datasets_cache_dir=Path("/tmp/test_cache"),
    )


def _make_mock_dataframe(data):
    """Crea un mock de DataFrame con iterrows()."""
    mock_df = MagicMock()
    mock_df.__len__ = MagicMock(return_value=len(data))

    class FakeRow:
        def __init__(self, d):
            self._d = d
        def get(self, key, default=""):
            return self._d.get(key, default)

    mock_df.iterrows.return_value = [
        (i, FakeRow(row)) for i, row in enumerate(data)
    ]
    return mock_df


# =============================================================================
# LO1: check_connection exitosa
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
def test_check_connection_success(mock_boto3):
    """head_bucket sin error -> True."""
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    from sandbox_mteb.loader import MinIOLoader
    loader = MinIOLoader(_make_storage_config())
    assert loader.check_connection() is True
    mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")


# =============================================================================
# LO2: check_connection falla
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_check_connection_failure(mock_boto3):
    """head_bucket lanza ClientError -> False."""
    mock_client = MagicMock()
    mock_client.head_bucket.side_effect = _FakeClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
    )
    mock_boto3.client.return_value = mock_client

    from sandbox_mteb.loader import MinIOLoader
    loader = object.__new__(MinIOLoader)
    loader.endpoint = "http://fake:9000"
    loader.bucket = "test-bucket"
    loader.prefix = "datasets/eval"
    loader.cache_dir = Path("/tmp/test_cache")
    loader.client = mock_client
    loader._manifest = None

    assert loader.check_connection() is False


# =============================================================================
# LO3: _populate_from_dataframes queries + corpus
# =============================================================================

def test_populate_from_dataframes_basic():
    """Puebla queries y corpus desde DataFrames."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _make_mock_dataframe([
        {"query_id": "q1", "text": "What is AI?", "answer": "Artificial Intelligence", "answer_type": "text"},
        {"query_id": "q2", "text": "Who is Bob?", "answer": "A person", "answer_type": "text"},
    ])
    corpus_df = _make_mock_dataframe([
        {"doc_id": "d1", "title": "AI", "text": "AI is a field of CS"},
        {"doc_id": "d2", "title": "People", "text": "Bob is a person"},
    ])

    result = LoadedDataset(
        name="test",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )
    MinIOLoader._populate_from_dataframes(result, queries_df, corpus_df, None)

    assert result.total_queries == 2
    assert result.total_corpus == 2
    assert len(result.queries) == 2
    assert len(result.corpus) == 2
    assert result.queries[0].query_id == "q1"
    assert result.queries[0].query_text == "What is AI?"
    assert "d1" in result.corpus
    assert result.corpus["d1"].content == "AI is a field of CS"


# =============================================================================
# LO4: _populate_from_dataframes con qrels
# =============================================================================

def test_populate_from_dataframes_with_qrels():
    """Qrels asignan relevant_doc_ids a queries."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _make_mock_dataframe([
        {"query_id": "q1", "text": "Q1?"},
    ])
    corpus_df = _make_mock_dataframe([
        {"doc_id": "d1", "title": "", "text": "doc1"},
        {"doc_id": "d2", "title": "", "text": "doc2"},
    ])
    qrels_df = _make_mock_dataframe([
        {"query_id": "q1", "doc_id": "d1"},
        {"query_id": "q1", "doc_id": "d2"},
    ])

    result = LoadedDataset(
        name="test",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )
    MinIOLoader._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

    assert result.queries[0].relevant_doc_ids == ["d1", "d2"]


# =============================================================================
# LO5: _populate_from_dataframes sin qrels
# =============================================================================

def test_populate_from_dataframes_no_qrels():
    """Sin qrels -> relevant_doc_ids vacios."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _make_mock_dataframe([
        {"query_id": "q1", "text": "Q1?"},
    ])
    corpus_df = _make_mock_dataframe([])

    result = LoadedDataset(
        name="test",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )
    MinIOLoader._populate_from_dataframes(result, queries_df, corpus_df, None)

    assert result.queries[0].relevant_doc_ids == []


# =============================================================================
# LO6: load_dataset con error
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_load_dataset_download_error(mock_boto3):
    """Error en descarga -> LoadedDataset con status error."""
    mock_client = MagicMock()
    mock_client.get_object.side_effect = _FakeClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
    )
    mock_boto3.client.return_value = mock_client

    from sandbox_mteb.loader import MinIOLoader
    loader = MinIOLoader(_make_storage_config())

    result = loader.load_dataset("nonexistent_dataset", use_cache=False)
    assert result.load_status == "error"
    assert result.error_message is not None
