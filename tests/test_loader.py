"""Tests unitarios para MinIOLoader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sandbox_mteb.config import MinIOStorageConfig
from sandbox_mteb.loader import MinIOLoader
from shared.types import DatasetType, LoadedDataset, MetricType


class _FakeClientError(Exception):
    """Simula botocore.exceptions.ClientError. conftest mockea botocore
    como MagicMock y `except MagicMock` no funciona."""

    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"{operation_name}: {error_response}")


def _storage_config():
    return MinIOStorageConfig(
        minio_endpoint="http://fake:9000",
        minio_access_key="test",
        minio_secret_key="test",
        minio_bucket="test-bucket",
        s3_datasets_prefix="datasets/eval",
        datasets_cache_dir=Path("/tmp/test_cache"),
    )


class _MockDataFrame:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, _MockRow(row)


class _MockRow(dict):
    def get(self, key, default=""):
        return super().get(key, default)


def _empty_result():
    return LoadedDataset(
        name="test",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )


def _populate(queries, corpus=None, qrels=None):
    result = _empty_result()
    MinIOLoader._populate_from_dataframes(
        result,
        _MockDataFrame(queries) if queries is not None else None,
        _MockDataFrame(corpus) if corpus is not None else None,
        _MockDataFrame(qrels) if qrels is not None else None,
    )
    return result


# =============================================================================
# check_connection
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
def test_check_connection_success(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    assert MinIOLoader(_storage_config()).check_connection() is True
    mock_client.head_bucket.assert_called_once()


@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_check_connection_failure(mock_boto3):
    mock_client = MagicMock()
    mock_client.head_bucket.side_effect = _FakeClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
    )
    mock_boto3.client.return_value = mock_client

    from tests.helpers import make_loader
    assert make_loader(mock_client=mock_client).check_connection() is False


# =============================================================================
# _populate_from_dataframes
# =============================================================================

def test_populate_basic():
    result = _populate(
        queries=[
            {"query_id": "q1", "text": "What is AI?", "answer": "Artificial Intelligence",
             "answer_type": "text", "question_type": "bridge", "level": "easy"},
            {"query_id": "q2", "text": "Is Python good?", "answer": "yes",
             "answer_type": "label", "question_type": "comparison", "level": "medium"},
        ],
        corpus=[
            {"doc_id": "d1", "title": "AI Intro", "text": "AI is the simulation of intelligence."},
            {"doc_id": "d2", "title": "Python", "text": "Python is a programming language."},
        ],
        qrels=[{"query_id": "q1", "doc_id": "d1"}, {"query_id": "q2", "doc_id": "d2"}],
    )
    assert [q.query_id for q in result.queries] == ["q1", "q2"]
    assert result.queries[0].query_text == "What is AI?"
    assert result.queries[0].expected_answer == "Artificial Intelligence"
    assert [q.answer_type for q in result.queries] == ["text", "label"]
    assert result.corpus["d1"].title == "AI Intro"
    assert "simulation" in result.corpus["d1"].content
    assert result.queries[0].relevant_doc_ids == ["d1"]
    assert result.queries[1].relevant_doc_ids == ["d2"]
    assert result.total_queries == 2
    assert result.total_corpus == 2


def test_populate_multiple_qrels_per_query():
    result = _populate(
        queries=[{"query_id": "q1", "text": "Multi-hop"}],
        corpus=[{"doc_id": "d1", "title": "A", "text": "CA"},
                {"doc_id": "d2", "title": "B", "text": "CB"}],
        qrels=[{"query_id": "q1", "doc_id": "d1"}, {"query_id": "q1", "doc_id": "d2"}],
    )
    assert set(result.queries[0].relevant_doc_ids) == {"d1", "d2"}


@pytest.mark.parametrize("queries,corpus,qrels", [
    (None, None, None),
    ([], [], []),
])
def test_populate_empty_or_none_no_crash(queries, corpus, qrels):
    result = _populate(queries, corpus, qrels)
    assert len(result.queries) == 0
    assert len(result.corpus) == 0


def test_populate_answer_type_inferred_when_missing():
    """answer presente sin answer_type → 'text'."""
    result = _populate([{"query_id": "q1", "text": "Q?", "answer": "Some answer"}], [])
    assert result.queries[0].answer_type == "text"


def test_populate_no_answer_no_answer_type():
    result = _populate([{"query_id": "q1", "text": "Q?"}], [])
    assert result.queries[0].expected_answer is None
    assert result.queries[0].answer_type is None


@pytest.mark.parametrize("row,expected_type", [
    ({"query_id": "q1", "text": "Q?", "question_type": "bridge", "level": "hard"}, "bridge"),
    ({"query_id": "q1", "text": "Q?", "type": "comparison"}, "comparison"),
])
def test_populate_question_type_metadata(row, expected_type):
    """question_type (o fallback a 'type') se guarda en metadata."""
    result = _populate([row], [])
    assert result.queries[0].metadata["question_type"] == expected_type


def test_populate_question_type_level_captured():
    result = _populate(
        [{"query_id": "q1", "text": "Q?", "question_type": "bridge", "level": "hard"}], []
    )
    assert result.queries[0].metadata["level"] == "hard"


def test_populate_query_without_qrels_empty_relevant_ids():
    result = _populate([{"query_id": "q1", "text": "Orphan"}], [], [])
    assert result.queries[0].relevant_doc_ids == []


@pytest.mark.parametrize("question_type,initial_answer_type,expected", [
    ("comparison", "text", "label"),   # comparison fuerza label
    ("comparison", "label", "label"),  # idempotente
    ("bridge", "text", "text"),        # no-comparison preserva
])
def test_populate_comparison_forces_label(question_type, initial_answer_type, expected):
    result = _populate([{
        "query_id": "q1", "text": "Q?", "answer": "a",
        "answer_type": initial_answer_type, "question_type": question_type,
    }], [])
    assert result.queries[0].answer_type == expected


# =============================================================================
# load_dataset error path
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_load_dataset_download_error(mock_boto3):
    mock_client = MagicMock()
    mock_client.get_object.side_effect = _FakeClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
    )
    mock_boto3.client.return_value = mock_client

    result = MinIOLoader(_storage_config()).load_dataset("nonexistent", use_cache=False)
    assert result.load_status == "error"
    assert result.error_message is not None


# =============================================================================
# load_collection (INGESTION_CONTRACT.md): ingesta de chunks de LI_AD
# =============================================================================

def _manifest(collection_id="col1", parts=None, num_chunks=2,
              max_chunk_chars=5000, **extra):
    parts = parts if parts is not None else [
        {"path": "chunks/docA.parquet", "document_id": "docA", "num_rows": 1},
        {"path": "chunks/docB.parquet", "document_id": "docB", "num_rows": 1},
    ]
    m = {
        "contract_version": "1",
        "schema_version": 1,
        "collection_id": collection_id,
        "generation": 3,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "chunking_fingerprint": "sha256:abc",
        "parts": parts,
    }
    m.update(extra)
    return m


def _chunk_row(chunk_id, text="contenido", collection_id="col1", **extra):
    row = {"chunk_id": chunk_id, "collection_id": collection_id, "text": text}
    row.update(extra)
    return row


def _make_collection_loader(manifest, parts_by_path):
    """make_loader con _download_json/_download_parquet mockeados (sin pandas/S3)."""
    from tests.helpers import make_loader
    loader = make_loader()
    loader._download_json = MagicMock(return_value=manifest)
    loader._download_parquet = MagicMock(side_effect=lambda key: parts_by_path.get(key))
    return loader


def test_load_collection_happy_path():
    manifest = _manifest()
    parts = {
        "col1/chunks/docA.parquet": _MockDataFrame([
            _chunk_row("docA:00000", text="alpha", document_id="docA",
                       chunk_index=0, page_start=1, page_end=1, token_count=10),
        ]),
        "col1/chunks/docB.parquet": _MockDataFrame([
            _chunk_row("docB:00000", text="beta", document_id="docB", chunk_index=0),
        ]),
    }
    result = _make_collection_loader(manifest, parts).load_collection("col1")

    assert result.load_status == "success"
    assert set(result.corpus.keys()) == {"docA:00000", "docB:00000"}
    assert result.corpus["docA:00000"].content == "alpha"
    assert result.corpus["docA:00000"].title is None  # source_file NO va a title
    md = result.corpus["docA:00000"].metadata
    assert md["document_id"] == "docA"
    assert md["chunk_index"] == 0
    assert md["page_start"] == 1
    assert md["token_count"] == 10
    assert result.total_corpus == 2
    assert len(result.queries) == 0  # ingesta: sin queries/qrels
    assert result.metadata["generation"] == 3
    assert result.metadata["chunking_fingerprint"] == "sha256:abc"
    assert result.metadata["max_chunk_chars"] == 5000


def test_load_collection_manifest_missing_raises():
    with pytest.raises(ValueError, match="collection.json"):
        _make_collection_loader(None, {}).load_collection("col1")


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.pop("collection_id"), "campos requeridos"),
    (lambda m: m.pop("parts"), "campos requeridos"),
    (lambda m: m.pop("num_chunks"), "campos requeridos"),
    (lambda m: m.pop("max_chunk_chars"), "campos requeridos"),
    (lambda m: m.update(collection_id="otra"), "no coincide"),
    (lambda m: m.update(parts=[]), "parts vacio"),
    (lambda m: m.update(max_chunk_chars=0), "max_chunk_chars invalido"),
    (lambda m: m.update(parts=[{"path": "x.parquet"}]), "sin 'path'"),
])
def test_load_collection_invalid_manifest_raises(mutate, match):
    manifest = _manifest()
    mutate(manifest)
    with pytest.raises(ValueError, match=match):
        _make_collection_loader(manifest, {}).load_collection("col1")


def test_load_collection_part_row_count_mismatch_raises():
    """Refinamiento A: num_rows del manifest != filas reales -> lectura sucia."""
    manifest = _manifest(parts=[{"path": "chunks/docA.parquet", "num_rows": 5}],
                         num_chunks=5)
    parts = {"col1/chunks/docA.parquet": _MockDataFrame([_chunk_row("docA:00000")])}
    with pytest.raises(ValueError, match="filas !="):
        _make_collection_loader(manifest, parts).load_collection("col1")


def test_load_collection_total_chunks_mismatch_raises():
    manifest = _manifest(parts=[{"path": "chunks/docA.parquet", "num_rows": 1}],
                         num_chunks=99)
    parts = {"col1/chunks/docA.parquet": _MockDataFrame([_chunk_row("docA:00000")])}
    with pytest.raises(ValueError, match="num_chunks=99"):
        _make_collection_loader(manifest, parts).load_collection("col1")


def test_load_collection_part_not_downloadable_raises():
    manifest = _manifest(parts=[{"path": "chunks/docA.parquet", "num_rows": 1}],
                         num_chunks=1)
    with pytest.raises(ValueError, match="no .*descargable"):
        _make_collection_loader(manifest, {}).load_collection("col1")


def test_load_collection_duplicate_chunk_id_raises():
    manifest = _manifest(parts=[{"path": "chunks/docA.parquet", "num_rows": 2}],
                         num_chunks=2)
    parts = {"col1/chunks/docA.parquet": _MockDataFrame([
        _chunk_row("dup"), _chunk_row("dup"),
    ])}
    with pytest.raises(ValueError, match="duplicado"):
        _make_collection_loader(manifest, parts).load_collection("col1")


def test_load_collection_text_over_cap_raises():
    manifest = _manifest(parts=[{"path": "chunks/docA.parquet", "num_rows": 1}],
                         num_chunks=1, max_chunk_chars=10)
    parts = {"col1/chunks/docA.parquet": _MockDataFrame([
        _chunk_row("docA:00000", text="x" * 50),
    ])}
    with pytest.raises(ValueError, match="max_chunk_chars"):
        _make_collection_loader(manifest, parts).load_collection("col1")


@pytest.mark.parametrize("bad_row,match", [
    ({"chunk_id": "", "collection_id": "col1", "text": "t"}, "chunk_id vacio"),
    ({"chunk_id": "c1", "collection_id": "col1", "text": ""}, "text vacio"),
    ({"chunk_id": "c1", "collection_id": "otra", "text": "t"}, "ajeno"),
])
def test_load_collection_bad_chunk_raises(bad_row, match):
    manifest = _manifest(parts=[{"path": "chunks/docA.parquet", "num_rows": 1}],
                         num_chunks=1)
    parts = {"col1/chunks/docA.parquet": _MockDataFrame([bad_row])}
    with pytest.raises(ValueError, match=match):
        _make_collection_loader(manifest, parts).load_collection("col1")


def test_load_collection_optional_columns_absent_ok():
    manifest = _manifest(parts=[{"path": "chunks/docA.parquet", "num_rows": 1}],
                         num_chunks=1)
    parts = {"col1/chunks/docA.parquet": _MockDataFrame([
        {"chunk_id": "docA:00000", "collection_id": "col1", "text": "solo requisito"},
    ])}
    result = _make_collection_loader(manifest, parts).load_collection("col1")
    assert result.corpus["docA:00000"].metadata == {"collection_id": "col1"}
