"""
MinIO Dataset Loader para sandbox MTEB.

Carga datasets MTEB/BeIR pre-descargados desde MinIO (formato Parquet).

Estructura esperada en MinIO:
    s3://{bucket}/{prefix}/
    +-- manifest.json
    +-- hotpotqa/
    |   +-- queries.parquet
    |   +-- corpus.parquet
    |   +-- qrels.parquet
    |   +-- metadata.json
    +-- fever/
        +-- ...
"""

import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import boto3
from botocore.exceptions import ClientError

from shared.types import (
    DatasetType,
    LoadedDataset,
    MetricType,
    NormalizedQuery,
    NormalizedDocument,
    get_dataset_config,
    parse_answer_type,
)
from .config import MinIOStorageConfig

logger = logging.getLogger(__name__)

# Campos obligatorios del manifest collection.json (INGESTION_CONTRACT.md §4).
_REQUIRED_MANIFEST_KEYS = ("collection_id", "num_chunks", "max_chunk_chars", "parts")


def _safe_str(val: object, default: str = "") -> str:
    """Convierte a str, tratando None y NaN como default."""
    if val is None:
        return default
    try:
        if val != val:  # NaN != NaN
            return default
    except (TypeError, ValueError):
        pass
    s = str(val)
    return default if s in ("None", "nan") else s


def _coerce_int(val: object) -> Optional[int]:
    """Convierte a int; None/NaN/no-numerico -> None."""
    if val is None:
        return None
    try:
        if val != val:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


class MinIOLoader:
    """
    Carga datasets de evaluacion desde MinIO.

    A diferencia de la version original, no tiene from_settings().
    Se construye con parametros explicitos desde MTEBConfig.
    """

    def __init__(self, storage_config: MinIOStorageConfig):
        endpoint = storage_config.minio_endpoint
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"

        self.endpoint = endpoint
        self.bucket = storage_config.minio_bucket
        self.prefix = storage_config.s3_datasets_prefix
        self.collections_prefix = storage_config.s3_collections_prefix
        self.cache_dir = storage_config.datasets_cache_dir

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=storage_config.minio_access_key,
            aws_secret_access_key=storage_config.minio_secret_key,
        )

        self._manifest: Optional[Dict] = None
        logger.info(f"MinIOLoader: {endpoint}/{self.bucket}/{self.prefix}")

    # -----------------------------------------------------------------
    # API PUBLICA
    # -----------------------------------------------------------------

    def check_connection(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as e:
            logger.error(f"Error conexion MinIO: {e}")
            return False

    def load_dataset(
        self,
        dataset_name: str,
        use_cache: bool = True,
    ) -> LoadedDataset:
        """
        Carga un dataset desde MinIO y lo normaliza.

        NO aplica limites de queries/corpus. El evaluador decide cuantos
        usar en su run. Esto elimina la mutacion in-place del original.
        """
        ds_config = get_dataset_config(dataset_name)

        # Intentar cache local primero
        if use_cache and self.cache_dir:
            cached = self._load_from_cache(dataset_name)
            if cached:
                logger.info(f"Dataset '{dataset_name}' cargado desde cache")
                return cached

        logger.info(f"Descargando dataset '{dataset_name}' desde MinIO...")

        result = LoadedDataset(
            name=dataset_name,
            dataset_type=ds_config["type"],
            primary_metric=ds_config["primary_metric"],
            secondary_metrics=ds_config.get("secondary_metrics", []),
        )

        try:
            queries_df = self._download_parquet(f"{dataset_name}/queries.parquet")
            corpus_df = self._download_parquet(f"{dataset_name}/corpus.parquet")
            qrels_df = self._download_parquet(f"{dataset_name}/qrels.parquet")

            # Detectar descargas fallidas (queries o corpus).
            if queries_df is None and corpus_df is None:
                raise ValueError(
                    f"No se pudo descargar queries ni corpus para '{dataset_name}'"
                )
            if queries_df is None:
                raise ValueError(
                    f"No se pudo descargar queries para '{dataset_name}'"
                )
            if corpus_df is None:
                raise ValueError(
                    f"No se pudo descargar corpus para '{dataset_name}'"
                )

            self._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

            # Metadata
            result.metadata = self._download_json(f"{dataset_name}/metadata.json") or {}
            result.load_status = "success"

            logger.info(f"Dataset '{dataset_name}' cargado: {result.get_statistics()}")

            if use_cache and self.cache_dir:
                self._save_to_cache(result)

            return result

        except Exception as e:
            logger.error(f"Error cargando dataset '{dataset_name}': {e}")
            result.load_status = "error"
            result.error_message = str(e)
            return result

    def load_collection(self, collection_id: str) -> LoadedDataset:
        """
        Carga una coleccion de chunks producida por LI_AD (INGESTION_CONTRACT.md).

        A diferencia de load_dataset (modo eval: queries + corpus + qrels), aqui
        se ingesta solo el corpus de chunks de produccion:
          - Entrada por el manifest `collection.json` (no glob de directorio).
          - Una coleccion = N parts (un Parquet por documento) listadas en el manifest.
          - Mapea cada chunk a NormalizedDocument (chunk_id->doc_id, text->content,
            procedencia->metadata).

        Valida el contrato y FALLA TEMPRANO (ValueError) ante incoherencias
        (columnas requisito, unicidad de chunk_id, text > cap, filas-por-part,
        num_chunks), para no quemar compute indexando datos rotos.
        """
        manifest = self._download_json(
            f"{collection_id}/collection.json", prefix=self.collections_prefix
        )
        if manifest is None:
            raise ValueError(
                f"Coleccion '{collection_id}': no se encontro collection.json en "
                f"{self.collections_prefix}/{collection_id}/"
            )
        self._validate_manifest(manifest, collection_id)

        max_chunk_chars = int(manifest["max_chunk_chars"])

        result = LoadedDataset(
            name=collection_id,
            dataset_type=DatasetType.RETRIEVAL_ONLY,
            primary_metric=MetricType.FAITHFULNESS,
            secondary_metrics=[],
        )

        seen_ids: Set[str] = set()
        total_rows = 0
        for part in manifest["parts"]:
            part_path = _safe_str(part.get("path"))
            chunks_df = self._download_parquet(
                f"{collection_id}/{part_path}", prefix=self.collections_prefix
            )
            if chunks_df is None:
                raise ValueError(
                    f"Coleccion '{collection_id}': part del manifest no "
                    f"descargable: '{part_path}'"
                )
            n_rows = len(chunks_df)
            declared = _coerce_int(part.get("num_rows"))
            if declared is not None and n_rows != declared:
                raise ValueError(
                    f"Coleccion '{collection_id}', part '{part_path}': {n_rows} "
                    f"filas != {declared} declaradas en el manifest (posible "
                    f"lectura a mitad de re-chunk; reintentar tras releer manifest)"
                )
            total_rows += n_rows
            self._populate_chunks_from_dataframe(
                result, chunks_df, collection_id, max_chunk_chars, seen_ids
            )

        declared_chunks = _coerce_int(manifest.get("num_chunks"))
        if declared_chunks is not None and total_rows != declared_chunks:
            raise ValueError(
                f"Coleccion '{collection_id}': {total_rows} filas totales != "
                f"num_chunks={declared_chunks} del manifest"
            )

        result.total_corpus = len(result.corpus)
        result.metadata = {
            "collection_id": collection_id,
            "generation": manifest.get("generation"),
            "chunking_fingerprint": manifest.get("chunking_fingerprint"),
            "contract_version": manifest.get("contract_version"),
            "schema_version": manifest.get("schema_version"),
            "max_chunk_chars": max_chunk_chars,
            "num_documents": manifest.get("num_documents"),
            "chunking": manifest.get("chunking"),
        }
        result.load_status = "success"
        logger.info(
            f"Coleccion '{collection_id}' cargada: {len(result.corpus)} chunks "
            f"(generation={manifest.get('generation')})"
        )
        return result

    # -----------------------------------------------------------------
    # PRIVATE: DataFrame -> LoadedDataset
    # -----------------------------------------------------------------

    @staticmethod
    def _populate_from_dataframes(
        result: LoadedDataset,
        queries_df,
        corpus_df,
        qrels_df,
    ) -> None:
        """
        Puebla un LoadedDataset a partir de DataFrames de queries, corpus y qrels.

        Extraido de load_dataset() y _load_from_cache() para eliminar
        duplicacion (~50 lineas identicas).
        """
        qrels: Dict[str, List[str]] = {}

        if queries_df is not None:
            result.total_queries = len(queries_df)
            for _, row in queries_df.iterrows():
                qid = _safe_str(row.get("query_id", ""))
                raw_answer = _safe_str(row.get("answer"))
                raw_answer_type = _safe_str(row.get("answer_type"))
                if not raw_answer_type and raw_answer:
                    raw_answer_type = "text"
                question_type = _safe_str(row.get("question_type")) or _safe_str(row.get("type"))
                # Comparison queries (yes/no) son clasificacion, no extractiva.
                if question_type == "comparison" and raw_answer_type != "label":
                    raw_answer_type = "label"

                result.queries.append(NormalizedQuery(
                    query_id=qid,
                    query_text=_safe_str(row.get("text", "")),
                    expected_answer=raw_answer or None,
                    answer_type=parse_answer_type(raw_answer_type),
                    metadata={
                        "question_type": question_type,
                        "level": _safe_str(row.get("level")),
                    },
                ))

        if corpus_df is not None:
            result.total_corpus = len(corpus_df)
            for _, row in corpus_df.iterrows():
                did = _safe_str(row.get("doc_id", ""))
                result.corpus[did] = NormalizedDocument(
                    doc_id=did,
                    title=_safe_str(row.get("title")),
                    content=_safe_str(row.get("text")),
                )

        if qrels_df is not None:
            for _, row in qrels_df.iterrows():
                qid = _safe_str(row.get("query_id", ""))
                did = _safe_str(row.get("doc_id", ""))
                qrels.setdefault(qid, []).append(did)

        for query in result.queries:
            query.relevant_doc_ids = qrels.get(query.query_id, [])

    @staticmethod
    def _validate_manifest(manifest: object, collection_id: str) -> None:
        """Valida collection.json contra INGESTION_CONTRACT.md §4/§7.

        Falla con ValueError claro (no status silencioso): el manifest es el
        punto de entrada y un drift aqui debe parar antes de indexar.
        """
        if not isinstance(manifest, dict):
            raise ValueError(
                f"Coleccion '{collection_id}': collection.json no es un objeto JSON"
            )
        missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in manifest]
        if missing:
            raise ValueError(
                f"Coleccion '{collection_id}': manifest sin campos requeridos: {missing}"
            )
        if _safe_str(manifest.get("collection_id")) != collection_id:
            raise ValueError(
                f"Coleccion '{collection_id}': manifest.collection_id="
                f"'{_safe_str(manifest.get('collection_id'))}' no coincide con el solicitado"
            )
        parts = manifest.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(
                f"Coleccion '{collection_id}': manifest.parts vacio o no es lista"
            )
        for i, part in enumerate(parts):
            if not isinstance(part, dict) or "path" not in part or "num_rows" not in part:
                raise ValueError(
                    f"Coleccion '{collection_id}': parts[{i}] sin 'path'/'num_rows'"
                )
        mcc = _coerce_int(manifest.get("max_chunk_chars"))
        if mcc is None or mcc <= 0:
            raise ValueError(
                f"Coleccion '{collection_id}': max_chunk_chars invalido: "
                f"{manifest.get('max_chunk_chars')!r}"
            )

    @staticmethod
    def _populate_chunks_from_dataframe(
        result: LoadedDataset,
        chunks_df,
        collection_id: str,
        max_chunk_chars: int,
        seen_ids: Set[str],
    ) -> None:
        """Mapea rows de un chunks.parquet a NormalizedDocument validando el contrato.

        Clave de indexacion: chunk_id (con collection_id constante por coleccion).
        Procedencia (document_id, chunk_index, source_file, page_*, token_count) va
        a metadata; source_file NO se usa como title para no sesgar el embedding.
        """
        for _, row in chunks_df.iterrows():
            chunk_id = _safe_str(row.get("chunk_id", ""))
            if not chunk_id:
                raise ValueError(
                    f"Coleccion '{collection_id}': chunk con chunk_id vacio/ausente"
                )
            if chunk_id in seen_ids:
                raise ValueError(
                    f"Coleccion '{collection_id}': chunk_id duplicado: '{chunk_id}'"
                )
            seen_ids.add(chunk_id)

            row_cid = _safe_str(row.get("collection_id", ""))
            if row_cid and row_cid != collection_id:
                raise ValueError(
                    f"Coleccion '{collection_id}': chunk '{chunk_id}' con "
                    f"collection_id='{row_cid}' ajeno a la coleccion"
                )

            text = _safe_str(row.get("text", ""))
            if not text:
                raise ValueError(
                    f"Coleccion '{collection_id}': chunk '{chunk_id}' con text vacio"
                )
            if len(text) > max_chunk_chars:
                raise ValueError(
                    f"Coleccion '{collection_id}': chunk '{chunk_id}' text de "
                    f"{len(text)} chars > max_chunk_chars={max_chunk_chars}"
                )

            metadata: Dict[str, object] = {"collection_id": collection_id}
            for col in ("document_id", "source_file"):
                v = _safe_str(row.get(col, ""))
                if v:
                    metadata[col] = v
            for col in ("chunk_index", "page_start", "page_end", "token_count"):
                iv = _coerce_int(row.get(col, None))
                if iv is not None:
                    metadata[col] = iv

            result.corpus[chunk_id] = NormalizedDocument(
                doc_id=chunk_id,
                content=text,
                title=None,
                metadata=metadata,
            )

    # -----------------------------------------------------------------
    # PRIVATE: S3
    # -----------------------------------------------------------------

    def _get_manifest(self, force_refresh: bool = False) -> Dict:
        if self._manifest and not force_refresh:
            return self._manifest
        try:
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/manifest.json",
            )
            self._manifest = json.loads(resp["Body"].read().decode("utf-8"))
            return self._manifest
        except ClientError as e:
            logger.warning(f"No se encontro manifest: {e}")
            return {"datasets": [], "error": str(e)}

    def _download_parquet(self, key: str, prefix: Optional[str] = None):
        import pandas as pd

        full_key = f"{prefix or self.prefix}/{key}"
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
            data = resp["Body"].read()
            return pd.read_parquet(io.BytesIO(data))
        except ClientError as e:
            logger.warning(f"No se pudo descargar {full_key}: {e}")
            return None

    def _download_json(self, key: str, prefix: Optional[str] = None) -> Optional[Dict]:
        full_key = f"{prefix or self.prefix}/{key}"
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            return dict(data) if isinstance(data, dict) else None
        except ClientError:
            return None

    # -----------------------------------------------------------------
    # PRIVATE: CACHE LOCAL
    # -----------------------------------------------------------------

    def _load_from_cache(self, dataset_name: str) -> Optional[LoadedDataset]:
        import pandas as pd

        cache_dir = self.cache_dir / dataset_name
        if not cache_dir.exists():
            return None

        queries_path = cache_dir / "queries.parquet"
        corpus_path = cache_dir / "corpus.parquet"
        qrels_path = cache_dir / "qrels.parquet"

        if not all(p.exists() for p in [queries_path, corpus_path]):
            return None

        ds_config = get_dataset_config(dataset_name)
        result = LoadedDataset(
            name=dataset_name,
            dataset_type=ds_config["type"],
            primary_metric=ds_config["primary_metric"],
            secondary_metrics=ds_config.get("secondary_metrics", []),
        )

        queries_df = pd.read_parquet(queries_path)
        corpus_df = pd.read_parquet(corpus_path)
        qrels_df = pd.read_parquet(qrels_path) if qrels_path.exists() else None

        self._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

        result.load_status = "success"
        return result

    def _save_to_cache(self, dataset: LoadedDataset) -> None:
        import pandas as pd

        if not self.cache_dir:
            return

        cache_dir = self.cache_dir / dataset.name
        cache_dir.mkdir(parents=True, exist_ok=True)

        queries_data = [
            {
                "query_id": q.query_id,
                "text": q.query_text,
                "answer": q.expected_answer or "",
                "answer_type": q.answer_type or "",
                "question_type": q.metadata.get("question_type", ""),
                "level": q.metadata.get("level", ""),
            }
            for q in dataset.queries
        ]
        pd.DataFrame(queries_data).to_parquet(
            cache_dir / "queries.parquet", index=False
        )

        corpus_data = [
            {"doc_id": d.doc_id, "title": d.title or "", "text": d.content}
            for d in dataset.corpus.values()
        ]
        pd.DataFrame(corpus_data).to_parquet(
            cache_dir / "corpus.parquet", index=False
        )

        qrels_data = [
            {"query_id": q.query_id, "doc_id": did, "relevance": 1}
            for q in dataset.queries
            for did in q.relevant_doc_ids
        ]
        pd.DataFrame(qrels_data).to_parquet(
            cache_dir / "qrels.parquet", index=False
        )

        logger.debug(f"Dataset '{dataset.name}' guardado en cache local")


__all__ = ["MinIOLoader"]
