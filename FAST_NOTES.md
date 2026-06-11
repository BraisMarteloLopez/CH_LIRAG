# FAST_NOTES — checkpoint del estado pendiente

> Notas rapidas para retomar entre sesiones. **No** sustituye `CLAUDE.md`
> ni los contratos; es solo el "donde estamos y que sigue".

## Lo que esta cerrado (en esta rama, pendiente de PR + merge a main)

- Reorientacion documental a fase de integracion (`CLAUDE.md`, `README.md`).
- Contratos: [`INGESTION_CONTRACT.md`](INGESTION_CONTRACT.md) (v1 acordada con
  LI_AD) y [`KG_CONTRACT.md`](KG_CONTRACT.md) (v0 export reciproco; implementacion P3).
- `MinIOLoader.load_collection(collection_id)` + validacion pre-carga + 36
  tests del loader verdes con fixtures sinteticos (suite completa: 480 passed, 7 skipped).
- `S3_COLLECTIONS_PREFIX` dedicado (default `admin/collections`), independiente
  de `S3_DATASETS_PREFIX` (eval HotpotQA).
- `KG_MAX_TEXT_CHARS=5000` como guard del contrato (>= `max_chunk_chars` del manifest).
- LI_AD confirmo alineacion: bucket `lakehouse`, prefix `admin/collections`,
  `ADMIN_ROOT_PREFIX=admin` (default).
- **Despliegue alternativo JupyterLab + OpenWebUI documentado**:
  - Smoke test verde: `ChatNVIDIA` + `NVIDIAEmbeddings` (langchain-nvidia 0.3.19)
    contra OpenWebUI `/api`, sync + async, sin tocar codigo del motor.
  - Modelos verificados: `coding-qwen3-coder-next` (chat, 32K ctx, ~1 s
    extraccion), `demo-bge-m3` (embeddings, 1024 dims, ~60 ms).
  - Plantilla `.env`: [`sandbox_mteb/env.example.jupyter`](sandbox_mteb/env.example.jupyter).
  - Scripts de bootstrap: [`scripts/jupyter/`](scripts/jupyter/) (venv, MinIO, bucket).
  - Seccion en `CLAUDE.md` -> "Despliegue alternativo: JupyterLab + OpenWebUI"
    con las particularidades del entorno (NFS persistente, GitHub bloqueado,
    `dl.min.io` bloqueado, sin tmux, etc.).

## Pendiente — orden estricto

### A. Bucle de validacion del contrato con LI_AD (entorno Linux+NIM)

Sigue **gated** por la generacion de prueba real. Sin acceso a MinIO desde claude_code.

1. **LI_AD coloca la generacion** en `lakehouse/admin/collections/{collection_id}/`
   (`collection.json` + `chunks/{stem}.parquet`).
2. **Usuario lanza `probe.py`** (snippet en §D) ajustando `COLLECTION_ID`.
3. **Usuario pega la salida** -> validacion + ajuste del loader si difiere.

### B. Bootstrap operativo en Jupyter (entorno B200 + OpenWebUI)

Smoke test de provider hecho. Falta el despliegue, en orden:

1. **Repo en `/home/jovyan/lirag`**: GitLab interno (mirror desde GitHub, que
   esta bloqueado en el pod) o tarball por la UI.
2. **Venv + deps**: `bash scripts/jupyter/01_setup_venv.sh` -> crea
   `/home/jovyan/lirag_venv` (NFS persistente).
3. **Binario MinIO**: descarga manual + upload UI a `/home/jovyan/`, luego
   `mv ... /home/jovyan/bin/minio && chmod +x`. Una sola vez (queda en NFS).
4. **MinIO arriba**: `bash scripts/jupyter/02_minio_run.sh`.
5. **Bucket**: `python scripts/jupyter/03_create_bucket.py`.
6. **`.env`**: `cp sandbox_mteb/env.example.jupyter sandbox_mteb/.env` y pega
   `NVIDIA_API_KEY` (la api_key de OpenWebUI).
7. **Transferencia de la coleccion**: tarball desde el MinIO de produccion
   (no alcanzable desde Jupyter) -> upload UI -> boto3 upload al MinIO local.
   **Script pendiente.**
8. **`probe.py`** contra la coleccion local.
9. **Mini-script de indexacion**: `load_collection` + `LightRAGRetriever.index_documents`.
   **Script pendiente.**

### C. Trabajo de codigo del motor — gated por A o B verdes

1. Ajustar `sandbox_mteb/loader.py` si el esquema real de LI_AD difiere del contrato.
2. Cablear `load_collection` en `sandbox_mteb/evaluator.py` / `sandbox_mteb/run.py`:
   selector de coleccion + separar **modo ingesta** de **modo eval** (HotpotQA).
3. Integrar `(collection_id, generation, chunking_fingerprint)` en
   `LightRAGRetriever._corpus_fingerprint` (`shared/retrieval/lightrag/retriever.py`)
   para invalidar cache del KG por **generacion**. Aborda parte de la deuda #10.

## D. Anexo — snippet `probe.py`

Guardar en la raiz del repo como `probe.py`, ajustar `COLLECTION_ID`, ejecutar
`python probe.py` (con el venv del entorno correspondiente activo):

```python
from dotenv import load_dotenv
load_dotenv("sandbox_mteb/.env")

from sandbox_mteb.config import MinIOStorageConfig
from sandbox_mteb.loader import MinIOLoader

COLLECTION_ID = "col_XXXXXXXXXXXXXX_xxxxxxxx"   # <- el id real de LI_AD

loader = MinIOLoader(MinIOStorageConfig.from_env())
try:
    ds = loader.load_collection(COLLECTION_ID)
    print(f"OK | chunks={len(ds.corpus)} | generation={ds.metadata.get('generation')}")
    print("manifest:", ds.metadata)
    if ds.corpus:
        cid, doc = next(iter(ds.corpus.items()))
        print("muestra:", cid, "| meta:", doc.metadata, "| text:", repr(doc.content[:150]))
except ValueError as e:
    print("CONTRATO INCUMPLIDO:", e)
```

## E. Referencias rapidas

- Contrato de ingesta: [`INGESTION_CONTRACT.md`](INGESTION_CONTRACT.md) (v1).
- Contrato de export del KG: [`KG_CONTRACT.md`](KG_CONTRACT.md) (v0, implementacion P3).
- Loader: `sandbox_mteb/loader.py::MinIOLoader.load_collection` (+ helpers
  `_validate_manifest`, `_populate_chunks_from_dataframe`).
- Tests: `tests/test_loader.py` (seccion "load_collection (INGESTION_CONTRACT.md)").
- Config: `sandbox_mteb/config.py::MinIOStorageConfig.s3_collections_prefix`.
- env templates:
  - Linux principal con NIM: `sandbox_mteb/env.example`.
  - JupyterLab con OpenWebUI: `sandbox_mteb/env.example.jupyter`.
- Scripts JupyterLab: [`scripts/jupyter/`](scripts/jupyter/) (venv, MinIO, bucket).
- Fase del proyecto: `CLAUDE.md` -> "Proximos pasos" -> "INTEGRACION".
- Despliegue Jupyter: `CLAUDE.md` -> "Despliegue alternativo: JupyterLab + OpenWebUI".
