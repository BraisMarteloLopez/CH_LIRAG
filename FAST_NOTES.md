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

**B1-B6 COMPLETADOS** (2026-06-11): repo clonado via GitLab interno
(`sim/CH_LIRAG`, sync manual Windows: `git pull origin main && git push gitlab main`
desde `~/lirag-sync`), venv completo (`igraph: True`), MinIO local arriba,
bucket `lakehouse` creado, `.env` con api_key, `--dry-run` valido.

**Arranque diario**: el pod se redespliega cada dia (procesos y /opt/conda se
pierden; NFS persiste). Al abrir el entorno:
`bash /home/jovyan/lirag/scripts/jupyter/00_daily_start.sh` (idempotente:
GPU, venv, MinIO relaunch, OpenWebUI, .env).

Pendiente del bucle de coleccion (scripts ya en `scripts/jupyter/`):

7. **Exportar la coleccion** del MinIO de produccion (corre en el LINUX):
   `python scripts/jupyter/04_export_collection.py <collection_id>` -> tgz
   -> subir a `/home/jovyan/` por la UI. **Requiere que LI_AD haya colocado
   la generacion de prueba en produccion.**
8. **Importar al MinIO local** (Jupyter):
   `python scripts/jupyter/05_import_collection.py /home/jovyan/<cid>.tgz`
   (parts primero, manifest el ultimo — manifest-as-commit).
9. **Indexar / construir el KG** (Jupyter, raiz del repo):
   `python scripts/jupyter/06_index_collection.py <cid> [--query "..."]`
   — valida contrato via `load_collection`, construye KG + VDBs, imprime
   KGStats/extractor stats, retrieval de sanidad opcional. Primer KG sobre
   B200 + OpenWebUI.

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
