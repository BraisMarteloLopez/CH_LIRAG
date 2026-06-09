# FAST_NOTES — estado al pausar (checkpoint)

> Notas rapidas para retomar tras el parentesis. **No** sustituye `CLAUDE.md`
> ni los contratos; es solo el "donde estamos y que sigue".

## Estado de la rama

- Rama activa: `claude/review-documentation-mnTCd`.
- **5 commits ahead / 0 behind de `main`**, **sin PR abierto**.
- Suite unitaria al pausar: **480 passed, 7 skipped** (verde).

Commits de la rama (oldest -> newest):

1. `b3d3d74` — docs: contrato de ingesta MinIO/Parquet v0
2. `5b20857` — docs: reorientar a fase de integracion (CLAUDE.md)
3. `c6e8372` — docs(readme): enlazar contratos de integracion
4. `76bdc27` — feat(loader): `load_collection` para ingesta de chunks (INGESTION_CONTRACT v1)
5. `1a94cb9` — feat(loader): `S3_COLLECTIONS_PREFIX` dedicado

## Lo que esta cerrado

- Reorientacion documental a fase de integracion (`CLAUDE.md`, `README.md`).
- Contratos en repo: `INGESTION_CONTRACT.md` (v1 acordada con LI_AD) y `KG_CONTRACT.md` (v0 export reciproco, implementacion P3).
- `MinIOLoader.load_collection(collection_id)` + validacion pre-carga + 19 tests nuevos con fixtures sinteticos (36 tests del loader en total).
- Prefijo dedicado `S3_COLLECTIONS_PREFIX` (default `admin/collections`), independiente de `S3_DATASETS_PREFIX` (eval HotpotQA).
- `KG_MAX_TEXT_CHARS=5000` como guard del contrato (>= `max_chunk_chars` del manifest).
- LI_AD confirmo alineacion: bucket `lakehouse`, prefix `admin/collections`, `ADMIN_ROOT_PREFIX=admin` (default).

## Pendiente — orden estricto

### A. Decision sobre la rama

- Hoy: 5 commits sin mergear, sin PR.
- Recomendacion al pausar: **esperar al `probe.py` verde antes de abrir PR**, para que el PR entre a `main` con el contrato validado contra muestra real.
- Alternativa: PR en *draft* ya, para tener revision empezada en paralelo.

### B. Bucle de validacion con LI_AD (corre en el entorno del usuario; claude_code no llega a MinIO)

1. **LI_AD coloca la generacion de prueba** en `lakehouse/admin/collections/{collection_id}/` (`collection.json` + `chunks/{stem}.parquet`).
2. **Usuario lanza `probe.py`** (snippet en §D), ajustando `COLLECTION_ID`.
3. **Usuario pega la salida** al chat: `OK | chunks=...` o `CONTRATO INCUMPLIDO: ...`.

### C. Trabajo del motor — *gated* por B verde

1. Ajustar `sandbox_mteb/loader.py` si el esquema real de LI_AD difiere del contrato (nombres de columna, tipos, campos del manifest).
2. Cablear `load_collection` en `sandbox_mteb/evaluator.py` / `sandbox_mteb/run.py`: selector de coleccion + separar **modo ingesta** de **modo eval** (HotpotQA). Decision de orquestacion pendiente.
3. Integrar `(collection_id, generation, chunking_fingerprint)` en `LightRAGRetriever._corpus_fingerprint` (`shared/retrieval/lightrag/retriever.py`) para invalidar la cache del KG por **generacion** en vez de hashear el contenido. Aborda parte de la deuda #10.

## D. Anexo — snippet `probe.py`

Guardar en la raiz del repo como `probe.py`, ajustar `COLLECTION_ID`, ejecutar `python probe.py`:

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
- Loader: `sandbox_mteb/loader.py::MinIOLoader.load_collection` (+ helpers `_validate_manifest`, `_populate_chunks_from_dataframe`).
- Tests: `tests/test_loader.py` (seccion "load_collection (INGESTION_CONTRACT.md)").
- Config: `sandbox_mteb/config.py::MinIOStorageConfig.s3_collections_prefix`.
- env template: `sandbox_mteb/env.example` (`S3_COLLECTIONS_PREFIX=admin/collections`, `KG_MAX_TEXT_CHARS=5000`).
- Fase del proyecto: `CLAUDE.md` -> "Proximos pasos" -> "INTEGRACION".
