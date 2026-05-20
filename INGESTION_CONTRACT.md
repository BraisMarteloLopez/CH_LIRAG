# Contrato de ingesta — Plataforma de administracion (productor) → motor CH_LIRAG (consumidor)

**Estado**: BORRADOR v0 — propuesta del lado consumidor (CH_LIRAG), pendiente de evaluacion y confirmacion por el equipo de la plataforma de administracion.

**Proposito**: definir el formato exacto (layout MinIO + esquema Parquet) que la plataforma de administracion debe **producir** y que el motor CH_LIRAG **consume** para indexar colecciones y responder consultas. El mecanismo de transporte (MinIO + Parquet, leido por el loader del motor) **no cambia**; lo que se especifica aqui es el **esquema de los datos**, distinto del que el motor lee hoy del harness HotpotQA.

**Como evaluar este documento**: cada elemento esta marcado como **REQUISITO** (no negociable sin romper la ingesta del motor), **PROPUESTO** (recomendado, ajustable) u **OPCIONAL**. La seccion §6 son **preguntas abiertas** que necesitamos que respondais segun lo que la plataforma puede emitir. Responded inline o devolved una v1 del documento.

---

## 1. Contexto (lado consumidor)

CH_LIRAG es un motor de RAG con grafo de conocimiento. Dado un corpus de **chunks de texto**, construye indices vectoriales + un grafo de entidades/relaciones y responde consultas en lenguaje natural.

- El motor **no trocea** documentos: consume los chunks **ya troceados** por la plataforma como unidad minima de recuperacion.
- El motor **calcula sus propios embeddings** y **construye su propio grafo**: la plataforma no provee ni vectores ni grafo.
- Hoy el motor lee datasets tipo HotpotQA (pasajes de Wikipedia) con un esquema `doc_id`/`title`/`text`. Este contrato **sustituye** ese esquema por uno orientado a chunks de documentos reales (PDFs) con su procedencia.

Estado actual de la plataforma (segun lo comunicado): ingesta PDFs → genera chunks de texto (≈600 tokens, configurable). Una **coleccion** = conjunto de documentos ingestados juntos.

## 2. Transporte y layout MinIO

- Almacenamiento: MinIO (S3-compatible). **REQUISITO**
- Formato de fichero: Apache Parquet (chunks) + JSON (manifest de coleccion). **REQUISITO**
- Codificacion de texto: UTF-8. **REQUISITO**

Layout **PROPUESTO** (un prefijo por coleccion):

```
s3://{bucket}/{prefix}/
  {collection_id}/
    chunks.parquet      # los chunks de la coleccion (esquema en §3)
    collection.json     # manifest de la coleccion (esquema en §4)
```

- `{bucket}` y `{prefix}` se acuerdan por despliegue (en el motor: `MINIO_BUCKET_NAME`, `S3_DATASETS_PREFIX`). **Integrar = apuntar el motor a `{prefix}`**; no hay rediseño de ingesta.
- `{collection_id}` es el **selector** con el que se le pide al motor que indexe una coleccion (analogo al nombre de dataset hoy).
- Multi-tenant y particionado: ver preguntas abiertas §6.

## 3. Esquema `chunks.parquet`

Un row = un chunk. Tipos en Arrow/Parquet.

| Columna | Tipo | Nivel | Nulos | Semantica |
|---|---|---|---|---|
| `chunk_id` | string (utf8) | **REQUISITO** | no | ID unico y **estable** del chunk. Es la **clave primaria** y la unidad de recuperacion del motor. |
| `collection_id` | string | **REQUISITO** | no | ID de la coleccion. Identico en todos los rows del fichero. |
| `text` | string | **REQUISITO** | no | Contenido textual del chunk. Es lo que el motor embebe e indexa. |
| `document_id` | string | PROPUESTO | no | ID del documento fuente (PDF) dentro de la coleccion. Habilita procedencia y agrupacion por documento. |
| `document_name` | string | OPCIONAL | si | Nombre/titulo legible del documento (p. ej. nombre del PDF). |
| `chunk_index` | int32 | PROPUESTO | no | Orden 0-based del chunk dentro de su `document_id`. |
| `page_start` | int32 | OPCIONAL | si | Primera pagina del span en el PDF (1-based). |
| `page_end` | int32 | OPCIONAL | si | Ultima pagina del span. |
| `token_count` | int32 | OPCIONAL | si | Numero de tokens del chunk segun el tokenizer de la plataforma. |

Reglas:

- `chunk_id` **unico** dentro del fichero y sin colisiones entre colecciones. Recomendado: `{collection_id}:{document_id}:{chunk_index}`, o un UUID estable.
- `text` **no vacio**, ya limpio (idealmente sin cabeceras/pies de pagina repetidos). El motor **no** re-trocea ni re-limpia.
- Columnas no-requeridas: si no se pueden poblar, omitir la columna o dejar `null`; el motor las trata como ausentes.
- Mapeo en el motor: `chunk_id` → id de recuperacion; `text` → contenido a embeber e indexar; el resto → **metadatos de procedencia** (no se inyectan en el embedding para no sesgarlo — en particular `document_name` **no** se antepone al texto).

Ejemplo ilustrativo (1 row):

| chunk_id | collection_id | text | document_id | document_name | chunk_index | page_start | page_end | token_count |
|---|---|---|---|---|---|---|---|---|
| `col1:docA:0` | `col1` | `"El sistema de frenado..."` | `docA` | `Manual_X.pdf` | 0 | 1 | 1 | 587 |

## 4. Manifest `collection.json`

JSON UTF-8 con metadatos de la coleccion y de la configuracion de chunking (necesaria para trazabilidad y para invalidar caches del motor si cambia el troceado).

```json
{
  "contract_version": "0",
  "collection_id": "col1",
  "name": "Catalogo X",
  "source_type": "pdf",
  "created_at": "2026-05-20T10:00:00Z",
  "num_documents": 12,
  "num_chunks": 3450,
  "chunking": {
    "strategy": "fixed_tokens",
    "chunk_size_tokens": 600,
    "overlap_tokens": 0,
    "tokenizer": "<nombre/version del tokenizer>"
  }
}
```

- `chunking.chunk_size_tokens` refleja el parametro configurable (hoy 600). **REQUISITO** (para trazabilidad).
- `num_chunks` debe coincidir con el numero de rows de `chunks.parquet` (chequeo de integridad). **REQUISITO**
- `contract_version` permite versionar este contrato (ver §8). **REQUISITO**

## 5. Que NO debe producir la plataforma

- **`queries.parquet` / `qrels.parquet`**: son ground-truth de *evaluacion* del harness MTEB del motor. En produccion no existen; las consultas las hace el usuario final en runtime.
- **Embeddings / vectores**: los calcula el motor.
- **Grafo de conocimiento (entidades/relaciones)**: lo construye el motor a partir de los chunks.

## 6. Preguntas abiertas (necesitamos respuesta de la plataforma)

1. **Metadatos disponibles**: de las columnas PROPUESTO/OPCIONAL (`document_id`, `chunk_index`, `page_start`/`page_end`, `token_count`), ¿cuales puede emitir la plataforma hoy? ¿Hay algo mas con valor de procedencia que si teneis (heading/seccion, bounding box, idioma del chunk, hash del documento)?
2. **Layout multi-coleccion / multi-tenant**: ¿os encaja "un prefijo por coleccion" (`{prefix}/{collection_id}/chunks.parquet`), o preferis un unico Parquet con columna `collection_id`? ¿El multi-tenant entra ya (`{prefix}/{tenant_id}/{collection_id}/...`) o lo dejamos previsto pero no implementado?
3. **Particionado**: ¿`chunks.parquet` sera un unico fichero por coleccion, o varios "parts" (`chunks/part-*.parquet`)? El motor puede soportar ambos si se acuerda.
4. **Estabilidad de IDs / versionado**: si se re-ingesta la misma coleccion, ¿`chunk_id` se mantiene estable? ¿Como versionais — sobrescribis el prefijo, o creais `{collection_id}/v2/`? (Impacta el reuso de cache e indexacion incremental del motor.)
5. **Garantias de integridad**: ¿`chunk_index` es contiguo y empieza en 0 por documento? ¿se garantiza no-duplicados de `chunk_id`? ¿hay un orden estable de filas?
6. **Tokenizer**: ¿que tokenizer usais para medir los 600 tokens? (Para alinear `token_count` con el presupuesto de contexto del motor.)

## 7. Validacion del contrato (lado motor)

Al integrar, el motor validara **antes de cargar**: presencia de columnas REQUISITO (`chunk_id`, `collection_id`, `text`), no-nulos en requeridas, unicidad de `chunk_id`, y coherencia `num_chunks` (manifest) vs numero de rows. Un drift de esquema debe fallar **temprano y con mensaje claro**, no a mitad de indexacion (que quemaria compute).

## 8. Versionado de este contrato

`contract_version` en `collection.json`. Esta es la **v0** (borrador). Cambios incompatibles incrementan la major; adiciones retrocompatibles (nuevas columnas opcionales), la minor.

---

> Documento generado desde el lado consumidor (CH_LIRAG) como punto de partida. Esperamos vuestra evaluacion de viabilidad (§6) para cerrar la v1.
