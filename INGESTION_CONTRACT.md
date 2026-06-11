# Contrato de ingesta — LI_AD (productor) y CH_LIRAG (consumidor)

**Estado**: v1 (acordada en la negociacion: v0 de CH_LIRAG -> respuesta de LI_AD -> esta v1). Pendiente de: implementacion por LI_AD de los puntos de §8 y ack final. El transporte (MinIO + Parquet leido por `sandbox_mteb/loader.py`) **no cambia**; este documento fija el **esquema**.

**Roles**: LI_AD **produce** (OCR PDF -> chunks -> Parquet en MinIO); CH_LIRAG **consume** (indexa, construye KG, responde). El motor no trocea, ni recibe vectores o grafo del admin.

## 0. Resolucion de contrapropuestas (v0 -> v1)

LI_AD respondio al v0 con contrapropuestas; CH_LIRAG las evaluo (verificado contra codigo del motor). Acuerdos:

| Tema | Acuerdo v1 |
|---|---|
| Layout: 1 Parquet por documento | **Aceptado** (mejor para incremental/borrado selectivo). |
| Manifest-as-entrypoint | **Aceptado**: el loader entra por `collection.json` (parts listadas), no por glob de directorio. |
| Clave `(collection_id, chunk_id)` | **Aceptado**: indexamos por la tupla; el motor carga 1 coleccion por indice. |
| `document_id`/`chunk_index` como columnas | **Aceptado** (no se parsea `chunk_id`). |
| `chunk_index` ordinal no denso | **Aceptado** (clave de orden, no de array; preserva estabilidad de IDs). |
| `chunking_fingerprint` + `generation` | **Aceptado** (invalidacion de cache + consistencia). |
| Tokenizer desacoplado (sizing vs embedder) | **Aceptado**: el motor no depende de `token_count`; el guard es el cap de chars. |
| Invariante de chars en vez de tokens | **Aceptado**: `max_chunk_chars` (ver §3, Refinamiento B). |
| `KG_CONTRACT.md` reciproco (export del grafo) | **Aceptado definirlo**; implementacion P3 (ver `KG_CONTRACT.md`). |
| `schema_version` separado de `contract_version` | **Aceptado**: dos campos. |

Refinamientos aportados por CH_LIRAG:

- **B (firme, semantica cerrada como "opcion A" el 2026-06-11)**: el manifest emite `max_chunk_chars = max(5000, max(len(text)) observado)` — **cota verdadera del dato**, no techo fijo; LI_AD **no** hace hard-split (cortar a 5000 romperia unidades semanticas de su chunker estructural, justo el insumo del KG). El motor: (1) valida cada `text <= max_chunk_chars` al cargar; (2) exige `KG_MAX_TEXT_CHARS >= max_chunk_chars` y **rechaza la coleccion con error claro** si no se cumple (fail-fast; nunca truncado silencioso — verificado: `triplet_extractor.py` hace `text[:max_text_chars]`). La alternativa B (hard-split a 5000 en el chunker, ~medio dia de LI_AD) queda disponible si el futuro lo exige.
- **A (prioridad baja)**: validar filas-por-part contra el manifest para detectar lectura a mitad de re-chunk. Solo muerde con lectura y re-chunk **concurrentes** (no en single-operador). No bloquea v1; ver §6.

## 1. Layout MinIO

```
{MINIO_BUCKET_NAME}/{ADMIN_ROOT_PREFIX}/collections/{collection_id}/
  collection.json              # manifest de contrato (§4) -- PUNTO DE ENTRADA del loader
  chunks/{stem}.parquet        # un Parquet por documento (§3)
  meta.json                    # admin-internal: el motor lo IGNORA
  raw/{file}.pdf               # admin-internal: IGNORADO
  ocr/{stem}/text.jsonl        # admin-internal: IGNORADO
```

- El motor configura `S3_COLLECTIONS_PREFIX = {ADMIN_ROOT_PREFIX}/collections` (default `admin/collections`), **independiente** de `S3_DATASETS_PREFIX` (modo eval HotpotQA) para que ambos modos convivan sin pisarse; `MINIO_BUCKET_NAME` compartido.
- `{collection_id}` (formato `col_{YYYYMMDDHHMMSS}_{hex8}`) es el selector de coleccion.
- El motor **solo** lee `collection.json` y las parts que este enumera. Los artefactos admin-internal quedan fuera del contrato por construccion.

## 2. Punto de entrada: el manifest, no el glob

El loader: lee `collection.json` -> obtiene `generation`, `chunking_fingerprint`, `max_chunk_chars` y la lista `parts` (path + `num_rows` por part) -> lee las parts listadas -> valida (§7). El naming de ficheros deja de ser parte del contrato (LI_AD puede cambiarlo sin romper al motor).

## 3. Esquema de `chunks/{stem}.parquet` (un row = un chunk)

| Columna | Tipo Arrow | Nivel | Semantica / mapeo motor |
|---|---|---|---|
| `chunk_id` | string | **REQUISITO** | Clave (con `collection_id`) y unidad de recuperacion. Opaco, estable a nivel de documento. Formato LI_AD: `{stem}:{i:05d}`. Mapea a `NormalizedDocument.doc_id`. |
| `collection_id` | string | **REQUISITO** | Identico en todo el fichero. Parte de la clave compuesta. |
| `text` | string | **REQUISITO** | Contenido del chunk. `len(text) <= max_chunk_chars`. Mapea a `content`. |
| `document_id` | string | PROVISTO | `{stem}` del PDF. A `metadata`. |
| `chunk_index` | int32 | PROVISTO | Ordinal **no denso** dentro del documento (monotono, unico, con huecos por filtrado). Clave de orden, no de array. A `metadata`. |
| `source_file` | string | PROVISTO | Filename con `.pdf`. A `metadata` (no a `title`, para no sesgar el embedding via `get_full_text`). |
| `page_start` | int32 | PROVISTO | 1-based. A `metadata`. |
| `page_end` | int32 | PROVISTO | 1-based. A `metadata`. |
| `token_count` | int32 | PROVISTO | Tokens segun tokenizer de LI_AD. **Informativo** (el motor no depende de el). A `metadata`. |
| `block_types` | list<string> | EXTRA LI_AD | Procedencia para la UI del admin. El motor lo **ignora**. |

**Clave**: `(collection_id, chunk_id)`, unica por construccion (sin prefijar). El motor carga una coleccion por indice -> indexa por `chunk_id`.

**Invariante de chars (Refinamiento B, semantica "opcion A")**: `max_chunk_chars = max(5000, max(len(text)))` — cota verdadera del dato emitida por LI_AD (sin hard-split en el chunker). El motor valida `len(text) <= max_chunk_chars` al cargar (§7) y **rechaza la coleccion** si `max_chunk_chars > KG_MAX_TEXT_CHARS` (fail-fast con mensaje accionable, nunca truncado silencioso). `token_count`/`target_tokens` son trazabilidad, no contrato.

## 4. Manifest `collection.json`

```json
{
  "contract_version": "1",
  "schema_version": 1,
  "collection_id": "col_20260429082254_3c22096a",
  "name": "test_003",
  "source_type": "pdf",
  "created_at": "2026-04-29T08:22:54+00:00",
  "generation": 3,
  "num_documents": 12,
  "num_chunks": 3450,
  "max_chunk_chars": 5000,
  "chunking": {
    "strategy": "structural_tokens",
    "target_tokens": 600,
    "overlap_tokens": 75,
    "tokenizer": "cl100k_base"
  },
  "chunking_fingerprint": "sha256:abc123...",
  "parts": [
    { "path": "chunks/1_CONCEPTOS_PREVIOS.parquet", "document_id": "1_CONCEPTOS_PREVIOS", "num_rows": 320 },
    { "path": "chunks/2_INSTALACION.parquet",       "document_id": "2_INSTALACION",       "num_rows": 210 }
  ]
}
```

- **Escrito como PASO FINAL** de la ingesta (marcador de commit): el motor considera la coleccion lista en generacion N solo cuando el manifest lo dice (§6).
- `generation` (int) sube en cada (re)chunk -> señal de invalidacion barata para el motor.
- `chunking_fingerprint` = hash sobre {version del chunker, tokenizer, target, overlap, config de filtros}; el motor invalida cache **solo si** cambia.
- `num_chunks` = suma de `num_rows` de `parts` (chequeo de integridad).
- `max_chunk_chars` = cota superior **verdadera** de `len(text)` en la coleccion: `max(5000, max observado)`. Crece con el dato; el motor debe leerla del manifest (no asumir 5000) y exigir `KG_MAX_TEXT_CHARS >=` este valor.

## 5. Que NO produce LI_AD

`queries.parquet`/`qrels.parquet` (ground-truth de eval del harness), embeddings/vectores, ni el grafo. El export del grafo **de vuelta** (motor -> LI_AD, para su Fase 2) es un contrato reciproco aparte: `KG_CONTRACT.md`.

## 6. Versionado, generacion y consistencia

- **`contract_version`** (semantica del contrato) y **`schema_version`** (columnas del parquet) versionan por separado: las columnas evolucionan a otro ritmo que el layout/manifest.
- **`generation`**: el motor compara su generacion indexada con la del manifest; re-indexa **solo si** avanza. Da deteccion incremental barata. (Rebuild completo por generacion; el append incremental es deuda #10 del motor, futuro.)
- **Consistencia (Refinamiento A, prioridad baja)**: `/rechunk` reescribe parts en sitio. MinIO da atomicidad por-objeto pero **no** del conjunto: un lector concurrente puede ver parts mezcladas (gen N + N+1). Mitigacion: el motor valida `num_rows` por part contra el manifest; si no cuadra (part sobreescrita bajo los pies) trata la coleccion como "en vuelo" y re-lee el manifest (ya gen N+1). Solo relevante con lectura+rechunk concurrentes; con single-operador no muerde. Alternativa mas fuerte si se prioriza: escribir la nueva generacion a staging y hacer flip atomico del manifest.

## 7. Validacion del contrato (lado motor, pre-carga)

Antes de indexar, el motor valida y **falla temprano y claro** si:

1. Faltan columnas REQUISITO (`chunk_id`, `collection_id`, `text`) o hay nulos en ellas.
2. `(collection_id, chunk_id)` no es unico.
3. Algun `len(text) > max_chunk_chars` (coherencia interna manifest-dato, Refinamiento B).
3b. (En indexacion) `max_chunk_chars > KG_MAX_TEXT_CHARS` del motor — la coleccion se rechaza con mensaje accionable ("sube KG_MAX_TEXT_CHARS a >= N") en vez de truncar en silencio.
4. La suma de `num_rows` de `parts` no es `num_chunks`, o el `num_rows` real de una part no cuadra con el manifest (Refinamiento A).
5. (En indexacion) `generation`/`chunking_fingerprint` incoherentes con lo ya indexado.

## 8. Implementacion pendiente (LI_AD, tras ack final)

- Emitir `collection.json` de contrato (separado de `meta.json` admin-internal) con los campos de §4, escrito al final de la ingesta.
- Añadir columnas `document_id` y `chunk_index` al parquet.
- Bundlear el vocab `cl100k_base` para un tokenizer determinista (eliminar el fallback word-level).
- Opcional, solo si el motor lo pide: `bbox`/heading como columnas; content-hash en `chunk_id`.

---

> v1 acordada salvo ack final + implementacion de §8 por LI_AD. El export reciproco del grafo se trata en `KG_CONTRACT.md`.
