# CH_LIRAG

Sistema de evaluacion RAG (Retrieval-Augmented Generation) para benchmarking de pipelines de recuperacion y generacion sobre datasets MTEB/BeIR (HotpotQA actualmente) con infraestructura NVIDIA NIM.

Soporta 2 estrategias de retrieval: busqueda vectorial pura (`SIMPLE_VECTOR`) y **LightRAG** (Vector + Knowledge Graph dual-level con extraccion de tripletas via LLM). Estrategia `HYBRID_PLUS` (BM25+Vector+NER) pendiente de eliminacion (DTm-83).

## Arquitectura

```
CH_LIRAG/
├── shared/                          # Libreria compartida
│   ├── types.py                     # NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
│   ├── metrics.py                   # F1, ExactMatch, Accuracy, Faithfulness (LLM-judge)
│   ├── llm.py                       # AsyncLLMService, load_embedding_model
│   ├── config_base.py               # InfraConfig, RerankerConfig, helpers _env_*
│   ├── report.py                    # RunExporter: JSON + CSV summary + CSV detail
│   ├── vector_store.py              # ChromaVectorStore
│   ├── structured_logging.py        # Logging JSONL estructurado
│   └── retrieval/
│       ├── __init__.py              # Factory get_retriever()
│       ├── core.py                  # BaseRetriever, SimpleVectorRetriever, RetrievalConfig, RRF
│       ├── reranker.py              # CrossEncoderReranker (NVIDIARerank)
│       ├── hybrid/                  # HYBRID_PLUS (pendiente eliminacion, DTm-83)
│       └── lightrag/                # Estrategia LIGHT_RAG
│           ├── retriever.py         # Vector + Knowledge Graph dual-level
│           ├── knowledge_graph.py   # KG in-memory (igraph): entidades, relaciones, traversal
│           └── triplet_extractor.py # Extraccion tripletas y query keywords via LLM
│
├── sandbox_mteb/                    # Evaluacion MTEB/BeIR
│   ├── config.py                    # MTEBConfig: .env -> dataclass validada
│   ├── loader.py                    # MinIO/Parquet -> LoadedDataset
│   ├── evaluator.py                 # Pipeline orquestador (<600 LOC)
│   ├── subset_selection.py          # Seleccion corpus: DEV_MODE, gold docs
│   ├── retrieval_executor.py        # Loop retrieval sync + metricas
│   ├── generation_executor.py       # Generacion async + metricas
│   ├── embedding_service.py         # Pre-embed queries batch (NIM REST)
│   ├── checkpoint.py                # Checkpoint/resume cada N queries
│   ├── result_builder.py            # Construccion EvaluationRun final
│   ├── preflight.py                 # Validacion pre-run (deps, NIM, MinIO)
│   ├── run.py                       # Entry point (--dry-run, -v, --resume)
│   └── env.example                  # Plantilla .env
│
├── tests/                           # pytest (unit + integration)
│   ├── conftest.py                  # Mocks condicionales (solo si paquete no instalado)
│   ├── test_*.py                    # Unit tests
│   └── integration/                 # Tests contra NIM + MinIO reales
│
├── pyproject.toml                   # Config pytest
├── mypy.ini                         # Config mypy
└── requirements.txt
```

## Estrategias de retrieval

| Estrategia | Indexacion | Busqueda | Reranker |
|---|---|---|---|
| `SIMPLE_VECTOR` | Embedding directo (NIM) | Cosine similarity (ChromaDB) | Opcional |
| `LIGHT_RAG` | LLM triplet extraction + KG + Embedding | Vector + Graph traversal + Fusion | Opcional |

> `HYBRID_PLUS` (BM25+Vector+NER cross-linking) existe en el codigo pero esta pendiente de eliminacion (DTm-83). LIGHT_RAG supersede su funcionalidad.

### LIGHT_RAG

Implementacion inspirada en [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779). Combina busqueda vectorial con un knowledge graph construido via LLM, sin BM25 — el grafo reemplaza la funcion de bridging lexical.

> **Estado de alineacion con el original (ver DAM-1 a DAM-8 en deuda tecnica):**
> Fases C-E implementaron entity VDB + relationship VDB + description merging,
> alineando la resolucion semantica con el original. Fase F implemento el modo
> `graph_primary` (grafo como retriever primario) y contexto estructurado JSON.
> Pendiente: validacion con run comparativo (F.5).

**Indexacion:**
1. Extrae tripletas (entidad, relacion, entidad) de cada documento via LLM (`TripletExtractor`)
2. Construye un `KnowledgeGraph` in-memory (igraph) con entidades, relaciones e indices invertidos
3. Indexa contenido original en ChromaDB para vector search

**Retrieval:**
1. Vector search (ChromaDB) → top_k candidatos con cosine similarity
2. Query analysis via LLM → extrae keywords de bajo nivel (entidades especificas) y alto nivel (temas abstractos)
3. Graph traversal dual-level:
   - **Low-level**: Entity VDB (cosine similarity) resuelve entidades semanticamente. BFS desde entidades resueltas, scoring `1/(1+depth)`. Fallback a string matching si VDB no disponible.
   - **High-level**: Relationship VDB (cosine similarity + edge weights) resuelve relaciones semanticamente. Fallback a token matching si VDB no disponible.
4. Modos configurables via `LIGHTRAG_MODE`: `hybrid` (default), `graph_primary`, `local`, `global`, `naive`.
5. Fusion via RRF (default) o modo `graph_primary` donde el grafo traza source docs directamente con vector search como fallback.

**Fallback:** Sin `igraph` o sin LLM service → degrada automaticamente a SimpleVectorRetriever puro (warning en log).

**Hardening (produccion):**
- Batching de coroutines en chunks de 500 docs para evitar presion de memoria en `extract_batch_async()` (DTm-22)
- Cap de entidades configurable (`KG_MAX_ENTITIES`, default 100K) para limitar crecimiento del grafo (DTm-21, DTm-63)
- Deduplicacion de relaciones en aristas (misma relacion + mismo doc no se duplica)
- Validacion post-parse del output LLM: entity types normalizados a enum (`PERSON|ORG|PLACE|CONCEPT|EVENT|OTHER`), nombres >= 1 char, descriptions truncadas a 200 chars (DTm-16)
- Estimacion de memoria en `get_stats()` para observabilidad
- Robustez para modelos de razonamiento (nemotron-3-nano thinking mode): strip de `<think>` tags en `llm.py` (incluye tags sin cerrar por truncamiento), `max_tokens` ampliados en extraccion (2048 tripletas, 512 keywords) para compensar tokens consumidos por razonamiento, y fallback `json.JSONDecoder.raw_decode()` para extraer JSON de respuestas con texto mixto
- Trazas de depuracion (nivel DEBUG): log de chars eliminados por strip de `<think>` tags, y primeros 200 chars del raw response en fallos de parse JSON — permite diagnosticar problemas con NIM sin activar logging verboso en produccion

**Optimizacion:** `pre_extract_query_keywords()` permite pre-extraer keywords de todas las queries en batch antes del loop de retrieval, analogo al pre-embed de vectores.

## Pipeline de evaluacion

```
.env -> MTEBConfig -> MinIO/cache(Parquet) -> LoadedDataset
     -> shuffle(seed) -> slice(max_corpus)
     -> [LLM triplet extraction + KG construction si LIGHT_RAG]
     -> index(ChromaDB)
     -> pre-embed queries (batch REST NIM)
     -> [pre-extract query keywords (batch LLM) si LIGHT_RAG]
     -> retrieve(local ChromaDB [+ Graph traversal + Fusion si LIGHT_RAG], sync)
     -> [rerank(cross-encoder) si habilitado]
     -> generate + metrics (async)
     -> EvaluationRun -> JSON + CSV
```

## Metricas

### Retrieval

Hit@K, MRR, Recall@K (K=1,3,5,10,20), NDCG@K sobre top `RETRIEVAL_K` documentos (pre-rerank).

### Generacion

Metrica primaria segun `answer_type`: `"label"` -> Accuracy, otro -> F1. EM siempre como secundaria. Faithfulness (LLM-judge) opcional.

### Retrieval efectivo (post-rerank)

Cuando el reranker esta activo: `generation_recall`, `generation_hit`, `reranker_rescue_count` (queries rescatadas por reranker). Sin reranker, no se emiten.

## Dataset: HotpotQA

| Propiedad | Valor |
|---|---|
| Queries | 7405 |
| Corpus | 66576 documentos (Wikipedia, deduplicados por titulo) |
| Qrels | 14810 (2.0 por query, solo supporting_facts) |
| Tipos de query | bridge (~80%), comparison (~20%) |
| Almacenamiento | MinIO (Parquet), `s3://lakehouse/datasets/evaluation/hotpotqa/` |

**Limitacion del corpus.** 10 pasajes por query (2 gold + 8 distractores). Resultados no comparables con benchmarks publicados (corpus completo ~5.2M). Solo comparaciones relativas entre estrategias son validas.

## Setup y uso

```bash
pip install -r requirements.txt
cp sandbox_mteb/env.example sandbox_mteb/.env
# Editar .env con endpoints NIM y MinIO
```

Para `LIGHT_RAG` (knowledge graph):

```bash
pip install python-igraph snowballstemmer
```

> **Nota:** LIGHT_RAG requiere un LLM service activo (NIM) tanto para indexacion (extraccion de tripletas) como para retrieval (analisis de queries). Sin `igraph` o sin LLM, degrada a vector search puro.

```bash
python -m sandbox_mteb.run                  # Run con .env
python -m sandbox_mteb.run --dry-run        # Solo validar config
python -m sandbox_mteb.run --env /path/.env # .env alternativo
python -m sandbox_mteb.run -v               # Verbose (DEBUG)
python -m sandbox_mteb.run --resume RUN_ID  # Reanudar run desde checkpoint

# Preflight check (recomendado antes de runs LIGHT_RAG largos)
python -m sandbox_mteb.preflight            # Verifica deps, NIM, MinIO, smoke test
python -m sandbox_mteb.preflight --skip-smoke  # Sin smoke test LLM
```

### Reproducibilidad (requirements.lock)

Para garantizar resultados comparables entre runs, pinnear dependencias en el entorno de produccion:

```bash
pip install -r requirements.txt
pip freeze > requirements.lock
```

`requirements.lock` se commitea al repo. Usar `pip install -r requirements.lock` para reproducir el entorno exacto.

## Configuracion (.env)

Referencia completa en `sandbox_mteb/env.example`. Variables criticas:

```bash
# Embedding (NIM)
EMBEDDING_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_BASE_URL=http://<nim-embedding-host>:8000/v1
EMBEDDING_MODEL_TYPE=asymmetric
EMBEDDING_BATCH_SIZE=5

# LLM (NIM)
LLM_BASE_URL=http://<nim-llm-host>:8000/v1
LLM_MODEL_NAME=nvidia/nemotron-3-nano
NIM_MAX_CONCURRENT_REQUESTS=32
NIM_REQUEST_TIMEOUT=120

# Retrieval
RETRIEVAL_STRATEGY=SIMPLE_VECTOR      # SIMPLE_VECTOR | LIGHT_RAG
RETRIEVAL_K=20

# Knowledge Graph (solo LIGHT_RAG)
KG_MAX_HOPS=1                         # Profundidad maxima BFS (DAM-7: 1-hop como el original)
KG_MAX_TEXT_CHARS=3000                 # Max chars de documento enviados al LLM para extraccion
KG_GRAPH_WEIGHT=0.3                   # Peso del score del grafo en fusion
KG_VECTOR_WEIGHT=0.7                  # Peso del score vectorial en fusion
KG_MAX_ENTITIES=0                     # Cap de entidades en KG (0 = default 100K)
KG_CACHE_DIR=./data/kg_cache          # Directorio para persistir KG entre runs (vacio = sin cache)
KG_FUSION_METHOD=rrf                  # rrf (default, robusto) o linear (legacy)
KG_RRF_K=60                           # Constante k para RRF (default 60)
KG_KEYWORD_MAX_TOKENS=1024            # Max tokens para keyword extraction LLM call
KG_GRAPH_OVERFETCH_FACTOR=2           # Graph traversal pide N * top_k candidatos
LIGHTRAG_MODE=hybrid                  # hybrid | graph_primary | local | global | naive

# Reranker (opcional)
RERANKER_ENABLED=false
RERANKER_BASE_URL=http://<nim-reranker-host>:9000/v1
RERANKER_MODEL_NAME=nvidia/llama-3.2-nv-rerankqa-1b-v2
RERANKER_TOP_N=5
RERANKER_FETCH_K=0                    # Candidatos para reranker (0 = top_n * 3)

# Dataset
MTEB_DATASET_NAME=hotpotqa
EVAL_MAX_QUERIES=50                   # 0 = todas (default en codigo: 50)
EVAL_MAX_CORPUS=1000                  # 0 = todo (default en codigo: 1000)
GENERATION_ENABLED=true
CORPUS_SHUFFLE_SEED=42

# DEV_MODE: subset con gold docs garantizados (metricas optimistas, solo comparacion relativa)
DEV_MODE=false
DEV_QUERIES=200
DEV_CORPUS_SIZE=4000

# MinIO
MINIO_ENDPOINT=http://<minio-host>:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minio123
MINIO_BUCKET_NAME=lakehouse
S3_DATASETS_PREFIX=datasets/evaluation
```

## Tests

```bash
pytest tests/                      # Unit + integracion
pytest tests/ -m "not integration" # Solo unit
pytest tests/integration/ -v       # Solo integracion (requiere NIM + MinIO)
```

## Historial de fases completadas

<details>
<summary>Fases 0-4: fundacion del sistema (completadas)</summary>

| Fase | Objetivo | Resumen |
|---|---|---|
| **0. Reproducibilidad** | Config determinista entre runs | `requirements.lock` pinneado, `preflight.py` para validacion pre-run. |
| **1. Fiabilidad** | No perder runs de horas | Checkpoint/resume cada 50 queries (`--resume RUN_ID`). Counter de entidades descartadas por cap KG. |
| **2. Calidad retrieval** | Mejorar Hit@5 y MRR | Apostrofes preservados en BM25, entity normalization alineada KG/linker, `MIN_ENTITY_NAME_LEN=1`. |
| **3. Eficiencia** | Corpus 66K sin OOM | Batch adaptativo (`semaphore*4`), dedup memoria. |
| **4. Mantenibilidad** | `evaluator.py` < 600 LOC | Descompuesto de 1225 a 592 LOC. Extraidos: `subset_selection.py`, `retrieval_executor.py`, `generation_executor.py`, `embedding_service.py`, `checkpoint.py`, `result_builder.py`. |

Issues resueltos: DTm-14 a DTm-38, DTm-45, DTm-46. Ver historial git para detalles.

</details>

<details>
<summary>Mejoras LIGHT_RAG — ronda 1: arquitectura (completada)</summary>

8 problemas de code review, agrupados en 5 sub-fases:

| Sub-fase | Objetivo | Estado |
|---|---|---|
| 0. Bugs + limpieza | Fingerprint truncado, race conditions, dead code | Completado |
| 1. Keyword search | Stemming en indices invertidos del KG | Completado |
| 2.1. Batch extraction | 1 doc/llamada → N docs/llamada (3-5x menos LLM calls) | Completado |
| 2.2. igraph | NetworkX → igraph (C-backed, 10-100x BFS) | Completado |
| 3. RRF fusion | Fusion lineal → Reciprocal Rank Fusion | Completado |
| 4. Test coverage | Tests evaluator, loader, pipeline e2e mockeado | Completado |

Cambios en dependencias: `networkx` → `python-igraph`, nuevo `snowballstemmer`.

</details>

<details>
<summary>Mejoras LIGHT_RAG — ronda 2: robustez (completada)</summary>

9 issues resueltos (DTm-47 a DTm-54, DTm-59):

| ID | Fix |
|---|---|
| DTm-47 | Cache de query keywords con LRU eviction (OrderedDict, max 10K) |
| DTm-48 | List comprehension en vez de `[([],[])]*n` (refs mutables) |
| DTm-49 | `triplets_dropped_by_cap` contador + WARNING resumen + `get_stats()` |
| DTm-50 | `assert` → `raise RuntimeError` (safe con `-O`) |
| DTm-51 | `threading.Lock` en cache de keywords |
| DTm-52 | WARNING cuando graph-only docs no se resuelven en ChromaDB |
| DTm-53 | Fusion lineal all-zero → fallback automatico a RRF |
| DTm-54 | Fallback batch→single-doc logueado como WARNING |
| DTm-59 | `stemWords()` envuelto en try-except |

</details>

## Resultados comparativos

> **Nota:** Todos los runs son **pre-Fases C-F** (sin entity VDB, sin relationship VDB,
> sin modo graph_primary). Pendiente run comparativo post-implementacion (F.5).

### Run 3: LIGHT_RAG post-refactor — 29 queries, 500 docs, DEV_MODE

| Metrica | Valor |
|---|---|
| Hit@5 | 0.793 |
| MRR | 0.518 |
| Recall@5 | 0.690 |
| Recall@10 | 0.931 |
| Recall@20 | 0.983 |
| Gen Recall | 0.983 |
| Gen Hit | 1.000 |
| F1 | 0.776 |
| EM | 0.586 |
| KG entidades | 3265 |
| KG relaciones | 2825 |
| KG componentes | 473 |
| KG build time | 850s (94% del run) |
| Reranker delta | +29.3pp (Recall@5 → Gen Recall) |

Config: `KG_GRAPH_WEIGHT=0.4, KG_VECTOR_WEIGHT=0.6, KG_FUSION_METHOD=rrf, RERANKER=ON`.
Embedding: llama-embed-nemotron-8b. LLM: nemotron-3-nano.

### Baseline: SIMPLE_VECTOR vs LIGHT_RAG

| Metrica | SIMPLE_VECTOR (50q, 3.5K docs) | LIGHT_RAG (29q, 500 docs) | Diagnostico |
|---|---|---|---|
| **Hit@5** | **1.000** | 0.793 | KG degrada -20.7pp |
| **MRR** | **1.000** | 0.518 | KG degrada -48.2pp |
| Recall@5 | 0.940 | 0.690 | KG degrada -25.0pp |
| Recall@20 | 0.970 | 0.983 | Similar (cobertura profunda OK) |
| Gen Recall | 0.970 | 0.983 | Reranker compensa |
| F1 | 0.754 | 0.776 | Similar |
| Tiempo | 138s | 901s | KG build = 94% del tiempo |

> Configs no identicas (50q/3.5K vs 29q/500), pero la tendencia es clara:
> LIGHT_RAG con KG activo degrada severamente el ranking. La causa raiz
> son las divergencias arquitectonicas documentadas en DAM-1 a DAM-8.

### Run 1 (KG vacio) vs Run 2 (KG funcional) — 125 queries, mismo seed

| Metrica | Run 1 (KG vacio, 33K corpus) | Run 2 (KG activo, 16.5K corpus) | Delta |
|---|---|---|---|
| KG entidades | 1208 (1.1%) | 50000 (70%) | +4040% |
| KG relaciones | 899 | 51841 | +5666% |
| **Hit@5** | **0.896** | **0.848** | **-4.8pp** |
| **MRR** | **0.864** | **0.531** | **-33.3pp** |
| Recall@5 | 0.816 | 0.660 | -15.6pp |
| NDCG@10 | 0.806 | 0.557 | -24.9pp |
| Gen Recall | 0.940 | 0.964 | +2.4pp |
| F1 | 0.756 | 0.784 | +2.8pp |
| EM | 0.568 | 0.624 | +5.6pp |

**Hallazgos clave:**

1. **El KG activo destruye el ranking.** MRR cae 33pp. Los scores normalizados del grafo desplazan docs vectoriales relevantes de posicion 1-2 a posiciones 3-10.
2. **El reranker compensa la degradacion.** Delta reranker (Gen Recall - Recall@5) paso de +12.4pp a +30.4pp. Sin reranker, este run seria significativamente peor.
3. **F1/EM mejoran pese a peor retrieval.** Gen Recall sube ligeramente (0.940→0.964) y el corpus mas pequeno reduce distractores. Recall@20=0.976 confirma mejor cobertura profunda.
4. **Entity cap 50K es insuficiente.** 26382/76382 entidades descartadas (34.5%). El orden de indexacion determina que entidades entran, sesgo arbitrario.
5. **Runs no comparables directamente.** Corpus distinto (33K vs 16.5K). Se necesita baseline SIMPLE_VECTOR con 16.5K para aislar efecto neto del KG.

## Deuda tecnica

### Deuda arquitectonica mayor (vs LightRAG original)

Analisis comparativo con [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025).
CH_LIRAG se inspira en el paper pero diverge en decisiones arquitectonicas clave que
explican la degradacion de ranking observada (MRR 0.52 vs ~0.86 con vector puro).

| ID | Severidad | Divergencia vs Original | Impacto en CH_LIRAG | Estado |
|---|---|---|---|---|
| DAM-1 | **Critica** | **Entity VDB**: entity names + descriptions indexados como embeddings en ChromaDB (cosine distance). Similarity search reemplaza string matching en low-level retrieval. Threshold de distancia para filtrar matches irrelevantes. | Fase C. Resuelve DTm-70. | Implementado |
| DAM-2 | **Critica** | **Relationship VDB**: relaciones indexadas como embeddings en ChromaDB (cosine distance). Similarity search reemplaza token matching en high-level retrieval. Scoring ponderado por edge weight. | Fase D. Resuelve DTm-71, DTm-74. | Implementado |
| DAM-3 | **Alta** | **Grafo como suplemento vs retriever primario**: en el original, los modos local/global/hybrid usan el grafo como mecanismo principal de retrieval. Modo `graph_primary` implementado en Fase F (F.2). | Fase F. | Implementado |
| DAM-4 | **Alta** | **Merging de descripciones**: descripciones multi-doc se acumulan y consolidan (dedup + top 5 por longitud, max 500 chars). Sin LLM synthesis aun (concatenacion simple vs map-reduce del original). Ver DTm-80. | Fase E. | Parcial |
| DAM-5 | **Media** | **Edge weights**: weight = docs unicos que mencionan la arista. Usado como factor de scoring en relationship VDB. | Fases D+E. | Implementado |
| DAM-6 | **Media** | **Gleaning**: re-extraccion con prompt de continuacion. Configurable via `KG_GLEANING_ROUNDS` (default 0 = desactivado). | Fase E. | Implementado |
| DAM-7 | **Media** | **BFS 1-hop**: default cambiado de 2 a 1 (como el original). Configurable via `KG_MAX_HOPS`. | Fase C. | Implementado |
| DAM-8 | **Baja** | **Contexto estructurado para generacion**: JSON con 3 secciones (Entity, Relationship, Document Chunks) cuando `graph_primary` activo. Implementado en Fase F (F.3). | Fase F. | Implementado |

**Estado (2026-04-06):** DAM-1, DAM-2, DAM-3, DAM-5, DAM-6, DAM-7, DAM-8 implementados (Fases C-F). DAM-4 parcial (concatenacion, sin LLM synthesis — ver DTm-80). **Pendiente: validacion con run comparativo (F.5).**

**Correcciones de validacion aplicadas (V.1-V.6):**
- V.1: VDBs usan cosine distance (no L2)
- V.2: Threshold de distancia (_ENTITY_VDB_MAX_DISTANCE=0.8) filtra matches irrelevantes
- V.3: Edge weight cuenta docs unicos, no relaciones en la arista
- V.4: Descripciones mergeadas limitadas a 500 chars (top 5 por longitud)
- V.5: Gleaning prompt incluye formato JSON explicito
- V.6: Conversion cosine distance [0,2] → similarity [1.0, 0.0] correcta + threshold

### Registro completo

| ID | Severidad | Descripcion | Ubicacion | Estado |
|---|---|---|---|---|
| DTm-66 | **Alta** | **`max_tokens=8192` en extraccion batch causa generacion masiva en thinking mode**. Fix: configurable via `KG_EXTRACTION_MAX_TOKENS` (default 4096). | `shared/retrieval/lightrag/triplet_extractor.py` | Resuelto |
| DTm-67 | **Alta** | **Batch docs/call configurable** via `KG_BATCH_DOCS_PER_CALL` (default 5). Subir a 10 reduce calls ~50%. | `shared/retrieval/lightrag/triplet_extractor.py` | Resuelto |
| DTm-62 | **Alta** | Fusion KG destruye ranking: MRR -33pp con KG activo. Scores normalizados del grafo compiten con vectoriales, desplazando docs relevantes. | `shared/retrieval/lightrag/retriever.py` fusion | Abierto |
| DTm-63 | **Alta** | Entity cap: subido de 50K a 100K (DTm-63). Sesgo FIFO persiste — orden de indexacion determina que entidades se descartan. | `shared/retrieval/lightrag/knowledge_graph.py` | Parcial |
| DTm-68 | **Media** | **Re-serializacion JSON eliminada**: `_build_entities_relations()` recibe dict directamente. | `shared/retrieval/lightrag/triplet_extractor.py` | Resuelto |
| DTm-69 | **Media** | **Token indexing diferido**: `build_keyword_indices()` como fase post-build, llamado desde `retriever.py:237`. | `shared/retrieval/lightrag/knowledge_graph.py` | Resuelto |
| DTm-64 | **Media** | Normalizacion [0,1] incomparable entre canales: distribuciones vector (concentrada) vs graph (uniforme) generan scores engañosos. RRF mitiga parcialmente. | `shared/retrieval/lightrag/retriever.py` | Abierto |
| DTm-65 | **Media** | Thinking-mode exhaustion: ~17% queries fallan en 1er intento. `KG_KEYWORD_MAX_TOKENS` configurable (default 1024). Puede requerir 2048 con modelos reasoning-heavy. | `shared/retrieval/lightrag/triplet_extractor.py` | Mitigado |
| DTm-55 | **Media** | Stats extractor se corrompen si KG build falla a mitad. `_has_graph=False` pero stats parciales persisten. | `shared/retrieval/lightrag/retriever.py` | Abierto |
| DTm-56 | **Media** | Fingerprint collision con corpus vacio: `sha256("")[:16]` es determinista pero edge case si dos configs distintas producen mismo hash. | `shared/retrieval/lightrag/retriever.py` | Abierto |
| DTm-12 | Baja | Sesgo LLM-judge en faithfulness para respuestas cortas. Inherente al LLM-judge. | `shared/metrics.py` | Aceptado |
| DTm-13 | Baja | No-determinismo HNSW: ChromaDB no expone `hnsw:random_seed`. ±0.02 entre runs. | `shared/vector_store.py` | Aceptado |
| DTm-24 | Baja | Naming ambiguo: `RRF_VECTOR_WEIGHT` vs `KG_VECTOR_WEIGHT`. Se resuelve con eliminacion de HYBRID_PLUS (DTm-83). | `sandbox_mteb/config.py` | Pendiente DTm-83 |
| DTm-57 | Baja | Normalizacion entidades agresiva: pierde apostrofes/guiones. Colisiones raras. | `shared/retrieval/lightrag/knowledge_graph.py` | Abierto |
| DTm-58 | Baja | No dedup queries identicas en batch keyword extraction: LLM calls duplicadas. | `shared/retrieval/lightrag/triplet_extractor.py` | Abierto |
| DTm-60 | Baja | Stats extractor acumulan entre llamadas. `reset_stats()` nunca se llama auto. | `shared/retrieval/lightrag/triplet_extractor.py` | Abierto |
| DTm-61 | Baja | No validacion tamano keywords del LLM: respuestas patologicas pasan sin limite. | `shared/retrieval/lightrag/triplet_extractor.py` | Abierto |
| DTm-70 | **Alta** | **Entity matching exacto en `query_entities` impide bridging**: BFS solo arranca si el keyword del LLM matchea exactamente el nombre normalizado de la entidad. **Mitigado por DAM-1 (entity VDB)**: `_resolve_entities_via_vdb()` resuelve por embedding similarity, bypass de string matching. Fallback lexico sigue activo si VDB no disponible. | `shared/retrieval/lightrag/knowledge_graph.py` | Mitigado (DAM-1) |
| DTm-71 | **Alta** | **`query_by_keywords` no hace graph traversal**: encuentra entidades por keyword pero solo devuelve sus docs directos (`source_doc_ids`), sin BFS. **Mitigado por DAM-2 (relationship VDB)**: `_resolve_relationships_via_vdb()` reemplaza `query_by_keywords` en high-level path. Fallback lexico sigue activo si VDB no disponible. | `shared/retrieval/lightrag/knowledge_graph.py` | Mitigado (DAM-2) |
| DTm-72 | **Media** | **BFS scoring ciego a la relacion**: `hop_score = 1/(1+depth)` trata todas las aristas igual. Una relacion `directed_by` pesa igual que `same_year_as`. Para queries como "nationality of the director", la relacion `directed_by` deberia puntuar mas. Fix: ponderar hop score por token overlap entre relation type y keywords de la query. | `shared/retrieval/lightrag/knowledge_graph.py` | Abierto |
| DTm-73 | **Alta** | **Grafo fragmentado (2831 componentes) limita bridging**: si los 2 gold docs de una query multi-hop generan entidades en componentes distintos (e.g. "Vlatko Gilić" vs "Gilić"), el BFS no puede cruzar. Fix: entity co-reference por token overlap de apellido, o aristas implicitas entre entidades co-ocurrentes en el mismo doc sin relacion explicita. | `shared/retrieval/lightrag/knowledge_graph.py` | Abierto |
| DTm-74 | Baja | **Scoring flat en `query_by_keywords`**: entidad match siempre puntua 1.0, relacion match siempre 0.5. **Mitigado por DAM-2**: relationship VDB usa cosine similarity + edge weight, reemplazando el scoring flat en high-level path. Solo afecta al fallback lexico. | `shared/retrieval/lightrag/knowledge_graph.py` | Mitigado (DAM-2) |
| DTm-75 | **Alta** | **Bug: loader silently succeeds when all downloads fail**. Fix: `ValueError` si queries y corpus son `None`. [#2](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/2) | `sandbox_mteb/loader.py` | Resuelto |
| DTm-76 | **Alta** | **Chunk selection desde grafo**. `_select_chunks_from_graph()` combina doc_ids de entity + relationship results. Implementado en Fase F (F.1). [#3](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/3) | `shared/retrieval/lightrag/retriever.py` | Resuelto |
| DTm-77 | **Media** | **Test gap: gleaning (DAM-6) tiene 0 tests**. `glean_from_doc_async()` implementada pero sin tests unitarios. Feature marcada como completada sin cobertura. [#4](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/4) | `tests/` | Abierto |
| DTm-78 | **Media** | **Test gap: E2E pipeline solo cubre SIMPLE_VECTOR**. `test_pipeline_e2e.py` hardcodeado con SIMPLE_VECTOR. No hay validacion E2E de LIGHT_RAG. [#5](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/5) | `tests/test_pipeline_e2e.py` | Abierto |
| DTm-79 | Baja | **Modos de query explicitos**: `LIGHTRAG_MODE` con 5 modos (hybrid, graph_primary, local, global, naive). Implementado en Fase F (F.4). [#6](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/6) | `shared/retrieval/lightrag/retriever.py` | Resuelto |
| DTm-80 | Baja | **DAM-4 parcial: falta LLM synthesis para merge de descripciones**. Actual: concatenacion con ` \| `. Original: LLM map-reduce cuando tokens exceden umbral. [#7](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/7) | `shared/retrieval/lightrag/knowledge_graph.py` | Abierto |
| DTm-81 | Baja | **Import fantasma `shared.retrieval.hybrid.core`**: corregido a `..core`. Se elimina completamente con DTm-83. [#8](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/8) | `shared/retrieval/hybrid/retriever.py:143` | Resuelto |
| DTm-82 | Baja | **23 errores mypy sin rastrear**: `union-attr` en evaluator.py, tipos incompatibles en vector_store.py, imports condicionales en reranker.py. [#9](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/9) | Multiples archivos | Abierto |
| DTm-83 | **Alta** | **Eliminar estrategia HYBRID_PLUS completa**. LIGHT_RAG supersede la funcionalidad (entity VDB + relationship VDB vs entity co-ocurrencia). Elimina ~2,100 LOC + 3 deps (spaCy, tantivy, rank-bm25). Resuelve DTm-81 y DTm-24. [#10](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/10) | `shared/retrieval/hybrid/`, tests, config | Abierto |

### Deuda resuelta (referencia)

DTm-14 a DTm-38, DTm-45 a DTm-54, DTm-59, DTm-66 a DTm-69 (35 issues). Ver historial git.

## Plan de desarrollo por fases

> **Nota (2026-04-06):** El objetivo es alinear la implementacion con la arquitectura
> de [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG). El analisis comparativo
> revelo divergencias criticas (DAM-1 a DAM-8) que son la causa raiz de la degradacion
> de ranking. La alineacion arquitectonica no es opcional — es el camino para que
> LIGHT_RAG funcione como fue disenado.

<details>
<summary>Fases A-E: alineacion arquitectonica con LightRAG original (completadas)</summary>

| Fase | Objetivo | DAMs resueltos |
|---|---|---|
| **A** | Documentacion de divergencias y baseline | — |
| **B** | Rendimiento KG build (DTm-66 a DTm-69) | — |
| **C** | Entity VDB: resolver entidades por embedding similarity | DAM-1, DAM-7 |
| **D** | Relationship VDB: semantic search para high-level retrieval | DAM-2, DAM-5 |
| **E** | Calidad grafo: description merging + gleaning + entity cap 100K | DAM-4 (parcial), DAM-6 |

Correcciones de validacion V.1-V.6 aplicadas post-implementacion.

</details>

### Fase F: Grafo como primary retriever (DAM-3, DAM-8) — completada (F.1-F.4)

**Objetivo:** Alinear el rol del grafo con el original — el grafo como mecanismo principal de
retrieval, no como suplemento de vector search.

| Tarea | Descripcion | DAM/DTm | Estado |
|---|---|---|---|
| F.1 Chunk selection desde grafo | `_select_chunks_from_graph()` combina doc_ids de entity + relationship results. | DTm-76 | Completada |
| F.2 Modo `graph_primary` | `_retrieve_via_graph()`: grafo primario, vector fallback si < top_k/2. | DAM-3 | Completada |
| F.3 Contexto estructurado | `format_structured_context()`: JSON con Entity + Relationship + Chunks. | DAM-8 | Completada |
| F.4 Modos de query | `LIGHTRAG_MODE`: hybrid, graph_primary, local, global, naive. | DTm-79 | Completada |
| F.5 Evaluar hibrido vs graph-primary | Runs comparativos naive vs hybrid vs graph_primary. | DTm-62 | **Pendiente** |

**Criterio de exito:** LIGHT_RAG supera o iguala SIMPLE_VECTOR en MRR y Hit@5.

### Fase G: Deuda tecnica menor (en paralelo)

**Objetivo:** Limpiar issues medios y bajos. Ejecutable en cualquier momento.

| Tarea | Descripcion | DTm | Esfuerzo |
|---|---|---|---|
| G.1 Stats extractor resilientes | Snapshot/restore si KG build falla. | DTm-55 | Bajo |
| G.2 Fingerprint robusto | Incluir `len(documents)` y config hash. | DTm-56 | Bajo |
| G.3 `KG_KEYWORD_MAX_TOKENS=2048` | Validar si elimina retries. | DTm-65 | Trivial |
| G.4 Dedup queries en batch keywords | Filtrar duplicadas pre-LLM. | DTm-58 | Bajo |
| G.5 Reset stats automatico | `reset_stats()` al inicio de `extract_batch()`. | DTm-60 | Trivial |
| G.6 Keyword size cap | Max 20 keywords/nivel. | DTm-61 | Trivial |
| G.7 Entity normalization | Preservar guiones internos en `_normalize_name`. | DTm-57 | Bajo |
| G.8 Fix bug loader | `load_dataset()` debe detectar cuando todas las descargas fallan. | DTm-75 | Trivial |
| G.9 Tests gleaning | Tests unitarios para `glean_from_doc_async()`. | DTm-77 | Bajo |
| G.10 E2E LIGHT_RAG | Test pipeline E2E con estrategia LIGHT_RAG (mocked). | DTm-78 | Medio |
| G.11 LLM synthesis merge | Descripciones multi-doc sintetizadas via LLM (map-reduce). | DTm-80 | Medio |
| G.12 Import fantasma | Eliminar import `shared.retrieval.hybrid.core`. | DTm-81 | Trivial |
| G.13 Mypy cleanup | Resolver 23 errores mypy (union-attr, dict-item, etc.). | DTm-82 | Bajo |
| G.14 Eliminar HYBRID_PLUS | Eliminar estrategia, tests, deps (spaCy, tantivy, rank-bm25). ~2,100 LOC. | DTm-83 | Medio |

### Resumen de dependencias entre fases

```
Fase A ✅ ──── Fase B ✅ [paralelo]
                  │
                  ▼
            Fase C ✅ (entity VDB — DAM-1)
                  │
                  ▼
            Fase D ✅ (relationship VDB — DAM-2)
                  │
                  ▼
            Fase E ✅ (calidad grafo — DAM-4, DAM-6)
                  │
                  ▼
            Fase F ✅ (graph primary — DAM-3, DAM-8) [F.5 pendiente: runs]
                  │
                  ▼
            *** RUN COMPARATIVO PENDIENTE (F.5) ***

Fase G (deuda tecnica) ─── en paralelo, sin bloqueo ────────────────
```

Fases A-F implementadas (F.5 pendiente: runs comparativos).
Fase G: DTm-55 a DTm-83 (bugs, test gaps, code quality, eliminacion HYBRID_PLUS).

### Pendiente: Run comparativo post-VDBs

**Objetivo:** Medir el impacto real de las Fases C-E (entity VDB, relationship VDB,
merging, 1-hop) comparando con el baseline SIMPLE_VECTOR (MRR=1.0) y el run
LIGHT_RAG pre-VDBs (MRR=0.52).

**Config sugerida para el run:**

```bash
# Mantener DEV_MODE para comparabilidad con runs anteriores
RETRIEVAL_STRATEGY=LIGHT_RAG
DEV_MODE=true
DEV_QUERIES=50
DEV_CORPUS_SIZE=3500
CORPUS_SHUFFLE_SEED=42
RERANKER_ENABLED=true

# Nuevos defaults post-refactor (ya aplicados)
KG_MAX_HOPS=1              # DAM-7: 1-hop como el original
KG_GLEANING_ROUNDS=0       # DAM-6: activar con 1 para probar gleaning

# IMPORTANTE: borrar KG_CACHE_DIR o limpiar cache para forzar rebuild
# con los nuevos VDBs
```

**Metricas a comparar:**

| Metrica | SIMPLE_VECTOR (baseline) | LIGHT_RAG pre-VDBs | LIGHT_RAG post-VDBs |
|---|---|---|---|
| MRR | 1.000 | 0.518 | *pendiente* |
| Hit@5 | 1.000 | 0.793 | *pendiente* |
| Recall@5 | 0.940 | 0.690 | *pendiente* |
| Gen Recall | 0.970 | 0.983 | *pendiente* |
| KG componentes | — | 473 | *pendiente* |
| Tiempo total | 138s | 901s | *pendiente* |

**Que buscar en el run:**
- MRR: deberia mejorar significativamente vs 0.52. Si sube a >0.80, los VDBs funcionan.
- KG componentes: ya no deberia ser bloqueante (entity VDB resuelve por semantica, no por nombre).
- Logs: verificar que aparecen "entity VDB construido" y "relationship VDB construido".
- Tiempo: los VDBs anaden overhead de embedding (~1 min para 3K entidades). Verificar que es aceptable.

### Variables de configuracion (referencia)

| Variable | Default | Descripcion |
|---|---|---|
| `KG_MAX_HOPS` | 1 | Profundidad BFS. DAM-7: 1-hop como el original (antes 2). |
| `KG_GLEANING_ROUNDS` | 0 | DAM-6: rounds de re-extraccion (0 = desactivado). Probar con 1. |
| `KG_KEYWORD_MAX_TOKENS` | 1024 | Max tokens para LLM call de keyword extraction. |
| `KG_EXTRACTION_MAX_TOKENS` | 4096 | Max tokens para LLM call de extraccion de tripletas. |
| `KG_BATCH_DOCS_PER_CALL` | 5 | Docs por LLM call en batch extraction. |
| `KG_GRAPH_OVERFETCH_FACTOR` | 2 | Multiplicador de candidatos en graph traversal (N * top_k). |
| `KG_FUSION_METHOD` | rrf | Metodo de fusion: `rrf` (robusto) o `linear` (legacy). |
| `KG_RRF_K` | 60 | Constante k de Reciprocal Rank Fusion. |
| `KG_MAX_ENTITIES` | 100000 | Cap de entidades en KG (DTm-63: subido de 50K a 100K). |
