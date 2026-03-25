# CH_LIRAG

Sistema de evaluacion RAG (Retrieval-Augmented Generation) para benchmarking de pipelines de recuperacion y generacion sobre datasets MTEB/BeIR (HotpotQA actualmente) con infraestructura NVIDIA NIM.

Soporta 3 estrategias de retrieval: busqueda vectorial pura, hibrida BM25+Vector con entity cross-linking, y **LightRAG** (Vector + Knowledge Graph dual-level con extraccion de tripletas via LLM).

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
│       ├── core.py                  # BaseRetriever, SimpleVectorRetriever, RetrievalConfig
│       ├── hybrid_retriever.py      # BM25 + Vector + RRF
│       ├── hybrid_plus_retriever.py # BM25+Vector+RRF + NER cross-linking
│       ├── lightrag_retriever.py    # Vector + Knowledge Graph dual-level (LIGHT_RAG)
│       ├── knowledge_graph.py       # KG in-memory (igraph): entidades, relaciones, traversal
│       ├── triplet_extractor.py     # Extraccion tripletas y query keywords via LLM
│       ├── entity_linker.py         # NER (spaCy) + indice invertido + cross-refs
│       ├── reranker.py              # CrossEncoderReranker (NVIDIARerank)
│       └── tantivy_index.py         # BM25 via Tantivy (Rust, fallback rank-bm25)
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
| `HYBRID_PLUS` | NER cross-refs + Embedding + BM25 | BM25 (Tantivy) + Vector + RRF | Opcional |
| `LIGHT_RAG` | LLM triplet extraction + KG + Embedding | Vector + Graph traversal + Fusion | Opcional |

### HYBRID_PLUS

Durante indexacion, spaCy NER extrae entidades, construye indice invertido in-memory, y genera cross-references textuales entre documentos que comparten entidades. BM25 captura terminos puente entre documentos, mejorando retrieval en bridge questions multi-hop. Sin dependencia de NIM para indexacion. Sin spaCy, se comporta como BM25+Vector+RRF puro (warning en log).

### LIGHT_RAG

Implementacion inspirada en [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779). Combina busqueda vectorial con un knowledge graph construido via LLM, sin BM25 — el grafo reemplaza la funcion de bridging lexical.

**Indexacion:**
1. Extrae tripletas (entidad, relacion, entidad) de cada documento via LLM (`TripletExtractor`)
2. Construye un `KnowledgeGraph` in-memory (igraph) con entidades, relaciones e indices invertidos
3. Indexa contenido original en ChromaDB para vector search

**Retrieval:**
1. Vector search (ChromaDB) → top_k candidatos con cosine similarity
2. Query analysis via LLM → extrae keywords de bajo nivel (entidades especificas) y alto nivel (temas abstractos)
3. Graph traversal dual-level:
   - **Low-level**: BFS desde entidades de la query, scoring inversamente proporcional a hops (`1/(1+depth)`)
   - **High-level**: token matching en nombres de entidad y descripciones de relaciones (indice invertido por token, DTm-30)
4. Fusion via Reciprocal Rank Fusion (RRF) con pesos configurables (`KG_FUSION_METHOD=rrf`, default). Fusion lineal disponible como alternativa (`KG_FUSION_METHOD=linear`).

**Fallback:** Sin `igraph` o sin LLM service → degrada automaticamente a SimpleVectorRetriever puro (warning en log).

**Hardening (produccion):**
- Batching de coroutines en chunks de 500 docs para evitar presion de memoria en `extract_batch_async()` (DTm-22)
- Cap de entidades configurable (`KG_MAX_ENTITIES`, default 50K) para limitar crecimiento del grafo (DTm-21)
- Deduplicacion de relaciones en aristas (misma relacion + mismo doc no se duplica)
- Validacion post-parse del output LLM: entity types normalizados a enum (`PERSON|ORG|PLACE|CONCEPT|EVENT|OTHER`), nombres >= 2 chars, descriptions truncadas a 200 chars (DTm-16)
- Estimacion de memoria en `get_stats()` para observabilidad
- Robustez para modelos de razonamiento (nemotron-3-nano thinking mode): strip de `<think>` tags en `llm.py` (incluye tags sin cerrar por truncamiento), `max_tokens` ampliados en extraccion (2048 tripletas, 512 keywords) para compensar tokens consumidos por razonamiento, y fallback `json.JSONDecoder.raw_decode()` para extraer JSON de respuestas con texto mixto
- Trazas de depuracion (nivel DEBUG): log de chars eliminados por strip de `<think>` tags, y primeros 200 chars del raw response en fallos de parse JSON — permite diagnosticar problemas con NIM sin activar logging verboso en produccion

**Optimizacion:** `pre_extract_query_keywords()` permite pre-extraer keywords de todas las queries en batch antes del loop de retrieval, analogo al pre-embed de vectores.

## Pipeline de evaluacion

```
.env -> MTEBConfig -> MinIO/cache(Parquet) -> LoadedDataset
     -> shuffle(seed) -> slice(max_corpus)
     -> [NER + cross-linking (spaCy) si HYBRID_PLUS]
     -> [LLM triplet extraction + KG construction si LIGHT_RAG]
     -> index(ChromaDB [+ Tantivy si HYBRID_PLUS])
     -> pre-embed queries (batch REST NIM)
     -> [pre-extract query keywords (batch LLM) si LIGHT_RAG]
     -> retrieve(local ChromaDB [+ BM25 + RRF si HYBRID_PLUS] [+ Graph traversal + Fusion si LIGHT_RAG], sync)
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

Para `HYBRID_PLUS` (entity cross-linking):

```bash
pip install spacy
python -m spacy download en_core_web_sm
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
RETRIEVAL_STRATEGY=SIMPLE_VECTOR      # SIMPLE_VECTOR | HYBRID_PLUS | LIGHT_RAG
RETRIEVAL_K=20
RETRIEVAL_PRE_FUSION_K=150
RETRIEVAL_RRF_K=60
RRF_BM25_WEIGHT=0.3
RRF_VECTOR_WEIGHT=0.7

# Knowledge Graph (solo LIGHT_RAG)
KG_MAX_HOPS=2                         # Profundidad maxima BFS en graph traversal
KG_MAX_TEXT_CHARS=3000                 # Max chars de documento enviados al LLM para extraccion
KG_GRAPH_WEIGHT=0.3                   # Peso del score del grafo en fusion
KG_VECTOR_WEIGHT=0.7                  # Peso del score vectorial en fusion
KG_MAX_ENTITIES=0                     # Cap de entidades en KG (0 = default 50K)
KG_CACHE_DIR=./data/kg_cache          # Directorio para persistir KG entre runs (vacio = sin cache)
KG_FUSION_METHOD=rrf                  # rrf (default) o linear
KG_RRF_K=60                           # Constante k para RRF (default 60)

# Reranker (opcional)
RERANKER_ENABLED=false
RERANKER_BASE_URL=http://<nim-reranker-host>:9000/v1
RERANKER_MODEL_NAME=nvidia/llama-3.2-nv-rerankqa-1b-v2
RERANKER_TOP_N=5
RERANKER_FETCH_K=0                    # Candidatos para reranker (0 = top_n * 3)

# Dataset
MTEB_DATASET_NAME=hotpotqa
EVAL_MAX_QUERIES=0                    # 0 = todas
EVAL_MAX_CORPUS=0                     # 0 = todo
GENERATION_ENABLED=true
CORPUS_SHUFFLE_SEED=42

# DEV_MODE: subset con gold docs garantizados (metricas optimistas, solo comparacion relativa)
DEV_MODE=false
DEV_QUERIES=200
DEV_CORPUS_SIZE=4000

# Entity cross-linking (solo HYBRID_PLUS)
ENTITY_MAX_CROSS_REFS=3
ENTITY_MIN_SHARED=1
ENTITY_MAX_DOC_FRACTION=0.05

# Graph expansion cap (HYBRID_PLUS / LIGHT_RAG). 0 = sin limite.
MAX_GRAPH_EXPANSION=30

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

## Fases de implementacion
### Completado

| Fase | Objetivo | Resumen |
|---|---|---|
| **0. Reproducibilidad** | Config determinista entre runs | `requirements.lock` pinneado, `preflight.py` para validacion pre-run. **Pendiente:** baseline LIGHT_RAG end-to-end (requiere NIM). |
| **1. Fiabilidad** | No perder runs de horas | Checkpoint/resume cada 50 queries (`--resume RUN_ID`). Counter de entidades descartadas por cap KG. |
| **2. Calidad retrieval** | Mejorar Hit@5 y MRR | Apostrofes preservados en BM25, entity normalization alineada KG/linker, `MIN_ENTITY_NAME_LEN=1`. |
| **3. Eficiencia** | Corpus 66K sin OOM | Batch adaptativo (`semaphore*4`), dedup memoria HYBRID_PLUS. |
| **4. Mantenibilidad** | `evaluator.py` < 600 LOC | Descompuesto de 1225 a 592 LOC. Extraidos: `subset_selection.py`, `retrieval_executor.py`, `generation_executor.py`, `embedding_service.py`, `checkpoint.py`, `result_builder.py`. |

Issues resueltos: DTm-14 a DTm-38, DTm-45 (22 issues), DTm-46 (robustez thinking mode). Ver historial git para detalles de cada fix.

### En progreso: mejoras LIGHT_RAG

Ronda de mejoras identificadas por code review. 8 problemas agrupados en 5 fases:

| Fase | Objetivo | Problemas | Estado |
|---|---|---|---|
| **0. Bugs + limpieza** | Corregir errores de datos y eliminar codigo muerto | Fingerprint truncado en cache KG (P5), race conditions en TripletExtractor (P6), dead code: `get_subgraph_context`, `context_utilization` (P7) | Completado |
| **1. Calidad keyword search** | Stemming en indices de entidades/relaciones del KG | `_tokenize()` sin normalizacion morfologica — "mechanics" no matchea "mechanical" (P3) | Completado |
| **2.1. Batch extraction** | Reducir llamadas LLM 3-5x en construccion del KG | 1 doc/llamada → N docs/llamada con prompt multi-documento (P2) | Completado |
| **2.2. igraph** | Reemplazar NetworkX por igraph (C-backed, 10-100x BFS) | Grafo in-memory lento y memory-heavy con NetworkX (P1) | Completado |
| **3. RRF fusion** | Reemplazar fusion lineal por Reciprocal Rank Fusion | Reusar `reciprocal_rank_fusion()` ya implementada en HYBRID_PLUS (P4) | Completado |
| **4. Test coverage** | Tests para evaluator, loader y pipeline e2e mockeado | Sin cobertura de tests para orquestacion y carga de datos (P8) | Completado |

Cambios en dependencias:
- `networkx` → `python-igraph` (Fase 2.2)
- Nuevo: `snowballstemmer` (Fase 1)
- Fusion LIGHT_RAG: combinacion lineal → RRF (Fase 3, configurable via `KG_FUSION_METHOD`)

### Deuda tecnica resuelta (ronda de robustez LightRAG)

Issues resueltos: DTm-47 a DTm-54, DTm-59.

| ID | Fix |
|---|---|
| DTm-47 | Cache de query keywords con LRU eviction (OrderedDict, max 10K entries) |
| DTm-48 | List comprehension en vez de `[([],[])]*n` (elimina sharing de refs mutables) |
| DTm-49 | `_triplets_dropped_by_cap` contador + WARNING en log + expuesto en `get_stats()` |
| DTm-50 | `assert` reemplazados por `raise RuntimeError` (safe con `-O`) |
| DTm-51 | `threading.Lock` en cache de keywords (thread-safe read/write/evict) |
| DTm-52 | WARNING cuando graph-only docs no se resuelven en ChromaDB |
| DTm-53 | Fusion lineal con scores all-zero hace fallback automatico a RRF |
| DTm-54 | Fallback batch→single-doc logueado como WARNING (no DEBUG) |
| DTm-59 | `_stemmer.stemWords()` envuelto en try-except |

### Deuda tecnica abierta

#### Alta — degradan metricas de retrieval en produccion

| ID | Descripcion | Ubicacion | Evidencia |
|---|---|---|---|
| DTm-62 | **Fusion KG destruye ranking**: MRR cae 33pp (0.864→0.531), NDCG@10 cae 25pp con KG activo. Scores del grafo normalizados compiten con scores vectoriales, desplazando docs relevantes de posicion 1-2 a 3-10. El reranker compensa (+30pp delta) pero no deberia ser necesario. **Fix**: reducir `KG_GRAPH_WEIGHT` a 0.1-0.15 (tiebreaker, no canal primario), o no normalizar scores del grafo y calibrar peso empiricamente. | `lightrag_retriever.py:398-430` | Run 2 vs Run 1 |
| DTm-63 | **Entity cap 50K insuficiente**: 26382/76382 entidades descartadas (34.5%). El orden de indexacion (shuffle seed) determina que entidades entran, introduciendo sesgo arbitrario. Docs indexados tarde tienen KG incompleto. **Fix**: subir cap a 100K+ o implementar eviction por frecuencia de menciones en vez de FIFO. | `knowledge_graph.py:247-252` | Run 2 stats |

#### Media — afectan fiabilidad o diagnosticabilidad

| ID | Descripcion | Ubicacion |
|---|---|---|
| DTm-55 | Stats del extractor se corrompen si construccion del KG falla a mitad. | `lightrag_retriever.py:150-162` |
| DTm-56 | Colision de fingerprint con corpus vacio (edge case teorico). | `lightrag_retriever.py:229-247` |
| DTm-64 | **Normalizacion de scores incomparable entre canales**: fusion lineal normaliza vector y graph scores a [0,1] independientemente, pero las distribuciones son incomparables (graph: 40 candidatos uniformes vs vector: 20 candidatos concentrados en 0.95-0.99). Candidatos mediocres del grafo se inflan a valores competitivos. RRF (default) mitiga esto parcialmente pero no lo elimina. | `lightrag_retriever.py:398-430` |
| DTm-65 | **Thinking-mode exhaustion en keyword extraction**: ~17% de queries fallan en primer intento (`LLM returned empty content after stripping reasoning tags`). El retry con max_tokens duplicado funciona pero anade latencia. `KG_KEYWORD_MAX_TOKENS` ahora configurable (default 1024, era 512 hardcoded). Subir a 2048 si el modelo piensa mucho. | `triplet_extractor.py:608-612` |

#### Baja — mejoras menores o limitaciones aceptadas

| ID | Descripcion | Ubicacion |
|---|---|---|
| DTm-12 | Sesgo LLM-judge en faithfulness para respuestas cortas (score 0.0-0.2 con F1=1.0). Inherente al LLM-judge, no actionable. | `shared/metrics.py` |
| DTm-13 | No-determinismo HNSW: ChromaDB no expone `hnsw:random_seed`. Recall@K varia ±0.02 entre runs. | `shared/vector_store.py` |
| DTm-24 | Naming ambiguo: `RRF_VECTOR_WEIGHT` vs `KG_VECTOR_WEIGHT`. Semantica distinta, nombre similar. | `sandbox_mteb/config.py` |
| DTm-57 | Normalizacion de entidades agresiva: pierde apostrofes, guiones. Puede causar colisiones raras. | `knowledge_graph.py:133-144` |
| DTm-58 | No dedup de queries identicas en batch keyword extraction — LLM calls duplicadas. | `triplet_extractor.py:624-668` |
| DTm-60 | Stats del extractor acumulan entre llamadas; `reset_stats()` existe pero nunca se llama automaticamente. | `triplet_extractor.py:127-146` |
| DTm-61 | No validacion de tamano de keywords del LLM — respuestas patologicas pasan sin limite. | `triplet_extractor.py:565-592` |

### Resultados comparativos LIGHT_RAG

#### Run 1 (KG vacio) vs Run 2 (KG funcional) — mismas 125 queries, mismo seed

| Metrica | Run 1 (KG vacio) | Run 2 (KG activo) | Delta |
|---|---|---|---|
| Corpus | 33000 | 16500 | -50% |
| KG entidades | 1208 (1.1%) | 50000 (70%) | +4040% |
| KG relaciones | 899 | 51841 | +5666% |
| **Hit@5** | **0.896** | **0.848** | **-4.8pp** |
| **MRR** | **0.864** | **0.531** | **-33.3pp** |
| Recall@5 | 0.816 | 0.660 | -15.6pp |
| NDCG@10 | 0.806 | 0.557 | -24.9pp |
| Gen Recall | 0.940 | 0.964 | +2.4pp |
| F1 | 0.756 | 0.784 | +2.8pp |
| EM | 0.568 | 0.624 | +5.6pp |

**Conclusion**: el KG activo degrada todas las metricas de ranking pero mejora generacion. El reranker compensa el desorden del KG (delta reranker: +12.4pp → +30.4pp). F1/EM mejoran porque Gen Recall sube ligeramente y el corpus mas pequeno reduce distractores.

### Acciones pendientes (prioridad por impacto en metricas)

1. **[CRITICO] Calibrar KG_GRAPH_WEIGHT**: probar 0.10, 0.15, 0.20 con mismo corpus/queries. Reducir el peso del grafo de 0.3 a tiebreaker level. Configurable via `KG_GRAPH_WEIGHT` en .env.
2. **[CRITICO] Baseline SIMPLE_VECTOR con corpus 16500**: aislar efecto neto del KG vs efecto de reduccion de corpus. Sin este dato no se puede validar si LIGHT_RAG aporta valor.
3. **[ALTO] Subir KG_MAX_ENTITIES a 100K+**: 34.5% de entidades descartadas es inaceptable. Evaluar impacto en memoria (~42MB estimado para 100K).
4. **[MEDIO] Evaluar fusion sin normalizacion**: usar scores raw del grafo con peso calibrado, en vez de normalizar a [0,1]. Alternativa: solo RRF (ya es default, pero linear sigue disponible).
5. **[BAJO] Subir KG_KEYWORD_MAX_TOKENS a 2048** si nemotron sigue con thinking-mode exhaustion post-fix de 512→1024.
