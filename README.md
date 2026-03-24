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
│       ├── knowledge_graph.py       # KG in-memory (NetworkX): entidades, relaciones, traversal
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
2. Construye un `KnowledgeGraph` in-memory (NetworkX) con entidades, relaciones e indices invertidos
3. Indexa contenido original en ChromaDB para vector search

**Retrieval:**
1. Vector search (ChromaDB) → top_k candidatos con cosine similarity
2. Query analysis via LLM → extrae keywords de bajo nivel (entidades especificas) y alto nivel (temas abstractos)
3. Graph traversal dual-level:
   - **Low-level**: BFS desde entidades de la query, scoring inversamente proporcional a hops (`1/(1+depth)`)
   - **High-level**: token matching en nombres de entidad y descripciones de relaciones (indice invertido por token, DTm-30)
4. Fusion: `vector_weight * vector_score + graph_weight * graph_score` (scores normalizados a [0,1])

**Fallback:** Sin `networkx` o sin LLM service → degrada automaticamente a SimpleVectorRetriever puro (warning en log).

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
pip install networkx
```

> **Nota:** LIGHT_RAG requiere un LLM service activo (NIM) tanto para indexacion (extraccion de tripletas) como para retrieval (analisis de queries). Sin `networkx` o sin LLM, degrada a vector search puro.

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

### Deuda tecnica abierta

| ID | Descripcion | Prioridad |
|---|---|---|
| DTm-12 | Sesgo LLM-judge en faithfulness para respuestas cortas (score 0.0-0.2 con F1=1.0). Faithfulness es informativa, no primaria. Inherente al LLM-judge. | Baja |
| DTm-13 | No-determinismo HNSW: ChromaDB no expone `hnsw:random_seed`. Recall@K varia ±0.02 entre runs. Aceptable para comparaciones relativas. | Baja |
| DTm-24 | Naming ambiguo: `RRF_VECTOR_WEIGHT` (peso vector en RRF) vs `KG_VECTOR_WEIGHT` (peso vector en fusion graph). Semantica distinta, nombre similar. | Baja |

### Pendiente de infraestructura

- Run baseline LIGHT_RAG end-to-end (requiere entorno con NIM activo)
- Benchmarks comparativos entre estrategias (Hit@K, F1) — sin datos no se puede validar el valor de HYBRID_PLUS/LIGHT_RAG sobre SIMPLE_VECTOR
