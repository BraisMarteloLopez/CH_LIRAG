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
│   ├── evaluator.py                 # Pipeline: pre-embed + retrieval + gen async
│   ├── run.py                       # Entry point (--dry-run, -v)
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
```

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
RETRIEVAL_BM25_WEIGHT=0.3
RETRIEVAL_VECTOR_WEIGHT=0.7

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

## Deuda tecnica abierta

| ID | Descripcion | Prioridad | Estado |
|---|---|---|---|
| DTm-12 | Sesgo LLM-judge en faithfulness para respuestas cortas (score 0.0-0.2 con F1=1.0). F1 es primaria; faithfulness solo informativa. | Baja | Abierto |
| DTm-13 | No-determinismo HNSW: ChromaDB no soporta `hnsw:random_seed`. Recall@K varia +/-0.02 entre runs. | Baja | Abierto |
| DTm-14 | Duplicacion contenido en memoria: `retrieved_contents` + `generation_contents` (~1.5GB con 7K queries). | Baja | Abierto |
| DTm-15 | ETL HotpotQA no asigna `answer_type="label"` a queries comparison (yes/no). Sin impacto numerico (F1=Accuracy para tokens unicos). | Baja | Abierto |
| DTm-16 | Validacion output LLM en triplet extraction: entity types normalizados a enum, nombres >= 2 chars, descriptions truncadas a 200 chars. | Media | **Resuelto** |
| DTm-18 | Entity normalization basica: no resuelve aliases (US/United States) ni formas parciales. Aplica a HYBRID_PLUS y LIGHT_RAG. | Baja | Abierto |
| DTm-20 | `question_type` en detail CSV requiere propagacion manual. Considerar metadata passthrough generico. | Baja | Abierto |
| DTm-21 | KG cap de memoria: `max_entities` configurable (default 50K), dedup de relaciones, estimacion de memoria en `get_stats()`. | Media | **Resuelto** |
| DTm-22 | Batching de coroutines en `extract_batch_async()`: chunks de 500 docs para limitar presion de memoria. | Baja | **Resuelto** |
| DTm-23 | Tests LIGHT_RAG: 63 tests unitarios cubriendo `KnowledgeGraph`, `TripletExtractor`, `_fuse_with_graph`, validacion y hardening. | Alta | **Resuelto** |
| DTm-24 | Naming ambiguo: `RETRIEVAL_VECTOR_WEIGHT` (peso vector en RRF/HYBRID_PLUS) vs `KG_VECTOR_WEIGHT` (peso vector en fusion graph/LIGHT_RAG). Semantica distinta, nombre similar. | Baja | Abierto |
| DTm-25 | Batch size de extraccion (500) sobredimensionado vs semaforo HTTP (32). Con 500 coroutines y semaforo de 32, 468 esperan en memoria. Batch de 64-128 (2-4x semaforo) seria mas eficiente. | Baja | Abierto |
| DTm-26 | `kg_max_entities` descarta entidades nuevas silenciosamente al llegar al cap. Las ultimas queries del corpus tendran KG incompleto. Considerar politica LRU o al menos counter de entidades descartadas para visibilidad. | Media | Abierto |
| DTm-27 | Filtro `len(name) < 2` en validacion de entidades rechaza entidades legitimas de 1 caracter (nombres chinos, siglas). Filtrar solo `name.strip() == ""`. | Baja | Abierto |
| DTm-28 | Sin dependencias pinneadas. `requirements.txt` sin versiones exactas. Un update de `networkx` o `chromadb` puede cambiar resultados silenciosamente entre runs. Necesita `pip freeze` versionado. | Media | Abierto |
| DTm-29 | BFS en `query_entities()` usa `collections.deque.popleft()` — O(1) por operacion en lugar de O(n) con `list.pop(0)`. | Media | **Resuelto** |
| DTm-30 | `query_by_keywords()` (`knowledge_graph.py`) sin indice: recorre toda `self._entities` y todas las aristas en cada llamada — O(entidades × keywords) por query. Resuelto con indice invertido por token. **Nota**: cambio semantico de substring matching a token (word-level) matching — puede afectar recall en nombres compuestos como "new york". | Media | **Resuelto** |
| DTm-31 | Corpus duplicado en memoria en HYBRID_PLUS: `HybridPlusRetriever._original_contents` y `HybridRetriever._doc_map` mantienen copias independientes del contenido completo. Analogamente a DTm-14 pero en otra capa. | Baja | Abierto |
| DTm-32 | `random.seed()` global en `evaluator.py:run()`: muta estado global del modulo `random`. DEV_MODE usa `random.Random(seed)` (instancia aislada) correctamente, pero el flujo estandar no. Contamina RNG de librerias de terceros. | Baja | Abierto |
| DTm-33 | Fallos silenciosos en extraccion de tripletas: `extract_from_doc_async()` (`triplet_extractor.py`) devuelve `([], [])` en excepcion con solo un `logger.warning`. `get_stats()` no reporta cuantos documentos fallaron. Fraccion del corpus puede quedar sin representacion en KG sin visibilidad. | Media | **Resuelto** |
| DTm-34 | Persistencia del Knowledge Graph entre runs via `KG_CACHE_DIR`. Serializa/deserializa el grafo completo (entidades, relaciones, indices, NetworkX) como JSON. Cache invalidado automaticamente por fingerprint del corpus (incluye `KG_MAX_TEXT_CHARS`). `kg_cache_dir` registrado en `config_snapshot`. Log de tamano del fichero al guardar con warning si > 100 MB. | Alta | **Resuelto** |
| DTm-35 | `pre_fusion_k` reutilizado con semantica incorrecta para el reranker en `evaluator.py:_execute_retrieval()`. El parametro define candidatos pre-RRF por canal, pero se usa como total de candidatos para el reranker — semantica distinta. Relacionado con DTm-24 (naming ambiguo) pero con impacto funcional directo. | Media | **Resuelto** |
| DTm-36 | `evaluator.py` es un God Object emergente (1268 LOC). Orquesta carga de datos, subset selection, indexado, pre-embedding, retrieval, reranking, generacion, metricas, agregacion y logging. Sin checkpoint/resume — un fallo mid-run en LIGHT_RAG (miles de llamadas LLM) pierde toda la ejecucion. Candidato a extraer subset selection, metric aggregation y result building a modulos separados. | Media | Abierto |
| DTm-37 | Query sanitization de Tantivy demasiado agresiva: `re.sub(r'[^\w\s]', ' ', query)` (`tantivy_index.py:183`) elimina todo caracter no alfanumerico, destruyendo apostrofes contractivos ("don't" → "don t", "it's" → "it s"). Afecta recall BM25 en queries con contracciones inglesas. Considerar preservar apostrofes intra-palabra: `re.sub(r"(?<!\w)[^\w\s]|[^\w\s'](?!\w)", ' ', query)` o similar. | Baja | Abierto |
