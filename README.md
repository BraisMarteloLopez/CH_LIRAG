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
│       ├── core.py                  # BaseRetriever, SimpleVectorRetriever, RetrievalConfig, RRF
│       ├── reranker.py              # CrossEncoderReranker (NVIDIARerank)
│       ├── hybrid/                  # Estrategia HYBRID_PLUS
│       │   ├── retriever.py         # BM25 + Vector + RRF (HybridRetriever)
│       │   ├── plus_retriever.py    # BM25+Vector+RRF + NER cross-linking
│       │   ├── entity_linker.py     # NER (spaCy) + indice invertido + cross-refs
│       │   └── tantivy_index.py     # BM25 via Tantivy (Rust, fallback rank-bm25)
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
KG_GRAPH_WEIGHT=0.3                   # Peso del score del grafo en fusion (DTm-62: probar 0.10-0.15)
KG_VECTOR_WEIGHT=0.7                  # Peso del score vectorial en fusion
KG_MAX_ENTITIES=0                     # Cap de entidades en KG (0 = default 50K; DTm-63: probar 100K)
KG_CACHE_DIR=./data/kg_cache          # Directorio para persistir KG entre runs (vacio = sin cache)
KG_FUSION_METHOD=rrf                  # rrf (default, robusto) o linear (legacy)
KG_RRF_K=60                           # Constante k para RRF (default 60)
KG_KEYWORD_MAX_TOKENS=1024            # Max tokens para keyword extraction LLM call
KG_GRAPH_OVERFETCH_FACTOR=2           # Graph traversal pide N * top_k candidatos

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

## Historial de fases completadas

<details>
<summary>Fases 0-4: fundacion del sistema (completadas)</summary>

| Fase | Objetivo | Resumen |
|---|---|---|
| **0. Reproducibilidad** | Config determinista entre runs | `requirements.lock` pinneado, `preflight.py` para validacion pre-run. |
| **1. Fiabilidad** | No perder runs de horas | Checkpoint/resume cada 50 queries (`--resume RUN_ID`). Counter de entidades descartadas por cap KG. |
| **2. Calidad retrieval** | Mejorar Hit@5 y MRR | Apostrofes preservados en BM25, entity normalization alineada KG/linker, `MIN_ENTITY_NAME_LEN=1`. |
| **3. Eficiencia** | Corpus 66K sin OOM | Batch adaptativo (`semaphore*4`), dedup memoria HYBRID_PLUS. |
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

## Resultados comparativos LIGHT_RAG

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
| DAM-1 | **Critica** | **Sin entity VDB**: el original indexa nombre+descripcion de cada entidad como embeddings en un vector DB (`entities_vdb`) y usa similarity search para resolver entidades en queries. CH_LIRAG usa matching lexico (exact + token overlap con stemming). Esto causa fallo sistematico cuando query y entidad difieren lexicamente ("Obama" vs "Barack Obama", "44th president" vs "Obama"). | Causa raiz de DTm-70, DTm-73. Explica fragmentacion (473 componentes/500 docs) y fallo de bridging. | Abierto |
| DAM-2 | **Critica** | **Sin relationship VDB**: el original indexa keywords+descripcion de cada relacion como embeddings en un segundo vector DB (`relationships_vdb`). El high-level retrieval busca relaciones por semantic similarity. CH_LIRAG usa token matching contra un indice invertido — mucho menos flexible. | Causa raiz de DTm-71, DTm-74. El high-level path esta castrado sin semantic search. | Abierto |
| DAM-3 | **Alta** | **Grafo como suplemento vs retriever primario**: en el original, los modos local/global/hybrid usan el grafo como mecanismo principal de retrieval (las entidades/relaciones apuntan a source chunks). En CH_LIRAG, vector search es siempre primario y el grafo intenta sumar via RRF — invirtiendo la arquitectura. | Causa raiz de DTm-62. El grafo inyecta candidatos ruidosos que degradan el ranking vectorial en vez de reemplazarlo con un ranking mejor. | Abierto |
| DAM-4 | **Alta** | **Sin merging de descripciones via LLM**: el original implementa `_merge_nodes_then_upsert()` — cuando la misma entidad aparece en N docs, un LLM sintetiza una descripcion unificada (map-reduce). CH_LIRAG sobrescribe con la ultima descripcion vista. | Entidades tienen descripciones pobres (parciales, de un solo doc). Embedding de entity VDB (si se implementa DAM-1) seria menos efectivo sin descripciones ricas. | Abierto |
| DAM-5 | **Media** | **Sin acumulacion de edge weights**: el original suma pesos en relaciones duplicadas (15 docs mencionan "Obama → president_of → USA" = weight 15) y usa weight como senal de importancia para scoring. CH_LIRAG deduplica pero no acumula ni usa weights. | Scoring no distingue relaciones frecuentes (importantes) de raras (ruido). Contribuye a DTm-72. | Abierto |
| DAM-6 | **Media** | **Sin gleaning (re-extraccion)**: el original soporta N pasadas de extraccion con prompt de continuacion para capturar entidades perdidas en la primera pasada. CH_LIRAG hace una sola pasada. | Entidades menos frecuentes o mencionadas indirectamente se pierden. Contribuye a fragmentacion (DTm-73). | Abierto |
| DAM-7 | **Media** | **BFS profundo (2 hops) vs 1-hop con matching semantico**: el original usa solo 1-hop pero compensa con entity/relationship VDBs. CH_LIRAG usa BFS hasta 2 hops con matching lexico — mas profundidad con peor precision introduce mas ruido. | Contribuye a DTm-62 (inundacion de candidatos irrelevantes). Contraintuitivamente, reducir a 1-hop podria mejorar resultados si se implementa DAM-1. | Abierto |
| DAM-8 | **Baja** | **Contexto raw vs estructurado para generacion**: el original pasa tablas CSV de entidades, relaciones y source chunks al LLM. CH_LIRAG pasa texto plano concatenado de los docs recuperados. | Menor impacto en retrieval pero afecta calidad de generacion. El LLM recibe menos estructura para razonar. | Abierto |

**Cadena causal:**

```
DAM-1 (sin entity VDB) + DAM-2 (sin relationship VDB)
    → Matching lexico falla en variaciones (DTm-70)
    → Grafo fragmentado en islas (DTm-73, 473 componentes)
    → BFS no puede cruzar islas
    → Graph candidates = ruido

DAM-3 (grafo como suplemento)
    → Ruido del grafo se fusiona con vector search via RRF
    → Candidatos irrelevantes desplazan gold docs en ranking
    → MRR baja de ~0.86 a 0.52 (DTm-62)
    → Reranker compensa (+29pp recall) pero enmascara el problema

DAM-4 (sin merging) + DAM-5 (sin edge weights) + DAM-6 (sin gleaning)
    → Entidades con descripciones pobres, relaciones sin pesos, extraccion incompleta
    → Grafo de menor calidad incluso si se resuelve DAM-1/DAM-2
```

**Priorizacion:** DAM-1 y DAM-2 son los cambios de mayor impacto. Requieren 2 colecciones adicionales
en ChromaDB (o vector store equivalente) para entidades y relaciones. El resto son mejoras incrementales
que aportan valor solo si la base semantica (VDBs) esta resuelta.

### Registro completo

| ID | Severidad | Descripcion | Ubicacion | Estado |
|---|---|---|---|---|
| DTm-66 | **Alta** | **`max_tokens=8192` en extraccion batch causa generacion masiva en thinking mode**: nemotron-3-nano genera ~6000 tokens de `<think>` + ~2000 de JSON por call. A ~30 tok/s, cada call tarda ~267s. Con 3,300 calls y 32 concurrencia = ~103 rondas × 267s = **7h 38min** (98% del tiempo de KG build). Fix: reducir a 4096 (configurable via `KG_EXTRACTION_MAX_TOKENS`). | `lightrag/triplet_extractor.py` | Abierto |
| DTm-67 | **Alta** | **Batch de 5 docs/call es conservador**: con 3000 chars/doc (~750 tokens) y context window 8K+, caben 10 docs/call. Reducir calls de 3,300 a 1,650 (50% menos). Hacer configurable via `KG_BATCH_DOCS_PER_CALL`. | `lightrag/triplet_extractor.py` | Abierto |
| DTm-62 | **Alta** | Fusion KG destruye ranking: MRR -33pp con KG activo. Scores normalizados del grafo compiten con vectoriales, desplazando docs relevantes. | `lightrag/retriever.py` fusion | Abierto |
| DTm-63 | **Alta** | Entity cap 50K insuficiente: 34.5% entidades descartadas. Orden de indexacion introduce sesgo arbitrario por FIFO. | `lightrag/knowledge_graph.py` | Abierto |
| DTm-68 | **Media** | **Re-serializacion JSON innecesaria en batch parse**: `_parse_batch_extraction_json` hace `json.dumps(entry)` para pasarlo a `_parse_extraction_json(raw_str)`. Se puede refactorizar para aceptar dict directamente, eliminando serialize+deserialize por doc. | `lightrag/triplet_extractor.py` | Abierto |
| DTm-69 | **Media** | **Token indexing secuencial durante graph build**: `_index_entity_tokens()` y `_index_relation_tokens()` ejecutan stemming por cada tripleta durante `add_triplets()`. Con 80K+ iteraciones es CPU-bound. Se podria diferir a una fase post-build (`build_keyword_indices()`) para separar I/O-bound (LLM) de CPU-bound (stemming). | `lightrag/knowledge_graph.py` | Abierto |
| DTm-64 | **Media** | Normalizacion [0,1] incomparable entre canales: distribuciones vector (concentrada) vs graph (uniforme) generan scores engañosos. RRF mitiga parcialmente. | `lightrag/retriever.py` | Abierto |
| DTm-65 | **Media** | Thinking-mode exhaustion: ~17% queries fallan en 1er intento. `KG_KEYWORD_MAX_TOKENS` configurable (default 1024). Puede requerir 2048 con modelos reasoning-heavy. | `lightrag/triplet_extractor.py` | Mitigado |
| DTm-55 | **Media** | Stats extractor se corrompen si KG build falla a mitad. `_has_graph=False` pero stats parciales persisten. | `lightrag/retriever.py` | Abierto |
| DTm-56 | **Media** | Fingerprint collision con corpus vacio: `sha256("")[:16]` es determinista pero edge case si dos configs distintas producen mismo hash. | `lightrag/retriever.py` | Abierto |
| DTm-12 | Baja | Sesgo LLM-judge en faithfulness para respuestas cortas. Inherente al LLM-judge. | `shared/metrics.py` | Aceptado |
| DTm-13 | Baja | No-determinismo HNSW: ChromaDB no expone `hnsw:random_seed`. ±0.02 entre runs. | `shared/vector_store.py` | Aceptado |
| DTm-24 | Baja | Naming ambiguo: `RRF_VECTOR_WEIGHT` vs `KG_VECTOR_WEIGHT`. | `sandbox_mteb/config.py` | Aceptado |
| DTm-57 | Baja | Normalizacion entidades agresiva: pierde apostrofes/guiones. Colisiones raras. | `lightrag/knowledge_graph.py` | Abierto |
| DTm-58 | Baja | No dedup queries identicas en batch keyword extraction: LLM calls duplicadas. | `lightrag/triplet_extractor.py` | Abierto |
| DTm-60 | Baja | Stats extractor acumulan entre llamadas. `reset_stats()` nunca se llama auto. | `lightrag/triplet_extractor.py` | Abierto |
| DTm-61 | Baja | No validacion tamano keywords del LLM: respuestas patologicas pasan sin limite. | `lightrag/triplet_extractor.py` | Abierto |
| DTm-70 | **Alta** | **Entity matching exacto en `query_entities` impide bridging**: BFS solo arranca si el keyword del LLM matchea exactamente el nombre normalizado de la entidad. Si el LLM extrae "Vlatko" pero la entidad es "vlatko gilić", el BFS no arranca y el bridging se pierde. Fix: fuzzy matching via token overlap usando `_kw_entity_index` existente como fallback cuando el exact match falla. | `lightrag/knowledge_graph.py` | Abierto |
| DTm-71 | **Alta** | **`query_by_keywords` no hace graph traversal**: encuentra entidades por keyword pero solo devuelve sus docs directos (`source_doc_ids`), sin recorrer el grafo via BFS. El bridging multi-hop solo funciona en `query_entities` (low-level), no en high-level. Fix: BFS limitado (1 hop) desde entidades matcheadas por keywords. | `lightrag/knowledge_graph.py` | Abierto |
| DTm-72 | **Media** | **BFS scoring ciego a la relacion**: `hop_score = 1/(1+depth)` trata todas las aristas igual. Una relacion `directed_by` pesa igual que `same_year_as`. Para queries como "nationality of the director", la relacion `directed_by` deberia puntuar mas. Fix: ponderar hop score por token overlap entre relation type y keywords de la query. | `lightrag/knowledge_graph.py` | Abierto |
| DTm-73 | **Alta** | **Grafo fragmentado (2831 componentes) limita bridging**: si los 2 gold docs de una query multi-hop generan entidades en componentes distintos (e.g. "Vlatko Gilić" vs "Gilić"), el BFS no puede cruzar. Fix: entity co-reference por token overlap de apellido, o aristas implicitas entre entidades co-ocurrentes en el mismo doc sin relacion explicita. | `lightrag/knowledge_graph.py` | Abierto |
| DTm-74 | Baja | **Scoring flat en `query_by_keywords`**: entidad match siempre puntua 1.0, relacion match siempre 0.5, sin distinguir proporcion de tokens matcheados. Una entidad que matchea 3/3 keywords deberia puntuar mas que 1/3. Fix: scoring proporcional a token overlap (TF-like). | `lightrag/knowledge_graph.py` | Abierto |

### Deuda resuelta (referencia)

DTm-14 a DTm-38, DTm-45 a DTm-54, DTm-59 (31 issues). Ver historial git.

## Plan de desarrollo por fases

> **Nota (2026-04-01):** Este plan reemplaza al anterior (Fases 5-8). El analisis
> comparativo con [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) revelo
> divergencias arquitectonicas criticas (DAM-1 a DAM-8) que invalidan el enfoque
> previo de calibrar pesos sin resolver la base semantica. El nuevo plan prioriza
> la alineacion arquitectonica con el original antes de optimizaciones incrementales.

### Fase A: Documentacion y baselines (sin cambios de codigo)

**Objetivo:** Dejar la documentacion alineada con la realidad y obtener datos de referencia.

**Prerrequisito:** Ninguno. Ejecutable inmediatamente.

| Tarea | Descripcion | Esfuerzo |
|---|---|---|
| A.1 Actualizar seccion LIGHT_RAG en README | Anadir nota de limitaciones conocidas con referencia a DAM-1/DAM-2. Advertir que la implementacion actual diverge del original en entity/relationship matching. | Bajo |
| A.2 Actualizar descripcion retrieval dual-level | Documentar que low-level falla por matching exacto (DTm-70) y high-level no hace traversal (DTm-71). | Bajo |
| A.3 Baseline SIMPLE_VECTOR | Run con `RETRIEVAL_STRATEGY=SIMPLE_VECTOR`, mismo corpus/queries que runs LIGHT_RAG, para cuantificar la degradacion neta del KG. | Bajo (1 run) |
| A.4 Tabla comparativa | Documentar runs: SIMPLE_VECTOR vs LIGHT_RAG con mismo corpus. Delta MRR y Hit@5 = coste actual del KG. | Bajo |

**Criterio de exito:** README refleja el estado real. Tabla con delta MRR cuantificado.

### Fase B: Rendimiento KG build (independiente)

**Objetivo:** Reducir tiempo de KG build para que las fases siguientes sean viables (iteraciones rapidas).

**Prerrequisito:** Ninguno. Ejecutable en paralelo con Fase A.

| Tarea | Descripcion | DTm | Impacto estimado |
|---|---|---|---|
| B.1 `KG_EXTRACTION_MAX_TOKENS` configurable | Reducir de 8192 a 4096. El JSON raramente excede 2000 tokens; el modelo generara menos thinking. | DTm-66 | ~2x mas rapido |
| B.2 `KG_BATCH_DOCS_PER_CALL` configurable | Subir de 5 a 10 docs/call. Reduce LLM calls ~50%. Validar robustez del parse. | DTm-67 | ~1.5x adicional |
| B.3 Eliminar re-serializacion JSON | Refactorizar `_parse_batch_extraction_json` para pasar dict directamente. | DTm-68 | Menor |
| B.4 Diferir token indexing | Separar `_index_entity_tokens()` a fase post-build (`build_keyword_indices()`). | DTm-69 | ~5-10 min menos |
| B.5 Activar `KG_CACHE_DIR` | Runs subsiguientes con mismo corpus cargan KG en milisegundos. | — | Runs 2+ instantaneos |

**Criterio de exito:** KG build < 2h para 16.5K docs. Con cache, < 5s.

### Fase C: Entity VDB — alineacion arquitectonica critica

**Objetivo:** Resolver DAM-1 (sin entity VDB) — la divergencia de mayor impacto con el original.

**Prerrequisito:** Fase B completada (iteraciones rapidas). Fase A.3 completada (baseline para comparar).

**Contexto:** El original indexa nombre+descripcion de cada entidad como embeddings en un vector
DB y usa similarity search para encontrar entidades relevantes a la query. CH_LIRAG usa string
matching. Este cambio es la causa raiz de DTm-70, DTm-73, y la degradacion de MRR.

| Tarea | Descripcion | DAM/DTm | Esfuerzo |
|---|---|---|---|
| C.1 Crear `entities_vdb` | Nueva coleccion ChromaDB que indexa `entity_name + ": " + description` como embedding por entidad. Poblar durante `add_triplets()` o como fase post-build. | DAM-1 | Medio |
| C.2 Reemplazar `_resolve_entity_names` | En `query_entities()`, resolver entidades via similarity search contra `entities_vdb` en vez de exact+fuzzy string matching. Top-k entidades por cosine similarity. | DAM-1, DTm-70 | Medio |
| C.3 Reducir BFS a 1-hop | Con matching semantico, 1-hop deberia bastar (como en el original). 2 hops con matching debil introduce ruido; 1 hop con matching fuerte lo reduce. Hacer configurable. | DAM-7 | Bajo |
| C.4 Tests unitarios entity VDB | Tests para: indexacion de entidades, similarity search, integracion con `query_entities()`. Mocks de ChromaDB. | — | Medio |
| C.5 Run comparativo | LIGHT_RAG con entity VDB vs baseline SIMPLE_VECTOR. Medir delta MRR y fragmentacion (componentes del grafo ya no determinan bridging). | — | 1 run |

**Criterio de exito:** MRR de LIGHT_RAG >= MRR de SIMPLE_VECTOR (o degradacion < 5pp).
Fragmentacion del grafo deja de ser bloqueante porque entity matching es semantico.

**Riesgo:** Si MRR no mejora significativamente, el grafo no aporta valor neto incluso
con matching semantico. En ese caso, evaluar DAM-3 (grafo como primary retriever) antes
de continuar.

### Fase D: Relationship VDB + high-level retrieval

**Objetivo:** Resolver DAM-2 (sin relationship VDB) — completar el dual-level retrieval.

**Prerrequisito:** Fase C completada y con resultados positivos.

| Tarea | Descripcion | DAM/DTm | Esfuerzo |
|---|---|---|---|
| D.1 Crear `relationships_vdb` | Nueva coleccion ChromaDB que indexa `keywords + ": " + description` como embedding por relacion. | DAM-2 | Medio |
| D.2 Reemplazar `query_by_keywords` | High-level retrieval via similarity search contra `relationships_vdb` en vez de token matching. Trazar de relaciones a source docs via `source_doc_id`. | DAM-2, DTm-71, DTm-74 | Medio |
| D.3 Acumular edge weights | Sumar weights en relaciones duplicadas cross-doc. Usar weight como factor de scoring (relaciones frecuentes = mas importantes). | DAM-5, DTm-72 | Bajo |
| D.4 Tests unitarios relationship VDB | Tests para: indexacion de relaciones, similarity search, integracion con high-level path. | — | Medio |
| D.5 Run comparativo | LIGHT_RAG con ambos VDBs vs Fase C. Medir delta MRR del high-level path. | — | 1 run |

**Criterio de exito:** High-level retrieval aporta valor medible (MRR o Recall mejoran vs solo entity VDB).

### Fase E: Calidad del grafo

**Objetivo:** Resolver DAM-4 y DAM-6 — mejorar la calidad de entidades y la cobertura de extraccion.

**Prerrequisito:** Fase D completada. Solo proceder si los VDBs muestran que el grafo aporta valor.

| Tarea | Descripcion | DAM/DTm | Esfuerzo |
|---|---|---|---|
| E.1 Merging de descripciones | Cuando la misma entidad aparece en N docs, concatenar descripciones unicas (dedup por contenido). Si excede umbral, sintetizar via LLM (map-reduce). | DAM-4 | Medio-Alto |
| E.2 Gleaning (re-extraccion) | Tras la primera pasada de extraccion, re-enviar al LLM con prompt de continuacion para capturar entidades perdidas. Configurable via `KG_GLEANING_ROUNDS` (default 0 = desactivado). | DAM-6 | Medio |
| E.3 Escalado entity cap | Subir `KG_MAX_ENTITIES` a 100K+. Con VDBs activos, el cap es menos critico pero sigue siendo guardrail de memoria. | DTm-63 | Bajo |
| E.4 Run comparativo | Medir impacto de descripciones ricas + gleaning en MRR y Recall. | — | 1 run |

**Criterio de exito:** Fragmentacion del grafo (componentes conectados) se reduce >50%.
Entidades con descripciones multi-doc mejoran similarity search en entity VDB.

### Fase F: Rol del grafo en la pipeline (opcional)

**Objetivo:** Evaluar DAM-3 — si el grafo deberia ser primary retriever en vez de suplemento.

**Prerrequisito:** Fases C-E completadas. Solo tiene sentido si el grafo ya aporta valor neto.

| Tarea | Descripcion | DAM | Esfuerzo |
|---|---|---|---|
| F.1 Modo `graph_primary` | Nuevo modo donde el grafo traza source chunks directamente (como el original). Vector search como fallback para entidades sin match en el grafo. | DAM-3 | Alto |
| F.2 Contexto estructurado | Pasar tablas de entidades, relaciones y source chunks al LLM para generacion (como el original). | DAM-8 | Medio |
| F.3 Calibracion de fusion | Sweep de pesos solo si se mantiene el modo hibrido (grafo + vector). | DTm-62, DTm-64 | Bajo (runs) |

**Criterio de exito:** Decidir si el modo hibrido (actual) o graph-primary (original) es mejor para HotpotQA.

### Fase G: Deuda tecnica menor (en paralelo)

**Objetivo:** Limpiar issues medios y bajos. Ejecutable en cualquier momento.

| Tarea | Descripcion | DTm | Esfuerzo |
|---|---|---|---|
| G.1 Stats extractor resilientes | Snapshot de stats antes de KG build, restaurar si falla. | DTm-55 | Bajo |
| G.2 Fingerprint robusto | Incluir `len(documents)` y config hash. `None` si corpus vacio. | DTm-56 | Bajo |
| G.3 `KG_KEYWORD_MAX_TOKENS=2048` | Validar si elimina retries de thinking-mode exhaustion. | DTm-65 | Trivial |
| G.4 Dedup queries en batch keywords | Filtrar duplicadas antes de enviar al LLM. | DTm-58 | Bajo |
| G.5 Reset stats automatico | `reset_stats()` al inicio de `extract_batch()`. | DTm-60 | Trivial |
| G.6 Keyword size cap | Limitar a 20 keywords/nivel. Truncar exceso. | DTm-61 | Trivial |
| G.7 Entity normalization | Preservar guiones internos en `_normalize_name`. | DTm-57 | Bajo |

### Resumen de dependencias entre fases

```
Fase A (docs + baseline) ──── Fase B (rendimiento KG build) [paralelo]
  │                                │
  │ Baseline cuantifica            │ Iteraciones rapidas
  │ degradacion actual             │
  │                                │
  └─────── Fase C (entity VDB) ◄──┘
                │
                ├── MRR no mejora → evaluar DAM-3 (Fase F) o STOP
                │
                └── MRR mejora
                      │
                      └── Fase D (relationship VDB)
                            │
                            ├── High-level no aporta → OK, entity VDB basta
                            │
                            └── High-level aporta
                                  │
                                  └── Fase E (calidad grafo)
                                        │
                                        └── Fase F (graph primary, opcional)

Fase G (deuda menor) ─── en paralelo, sin bloqueo ───────────────────────
```

**Decision gates:**
- Post-Fase A: Si SIMPLE_VECTOR ya supera LIGHT_RAG por >20pp MRR, el KG no tiene potencial → STOP.
- Post-Fase C: Si entity VDB no cierra el gap → evaluar cambio de rol (DAM-3) antes de seguir.
- Post-Fase D: Si relationship VDB no aporta valor medible → Fase E sigue siendo util (calidad entidades) pero Fase F pierde prioridad.

### Variables de configuracion nuevas (referencia)

Anadidas en esta ronda, disponibles via .env:

| Variable | Default | Descripcion |
|---|---|---|
| `KG_KEYWORD_MAX_TOKENS` | 1024 | Max tokens para LLM call de keyword extraction. Subir si thinking-mode exhaustion. |
| `KG_EXTRACTION_MAX_TOKENS` | 4096 | Max tokens para LLM call de extraccion de tripletas. Antes hardcoded a 8192 (DTm-66). |
| `KG_BATCH_DOCS_PER_CALL` | 5 | Docs por LLM call en batch extraction. Subir a 10 reduce calls 50% (DTm-67). |
| `KG_GRAPH_OVERFETCH_FACTOR` | 2 | Multiplicador de candidatos en graph traversal (N * top_k). |
| `KG_FUSION_METHOD` | rrf | Metodo de fusion: `rrf` (robusto) o `linear` (legacy, candidato a deprecar). |
| `KG_RRF_K` | 60 | Constante k de Reciprocal Rank Fusion. |
