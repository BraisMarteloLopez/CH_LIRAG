# CH_LIRAG

Sistema de evaluacion RAG para benchmarking de pipelines de recuperacion y generacion sobre el dataset HotpotQA (MTEB/BeIR) con infraestructura NVIDIA NIM.

- **`SIMPLE_VECTOR`**: embedding + ChromaDB (cosine similarity), baseline.
- **`LIGHT_RAG`**: LLM triplet extraction + chunk keywords + KG + Embedding (chunks, entidades, relaciones) | Chunks via KG: tres canales (Entity VDB, Relationship VDB, Chunk Keywords VDB) agregados al mismo scoring, LLM synthesis del contexto.

**LIGHT_RAG** implementa la arquitectura de [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025, [arxiv](https://arxiv.org/abs/2410.05779)) con adaptaciones operativas (cache KG a disco, fallbacks, observabilidad). LightRAG extrae via LLM entidades, relaciones y high-level keywords por chunk para construir un knowledge graph + indices tematicos; las queries se resuelven en dos niveles (entidades concretas y temas abstractos) contra tres VDBs dedicadas — Entity VDB (entidades), Relationship VDB (relaciones) y Chunk Keywords VDB (temas high-level por chunk) — cuyos resultados agregan al mismo scoring de chunks. Chunks seleccionados via `source_doc_ids` del KG y matching directo a temas (paper-aligned), con fallback a vector search directo cuando el KG no produce doc_ids. Tras el retrieval, una capa de synthesis LLM (query-aware, `KG_SYNTHESIS_ENABLED`) reescribe el contexto multi-seccion (entidades + relaciones + chunks) como narrativa coherente antes de la generacion final.

El harness en `sandbox_mteb/` (datasets MTEB/BeIR) es **instrumento de verificacion temporal** — no es el producto final (en proceso de desarrollo).

**Documentacion tecnica y guia de desarrollo**: [`CLAUDE.md`](CLAUDE.md) (fuente unica para convenciones, divergencias con el paper, deuda tecnica, fases del proyecto, observabilidad, configuracion).
**Referencia de tests**: [`TESTS.md`](TESTS.md).

## Estructura

```
shared/                          # Libreria core (motor)
  types.py                       # NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
  metrics.py                     # F1, ExactMatch, Accuracy, Faithfulness (LLM-judge)
  llm.py                         # AsyncLLMService (NIM client, async/sync bridge)
  config_base.py                 # InfraConfig, RerankerConfig, helpers _env_*
  vector_store.py                # ChromaVectorStore (wrapper ChromaDB)
  report.py                      # RunExporter: JSON + CSV summary + CSV detail
  structured_logging.py          # Logging JSONL estructurado
  retrieval/
    __init__.py                  # Factory get_retriever() — punto de entrada
    core.py                      # RetrievalStrategy, RetrievalConfig, SimpleVectorRetriever
    reranker.py                  # CrossEncoderReranker (NVIDIARerank)
    lightrag/
      retriever.py               # LightRAGRetriever: vector + KG dual-level
      knowledge_graph.py         # KnowledgeGraph in-memory (igraph)
      triplet_extractor.py       # Extraccion de tripletas y keywords via LLM

sandbox_mteb/                    # Harness de evaluacion MTEB/BeIR (verifica el motor)
  config.py                      # MTEBConfig: .env -> dataclass validada
  evaluator.py                   # Pipeline orquestador
  run.py                         # Entry point CLI (--dry-run, -v, --resume)
  loader.py                      # MinIO/Parquet -> LoadedDataset
  retrieval_executor.py          # Loop retrieval sync + reranking
  generation_executor.py         # Generacion async + metricas
  embedding_service.py           # Pre-embed queries batch (NIM REST)
  checkpoint.py                  # Checkpoint/resume cada N queries
  result_builder.py              # Construccion EvaluationRun final
  preflight.py                   # Validacion pre-run (deps, NIM, MinIO)
  subset_selection.py            # DEV_MODE: gold docs + distractores
  env.example                    # Plantilla .env (referencia completa de variables)

tests/                           # pytest — ver CLAUDE.md "Test coverage"
  conftest.py                    # Mocks condicionales de infra
  integration/                   # Requiere NIM + MinIO reales
```

## Setup

```bash
pip install -r requirements.txt
cp sandbox_mteb/env.example sandbox_mteb/.env
# Editar .env con endpoints NIM y MinIO
```

## Uso

```bash
python -m sandbox_mteb.run                  # Run con .env por defecto
python -m sandbox_mteb.run --dry-run        # Solo validar config
python -m sandbox_mteb.run --env /path/.env # .env alternativo
python -m sandbox_mteb.run -v               # Verbose (DEBUG)
python -m sandbox_mteb.run --resume RUN_ID  # Reanudar desde checkpoint

python -m sandbox_mteb.preflight            # Preflight check (deps, NIM, MinIO)
```

## Configuracion (.env)

Referencia completa en `sandbox_mteb/env.example`. Variables principales:

```bash
# Embedding (NIM)
EMBEDDING_BASE_URL=http://<nim-host>:8000/v1
EMBEDDING_MODEL_TYPE=asymmetric
EMBEDDING_BATCH_SIZE=5

# LLM (NIM) - generacion + LLM-judge
LLM_MODEL_NAME=nvidia/nemotron-3-nano
NIM_MAX_CONCURRENT_REQUESTS=32        # [DEFAULT]
NIM_REQUEST_TIMEOUT=120               # [DEFAULT]
NIM_MAX_RETRIES=3                     # [DEFAULT]

# Retrieval
RETRIEVAL_STRATEGY=LIGHT_RAG          # SIMPLE_VECTOR | LIGHT_RAG
RETRIEVAL_K=20

# Knowledge Graph (solo LIGHT_RAG)
LIGHTRAG_MODE=hybrid                  # hybrid | local | global | naive
KG_MAX_HOPS=1                         # Profundidad BFS (1-hop como el original)
KG_BATCH_DOCS_PER_CALL=2              # Docs por LLM call en batch
KG_MAX_TEXT_CHARS=5000                # Caracteres de documento al LLM para extraccion de tripletas.
KG_MAX_NEIGHBORS_PER_ENTITY=5         # Vecinos 1-hop por entidad resuelta.
KG_MAX_ENTITIES=0                     # Cap entidades (0 = default interno 100K)
KG_KEYWORD_MAX_TOKENS=2048            # Max tokens de la llamada LLM de keyword extraction.
KG_EXTRACTION_MAX_TOKENS=4096         # Max tokens para triplet extraction
KG_GLEANING_ROUNDS=1                  # Rondas de re-extraccion para capturar entidades perdidas.
KG_CACHE_DIR=                         # Directorio para persistir KG (vacio = sin cache)
KG_DESCRIPTION_SYNTHESIS=false        # LLM synthesis para descripciones multi-doc (DAM-4)
KG_SYNTHESIS_CHAR_THRESHOLD=350       # Chars minimos para trigger LLM synthesis (DAM-4)
KG_CHUNK_KEYWORDS_ENABLED=true        # Piggyback extraccion de temas por chunk + 3er canal en retrieval
KG_CHUNK_KEYWORDS_TOP_K=20            # Top-K devueltos por keyword al consultar Chunk Keywords VDB

# KG context synthesis en generacion
KG_SYNTHESIS_ENABLED=true             # LLM reescribe contexto multi-seccion como narrativa
KG_SYNTHESIS_MAX_CHARS=0              # 0 = usar max_context_chars del run
KG_SYNTHESIS_TIMEOUT_S=180.0

# LLM judge instrumentation
JUDGE_FALLBACK_THRESHOLD=0.02         # Max tasa de fallback a 0.5 permitida (0 = desactiva). Ver CLAUDE.md "Observabilidad de runs"
```

## Tests

```bash
pytest tests/                               # Unit + integracion
pytest tests/ -m "not integration"          # Solo unit (sin NIM/MinIO)
pytest tests/integration/ -v                # Solo integracion (requiere NIM + MinIO)
```
