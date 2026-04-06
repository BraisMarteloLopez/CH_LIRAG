# CH_LIRAG

Sistema de evaluacion RAG (Retrieval-Augmented Generation) para benchmarking de pipelines de recuperacion y generacion sobre datasets MTEB/BeIR (HotpotQA) con infraestructura NVIDIA NIM.

## Estrategias de retrieval

| Estrategia | Indexacion | Busqueda | Reranker |
|---|---|---|---|
| `SIMPLE_VECTOR` | Embedding directo (NIM) | Cosine similarity (ChromaDB) | Opcional |
| `LIGHT_RAG` | LLM triplet extraction + KG + Embedding | Vector + Graph traversal + Fusion RRF | Opcional |

**LIGHT_RAG** es una implementacion inspirada en [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779). Combina busqueda vectorial con un knowledge graph construido via LLM. Entity VDB + Relationship VDB para resolucion semantica. 5 modos configurables via `LIGHTRAG_MODE`: `hybrid` (default), `graph_primary`, `local`, `global`, `naive`. Sin `igraph` o sin LLM → degrada a vector search puro.

## Arquitectura

```
shared/                          # Libreria compartida
  types.py                       # NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
  metrics.py                     # F1, ExactMatch, Accuracy, Faithfulness (LLM-judge)
  llm.py                         # AsyncLLMService, load_embedding_model
  config_base.py                 # InfraConfig, RerankerConfig, helpers _env_*
  report.py                      # RunExporter: JSON + CSV summary + CSV detail
  vector_store.py                # ChromaVectorStore
  structured_logging.py          # Logging JSONL estructurado
  retrieval/
    __init__.py                  # Factory get_retriever()
    core.py                      # BaseRetriever, SimpleVectorRetriever, RetrievalConfig, RRF
    reranker.py                  # CrossEncoderReranker (NVIDIARerank)
    lightrag/
      retriever.py               # LightRAGRetriever: vector + KG dual-level
      knowledge_graph.py         # KG in-memory (igraph): entidades, relaciones, traversal
      triplet_extractor.py       # Extraccion tripletas y query keywords via LLM

sandbox_mteb/                    # Pipeline de evaluacion MTEB/BeIR
  config.py                      # MTEBConfig: .env -> dataclass validada
  evaluator.py                   # Pipeline orquestador (<600 LOC)
  run.py                         # Entry point (--dry-run, -v, --resume)
  loader.py                      # MinIO/Parquet -> LoadedDataset
  retrieval_executor.py          # Loop retrieval sync + reranking
  generation_executor.py         # Generacion async + metricas
  embedding_service.py           # Pre-embed queries batch (NIM REST)
  checkpoint.py                  # Checkpoint/resume cada N queries
  result_builder.py              # Construccion EvaluationRun final
  preflight.py                   # Validacion pre-run (deps, NIM, MinIO)
  env.example                    # Plantilla .env

tests/                           # pytest (447 unit + 19 integration tests, 38 archivos)
```

## Pipeline

```
.env -> MTEBConfig -> MinIO/cache(Parquet) -> LoadedDataset
     -> shuffle(seed) -> slice(max_corpus)
     -> [LLM triplet extraction + KG build si LIGHT_RAG]
     -> index(ChromaDB)
     -> pre-embed queries (batch REST NIM)
     -> [pre-extract query keywords (batch LLM) si LIGHT_RAG]
     -> retrieve (sync) -> [rerank si habilitado] -> generate + metrics (async)
     -> EvaluationRun -> JSON + CSV
```

## Setup

```bash
pip install -r requirements.txt
cp sandbox_mteb/env.example sandbox_mteb/.env
# Editar .env con endpoints NIM y MinIO
```

## Uso

```bash
python -m sandbox_mteb.run                  # Run con .env
python -m sandbox_mteb.run --dry-run        # Solo validar config
python -m sandbox_mteb.run --env /path/.env # .env alternativo
python -m sandbox_mteb.run -v               # Verbose (DEBUG)
python -m sandbox_mteb.run --resume RUN_ID  # Reanudar desde checkpoint

python -m sandbox_mteb.preflight            # Preflight check (deps, NIM, MinIO)
```

## Tests

```bash
pytest tests/                      # Unit + integracion
pytest tests/ -m "not integration" # Solo unit (sin NIM/MinIO)
pytest tests/integration/ -v       # Solo integracion (requiere NIM + MinIO)
```

## Configuracion (.env)

Referencia completa en `sandbox_mteb/env.example`. Variables principales:

```bash
# Embedding (NIM)
EMBEDDING_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_BASE_URL=http://<nim-host>:8000/v1
EMBEDDING_BATCH_SIZE=5

# LLM (NIM) — requerido para LIGHT_RAG y generacion
LLM_BASE_URL=http://<nim-host>:8000/v1
LLM_MODEL_NAME=nvidia/nemotron-3-nano
NIM_MAX_CONCURRENT_REQUESTS=32

# Retrieval
RETRIEVAL_STRATEGY=SIMPLE_VECTOR      # SIMPLE_VECTOR | LIGHT_RAG
RETRIEVAL_K=20

# Knowledge Graph (solo LIGHT_RAG)
LIGHTRAG_MODE=hybrid                  # hybrid | graph_primary | local | global | naive
KG_MAX_HOPS=1                         # Profundidad BFS (1-hop como el original)
KG_GRAPH_WEIGHT=0.3                   # Peso graph en fusion (vector = 0.7)
KG_FUSION_METHOD=rrf                  # rrf (default) o linear
KG_EXTRACTION_MAX_TOKENS=4096         # Max tokens para triplet extraction
KG_BATCH_DOCS_PER_CALL=5             # Docs por LLM call en batch
KG_MAX_ENTITIES=0                     # Cap entidades (0 = default interno 100K)
KG_CACHE_DIR=                         # Directorio para persistir KG (vacio = sin cache)
KG_DESCRIPTION_SYNTHESIS=false        # LLM synthesis para descripciones multi-doc (DAM-4)
KG_SYNTHESIS_CHAR_THRESHOLD=200       # Chars minimos para trigger LLM synthesis

# Reranker (opcional)
RERANKER_ENABLED=false
RERANKER_TOP_N=5

# Dataset
MTEB_DATASET_NAME=hotpotqa
EVAL_MAX_QUERIES=50
EVAL_MAX_CORPUS=1000
GENERATION_ENABLED=true

# DEV_MODE: subset con gold docs garantizados
DEV_MODE=false
DEV_QUERIES=200
DEV_CORPUS_SIZE=4000
```

## Metricas

**Retrieval:** Hit@K, MRR, Recall@K, NDCG@K (K=1,3,5,10,20) sobre top `RETRIEVAL_K` docs pre-rerank.

**Generacion:** F1 (primary), Exact Match (secondary), Faithfulness (LLM-judge, opcional).

**Post-rerank:** `generation_recall`, `generation_hit`, `reranker_rescue_count` (solo con reranker activo).

## Dataset: HotpotQA

| Propiedad | Valor |
|---|---|
| Queries | 7405 |
| Corpus | 66576 docs (Wikipedia) |
| Qrels | 14810 (2.0/query) |
| Tipos | bridge (~80%), comparison (~20%) |

Limitacion: 10 pasajes/query (2 gold + 8 distractores). No comparable con benchmarks publicados (corpus 5.2M). Solo comparaciones relativas entre estrategias.

## Guia de desarrollo

Ver [`CLAUDE.md`](CLAUDE.md) para convenciones, divergencias con el paper, deuda tecnica, y proximos pasos. Ver [`TESTS.md`](TESTS.md) para referencia de la suite de tests.

<details>
<summary>Historial de desarrollo</summary>

**Fases 0-4 (fundacion):** Reproducibilidad, checkpoint/resume, batch adaptativo, evaluator descompuesto de 1225 a 592 LOC.

**Ronda 1 (arquitectura LightRAG):** NetworkX → igraph, batch extraction, RRF fusion, stemming.

**Ronda 2 (robustez):** LRU cache keywords, thread safety, fallbacks, stats.

**Fases A-F (alineacion con original):** Entity VDB, Relationship VDB, description merging, gleaning, graph_primary mode, contexto estructurado. Pendiente: F.5 (run comparativo).

**Fase H (hardening):** Bare excepts con logging, dead code eliminado, validacion sub-configs.

**Fase I (test coverage):** +42 tests nuevos. Cobertura 86% → 93%.

**Fase G (deuda tecnica):** Stats resilientes, query dedup, keyword cap, entity normalization apostrofes, fingerprint robusto.

**DTm-62/63/72/73/82/83:** Conditional fusion, eviction mejorada, BFS weighted, co-occurrence bridging, mypy zero errors, eliminacion HYBRID_PLUS.

**Audit Fases 1-5:** 8 bugfixes, 48 tests nuevos, DAM-4 LLM synthesis, eviction con score compuesto. 447 tests, 0 fallos.

80+ issues resueltos. Ver historial git.

</details>

<details>
<summary>Resultados comparativos (pre-Fases C-F, obsoletos)</summary>

> Todos los runs son pre-VDBs. Pendiente run comparativo post-implementacion (F.5).

| Metrica | SIMPLE_VECTOR | LIGHT_RAG (KG activo) | Delta |
|---|---|---|---|
| MRR | 1.000 | 0.518 | -48.2pp |
| Hit@5 | 1.000 | 0.793 | -20.7pp |
| Recall@5 | 0.940 | 0.690 | -25.0pp |
| F1 | 0.754 | 0.776 | +2.2pp |

Causa raiz: divergencias arquitectonicas (DAM-1..8), corregidas en Fases A-F + Audit.

</details>
