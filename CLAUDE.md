# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Estructura clave

```
shared/                        # Libreria core
  types.py                     # Tipos: NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
  metrics.py                   # F1, EM, Accuracy, Faithfulness (LLM-judge)
  llm.py                       # AsyncLLMService (NIM client)
  config_base.py               # InfraConfig, RerankerConfig, _env helpers
  vector_store.py              # ChromaVectorStore
  retrieval/
    core.py                    # RetrievalStrategy enum, RetrievalConfig, SimpleVectorRetriever, RRF
    __init__.py                # Factory get_retriever() — punto de entrada para crear retrievers
    reranker.py                # CrossEncoderReranker (NVIDIARerank)
    lightrag/
      retriever.py             # LightRAGRetriever: vector + KG dual-level
      knowledge_graph.py       # KnowledgeGraph in-memory (igraph): entidades, relaciones, BFS
      triplet_extractor.py     # Extraccion de tripletas y keywords via LLM

sandbox_mteb/                  # Pipeline de evaluacion
  config.py                    # MTEBConfig: .env → dataclass validada
  evaluator.py                 # Orquestador principal (<600 LOC)
  run.py                       # Entry point CLI
  loader.py                    # MinIO/Parquet → LoadedDataset
  retrieval_executor.py        # Loop retrieval + reranking
  generation_executor.py       # Generacion async + metricas
  embedding_service.py         # Pre-embed queries batch
  result_builder.py            # Construccion EvaluationRun final
  preflight.py                 # Validacion pre-run

tests/                         # pytest (unit + integration)
  conftest.py                  # Mocks condicionales de infra
```

## Comandos

```bash
# Tests (unit only, sin infra)
pytest tests/ -m "not integration"

# Run de evaluacion
python -m sandbox_mteb.run
python -m sandbox_mteb.run --dry-run
python -m sandbox_mteb.run -v

# Preflight check
python -m sandbox_mteb.preflight
```

## Convenciones

- **Config via .env**: toda la parametrizacion en `sandbox_mteb/.env`, leida por `MTEBConfig.from_env()` una sola vez
- **Factory pattern**: `get_retriever(config, embedding_model)` en `shared/retrieval/__init__.py` crea el retriever correcto
- **2 estrategias**: `SIMPLE_VECTOR` y `LIGHT_RAG` — no hay mas. `HYBRID_PLUS` fue eliminada (DTm-83)
- **Enum en core.py**: `RetrievalStrategy` define las estrategias validas. `VALID_STRATEGIES` en `sandbox_mteb/config.py` debe coincidir
- **Tests**: `conftest.py` mockea modulos de infra (boto3, langchain, chromadb) si no estan instalados. Tests de integracion requieren NIM + MinIO reales
- **Logging**: JSONL estructurado via `shared/structured_logging.py`
- **Idioma**: codigo y comentarios en ingles/espanol mezclado (historico). Docstrings y variables en ingles

## Estrategia LIGHT_RAG — como funciona

Inspirada en [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779).

**Indexacion**: LLM extrae tripletas (entidad, relacion, entidad) de cada doc → KnowledgeGraph in-memory (igraph) + ChromaDB para vector search. Entity VDB y Relationship VDB para resolucion semantica.

**Retrieval**: vector search + query keywords via LLM + graph traversal dual-level (entity VDB low-level + relationship VDB high-level) + fusion RRF.

**Modos** (`LIGHTRAG_MODE`): `hybrid` (default), `graph_primary`, `local`, `global`, `naive`.

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro.

## Deuda tecnica vigente (priorizada)

### Alta — afectan calidad de resultados
- **DTm-62**: Fusion KG destruye ranking (MRR -33pp). Pendiente run comparativo F.5
- **DTm-63**: Entity cap 100K con sesgo FIFO en orden de indexacion
- **DTm-73**: Grafo fragmentado impide bridging entre componentes desconectados

### Media — mejoras funcionales
- **DTm-64**: Normalizacion scores incomparable entre canales vector/graph
- **DTm-55**: Stats se corrompen si KG build falla a mitad
- **DTm-56**: Fingerprint collision edge case con corpus vacio
- **DTm-72**: BFS scoring ciego a la relacion (todas las aristas pesan igual)
- **DTm-77**: Test gap: gleaning sin tests (`glean_from_doc_async`)
- **DTm-78**: Test gap: E2E solo cubre SIMPLE_VECTOR, no LIGHT_RAG
- **DTm-80**: DAM-4 parcial: merge de descripciones por concatenacion, sin LLM synthesis

### Baja — code quality
- **DTm-82**: Errores mypy sin resolver
- **DTm-57**: Normalizacion entidades pierde apostrofes
- **DTm-58**: No dedup queries en batch keyword extraction
- **DTm-60**: `reset_stats()` nunca se llama automaticamente
- **DTm-61**: Sin validacion de tamano de keywords del LLM

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `reciprocal_rank_fusion()` en `core.py` — la usa LIGHT_RAG, no es legacy
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon

## Proximos pasos

1. **F.5**: Run comparativo post-VDBs (SIMPLE_VECTOR vs LIGHT_RAG) — necesario para validar que las Fases C-F mejoraron el ranking
2. **Fase G**: 10 tareas de deuda tecnica menor (ver tabla arriba)
