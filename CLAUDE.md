# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Estructura clave

```
shared/                        # Libreria core
  types.py                     # Tipos: NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
  metrics.py                   # F1, EM, Accuracy, Faithfulness (LLM-judge)
  llm.py                       # AsyncLLMService (NIM client, async/sync bridge)
  config_base.py               # InfraConfig, RerankerConfig, _env helpers
  vector_store.py              # ChromaVectorStore (wrapper ChromaDB)
  report.py                    # RunExporter: JSON + CSV summary + CSV detail
  structured_logging.py        # Logging JSONL estructurado
  retrieval/
    core.py                    # RetrievalStrategy enum, RetrievalConfig, SimpleVectorRetriever, RRF
    __init__.py                # Factory get_retriever() — punto de entrada para crear retrievers
    reranker.py                # CrossEncoderReranker (NVIDIARerank)
    lightrag/
      retriever.py             # LightRAGRetriever: vector + KG dual-level
      knowledge_graph.py       # KnowledgeGraph in-memory (igraph): entidades, relaciones, BFS
      triplet_extractor.py     # Extraccion de tripletas y keywords via LLM

sandbox_mteb/                  # Pipeline de evaluacion
  config.py                    # MTEBConfig: .env → dataclass validada (+RerankerConfig.validate)
  evaluator.py                 # Orquestador principal
  run.py                       # Entry point CLI (--dry-run, -v, --resume)
  loader.py                    # MinIO/Parquet → LoadedDataset
  retrieval_executor.py        # Loop retrieval + reranking
  generation_executor.py       # Generacion async + metricas
  embedding_service.py         # Pre-embed queries batch + context window detection
  checkpoint.py                # Checkpoint/resume cada N queries (atomic writes)
  result_builder.py            # Construccion EvaluationRun final
  preflight.py                 # Validacion pre-run (deps, NIM, MinIO)
  subset_selection.py          # DEV_MODE: gold docs + distractores

tests/                         # pytest (447 unit tests, 38 files)
  conftest.py                  # Mocks condicionales de infra (boto3, langchain, chromadb)
  test_*.py                    # Unit test files
  integration/                 # 3 files, requieren NIM + MinIO reales
```

## Comandos

```bash
# Tests (unit only, sin infra)
pytest tests/ -m "not integration"

# Run de evaluacion
python -m sandbox_mteb.run
python -m sandbox_mteb.run --dry-run
python -m sandbox_mteb.run -v
python -m sandbox_mteb.run --resume RUN_ID

# Preflight check
python -m sandbox_mteb.preflight
```

## Convenciones

- **Config via .env**: toda la parametrizacion en `sandbox_mteb/.env`, leida por `MTEBConfig.from_env()` una sola vez. Sub-configs delegadas a `InfraConfig`, `RerankerConfig`, `RetrievalConfig` en shared/. `MTEBConfig.validate()` propaga validacion a sub-configs
- **Factory pattern**: `get_retriever(config, embedding_model)` en `shared/retrieval/__init__.py` crea el retriever correcto
- **2 estrategias**: `SIMPLE_VECTOR` y `LIGHT_RAG` — no hay mas. `HYBRID_PLUS` fue eliminada (DTm-83)
- **Enum en core.py**: `RetrievalStrategy` define las estrategias validas. `VALID_STRATEGIES` en `sandbox_mteb/config.py` debe coincidir
- **Tests**: `conftest.py` mockea modulos de infra (boto3, langchain, chromadb) si no estan instalados. Tests de integracion requieren NIM + MinIO reales. Mocks siempre a nivel de funcion, nunca modulos enteros
- **Logging**: JSONL estructurado via `shared/structured_logging.py`. Bare excepts tienen `logger.debug(...)` — no hay excepts silenciosos
- **Idioma**: codigo y comentarios en ingles/espanol mezclado (historico). Docstrings y variables en ingles

## Estrategia LIGHT_RAG — como funciona

Inspirada en [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779).

**Indexacion**: LLM extrae tripletas (entidad, relacion, entidad) de cada doc → KnowledgeGraph in-memory (igraph) + ChromaDB para vector search. Entity VDB y Relationship VDB para resolucion semantica. Stats se resetean automaticamente al inicio de cada batch (G.5). Gleaning opcional via `KG_GLEANING_ROUNDS`.

**Retrieval**: vector search + query keywords via LLM (dedup automatico de queries identicas, cap 20 keywords/nivel) + graph traversal dual-level (entity VDB low-level + relationship VDB high-level) + fusion RRF.

**Modos** (`LIGHTRAG_MODE`): `hybrid` (default), `graph_primary`, `local`, `global`, `naive`.

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro.

**Alineacion con original (DAM-1 a DAM-8)**: Entity VDB, Relationship VDB, edge weights (log1p), gleaning, BFS 1-hop weighted, graph_primary mode, co-occurrence bridging, LLM description synthesis — todo implementado.

**Divergencia arquitectonica critica**: pese a lo anterior, el pipeline de retrieval+generacion **no replica la arquitectura del paper**. La indexacion es fiel, pero la forma en que se consumen los resultados del KG difiere fundamentalmente (ver seccion siguiente).

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original (HKUDS/LightRAG, EMNLP 2025)](https://arxiv.org/abs/2410.05779). Validadas empiricamente en F.5 (125q, 4000 docs, seed=42): LIGHT_RAG produjo metricas de retrieval **identicas** a SIMPLE_VECTOR, confirmando que las divergencias arquitectonicas anulan la contribucion del KG.

### Divergencias arquitectonicas (descubiertas en F.5)

| # | Divergencia | Criticidad | Detalle |
|---|---|---|---|
| 4 | **Contexto aplanado a ranking de docs** — el original presenta entidades, relaciones y chunks como **secciones separadas** en el prompt del LLM; aqui todo se aplana a un ranking unico de doc_ids | **9/10** | El LLM del original recibe contexto estructurado (`"Knowledge Graph Data:"` + `"Document Chunks:"`) con presupuesto de tokens por tipo. Aqui el KG solo sirve como puntero a documentos, perdiendo las descripciones de entidades/relaciones. `retrieval_executor.py:format_structured_context()` intenta mitigar esto pero opera post-reranker, sobre docs ya filtrados. |
| 5 | **RRF fusiona KG + vector en un ranking unico** — el original no fusiona, presenta ambos canales por separado al LLM | **8/10** | El original no usa RRF ni fusion lineal. Entidades y relaciones son secciones independientes con token budgets propios (`max_entity_tokens`, `max_relation_tokens`). CH_LIRAG convierte todo a scores de documentos y fusiona via RRF (`core.py:reciprocal_rank_fusion()`), descartando la granularidad entity/relation. |
| 6 | **Reranker post-fusion anula senal del grafo** — el original no usa reranker (adicion post-publicacion al repo) | **8/10** | El `CrossEncoderReranker` se aplica incondicionalmente a ambas estrategias (`retrieval_executor.py:100-146`). Evalua similitud query↔doc en aislamiento — los docs multi-hop que el KG aporta (indirectamente relevantes) puntuan bajo y caen del top-N. En F.5, el reranker colapso el ranking LIGHT_RAG al mismo top-20 que SIMPLE_VECTOR en las 125 queries. |
| 7 | **Sin token budgets separados por tipo** — el original asigna presupuesto independiente a entidades, relaciones y chunks | **5/10** | Un unico `max_context_chars` se aplica al contexto concatenado. Las descripciones de entidades/relaciones compiten con los chunks por el mismo espacio, en vez de tener presupuesto garantizado. |

### Divergencias menores (preexistentes)

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 2 | Sin LLM synthesis en fusion final de contexto | **5/10** | Subsumida por #4/#5: la fusion synthesis del paper opera sobre el contexto estructurado (secciones separadas). Sin ese contexto, synthesis no resolveria el problema de fondo. |
| 3 | Entity cap 100K | **3/10** | Eviction mejorada con score compuesto. Para HotpotQA (66K docs, ~24K entidades) no se alcanza. |

### Resueltas (indexacion)

- ~~DAM-4 (7/10)~~: LLM synthesis para descripciones → `KG_DESCRIPTION_SYNTHESIS=true` (A5.1)
- ~~Grafo fragmentado (3/10)~~: Co-occurrence bridging (DTm-73)
- ~~BFS scoring uniforme (3/10)~~: Edge weights via `log(1 + n_docs)` (DTm-72)

### Resultado F.5 (validacion empirica)

Runs ejecutadas: SIMPLE_VECTOR y LIGHT_RAG hybrid (125q, 4000 docs, DEV_MODE, seed=42, reranker ON).

| Metrica | SIMPLE_VECTOR | LIGHT_RAG | Delta |
|---|---|---|---|
| Hit@5 | 1.000 | 1.000 | 0 |
| MRR | 0.992 | 0.992 | 0 |
| Recall@5 | 0.968 | 0.968 | 0 |
| Recall@20 | 0.988 | 0.988 | 0 |
| Avg gen score | 0.7764 | 0.7877 | +0.011 (ruido LLM) |
| Tiempo total | 194.7s | 9002.1s | ×46 |

**Todas las metricas de retrieval son identicas query por query.** El KG aporto ~49 docs exclusivos por query, pero el reranker colapso el ranking final al mismo top-20 que SIMPLE_VECTOR. Las 10 diferencias en generacion son no-determinismo del LLM (mismo contexto, distinta respuesta).

**Conclusion**: la indexacion KG funciona correctamente (23K entidades, 55K relaciones, 32K co-occurrence edges), pero las divergencias #4/#5/#6 impiden que su senal llegue al LLM.

## Deuda tecnica vigente

| # | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB no se limpia entre runs | **ALTO** | `evaluator.py:_cleanup()` llama `clear_index()` pero solo pone `_is_indexed=False` — no borra la coleccion ChromaDB. Cada run crea `eval_{run_id}`, las viejas se acumulan (~500MB-1GB tras 10 runs) | Borrar manualmente `VECTOR_DB_DIR` antes de cada run |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO (`head_bucket` + `list_objects MaxKeys=1`). No descarga ni parsea Parquet — dataset corrupto solo falla horas despues. No verifica espacio en disco | `--dry-run` primero y verificar que el dataset carga |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 4 | LLM Judge puede devolver scores por defecto | **MEDIO-BAJO** | `metrics.py:_extract_score_fallback()` intenta 4 regex patterns; si todos fallan retorna 0.5 — sesga metricas silenciosamente. Se logea a WARNING | Post-run, buscar `"Score extraction fallback"` en logs y contar ocurrencias |
| 5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, usa fallback de 4000 chars (~1000 tokens). Puede truncar docs importantes. Se logea WARNING | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicitamente en `.env` |

Las divergencias arquitectonicas #4/#5/#6 son la causa raiz de que LIGHT_RAG no aporte valor. Ver seccion "Divergencias con el paper original".

## Bare excepts aceptados (no criticos)

Estos `except Exception as e:` logean el error pero no lo re-lanzan. Aceptable para wrappers de infraestructura:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:125, 141, 178` | Operaciones ChromaDB — retorna lista vacia |

## Test coverage

| Metrica | Valor |
|---|---|
| Tests unitarios | 447 en 38 archivos |
| Tests integracion | 19 en 3 archivos |
| mypy | 0 errores (27 source files) |
| Modulos con tests dedicados | 21/23 (91%) |
| Modulos sin tests | structured_logging.py (bajo riesgo) |
| Tests con assertions | 100% |
| Mocks a nivel funcion | 100% |

**Referencia completa**: ver `TESTS.md` — mapa test→produccion, atributos `object.__new__()`, trampas de mock, gaps de cobertura, reglas de modificacion.

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `reciprocal_rank_fusion()` en `core.py` — la usa LIGHT_RAG, no es legacy
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio (DTm-45). Parece complejo pero es necesario

## Proximos pasos

### Alinear pipeline LIGHT_RAG con el paper original

F.5 demostro que la indexacion KG funciona pero el pipeline de consumo no. Acciones ordenadas por impacto:

| Prioridad | Tarea | Divergencia | Descripcion |
|---|---|---|---|
| **P0** | Contexto estructurado al LLM | #4 | Pasar entidades + relaciones + chunks como secciones separadas en el prompt, con token budgets independientes. Requiere refactor de `retrieval_executor.py:format_structured_context()` y `generation_executor.py` |
| **P0** | Eliminar RRF entre KG y vector | #5 | No fusionar en un ranking unico. Cada canal (entidades, relaciones, vector chunks) mantiene su propio conjunto de resultados. El LLM decide que es relevante |
| **P1** | Desactivar reranker para LIGHT_RAG | #6 | Flag `RERANKER_SKIP_FOR_LIGHT_RAG` o desactivar incondicionalmente cuando strategy=LIGHT_RAG. El cross-encoder single-hop es incompatible con retrieval multi-hop |
| **P2** | Token budgets por tipo | #7 | `max_entity_tokens`, `max_relation_tokens`, `max_chunk_tokens` en config, en vez de un unico `max_context_chars` |
| **P3** | Re-run F.5 post-refactor | — | Repetir comparativa SIMPLE_VECTOR vs LIGHT_RAG con pipeline alineado |

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`)
