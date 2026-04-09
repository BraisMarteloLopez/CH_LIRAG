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

tests/                         # pytest (447 tests declarados, ver estado real abajo)
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

**Modos** (`LIGHTRAG_MODE`): `naive` (solo chunks), `local` (entidades + chunks), `global` (relaciones + chunks), `hybrid` (default, entidades + relaciones + chunks). Todos los modos (excepto naive) deben presentar secciones separadas al LLM. Nota: `graph_primary` existe en el codigo (DAM-3) pero no en el paper original — pendiente de eliminar.

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro.

**Alineacion con original (DAM-1 a DAM-8)**: Entity VDB, Relationship VDB, edge weights (log1p), gleaning, BFS 1-hop weighted, co-occurrence bridging, LLM description synthesis — todo implementado.

**Divergencia arquitectonica critica**: pese a lo anterior, el pipeline de retrieval+generacion **no replica la arquitectura del paper**. La indexacion es fiel, pero la forma en que se consumen los resultados del KG difiere fundamentalmente (ver seccion siguiente).

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original (HKUDS/LightRAG, EMNLP 2025)](https://arxiv.org/abs/2410.05779). Validadas empiricamente en F.5 (125q, 4000 docs, seed=42): LIGHT_RAG produjo metricas de retrieval **identicas** a SIMPLE_VECTOR, confirmando que las divergencias arquitectonicas anulan la contribucion del KG.

### Divergencias arquitectonicas (descubiertas en F.5)

| # | Divergencia | Criticidad | Detalle |
|---|---|---|---|
| 4+5 | **Pipeline de consumo diverge del paper** — el original presenta entidades, relaciones y chunks como secciones separadas al LLM sin fusion; aqui se fusionan via RRF en un ranking unico de doc_ids | **9/10** | Cada modo (`local`, `global`, `hybrid`) debe recopilar entidades, relaciones y/o chunks de forma independiente y presentarlos como secciones separadas al LLM (`"Knowledge Graph Data:"` + `"Document Chunks:"`) con token budgets propios (`max_entity_tokens`, `max_relation_tokens`). CH_LIRAG convierte todo a scores de documentos y fusiona via RRF (`core.py:reciprocal_rank_fusion()`, `retriever.py:_full_fusion()`/`_vector_first_fusion()`), descartando la granularidad entity/relation. El codigo para generar contexto estructurado existe (`format_structured_context()` en `retrieval_executor.py:195-266`, invocado desde `generation_executor.py:89-103`), pero `_fuse_with_graph()` no propaga `kg_entities`/`kg_relations` al metadata — el contexto estructurado nunca se genera. Ademas, `graph_primary` (DAM-3) no existe en el paper y debe eliminarse. Fix: (1) eliminar RRF entre canales KG y vector — cada canal mantiene resultados independientes, (2) propagar `kg_entities`/`kg_relations` y presentar secciones separadas al LLM, (3) eliminar `graph_primary`. |
| 6 | **Reranker post-fusion anula senal del grafo** — el paper no usa reranker | **8/10** | El `CrossEncoderReranker` se aplica incondicionalmente a ambas estrategias (`retrieval_executor.py:100-146`). Evalua similitud query↔doc en aislamiento — los docs multi-hop que el KG aporta (indirectamente relevantes) puntuan bajo y caen del top-N. En F.5, el reranker colapso el ranking LIGHT_RAG al mismo top-20 que SIMPLE_VECTOR en las 125 queries. Fix: desactivar reranker cuando `strategy=LIGHT_RAG`; mantener para `SIMPLE_VECTOR`. |
| 7 | **Sin token budgets separados por tipo** — el original asigna presupuesto independiente a entidades, relaciones y chunks | **5/10** | Un unico `max_context_chars` se aplica al contexto concatenado. Las descripciones de entidades/relaciones compiten con los chunks por el mismo espacio, en vez de tener presupuesto garantizado. Subsumida por #4+5: el fix de presentar secciones separadas requiere token budgets independientes (`max_entity_tokens`, `max_relation_tokens`, `max_chunk_tokens`). |

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

**Conclusion**: la indexacion KG funciona correctamente (23K entidades, 55K relaciones, 32K co-occurrence edges), pero las divergencias #4+5 y #6 impiden que su senal llegue al LLM.

## Deuda tecnica vigente

| # | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB: colecciones huerfanas si el proceso se interrumpe | **BAJO** | `evaluator.py:_cleanup()` ahora elimina la coleccion correctamente via `delete_all_documents()` (que llama `delete_collection()` + recrea). Sin embargo, cada run crea `eval_{run_id}` — si el proceso se interrumpe antes de cleanup, la coleccion queda huerfana. Con `PersistentClient`, se acumulan en disco | Aceptable; borrar manualmente `VECTOR_DB_DIR` si se acumulan |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO (`head_bucket` + `list_objects MaxKeys=1`). No descarga ni parsea Parquet — dataset corrupto solo falla horas despues. No verifica espacio en disco | `--dry-run` primero y verificar que el dataset carga |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 4 | LLM Judge puede devolver scores por defecto | **MEDIO-BAJO** | `metrics.py:_extract_score_fallback()` intenta 3 regex patterns (fraccion, decimal, entero con prefijo); si todos fallan retorna 0.5 — sesga metricas silenciosamente. Se logea a WARNING. Deuda a largo plazo: la mitigacion real requiere mas contexto de ventana y/o un modelo judge mas capaz que produzca respuestas estructuradas consistentemente | Post-run, buscar `"Score extraction fallback"` en logs y contar ocurrencias |
| 5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, usa fallback de 4000 chars (~1000 tokens). Puede truncar docs importantes. Se logea WARNING | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicitamente en `.env` |
| 6 | Suite de tests no portable | **CRITICO** | `conftest.py` mockea boto3/langchain/chromadb pero no `python-dotenv` ni `igraph`. 17 archivos fallan en coleccion (dotenv), 65 tests fallan (igraph). Solo 14 de ~447 pasan en un entorno limpio. Ver seccion "Estado real de tests" abajo | Crear `requirements-test.txt` con todas las dependencias de test, o añadir dotenv+igraph a la lista de mocks en conftest.py |
| 7 | Validacion empirica pendiente post-refactor | **MEDIO** | F.5 completado para ambas estrategias (125q, 4000 docs, seed=42). Resultados confirman que LIGHT_RAG no aporta valor con la arquitectura actual (metricas identicas a SIMPLE_VECTOR). Pendiente: re-ejecutar tras alinear pipeline con el paper (ver divergencias #4+5 y #6) para validar que el KG aporta mejora medible | Implementar fixes de divergencias #4+5 y #6 primero, luego re-run F.5 |
| 8 | Infraestructura pesada para el scope | **BAJO** | Para 1 dataset y 2 estrategias, la infraestructura (checkpoint, preflight, JSONL, export dual, subset selection, DEV_MODE) es considerable. Sin embargo, el run F.5 demostro que esta infraestructura funciona y es util en practica | Aceptado — la infraestructura se justifica con uso real |
| 9 | Lock-in a NVIDIA NIM | **MEDIO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |

Las divergencias arquitectonicas #4/#5/#6 son la causa raiz de que LIGHT_RAG no aporte valor. Ver seccion "Divergencias con el paper original".

## Bare excepts aceptados (no criticos)

Estos `except Exception as e:` logean el error pero no lo re-lanzan. Aceptable para wrappers de infraestructura:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:126, 142, 179, 232, 247` | Operaciones ChromaDB — retorna fallback (lista vacia, dict vacio, o continua cleanup) |

## Test coverage

| Metrica | Valor declarado | Valor real (entorno limpio, abril 2026) |
|---|---|---|
| Tests unitarios | 447 en 38 archivos | **14 pasan**, 65 fallan, 17 archivos con error de coleccion |
| Tests integracion | 19 en 3 archivos | No verificados (requieren NIM + MinIO) |
| mypy | 0 errores (27 source files) | No verificado |

### Estado real de tests

**La suite de tests no es portable.** Ejecutar `pytest tests/ -m "not integration"` en un entorno limpio produce:

- **17 archivos con ERROR de coleccion**: `ModuleNotFoundError: No module named 'dotenv'`. `conftest.py` mockea boto3, langchain y chromadb, pero omite `python-dotenv` (importado por `shared/config_base.py`).
- **65 tests FALLAN** (tras instalar dotenv): todos los tests de `test_knowledge_graph.py` requieren `igraph` real — no hay mock ni skip condicional.
- **14 tests pasan**: solo los que no dependen de config_base ni igraph.

**Causa raiz**: `conftest.py` lineas 33-44 lista los modulos a mockear, pero falta `dotenv` e `igraph`. Los tests de KG asumen igraph instalado sin fallback.

**Implicacion**: la cifra "447 tests passing" solo es valida en el entorno de desarrollo del autor. Nadie mas puede verificarla.

**Fix necesario**: añadir `python-dotenv` e `python-igraph` a un `requirements-test.txt`, o incluirlos en la lista de mocks de `conftest.py`.

**Referencia completa**: ver `TESTS.md` — mapa test→produccion, atributos `object.__new__()`, trampas de mock, gaps de cobertura, reglas de modificacion.

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio (DTm-45). Parece complejo pero es necesario

## Proximos pasos

### Alinear pipeline LIGHT_RAG con el paper original

F.5 demostro que la indexacion KG funciona pero el pipeline de consumo no. Acciones ordenadas por impacto:

| Prioridad | Tarea | Divergencia | Descripcion |
|---|---|---|---|
| **P0** | Desactivar reranker para LIGHT_RAG | #6 | Cambio trivial, desbloquea senal KG. Mantener para `SIMPLE_VECTOR` |
| **P0** | Contexto estructurado al LLM | #4+5 | Eliminar RRF entre canales KG y vector. Cada canal mantiene resultados independientes y se presenta como secciones separadas al LLM con token budgets propios (`max_entity_tokens`, `max_relation_tokens`, `max_chunk_tokens`). Eliminar `graph_primary` (DAM-3). Limpiar funciones huerfanas: `reciprocal_rank_fusion()`, `_full_fusion()`, `_vector_first_fusion()`, `_fuse_with_graph()`. Requiere refactor de `retriever.py`, `retrieval_executor.py:format_structured_context()` y `generation_executor.py` |
| **P1** | Re-run F.5 post-refactor | — | Repetir comparativa SIMPLE_VECTOR vs LIGHT_RAG con pipeline alineado |

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`)
