# CLAUDE.md

## Que es este proyecto

**Motor de ejecucion para construccion y consulta de grafos de conocimiento sobre corpus arbitrarios.** Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB, baseline) y `LIGHT_RAG` (vector + KG via LLM, motor objetivo). Implementa la arquitectura de [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)).

El harness en `sandbox_mteb/` (datasets MTEB/BeIR + NVIDIA NIM) es **instrumento de verificacion temporal** de la correctitud del motor frente al paper. No es el producto.

## Contexto del producto

**Vision a largo plazo**: este motor se integrara en un sistema administrador de colecciones de datos (versionado de KGs, multi-tenant, ciclo de vida de corpus). Contrato compartido: MinIO + Parquet — la integracion es apuntar a otro prefijo, no rediseñar la ingesta. **Condicionada a que P0 cierre con exito**; sin replicacion del paper, solo SIMPLE_VECTOR es integrable y el trabajo sobre KG se vuelve inutil.

**Escenario objetivo**: colecciones pequenas (10-50 PDFs) de dominio especializado con terminologia/entidades fuera del pre-entrenamiento del embedding. Es el caso donde el KG aporta diferencial. HotpotQA NO es ese escenario — es harness de verificacion previa.

**Implicacion de diseño**: las decisiones favorecen la embedibilidad futura (config declarativa, interfaces claras, sin side-effects globales, corpus arbitrarios, muy alta calidad de generacion). Mientras P0 no este verde, la embedibilidad es objetivo de diseño, no trabajo activo.

**Fases del proyecto**:
1. **Pre-P0 — Cerrada (2026-04-19)**: completitud arquitectural de LIGHT_RAG + ejecucion estable. Las 7 divergencias arquitectonicas (#2, #4+5, #6, #7, #8, #9, #10) resueltas en codigo y tests. Validacion empirica:
   - **Validacion directa**: #2, #6, #8 (observables discriminantes durante el cierre).
   - **Observable per-query + validacion empirica complementaria**: #4+5, #7, #9 (los canales operan; #7 tiene hallazgo abierto: el generador final NO propaga `[ref:N]` al usuario, faithfulness intacta porque se evalua contra el contexto estructurado original).
   - **Presencia validada, calidad NO validada**: #10 (piggyback puede degradar output high-level vs llamada dedicada; el observable solo mide matches de VDB, no calidad semantica). **Bloqueante antes de P2, no de P0**.
   - Ejecucion estable: `fallback_rate = 2.86%` (< 10% umbral), `judge.default_return_rate = 0%`, `kg_fallback=null` en 35/35 queries.
2. **P0 — Fase actual**: replicacion empirica sobre benchmark donde el paper muestre ventaja (`LIGHT_RAG > SIMPLE_VECTOR`).

**Export de KG (P3)**: cuando LIGHT_RAG sea produccion, los KGs deberan persistirse para tres usos del administrador: versionado, reuso entre runs (no re-indexar por query), multi-tenant. Hoy efimero (igraph + ChromaDB en memoria).

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
    core.py                    # RetrievalStrategy enum, RetrievalConfig, SimpleVectorRetriever
    __init__.py                # Factory get_retriever() — punto de entrada para crear retrievers
    reranker.py                # CrossEncoderReranker (NVIDIARerank)
    lightrag/
      retriever.py             # LightRAGRetriever: vector + KG dual-level
      knowledge_graph.py       # KnowledgeGraph in-memory (igraph): entidades, relaciones, BFS
      triplet_extractor.py     # Extraccion de tripletas y keywords via LLM

sandbox_mteb/                  # Harness de evaluacion (verifica el motor; no es el motor)
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

tests/                         # pytest (contador exacto: ver seccion "Test coverage")
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

Adaptaciones operativas propias del entorno: cache de KG a disco, fallbacks ante errores del LLM/igraph, instrumentacion de observabilidad.

**Indexacion**: LLM extrae tripletas `(entidad, relacion, entidad)` + `high_level_keywords` tematicas por chunk (piggyback en la misma llamada, divergencia #10) → KnowledgeGraph in-memory (igraph) + 3 VDBs ChromaDB: Entity VDB, Relationship VDB, Chunk Keywords VDB. Gleaning opcional via `KG_GLEANING_ROUNDS` (solo entidades/relaciones perdidas; no re-extrae keywords).

**Retrieval (modos `local`/`global`/`hybrid`)**: los chunks se obtienen **via KG** con tres canales que suman al mismo `doc_scores` con `1/(1+rank) × similarity [× edge_weight]`:
- **Entidades** (local/hybrid): low-level keywords → Entity VDB → `source_doc_ids`
- **Relaciones** (global/hybrid): high-level keywords → Relationship VDB → `source_doc_ids` de endpoints
- **Chunk keywords** (global/hybrid, div. #10): high-level keywords → Chunk Keywords VDB → doc_ids directos

Contenido real fetcheado por `get_documents_by_ids`. Fallback a vector directo si KG no produce doc_ids (logueado en `retrieval_metadata.kg_fallback`). Resultados intermedios anotados en `retrieval_metadata.kg_entities`/`kg_relations`/`kg_chunk_keyword_matches`. Scoring usa `1/(1+rank)` (paper usa decay lineal — misma intencion, curva distinta).

**Enriquecimiento KG (div. #9)**: cada entidad resuelta incluye vecinos 1-hop rankeados por `edge_weight + degree_centrality` (`get_neighbors_ranked`), configurable via `KG_MAX_NEIGHBORS_PER_ENTITY` (default 5, 0=off). Cada relacion incluye descripcion+tipo de endpoints. Degradacion graceful en lookup fallido.

**Modos** (`LIGHTRAG_MODE`): `naive` (solo chunks), `local` (+entidades), `global` (+relaciones), `hybrid` (default, todo).

**Synthesis del contexto (div. #2 — value-add del proyecto, no esta en el paper)**: `GenerationExecutor._synthesize_kg_context_async()` reescribe las 3 secciones como narrativa coherente via LLM antes de la generacion final, con citas `[ref:N]` inline (prompt en `sandbox_mteb/config.py:KG_SYNTHESIS_SYSTEM_PROMPT`). **Faithfulness se evalua contra el contexto estructurado original**, no contra la narrativa — control anti-fabricacion para penalizar alucinacion de la propia synthesis. Degradacion graceful: error/vacio/timeout → fallback al contexto estructurado.

**Fallback global**: sin igraph o sin LLM → SimpleVectorRetriever puro. El run nunca se rompe.

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original](https://github.com/HKUDS/LightRAG) (HKUDS, EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)).

### Status de validacion por divergencia arquitectonica

Tabla resumen del estado de cada divergencia arquitectonica frente al paper. **Validada end-to-end** = un observable concreto (flag en `retrieval_metadata`, stat en `kg_synthesis_stats`) confirmo el comportamiento durante el cierre de Pre-P0. **Resuelta en codigo** = implementado y cubierto por tests unitarios; ejercitado implicitamente en runs reales (sin observable dedicado). P0 sobre benchmark con contra-referencia validara implicitamente #4+5/#7/#9 si el delta reproduce la direccion del paper.

| # | Divergencia | Status | Evidencia empirica |
|---|---|---|---|
| 2 | Synthesis layer con citas `[ref:N]` | **Resuelta + Validada end-to-end** | `kg_synthesis_stats.fallback_rate=2.86%`, `kg_synthesis_used=True` en 34/35 queries |
| 4+5 | Budgets proporcionales entre secciones del contexto | **Resuelta en codigo + observable per-query** | Tests `test_structured_context.py` (+5 tests para `is_kg_budget_cap_triggered`) y `test_generation_executor.py` (+3 tests para la anotacion en `retrieval_metadata`). `retrieval_metadata.kg_budget_cap_triggered` discrimina per-query si el cap al 50% escalo las secciones KG (runs post-auditoria) |
| 6 | Reranker desactivado para LIGHT_RAG | **Resuelta + Validada end-to-end** | Guard en `_init_components()`: log `"Reranker habilitado en .env pero estrategia es LIGHT_RAG; omitiendo inicializacion"` confirmado durante el cierre de Pre-P0 |
| 7 | Contexto JSON-lines con `reference_id` + observable de citas | **Resuelta + Validada (synth) + hallazgo abierto (gen)** | Parser `shared/citation_parser.py::parse_citation_refs` emite 7 contadores × 2 prefijos (`citation_refs_{synth,gen}_*`) = 14 campos cuando `has_kg_data AND kg_synthesis_enabled`. Senales rojas: `out_of_range>0`, `malformed>0`, `gen_valid>synth_valid`. **Validado**: synth narrativa cita bien (alta cobertura, 0 out_of_range, 0 malformed). **Hallazgo abierto**: `gen_*=0` — el generador final NO propaga `[ref:N]` al usuario. Faithfulness intacta (se evalua contra contexto estructurado original). Mejora pendiente antes de P0/P2 si se quiere respuesta anclada. Acoplamiento: parser ligado al formato de `KG_SYNTHESIS_SYSTEM_PROMPT`; tocar el prompt exige revisar regex `_VALID_RE`/`_CANDIDATE_RE`. |
| 8 | Chunks obtenidos via `source_doc_ids` del KG | **Resuelta + Validada end-to-end** | `retrieval_metadata.kg_fallback=null` en 35/35 queries (el KG produjo doc_ids en todas, sin fallback al vector search) |
| 9 | Enriquecimiento con vecinos 1-hop y endpoints | **Resuelta en codigo + observable per-query** | `retrieval_metadata.kg_entities_with_neighbors` y `kg_mean_neighbors_per_entity` por query (JSON + CSV detail). Validacion empirica: canal operando en 100% de queries en el cierre de Pre-P0 |
| 10 | High-level keywords por chunk durante indexacion | **Presencia validada; calidad NO validada** | Canal arquitectonicamente presente: `retrieval_metadata.kg_chunk_keyword_matches > 0` en 35/35 queries. ⚠️ **Riesgo conocido de piggyback**: el paper hace una llamada LLM dedicada por chunk; aqui las keywords se emiten en la misma llamada que entities/relations para ahorrar ~50% de coste de indexacion. Coste teorico: el LLM puede emitir keywords genericas ("event", "person") en vez de tematicas reales. El observable actual mide presencia, no calidad. **Bloqueante antes de P2** (catalogo especializado donde el embedding no satura), no bloqueante para P0 (HotpotQA). Accion pendiente antes de P2: anadir observable de calidad de keywords (mean por doc, diversidad Jaccard, ratio generico/especifico) y/o exponer toggle `KG_CHUNK_KEYWORDS_DEDICATED_CALL=true` |

### Detalle del riesgo de piggyback en divergencia #10

La extraccion de high-level keywords por chunk se implementa como **piggyback en la misma llamada LLM que entities/relations**, no como llamada dedicada como en el paper. Motivacion: ahorrar ~50% de llamadas LLM en indexacion. **Coste teorico**: cuando un prompt pide tres outputs simultaneos (entities + relations + high_level_keywords), los outputs concretos dominan y los temas abstractos (keywords) tienden a degradarse a terminos genericos ("event", "person", "organization") en vez de temas especificos del chunk. **Lo que el observable actual NO captura**: `kg_chunk_keyword_matches > 0` solo confirma matches semanticos en la VDB, no que las keywords representen los temas reales del chunk — un LLM que emita `["article", "document", "text"]` pasaria trivialmente este observable. **Cuando importa**: HotpotQA es ciego a esta degradacion porque el canal vector directo satura el retrieval. En el escenario objetivo (P2, catalogo de 10-50 PDFs especializados donde las entidades son desconocidas para el embedding), el canal high-level es el unico que opera sobre conceptos que el embedding SI conoce; si piggyback lo degrada silenciosamente, rompemos la pata diferencial de LightRAG en su caso de uso objetivo. **Accion pendiente antes de P2** (no bloquea P0): (1) anadir observable de calidad de keywords (diversidad Jaccard intra-tema, ratio genericas/especificas via IDF intra-corpus); (2) si el observable muestra degradacion, exponer toggle `KG_CHUNK_KEYWORDS_DEDICATED_CALL=true`.

### Divergencias menores (cosmeticas / no funcionales)

- **#3 — Entity cap 100K**: eviction con score compuesto. Para HotpotQA (66K docs, ~24K entidades) no se alcanza.
- **#11 — Chunks en coleccion principal de ChromaDB, no en `text_chunks_vdb` dedicado**: Entity VDB y Relationship VDB si son colecciones separadas; los chunks residen en la coleccion principal de `ChromaVectorStore`. Post-#8 actua de facto como `text_chunks_vdb`. Separar formalmente seria naming sin efecto funcional. Re-evaluar si el schema de export hacia el administrador (P3) requiere distinguir VDBs por rol.
- **#12 — Formato de contexto JSON-lines, no CSV con headers del paper**: ambos son estructurados; JSON-lines tiene la ventaja de que la capa de synthesis (#2) usa `reference_id` para anclar citas `[ref:N]`. Cambiar a CSV romperia el esquema de citas sin beneficio.

## Deuda tecnica vigente

| # ID | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB: colecciones huerfanas si el proceso se interrumpe | **BAJO** | `evaluator.py:_cleanup()` borra la coleccion al terminar; si el proceso muere antes, queda `eval_*` en disco. Con `PersistentClient` se acumulan | Auditar `VECTOR_DB_DIR` entre campanas y purgar `eval_*` huerfanas; automatizar en `preflight.py` si el tamano supera presupuesto |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO; no descarga ni parsea Parquet, no valida schema contra `DATASET_CONFIG`. Riesgo real: **schema drift del contrato upstream** (administrador produce catalogo con columnas/tipos distintos) — fallo horas despues en `_populate_from_dataframes()`, quemando compute | `--dry-run` primero; cerrar con contract testing al integrar con administrador (P3) |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, fallback 4000 chars (presupuesto TOTAL de contexto, no truncado del doc; ~4-8 chunks). Riesgo: dejar senal en la mesa si el modelo soporta mucho mas (p.ej. 192K). Loguea INFO (no WARNING) → puede pasar desapercibido | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicito en `.env`. Auditar `config_snapshot._runtime.max_context_chars` antes de aceptar un run |
| 8 | Infraestructura pesada para el scope | **BAJO** | Para 1 dataset y 2 estrategias, checkpoint/preflight/JSONL/export dual/DEV_MODE es mucho. Componentes ejercitados sin incidentes en validacion previa | Revisar post-P0: si algun componente queda sin uso tras 3 runs reales, candidato a eliminacion |
| 9 | Lock-in a NVIDIA NIM | **BAJO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |
| 10 | Sin indexacion incremental del KG | **MEDIO** | `LightRAGRetriever.index_documents()` siempre rebuild completo o carga de cache. No hay `append_documents()` para integrar docs nuevos sin reconstruir. El paper soporta `insert()` incremental. Para PDFs en tandas (escenario administrador), obliga a re-indexar todo | Cache de disco mitiga re-extraccion LLM pero no rebuild del grafo ni VDBs. P3 lo requiere |
| 11 | Duplicacion parcial de iteracion de vecinos en KG | **BAJO** | `_get_neighbors_weighted()` y `get_neighbors_ranked()` (div #9) iteran los mismos edges con campos distintos (uno extrae `edge_weight`, el otro ademas `degree_centrality` + relacion). Duplicacion deliberada para no complicar la API interna | Refactorizar a iterador comun solo si aparece un tercer consumidor |
| 12 | Tests acoplados a mensajes de error de `AsyncLLMService` | **BAJO** | Tests en `test_llm.py` usan regex laxa sobre `RuntimeError` porque no hay excepciones custom (`EmptyResponseError`/`RetriesExhaustedError`). Si cambia el texto, los tests fallan por regex, no por comportamiento | Refactor de excepciones en el proximo PR funcional a `shared/llm.py`. Mientras tanto, sincronizar regex al tocar mensajes |
| 14 | Acceso a `_vector_store.collection_name` desde `LightRAGRetriever` | **BAJO** | 3 builders en `retriever.py` (entities/relationships/chunk_keywords VDB) leen el atributo privado para derivar nombres. Si se renombra sin buscar refs externas, rompen en runtime al indexar. Sin test que lo detecte | Exponer property `collection_name` en `SimpleVectorRetriever` al proximo cambio en esa clase, o al aparecer cuarto consumidor |
| 17 | Parametros fijos del canal de chunk keywords (div #10) | **BAJO** | 3 parametros hardcoded en `triplet_extractor.py`/`retriever.py`: `MAX_CHUNK_KEYWORDS_PER_DOC=10`, `MIN/MAX_CHUNK_KEYWORD_LEN=2/80`, `_CHUNK_KEYWORDS_VDB_MAX_DISTANCE=0.8`. Solo `KG_CHUNK_KEYWORDS_ENABLED`/`_TOP_K` expuestos | Exponer al `.env` solo si un run real demuestra que algun caso cae fuera de defaults |
| 18 | Observable de citaciones #7 acoplado al prompt de synthesis | **BAJO** | Parser usa regex `\[ref:(\d+)\]` alineado con `KG_SYNTHESIS_SYSTEM_PROMPT`. Si alguien cambia formato (p.ej. `(ref N)` o `[ref:3,4,5]`), los 14 campos dejan de medir lo que dicen. Acoplamiento semantico, sin test automatico | Al tocar `KG_SYNTHESIS_SYSTEM_PROMPT`, revisar parser y actualizar regex `_VALID_RE`/`_CANDIDATE_RE` en sync |

**Items resueltos (historico)**: #13 (guard de reranker para LIGHT_RAG), #15 (per-query `retrieval_metadata` en exports JSON/CSV unificados), #16 (timeouts de synthesis en corpus pequeno: instrumentacion timing + config calibrada con `NIM_MAX_CONCURRENT_REQUESTS=32`, `KG_SYNTHESIS_MAX_CHARS=50000`, `KG_SYNTHESIS_TIMEOUT_S=180`).

## Observabilidad de runs

Los `EvaluationRun` exportados a JSON incluyen en `config_snapshot._runtime` dos bloques de stats para auditoria post-run:

**`judge_fallback_stats`**: solo aparecen las `MetricType` del judge **efectivamente invocadas**. Por metrica: `invocations`, `parse_failures`, `default_returns` (0.5), `parse_failure_rate`, `default_return_rate`. Si `default_return_rate > JUDGE_FALLBACK_THRESHOLD` (default 2%) en cualquier metrica invocada, el run falla con `RuntimeError`. Que una metrica no aparezca no es bug — significa no invocada; auditar `primary_metric_type`/`secondary_metrics` en `query_results`.

```bash
jq '.config_snapshot._runtime.judge_fallback_stats' <run_export.json>
```

**`kg_synthesis_stats`** (solo `LIGHT_RAG` con `KG_SYNTHESIS_ENABLED=true`): contadores (`invocations`, `successes`, `errors`, `empty_returns`, `truncations`, `timeouts`, `fallback_rate`) + timing per-invocacion para discriminar causas: `p50/p95/max_total_ms` (wall-clock completo), `p50/p95/max_queue_ms` (espera del semaforo `NIM_MAX_CONCURRENT_REQUESTS`), `p50/p95/max_llm_ms` (llamada al NIM tras acquire). `n_total_samples`/`n_queue_samples`/`n_llm_samples` reportan cuantas invocaciones contribuyeron — queue/llm pueden tener menos cuando un timeout cancelo la coroutine antes. **Umbrales**: `fallback_rate > 10%` bloquea. **Diagnostico**: `p50_queue_ms` alto → subir `NIM_MAX_CONCURRENT_REQUESTS` o bajar paralelismo; `p50_llm_ms` alto → reducir `KG_SYNTHESIS_MAX_CHARS` o subir `KG_SYNTHESIS_TIMEOUT_S`.

```bash
jq '.config_snapshot._runtime.kg_synthesis_stats' <run_export.json>
```

Ambos tambien se emiten en el evento estructurado `run_complete` del JSONL y en logs INFO al final de cada run.

**`citation_refs_{synth,gen}_*` (div #7, 14 campos per-query)**: solo LIGHT_RAG con synthesis activa. Parser `shared/citation_parser.py::parse_citation_refs` invocado dos veces en `_process_single_async`: sobre narrativa synth y sobre respuesta final, cada una con `n_chunks_emitted` como rango valido. 7 contadores por prefijo: `total`, `valid` (formato estricto `[ref:N]`), `malformed` (variantes), `in_range`, `out_of_range` (**senal roja**), `distinct`, `coverage_ratio`.

**Interpretacion cruzada faithfulness × citations**: el judge de faithfulness y los observables de citacion miden cosas distintas. La matriz discriminante es:

| Faithfulness | Citations (`out_of_range`, `valid`) | Diagnostico |
|---|---|---|
| Alta | `out_of_range = 0`, `valid > 0` | **Narrativa ideal**: respuesta consistente con contexto + anclaje trazable per-claim |
| Alta | `out_of_range > 0` | Respuesta OK pero alucinacion de referencias — revisar prompt del synth |
| Alta | `valid = 0, malformed = 0, total = 0` | Respuesta OK pero LLM ignora la instruccion de citar — revisar prompt del synth |
| Baja | `out_of_range = 0` | LLM invento contenido pero cito correctamente — inutil: las citas apuntan a algo real pero lo que dice no viene de ahi |
| Baja | `out_of_range > 0` | Peor caso: contenido fabricado Y citas fabricadas juntas. Alarma maxima |
| Cualquiera | `gen_valid > synth_valid` | Generador invento citas sobre la narrativa — separar del riesgo #2 (synth) |
| Cualquiera | `malformed > 0` en el run | Problema de prompt/modelo con formato — revisar `KG_SYNTHESIS_SYSTEM_PROMPT` o cambiar modelo |

`out_of_range > 0` o `malformed > 0` agregados a nivel de run son **senales rojas equivalentes a `judge.default_return_rate > 2%`**: el run sigue ejecutando pero el observable discriminativo esta comprometido antes de interpretar deltas.

## Bare excepts tolerados (con criterio)

Estos `except Exception as e:` logean el error y devuelven un fallback en vez de re-lanzar. El criterio para tolerarlos es que esten en wrappers de infraestructura donde el run debe continuar ante errores operacionales puntuales (ChromaDB transitorio, NIM latencia), cada uno contabiliza el evento en stats (`kg_synthesis_stats`, `rerank_failures`) o loguea a `logger.warning`/`debug` para trazabilidad post-mortem, y el fallback es observable desde el JSON del run. Si un bare except no cumple las tres condiciones, es candidato a reclasificar:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:126, 142, 179, 232, 247` | Operaciones ChromaDB — retorna fallback (lista vacia, dict vacio, o continua cleanup) |
| `generation_executor.py` (`_synthesize_kg_context_async`) | `asyncio.TimeoutError` + `Exception` genericos durante synthesis KG — fallback al contexto estructurado. Todos los eventos contabilizados en `kg_synthesis_stats` (errors/timeouts) |

## Test coverage

| Metrica | Valor orientativo |
|---|---|
| Tests unitarios | **~465 pasan**, 6 skipped con `python-igraph` + `snowballstemmer` instalados; **~392 pasan**, 7 skipped sin igraph (los tests que lo requieren se saltan limpiamente). Cifras orientativas — drift con cada PR; para el valor exacto del entorno: `pytest --collect-only -q tests/ \| tail -1` y `pytest -m "not integration" tests/`. Ultimas adiciones reseñables: `test_judge_fallback_tracker.py` (deuda #4), `test_kg_synthesis.py` (divergencia #2) |
| Tests integracion | **~15** en 3 archivos (`tests/integration/`), requieren NIM + MinIO reales. Cifra orientativa — verificar via `pytest --collect-only -q tests/integration/` (solo si `.env` esta configurado) |
| mypy | 0 errores nuevos en ficheros modificados; 3 errores preexistentes no relacionados (dotenv/numpy sin stubs, `retrieval_executor.py:124` union-attr) |

### Portabilidad de tests

`conftest.py` mockea modulos de infra (dotenv, boto3, langchain, chromadb) si no estan instalados. `test_knowledge_graph.py` usa `pytest.importorskip("igraph")` para skip limpio sin igraph. Dependencias opcionales para suite completa: `python-igraph`, `snowballstemmer`.

**Referencia completa**: ver `TESTS.md` — mapa test→produccion, atributos `object.__new__()`, trampas de mock, gaps de cobertura, reglas de modificacion.

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio (DTm-45). Parece complejo pero es necesario

## Limitaciones de claude_code sobre este proyecto

**Encuadre estructural — claude_code no tiene acceso a la infraestructura del usuario.** NIM + MinIO viven en el entorno privado del usuario (deuda #9). Solo el usuario puede replicar la infra y ejecutar runs reales. claude_code nunca ve `kg_synthesis_stats`, `judge_fallback_stats` ni `retrieval_metadata` reales antes de declarar un cambio como completo — todo lo que entrega es **hipotesis pendiente de verificacion humana**. Cada iteracion fallida son horas perdidas: un run LIGHT_RAG completo consume ~1h30min y presupuesto de NIM; un parametro mal capeado o un test que no estresa el edge case real solo se descubre cuando las metricas ya no son interpretables.

**Implicaciones operativas**:
- Cambios al flujo retrieval→synthesis→generation solo se validan cuando el usuario lanza el run.
- Antes de declarar completo, claude_code debe enumerar los **criterios observables** que el usuario comprobara: que variable de `config_snapshot._runtime` esperar, que rango en `kg_synthesis_stats`/`judge_fallback_stats`, que metrica agregada.
- "Falta test e2e de X" NO es deuda pendiente — es consecuencia estructural. Listarla induce intentos de construir algo imposible de validar en sesion. 

**Contramedidas de Claude_code (vehiculos complementarios pre-run; ninguno equivale a validar)**:
1. **Auto-evaluacion como parte del entregable**: enumerar limitaciones y criterios observables ANTES de entregar, no despues.
2. **Distinguir "menor" de "conveniente de no arreglar"**: si la razon para clasificarlo como menor es el coste de arreglarlo, es deuda real.
3. **Parametros que dependen del LLM usado** (timeouts, contexto, concurrencia) van al `.env`, no a `constants.py`. `constants.py` solo para cosas que nunca deberian tocarse (p.ej. `CHARS_PER_TOKEN`).
4. **Tests unitarios adversariales** para todo cambio que afecte el contexto que ve el LLM: estresar el budget, no solo el caso holgado.

## Proximos pasos

### Orden de prioridades

```
Pre-P0 (completitud arquitectural + ejecucion estable)  <-- GATE CERRADO (2026-04-19)
  |
P0 (replicacion empirica del paper)                     <-- FASE ACTUAL
  |
  +-- P2 (experimento 3: catalogo especializado)        <-- SOLO si P0 pasa + riesgo piggyback #10 resuelto
  |
  +-- P3 (embedibilidad + export KG + integracion)      <-- SOLO si P2 pasa
```

** Alerta sobre Pre-P0 cerrado con matiz**: El piggyback en la extraccion de keywords high-level puede degradar la calidad del canal **exactamente en el escenario donde LightRAG deberia brillar** (catalogo especializado, dominio fuera del pre-entrenamiento del embedding). HotpotQA no discrimina este riesgo por saturacion del vector directo. **Antes de lanzar P2 — no P0 — hace falta anadir observable de calidad de keywords y/o toggle a llamada dedicada.**

### Pre-P0 — Completitud arquitectural de LIGHT_RAG · **GATE CERRADO (2026-04-19)**

Tres condiciones cumplidas simultaneamente:

1. **Arquitectonicamente completa**: las 7 divergencias (#2, #4+5, #6, #7, #8, #9, #10) resueltas en codigo + tests. Validacion directa para #2, #6, #8; ejercitadas implicitamente para #4+5, #7, #9; presencia validada (calidad NO) para #10. Detalle por divergencia en la tabla "Status de validacion".
2. **Ejecucion estable**: `kg_synthesis_stats.fallback_rate = 2.86%` (< 10% umbral), `judge.default_return_rate = 0%`, `retrieval_metadata.kg_fallback=null` en 35/35 queries, `queries_failed = 0`.
3. **Funcionalidades extra documentadas**: cache de KG, fallbacks ante errores LLM/igraph, instrumentacion de timing (queue/LLM split) — adaptaciones operativas, no sustitutos de piezas del paper.

**Config validada** para runs LIGHT_RAG en infra actual (defaults en `sandbox_mteb/env.example`, marcados `[PRE-P0 VALIDATED]`):
- `NIM_MAX_CONCURRENT_REQUESTS=32`
- `KG_SYNTHESIS_MAX_CHARS=50000`
- `KG_SYNTHESIS_TIMEOUT_S=180`

**Palanca post-Pre-P0**: `KG_GLEANING_ROUNDS` (default `0`) ejecuta una pasada extra de extraccion para recuperar entidades/relaciones perdidas (no re-extrae keywords de chunk). Coste: ~2x llamadas LLM en indexacion. Usar solo si la cobertura del KG (`num_docs_with_entities / total_docs`) baja de ~95%.


### P0 — Demostrar calidad de generacion · **FASE ACTUAL**

**Objetivo**: demostrar que sobre al menos un benchmark donde una run reporta `LIGHT_RAG > baseline vector`, nuestra implementacion reproduce la **direccion** del delta (magnitudes exactas son secundarias; el signo y su significancia sobre el ruido es lo que importa).

**Estado**: desbloqueado tras cierre de Pre-P0 el 2026-04-19. La arquitectura esta completa y ejecuta estable; el siguiente trabajo es seleccionar benchmark y correr la comparativa.

**Prerequisitos arquitectonicos**: todos cumplidos para P0 sobre HotpotQA. **Para avanzar a P2 (catalogo especializado) hace falta adicionalmente cerrar el riesgo de piggyback en #10** — anadir observable de calidad de keywords y/o exponer toggle a llamada dedicada.

Ninguno esta en formato MTEB/BeIR nativo; todos requieren ETL propio al contrato MinIO/Parquet de `loader.py`.

**Trabajo necesario**:
1. **Comparativa SIMPLE_VECTOR vs LIGHT_RAG hybrid** con synthesis on.

**Criterio de exito**: delta `LIGHT_RAG > SIMPLE_VECTOR` en la metrica principal del benchmark, distinguible del ruido (seed×LLM), con signo consistente con el paper.

**Criterio de fallo**: deltas dentro del ruido o invertidos → debug (¿synthesis llega al generador? ¿KG se construye? ¿indexacion falla silenciosamente?), no avance a P2/P3.

### P2 — Experimento 3: catalogo especializado · **futuro, contingente a P0**

**No empezar sin P0 verde.** Sobre un catalogo privado de 10-50 PDFs especializados (upstream del sistema administrador), LIGHT_RAG vs SIMPLE_VECTOR se evalua en dos ejes:

- **Eje 1 — Precision/calidad**: se espera delta LIGHT_RAG > SIMPLE_VECTOR >3-5pp en gen score, >5-10pp en Recall@K, porque el embedding no ha visto el dominio y el KG se construye del propio corpus.
- **Eje 2 — Resistencia a alucinacion**: el KG aporta grounding explicito que deberia reducir la tasa de fabricacion cuando el retrieval devuelve chunks poco relevantes. Se mide via `faithfulness` (LLM-judge); la fiabilidad depende de que `judge_fallback_stats` reporte tasas sanas (ver "Observabilidad de runs").

**Criterio de decision**: validar cualquiera de los dos ejes (idealmente ambos) → LIGHT_RAG pasa a estrategia default. Ninguno → reconsiderar el rol de LIGHT_RAG en el producto.

**Bloqueado por**: P0 verde + disponibilidad del catalogo upstream.

### P3 — Embedibilidad + export de KG + integracion · **futuro lejano, contingente a P2**

**No empezar sin P0 y P2 verdes.** Trabajo de producto, no de investigacion:

- **Embedibilidad**: configuracion via dict inyectado (no solo `.env` global), corpus en memoria, separar "cargar/indexar" de "evaluar" para reusar indices entre runs, sin asunciones sobre el filesystem excepto `EVALUATION_RESULTS_DIR` explicito.
- **Export de KG a MinIO/Parquet**: serializador `KnowledgeGraph` → Parquet (nodos + aristas + pesos + metadatos de co-ocurrencia) + VDBs. Schema a acordar con el administrador. Hoy el KG es efimero (igraph + ChromaDB en memoria, descartado en `_cleanup()`). Sin export, multi-tenant y versionado son imposibles.
- **Contract testing con el administrador**: validar el schema Parquet upstream contra `DATASET_CONFIG` desde preflight (cierra deuda #2).

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`; deuda #3)
- Lock-in a NVIDIA NIM (deuda #9) — solo reproducible con acceso a NIM
