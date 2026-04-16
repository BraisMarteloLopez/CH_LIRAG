# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Contexto del producto

Este proyecto es un subsistema de evaluacion RAG, no un producto final. Su proposito **inmediato** es validar empiricamente que nuestra implementacion de LIGHT_RAG replica las mejoras reportadas en el paper original ([HKUDS/LightRAG, EMNLP 2025](https://arxiv.org/abs/2410.05779)). Las cuatro divergencias arquitectonicas con el paper estan resueltas en codigo, pero aun no demostradas empiricamente (HotpotQA no discrimina; ver "Proximos pasos · P0"). Sin esa demostracion no se puede promover LIGHT_RAG a produccion ni proponerlo al sistema administrador.

**Vision a largo plazo — sistema administrador**: eventualmente este subsistema se integrara dentro de un sistema mas amplio cuya mision es administrar colecciones de datos, orquestar el ciclo de vida de corpus, versionado de KGs, consultas multi-tenant y APIs de uso. El administrador compartira infraestructura con este subsistema (MinIO + Parquet como contrato), asi que la integracion no implica cambiar como consumimos datos, solo apuntar a un prefijo MinIO distinto. **La integracion esta condicionada a que P0 (replicacion del paper) cierre con exito**; si no replicamos, lo unico integrable es SIMPLE_VECTOR y el trabajo sobre KG se vuelve inutil. Trabajo concreto en "Proximos pasos · P3".

**Implicacion de diseno**: las decisiones estructurales favorecen la embedibilidad futura — configuracion declarativa, interfaces claras, sin side-effects globales, capacidad de operar sobre corpus arbitrarios. **El valor de este subsistema no es resolver HotpotQA, es producir metricas fiables sobre cualquier corpus que el administrador le entregue.** Pero mientras P0 no este verde, la embedibilidad es solo un objetivo de diseno, no trabajo activo.

**Escenario de uso esperado (no confundir con el experimento 3)**: colecciones pequenas (10-50 PDFs) de dominio especializado, no publico, con idiosincrasia propia — terminologia tecnica, entidades internas, relaciones que no estan en el pre-entrenamiento de los embeddings. Este es el escenario tipico del producto a largo plazo. El experimento 3 (P2) es la prueba empirica concreta de ese escenario; el marco general queda aqui.

**Export de KG — proposito del futuro serializador**: cuando LIGHT_RAG sea estrategia de produccion (post-P0), los KGs deberan persistirse **para tres usos concretos del administrador**: versionado (distintas revisiones del mismo corpus), reuso entre runs (no re-indexar cada vez que entra una query nueva) y consultas multi-tenant (un mismo KG servido a varios consumidores). Detalle tecnico del serializador en "Proximos pasos · P3".

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

**Retrieval (⚠ divergente del paper, ver divergencia #8)**: el vector search query↔chunks produce el ranking de documentos (heredado de `SimpleVectorRetriever`). En modos `local`/`global`/`hybrid`, query keywords via LLM (dedup automatico, cap 20 keywords/nivel) resuelven entidades y relaciones del KG via sus VDBs y se adjuntan como secciones separadas en `retrieval_metadata.kg_entities`/`kg_relations`. Sin fusion RRF — cada canal es independiente. **Importante**: los chunks que entran al contexto NO se derivan del KG; se obtienen por similitud semantica directa query↔chunk. En el paper original los chunks vienen *a traves* del KG (via `source_doc_ids` de entidades y relaciones resueltas). Ver divergencia #8.

**Modos** (`LIGHTRAG_MODE`): `naive` (solo chunks), `local` (entidades + chunks), `global` (relaciones + chunks), `hybrid` (default, entidades + relaciones + chunks). Todos los modos (excepto naive) presentan secciones separadas al LLM.

**Synthesis del contexto**: para queries con datos KG, `GenerationExecutor._synthesize_kg_context_async()` reescribe las 3 secciones como narrativa coherente via LLM *antes* de la generacion final. Detalle y estado en la tabla "Divergencias con el paper original · #2". Stats en `config_snapshot._runtime.kg_synthesis_stats` (ver "Observabilidad de runs"). Toggle `KG_SYNTHESIS_ENABLED` (default `true` para LIGHT_RAG).

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro. Fallos en la capa de synthesis → degrada al contexto estructurado (el run nunca se rompe).

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original (HKUDS/LightRAG, EMNLP 2025)](https://arxiv.org/abs/2410.05779). Las divergencias 4+5, 6, 7, 2 estan resueltas en codigo. **Queda abierta la divergencia #8 (estructural, 9/10)**: los chunks se obtienen por vector search directo, no via `source_doc_ids` del KG. La validacion empirica sobre un benchmark donde el paper muestra ventaja esta pendiente ("Proximos pasos · P0") y depende de la decision sobre #8.

### Divergencias arquitectonicas

| # | Divergencia | Criticidad | Detalle |
|---|---|---|---|
| 4+5 | ~~**Pipeline de consumo diverge del paper**~~ | ~~9/10~~ | **Resuelto.** RRF eliminado (`reciprocal_rank_fusion()`, `_full_fusion()`, `_vector_first_fusion()`, `_fuse_with_graph()` eliminados). `_enrich_with_graph()` recopila entidades y relaciones relevantes a la query via VDB y las propaga en `retrieval_metadata` como `kg_entities`/`kg_relations`. `generation_executor.py` invoca `format_structured_context()` que presenta secciones separadas al LLM. `graph_primary` (DAM-3) eliminado. |
| 6 | ~~**Reranker post-fusion anula senal del grafo**~~ | ~~8/10~~ | **Resuelto.** El `CrossEncoderReranker` ahora se desactiva automaticamente cuando `strategy=LIGHT_RAG` (`retrieval_executor.py:100-105`). Se mantiene activo para `SIMPLE_VECTOR`. |
| 7 | ~~**Sin token budgets separados por tipo**~~ | ~~5/10~~ | **Resuelto.** `format_structured_context()` en `retrieval_executor.py` ahora divide `max_context_chars` en presupuestos proporcionales segun el modo LightRAG: `hybrid` (20%/20%/60%), `local` (30%/0%/70%), `global` (0%/30%/70%). Budget no usado por KG se redistribuye a chunks, evitando que ninguna seccion aplaste a las otras. `generation_executor.py` propaga `lightrag_mode` desde `retrieval_metadata`. |
| 2 | ~~**Sin LLM synthesis en fusion final**~~ | ~~7/10~~ | **Resuelto.** `GenerationExecutor._synthesize_kg_context_async()` reescribe el contexto multi-seccion (entidades + relaciones + chunks) como narrativa coherente via LLM antes de la generacion final. Prompt query-aware en `sandbox_mteb/config.py:KG_SYNTHESIS_SYSTEM_PROMPT` con reglas anti-fabricacion y citas `[ref:N]` inline. Se activa automaticamente para LIGHT_RAG (`KG_SYNTHESIS_ENABLED=true` default). **Faithfulness se evalua contra el contexto estructurado original**, no contra la narrativa, para penalizar cualquier alucinacion introducida por la propia capa de synthesis. Degradacion graceful: error LLM / vacio / timeout → fallback al contexto estructurado; stats por evento en `config_snapshot._runtime.kg_synthesis_stats`. |
| 8 | **Chunks via vector search directo, no via `source_doc_ids` del KG** | **9/10** | **ABIERTA.** En el paper original (HKUDS/LightRAG `operate.py`: `_find_most_related_text_unit_from_entities` y `_find_related_text_unit_from_relationships`), los chunks que entran al contexto se obtienen **a traves del KG**: low-level keywords → `entities_vdb` → entidades → sus `source_doc_ids` guardados al indexar → esos son los chunks. Mismo patron para high-level keywords → relaciones → chunks. **No hay similarity search query↔chunk independiente**. <br><br>En esta implementacion (`shared/retrieval/lightrag/retriever.py:660-668` y `:689-700`): <br>`LightRAGRetriever.retrieve()` → `self._vector_retriever.retrieve(query, top_k=k)` (similitud query↔chunks pura, heredada de SimpleVectorRetriever) → ESTO determina el ranking de chunks. Luego `_enrich_with_graph()` anade entidades/relaciones relevantes a `retrieval_metadata` **sin tocar el ranking**. El ranking de chunks es estructuralmente identico al de SIMPLE_VECTOR. <br><br>**Consecuencia empirica (F.5, pre y post refactor)**: las metricas de retrieval (Hit@K, MRR, Recall@K, NDCG) salen identicas query-por-query entre SIMPLE_VECTOR y LIGHT_RAG porque ambas estrategias comparten el motor de ranking. No es saturacion de HotpotQA ni ruido — es causal. Mientras #8 siga abierta, **ningun dataset puede producir deltas de retrieval** entre estrategias: el KG puede mover metricas de generacion (via secciones separadas + synthesis) pero no las de retrieval. <br><br>**Consecuencia para P0 (replicacion empirica del paper)**: si el benchmark elegido evalua principalmente metricas de retrieval sobre chunks, #8 impide replicar la ventaja reportada por construccion — no por fallo de implementacion. Si el benchmark evalua generacion (QA con respuesta textual), #8 no bloquea pero tampoco aprovecha completamente la contribucion del KG descrita en el paper. Hay que decidir #8 **antes** de elegir benchmark para P0. <br><br>**Codigo heredado de la ruta paper-aligned, presente pero no conectado**: ver seccion "Codigo heredado no conectado" mas abajo. Son 4 metodos + varios indices en `KnowledgeGraph` que implementarian la ruta del paper; hoy estan definidos, testeados y pueblados al indexar, pero ninguna ruta de retrieval de produccion los llama. <br><br>**Opciones de resolucion** (decision estrategica, no tomar sin discusion): <br>• **A — Alineacion total con el paper**: chunks solo via KG. `LightRAGRetriever.retrieve()` usa `query_entities` / `query_by_keywords` (o VDB + `source_doc_ids`) para producir el ranking. Fallback a vector search solo cuando el KG devuelve vacio, logueado y contabilizado. Separa estructuralmente LIGHT_RAG de SIMPLE_VECTOR — las metricas de retrieval pueden por fin diverger. <br>• **B — Fusion explicita de dos rankings**: mantener el ranking vectorial + producir un segundo ranking KG-derived (union de chunks de entidades + relaciones resueltas) + fusion declarada (RRF o weighted). Revive parte de lo eliminado en el fix #4+5 pero con razon arquitectonica distinta: es fusion de dos rankings de chunks, no mezcla ranking+metadata. Requiere justificar pesos. <br>• **C — Asumir la divergencia y restringir P0 a eje generacion**: documentar que este proyecto no pretende replicar el flujo chunks-via-KG; LIGHT_RAG aqui es "vector search + KG como contexto complementario + synthesis". P0 mide solo generacion y faithfulness. Coste: imposible replicar deltas de retrieval del paper. <br><br>**Para claude_code (desarrollador)**: si tocas `LightRAGRetriever.retrieve()`, piensa cual de A/B/C esta activo antes de anadir codigo. No intentes "arreglar" la aparente redundancia entre el path vectorial y los metodos `query_by_*` del `KnowledgeGraph` — es la divergencia #8, no dead code accidental. Consulta con el dueno antes de cambiar el path de obtencion de chunks. |

### Divergencias menores (preexistentes)

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 3 | Entity cap 100K | **3/10** | Eviction mejorada con score compuesto. Para HotpotQA (66K docs, ~24K entidades) no se alcanza. |

### Resueltas (indexacion)

- ~~DAM-4 (7/10)~~: LLM synthesis para descripciones → `KG_DESCRIPTION_SYNTHESIS=true` (A5.1)
- ~~Grafo fragmentado (3/10)~~: Co-occurrence bridging (DTm-73)
- ~~BFS scoring uniforme (3/10)~~: Edge weights via `log(1 + n_docs)` (DTm-72)

Runs F.5 (resultados empiricos pre-refactor y post-refactor) en "Proximos pasos · Resultado F.5".

## Deuda tecnica vigente

| # | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB: colecciones huerfanas si el proceso se interrumpe | **BAJO** | `evaluator.py:_cleanup()` ahora elimina la coleccion correctamente via `delete_all_documents()` (que llama `delete_collection()` + recrea). Sin embargo, cada run crea `eval_{run_id}` — si el proceso se interrumpe antes de cleanup, la coleccion queda huerfana. Con `PersistentClient`, se acumulan en disco | Aceptable; borrar manualmente `VECTOR_DB_DIR` si se acumulan |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO (`head_bucket` + `list_objects MaxKeys=1`). No descarga ni parsea Parquet, no valida schema contra `DATASET_CONFIG`, no verifica espacio en disco. El riesgo principal no es infra (MinIO ya es compartido con el administrador) sino **schema drift del contrato upstream**: cuando el administrador produzca un catalogo nuevo con columnas/tipos/ids diferentes, el fallo ocurre horas despues del start en `_populate_from_dataframes()`, quemando compute | `--dry-run` primero y verificar que el dataset carga |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 4 | ~~LLM Judge puede devolver scores por defecto~~ | **RESUELTO** (instrumentacion) | `_extract_score_fallback` devuelve 0.5 silenciosamente si ni JSON ni regex extraen score, sesgando metricas hacia el centro (especialmente faithfulness). Instrumentado con `_JudgeFallbackTracker` thread-safe en `shared/metrics.py`. APIs publicas: `get_judge_fallback_stats()`, `reset_judge_fallback_stats()`, `max_judge_default_return_rate()`. `_extract_score_fallback` refactorizado a variante con status `(score, was_default)` preservando la API publica existente. `_parse_judge_result` loguea WARNING con raw response (trunc 200) cuando se devuelve el default. El evaluator resetea stats al inicio del run, las propaga a `config_snapshot._runtime.judge_fallback_stats`, y aplica threshold check post-run via `JUDGE_FALLBACK_THRESHOLD` (default 2%): si alguna metrica supera el umbral, el run lanza `RuntimeError` y emite evento estructurado `run_judge_threshold_exceeded`. La mitigacion de la causa raiz (constrained decoding / judge mas capaz) sigue pendiente pero ya no silenciosa | N/A. Ver campos y comandos jq en "Observabilidad de runs" |
| 5 | Context window fallback silencioso | **BAJO** (casi aceptable) | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, usa fallback de 4000 chars (~1000 tokens). **Aclaracion importante**: este valor es el **presupuesto total de contexto** pasado al LLM, que `format_context()`/`format_structured_context()` usan para seleccionar cuantos fragmentos (chunks) caben. No es "el LLM recibe un documento truncado a 4000 chars". Con chunks tipicos de 500-1000 chars, 4000 = ~4-8 chunks relevantes, suficiente para casos tipo HotpotQA (2 docs gold) y para catalogos pequenos de PDFs especializados donde las respuestas suelen estar en 2-5 chunks. Se logea WARNING. El unico riesgo real es dejar senal en la mesa cuando el modelo soporta mucho mas (p.ej. 192K chars): no se "rompe" nada, simplemente no se aprovecha toda la ventana | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicitamente en `.env` si se quiere mayor cobertura |
| 6 | ~~Suite de tests no portable~~ | ~~CRITICO~~ | **Resuelto.** `conftest.py` ahora mockea `dotenv` (ademas de boto3/langchain/chromadb). `test_knowledge_graph.py` usa `pytest.importorskip("igraph")` para skip limpio sin igraph. Con `python-igraph` + `snowballstemmer` instalados: 409 pasan, 6 skipped. Sin igraph: 344 pasan, 65 skipped |
| 7 | ~~Validacion empirica pendiente post-refactor~~ | **RESUELTO** | F.5 re-ejecutado post-refactor (abril 2026) con los fixes aplicados. Ver tabla "Resultado F.5" en "Proximos pasos" para cifras. La siguiente validacion requiere cambiar de dataset, no mas fixes — ese es P0 | N/A |
| 8 | Infraestructura pesada para el scope | **BAJO** | Para 1 dataset y 2 estrategias, la infraestructura (checkpoint, preflight, JSONL, export dual, subset selection, DEV_MODE) es considerable. Sin embargo, el run F.5 demostro que esta infraestructura funciona y es util en practica | Aceptado — la infraestructura se justifica con uso real |
| 9 | Lock-in a NVIDIA NIM | **MEDIO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |

## Observabilidad de runs

Los `EvaluationRun` exportados a JSON incluyen en `config_snapshot._runtime` dos bloques de stats para auditoria post-run:

**`judge_fallback_stats`** (deuda #4): por cada `MetricType` del judge (`faithfulness`, `answer_relevance`) reporta `invocations`, `parse_failures` (JSON no parseable), `default_returns` (0.5 por defecto), `parse_failure_rate`, `default_return_rate`. Si `default_return_rate > JUDGE_FALLBACK_THRESHOLD` (default 2%) en cualquier metrica, el run falla con `RuntimeError`.

```bash
jq '.config_snapshot._runtime.judge_fallback_stats' data/results/<run_id>.json
```

**`kg_synthesis_stats`** (divergencia LightRAG #2): cuando `KG_SYNTHESIS_ENABLED=true` y la estrategia es `LIGHT_RAG`, reporta `invocations`, `successes`, `errors`, `empty_returns`, `truncations`, `timeouts`, `fallback_rate`. `fallback_rate > 10%` indica degradacion frecuente — el run no refleja la arquitectura completa.

```bash
jq '.config_snapshot._runtime.kg_synthesis_stats' data/results/<run_id>.json
```

Ambos tambien se emiten en el evento estructurado `run_complete` del JSONL y en logs INFO al final de cada run.

## Codigo heredado no conectado (ruta paper-aligned a medio construir)

Varias funciones y estructuras de `shared/retrieval/lightrag/knowledge_graph.py` implementan la ruta chunks-via-KG del paper pero **no estan conectadas al retriever actual** (ver divergencia #8). Estan definidas, testeadas y se pueblan al indexar; simplemente ningun path de retrieval de produccion las invoca. Se mantienen intencionadamente para facilitar la resolucion futura de la divergencia #8 (opcion A) sin reescribir desde cero.

| Pieza | Ubicacion | Proposito paper-aligned | Estado en produccion |
|---|---|---|---|
| `KnowledgeGraph.query_entities(entity_names, max_hops, max_docs, pre_resolved)` | `knowledge_graph.py:614` | BFS ponderado desde entidades query → ranking de `doc_ids` con scoring por hops + edge weight (DTm-72) + confidence. Es el equivalente a `_find_most_related_text_unit_from_entities` del paper | Definida, con tests (KG*), **no llamada desde `shared/` ni `sandbox_mteb/`** (sí aparece mockeada en `tests/test_dtm38_strategy_guardrail.py:122,144`, test sobre un path que ya no existe en produccion) |
| `KnowledgeGraph.query_by_keywords(keywords, max_docs)` | `knowledge_graph.py:674` | Indice invertido keyword → docs (DTm-30), match en nombres de entidad y descripciones de relacion. Complementa el path high-level del paper | Definida, con tests, **no llamada en produccion** |
| `KnowledgeGraph.get_entities_for_docs(doc_ids)` | `knowledge_graph.py:166` | Lookup inverso: dado un conjunto de chunks ya recuperados, devolver sus entidades KG (DAM-8). Util si el ranking se produce por vector search y se quiere cruzar con KG | Definida, con tests, **no llamada en produccion** |
| `KnowledgeGraph.get_relations_for_docs(doc_ids)` | `knowledge_graph.py:191` | Idem para relaciones | Definida, con tests, **no llamada en produccion** |
| `_doc_to_entities`, `_entity_to_docs`, `_doc_to_relations` | atributos de `KnowledgeGraph` | Indices bidireccionales doc↔KG. Se pueblan en `add_document()` al indexar | Mantenidos, **solo los consumen las 4 funciones de arriba** |
| Indice invertido de keywords (DTm-30) | `knowledge_graph.py:863` | Tokenizacion + stem para acelerar `query_by_keywords` | Construido al indexar, **solo lo consume `query_by_keywords`** |

**Para claude_code (desarrollador), reglas al tocar este codigo**:

1. **No borres estas funciones aunque el linter las marque como no usadas**. Son el puente hacia la opcion A de la divergencia #8. Borrarlas cuesta horas de reimplementacion si se decide cerrar #8.
2. **No anadas uso de estas funciones sin discutir la divergencia #8 primero**. Conectar `query_entities` al ranking cambia el significado de las metricas de retrieval del proyecto — es una decision estrategica, no una mejora mecanica.
3. **Si el benchmark de P0 exige replicar metricas de retrieval del paper, volver a esta seccion**. La decision A/B/C de #8 hay que tomarla antes de invertir compute en P0, no despues.
4. **No mezcles** `get_entities_for_docs` / `get_relations_for_docs` **con el flujo actual** `_enrich_with_graph` que usa los VDBs directamente. Son dos caminos distintos de enriquecimiento KG; combinarlos sin decision explicita duplica entidades y distorsiona los budgets de `format_structured_context`.
5. **Los tests** `test_knowledge_graph.py` **KG9, KG10, KG11, KG24-KG28, y los tests de `get_entities_for_docs` / `get_relations_for_docs`** siguen verdes porque prueban las funciones aisladas. Si se resuelve #8 y se conectan al retriever, habra que extender los tests con escenarios end-to-end.

## Bare excepts aceptados (no criticos)

Estos `except Exception as e:` logean el error pero no lo re-lanzan. Aceptable para wrappers de infraestructura:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:126, 142, 179, 232, 247` | Operaciones ChromaDB — retorna fallback (lista vacia, dict vacio, o continua cleanup) |
| `generation_executor.py` (`_synthesize_kg_context_async`) | `asyncio.TimeoutError` + `Exception` genericos durante synthesis KG — fallback al contexto estructurado. Todos los eventos contabilizados en `kg_synthesis_stats` (errors/timeouts) |

## Test coverage

| Metrica | Valor (abril 2026) |
|---|---|
| Tests unitarios | **~441 pasan**, 6 skipped en entorno con igraph+snowballstemmer. Sin igraph: **382 pasan**, 7 skipped (verificado). Ultimas adiciones: +19 `test_judge_fallback_tracker.py` (deuda #4) +13 `test_kg_synthesis.py` (divergencia #2) |
| Tests integracion | 19 en 3 archivos, requieren NIM + MinIO reales |
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

## Proximos pasos

### Orden de prioridades

```
P0 (replicacion del paper)                            <-- GATE activo, varias sesiones
  |
  +-- P1 (sanity on/off synthesis sobre HotpotQA)     <-- barato, en paralelo
  |
  +-- P2 (experimento 3: catalogo especializado)      <-- SOLO si P0 pasa
  |
  +-- P3 (embedibilidad + export KG + integracion)    <-- SOLO si P2 pasa
```

### Resultado F.5 (referencia historica)

F.5 se ejecuto dos veces sobre HotpotQA (125q, DEV_MODE, seed=42, reranker ON).

**F.5 pre-refactor** (divergencias #4+5/#6/#7 abiertas, 4000 docs):

| Metrica | SIMPLE_VECTOR | LIGHT_RAG | Delta |
|---|---|---|---|
| Hit@5 | 1.000 | 1.000 | 0 |
| MRR | 0.992 | 0.992 | 0 |
| Recall@5 | 0.968 | 0.968 | 0 |
| Recall@20 | 0.988 | 0.988 | 0 |
| Avg gen score | 0.7764 | 0.7877 | +0.0113 |
| Tiempo total | 194.7s | 9002.1s | ×46 |

Todas las metricas de retrieval identicas query por query. El KG aporto ~49 docs exclusivos por query, pero el reranker colapso el ranking final al mismo top-20 que SIMPLE_VECTOR. KG indexado correctamente (23K entidades, 55K relaciones, 32K co-occurrence edges), pero las divergencias #4+5 y #6 impedian que su senal llegara al LLM. Las 10 diferencias en generacion son no-determinismo del LLM con mismo contexto.

**F.5 post-refactor** (#4+5/#6/#7 resueltos, 2500 docs):

| Metrica | SIMPLE_VECTOR | LIGHT_RAG hybrid | Delta |
|---|---|---|---|
| Hit@5 / MRR | 1.000 / 1.000 | 1.000 / 1.000 | 0 (saturado) |
| Avg gen score | 0.8038 | 0.8157 | +0.0119 |
| Tiempo | 144s | 4589s | ×31.8 |

Delta pre → post se movio 0.6 decimas de porcentaje — dentro del ruido del LLM judge. **HotpotQA no discrimina** en generacion: Wikipedia en el pre-entrenamiento del embedding + DEV_MODE saturando gold docs + ventana 192K chars → el embedding resuelve por si solo sin necesitar el KG. Los fixes estan correctamente implementados (el KG se construye, las secciones llegan al LLM con budgets proporcionales, el reranker no colapsa el ranking); simplemente este dataset no es util para validar la arquitectura por el lado de la generacion.

**Nota estructural sobre metricas de retrieval identicas en F.5**: el que Hit@K / MRR / Recall@K / NDCG salieran identicos query-por-query en ambas pasadas de F.5 no es solo "HotpotQA no discrimina". Es consecuencia directa de la divergencia #8 (chunks via vector search directo, no via KG): las dos estrategias comparten literalmente el motor de ranking de chunks, asi que las metricas de retrieval son necesariamente iguales en cualquier dataset mientras #8 siga abierta. El efecto del dataset aplica a las metricas de generacion; el efecto de #8 aplica a las de retrieval. Son causas distintas que hasta ahora se confundian.

### P0 — Replicacion empirica del paper · **activo, bloqueante**

**Objetivo**: demostrar que sobre al menos un benchmark donde el paper reporta `LIGHT_RAG > baseline vector`, nuestra implementacion reproduce la **direccion** del delta (magnitudes exactas son secundarias; el signo y su significancia sobre el ruido es lo que importa).

**⚠ Prerequisito arquitectonico — decidir divergencia #8 antes de elegir benchmark**: mientras #8 siga abierta (chunks via vector search directo), las metricas de retrieval seran identicas entre estrategias en cualquier dataset. Si el benchmark elegido para P0 reporta deltas principalmente en retrieval (Hit@K, MRR, Recall@K sobre chunks), la replicacion es estructuralmente imposible sin cerrar #8 primero (opcion A o B). Si el benchmark reporta deltas en generacion (QA end-to-end), P0 se puede intentar con #8 abierta pero la contribucion del KG estara subutilizada. **Antes de invertir en ETL de benchmark, decidir A/B/C sobre #8** (ver tabla de divergencias).

**Candidatos de dataset**:
- UltraDomain subsets (Legal-QA, Agriculture, CS, Mix) — los del paper original
- Cualquier QA especializado con domain shift real al embedding

Ninguno esta en formato MTEB/BeIR nativo; todos requieren ETL propio al contrato MinIO/Parquet de `loader.py`.

**Trabajo necesario (varias sesiones adicionales de claude_code)**:
1. **Seleccion de benchmark** publico donde exista contra-referencia publicada (paper u otra fuente revisada) a favor de LightRAG/GraphRAG. Sin contra-referencia no hay "replicacion" que validar.
2. **ETL al contrato Parquet**: mapear queries, corpus, qrels; extender `DATASET_CONFIG` en `shared/types.py`.
3. **Protocolo experimental**: seed fijo, N>=3 runs por estrategia (mitiga deuda #3), reranker segun config del paper, metricas alineadas con lo que reporta el paper.
4. **Comparativa SIMPLE_VECTOR vs LIGHT_RAG hybrid** con synthesis on. Opcionalmente ablacion off para aislar la aportacion de la synthesis.
5. **Analisis**: validar `judge_fallback_stats` y `kg_synthesis_stats` antes de interpretar deltas (ver "Observabilidad de runs"). Si alguno degrada, los resultados no son interpretables.

**Criterio de exito**: delta `LIGHT_RAG > SIMPLE_VECTOR` en la metrica principal del benchmark, distinguible del ruido (seed×LLM), con signo consistente con el paper.

**Criterio de fallo**: deltas dentro del ruido o invertidos → debug (¿synthesis llega al generador? ¿KG se construye? ¿indexacion falla silenciosamente?), no avance a P2/P3.

### P1 — Sanity de synthesis sobre HotpotQA · barato, paralelo a P0

Dos controles gratuitos que no sustituyen P0 (por la razon de arriba):

1. **Ablacion synthesis on/off** (F.7): repetir F.5 con `KG_SYNTHESIS_ENABLED=true` y `false`. Objetivo: detectar regresion introducida por la capa de synthesis antes de invertir en P0.
2. **Full corpus sin DEV_MODE** (66K docs): senal intermedia sobre robustez del KG cuando el retrieval deja de saturar.

Coste: F.7 ~2-3h, full corpus ~4-6h.

### P2 — Experimento 3: catalogo especializado · **futuro, contingente a P0**

**No empezar sin P0 verde.** Sobre un catalogo privado de 10-50 PDFs especializados (upstream del sistema administrador), LIGHT_RAG vs SIMPLE_VECTOR se evalua en dos ejes:

- **Eje 1 — Precision/calidad**: se espera delta LIGHT_RAG > SIMPLE_VECTOR >3-5pp en gen score, >5-10pp en Recall@K, porque el embedding no ha visto el dominio y el KG se construye del propio corpus.
- **Eje 2 — Resistencia a alucinacion**: el KG aporta grounding explicito que deberia reducir la tasa de fabricacion cuando el retrieval devuelve chunks poco relevantes. Se mide via `faithfulness` (LLM-judge); la fiabilidad depende de que la instrumentacion de deuda #4 reporte tasas sanas.

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
