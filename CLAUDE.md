# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Contexto del producto

Subsistema de evaluacion RAG, no producto final. Dos fases secuenciales:

1. **Pre-P0 (CERRADA 2026-04-19)** — completitud arquitectural de LIGHT_RAG frente al paper ([HKUDS/LightRAG](https://github.com/HKUDS/LightRAG), EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)). Criterios de cierre en "Proximos pasos · Pre-P0".
2. **P0 (FASE ACTUAL)** — replicacion empirica: demostrar sobre un benchmark donde el paper reporte ventaja que esta implementacion reproduce la direccion del delta `LIGHT_RAG > SIMPLE_VECTOR`. Ver "Proximos pasos · P0".

Estado sumario: **las 7 divergencias arquitectonicas (#2, #4+5, #6, #7, #8, #9, #10) estan resueltas en codigo con tests unitarios y con observables empiricos per-query**. Unica excepcion cualitativa: la **calidad** del canal high-level del #10 (piggyback vs llamada dedicada del paper) — **bloqueante antes de P2, no de P0** (HotpotQA es ciego a esta degradacion por saturacion del vector directo). Detalle por divergencia en la tabla de la seccion "Divergencias".

**Vision a largo plazo — sistema administrador**: eventualmente este subsistema se integrara dentro de un sistema mas amplio cuya mision es administrar colecciones de datos, orquestar el ciclo de vida de corpus, versionado de KGs, consultas multi-tenant y APIs de uso. El administrador compartira infraestructura con este subsistema (MinIO + Parquet como contrato). **La integracion esta condicionada a que P0 cierre con exito**; si no replicamos, lo unico integrable es SIMPLE_VECTOR y el trabajo sobre KG se vuelve inutil. Detalle en "Proximos pasos · P3".

**Implicacion de diseno**: configuracion declarativa, interfaces claras, sin side-effects globales, capacidad de operar sobre corpus arbitrarios. **El valor de este subsistema no es resolver HotpotQA, es producir metricas fiables sobre cualquier corpus que el administrador le entregue.** Mientras P0 no este verde, la embedibilidad es objetivo de diseno, no trabajo activo.

**Escenario de uso esperado**: colecciones pequenas (10-50 PDFs) de dominio especializado, no publico, con terminologia tecnica y entidades fuera del pre-entrenamiento de los embeddings. El experimento P2 es la prueba empirica concreta de este escenario.

**Export de KG — proposito del futuro serializador**: cuando LIGHT_RAG sea estrategia de produccion, los KGs deberan persistirse para versionado, reuso entre runs y consultas multi-tenant. Detalle en "Proximos pasos · P3".

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
- **2 estrategias**: `SIMPLE_VECTOR` y `LIGHT_RAG` — no hay mas
- **Enum en core.py**: `RetrievalStrategy` define las estrategias validas. `VALID_STRATEGIES` en `sandbox_mteb/config.py` debe coincidir
- **Tests**: `conftest.py` mockea modulos de infra (boto3, langchain, chromadb) si no estan instalados. Tests de integracion requieren NIM + MinIO reales. Mocks siempre a nivel de funcion, nunca modulos enteros
- **Logging**: JSONL estructurado via `shared/structured_logging.py`. Bare excepts tienen `logger.debug(...)` — no hay excepts silenciosos
- **Idioma**: codigo y comentarios en ingles/espanol mezclado (historico). Docstrings y variables en ingles

## Estrategia LIGHT_RAG — como funciona

Implementa la arquitectura de [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)), con adaptaciones operativas propias del entorno (cache de KG a disco, fallbacks ante errores del LLM/igraph, instrumentacion de observabilidad).

**Indexacion**: LLM extrae tripletas (entidad, relacion, entidad) de cada doc + **high-level keywords tematicas del chunk** (divergencia #10, piggyback en la misma llamada LLM) → KnowledgeGraph in-memory (igraph) + ChromaDB para vector search. Entity VDB, Relationship VDB y Chunk Keywords VDB (tercera VDB, divergencia #10) para resolucion semantica. Stats se resetean automaticamente al inicio de cada batch. Gleaning opcional via `KG_GLEANING_ROUNDS` (no re-extrae keywords; solo entidades/relaciones perdidas).

**Retrieval (paper-aligned, divergencia #8 resuelta, #10 resuelta en implementacion con riesgo de piggyback documentado)**: en modos `local`/`global`/`hybrid`, los chunks se obtienen **a traves del KG** con tres canales:
— **Canal de entidades** (local/hybrid): query low-level keywords via LLM → Entity VDB → entidades resueltas → `source_doc_ids`.
— **Canal de relaciones** (global/hybrid): query high-level keywords → Relationship VDB → relaciones resueltas → `source_doc_ids` de los endpoints.
— **Canal de chunk keywords** (global/hybrid, divergencia #10): query high-level keywords → Chunk Keywords VDB → doc_ids directos.

Los tres canales suman al mismo `doc_scores` con formula simetrica `1/(1+rank) × similarity [× edge_weight]`. Contenido real se fetcha desde el vector store via `get_documents_by_ids`. Fallback a vector search directo cuando el KG no produce doc_ids (logueado en `metadata["kg_fallback"]`). Entidades y relaciones resueltas van en `retrieval_metadata.kg_entities`/`kg_relations` para la capa de synthesis; el conteo de matches del canal de chunk keywords va en `retrieval_metadata.kg_chunk_keyword_matches`. **Nota sobre scoring**: la formula usa `1/(1+rank)` como decay por posicion (inverse-rank), mientras el paper usa un contador descendiente lineal; la intencion es la misma pero la curva difiere. Las entidades y chunks no ponderan por weight (no hay peso equivalente almacenado); `len(source_doc_ids)` seria proxy viable para entidades pero el paper no lo hace explicitamente. Ver docstring de `_retrieve_via_kg` para detalle.

**Enriquecimiento KG (paper-aligned, divergencia #9 resuelta)**: cada entidad resuelta incluye sus vecinos 1-hop del grafo, rankeados por `edge_weight + degree_centrality` (`get_neighbors_ranked` en `knowledge_graph.py`), anidados como lista `"neighbors"` dentro del dict de entidad. Configurable via `KG_MAX_NEIGHBORS_PER_ENTITY` (default 5, 0 = desactivado). Cada relacion resuelta incluye las descripciones y tipos de sus entidades endpoint (`source_description`, `source_type`, `target_description`, `target_type`) obtenidos del KG. Degradacion graceful: si el lookup de vecinos o endpoints falla, la entidad/relacion aparece sin los campos extra.

**Modos** (`LIGHTRAG_MODE`): `naive` (solo chunks), `local` (entidades + chunks), `global` (relaciones + chunks), `hybrid` (default, entidades + relaciones + chunks). Todos los modos (excepto naive) presentan secciones separadas al LLM.

**Synthesis del contexto**: para queries con datos KG, `GenerationExecutor._synthesize_kg_context_async()` reescribe las 3 secciones como narrativa coherente via LLM *antes* de la generacion final. Prompt query-aware en `sandbox_mteb/config.py:KG_SYNTHESIS_SYSTEM_PROMPT` con reglas anti-fabricacion y citas `[ref:N]` inline. Se activa automaticamente para LIGHT_RAG (`KG_SYNTHESIS_ENABLED=true` default). **Faithfulness se evalua contra el contexto estructurado original**, no contra la narrativa, para penalizar cualquier alucinacion introducida por la propia capa de synthesis. Degradacion graceful: error LLM / vacio / timeout → fallback al contexto estructurado; stats por evento en `config_snapshot._runtime.kg_synthesis_stats` (ver "Observabilidad de runs"). **Nota sobre divergencia #2**: validacion externa (revisor independiente, abril 2026) confirma que esta capa no esta en el paper ni en `lightrag-hku`; es value-add del proyecto. La decision de medir faithfulness contra el contexto estructurado original (no contra la narrativa) es un control anti-fabricacion que el paper no tiene — penaliza cualquier alucinacion introducida por la propia synthesis.

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro. Fallos en la capa de synthesis → degrada al contexto estructurado (el run nunca se rompe).

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original](https://github.com/HKUDS/LightRAG) (HKUDS, EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)). Las divergencias #11 y #12 son menores (cosmeticas/no funcionales) con descripcion y criterio de re-evaluacion en su fila. La validacion empirica sobre un benchmark donde el paper muestra ventaja es el objetivo de P0 ("Proximos pasos").

### Status de validacion por divergencia arquitectonica

Tabla resumen del estado de cada divergencia arquitectonica frente al paper. **Validada end-to-end** significa que el run `mteb_hotpotqa_20260419_032905` expuso un observable concreto que confirma el comportamiento (flag en `retrieval_metadata`, stat en `kg_synthesis_stats`, o similar). **Resuelta en codigo** significa que el codigo esta implementado y cubierto por tests unitarios; en el run se ejercito implicitamente (sin observable dedicado). P0 sobre benchmark con contra-referencia validara implicitamente #4+5/#7/#9 si el delta reproduce la direccion del paper.

| # | Divergencia | Status | Evidencia empirica |
|---|---|---|---|
| 2 | Synthesis layer con citas `[ref:N]` | **Resuelta + Validada end-to-end** | `kg_synthesis_stats.fallback_rate=2.86%`, `kg_synthesis_used=True` en 34/35 queries |
| 4+5 | Budgets proporcionales entre secciones del contexto | **Resuelta en codigo + observable per-query** | Tests `test_structured_context.py` (+5 tests para `is_kg_budget_cap_triggered`) y `test_generation_executor.py` (+3 tests para la anotacion en `retrieval_metadata`). `retrieval_metadata.kg_budget_cap_triggered` discrimina per-query si el cap al 50% escalo las secciones KG (runs post-auditoria) |
| 6 | Reranker desactivado para LIGHT_RAG | **Resuelta + Validada end-to-end** | Fix #13 cerrado esta sesion; log linea 29 del run `20260419_032905` confirma `"Reranker habilitado en .env pero estrategia es LIGHT_RAG; omitiendo inicializacion"` |
| 7 | Contexto estructurado JSON-lines con `reference_id` | **Resuelta en codigo + observable per-query (synth validado, gen con hallazgo abierto)** | Tests `test_structured_context.py` + 13 tests unitarios en `test_citation_parser.py` + 7 tests de integracion en `test_generation_executor.py` (TestCitationRefsIntegration). Parser `shared/citation_parser.py::parse_citation_refs` emite 7 contadores (`total, valid, malformed, in_range, out_of_range, distinct, coverage_ratio`) aplicados a dos textos: narrativa synthesized (prefijo `citation_refs_synth_*`) y respuesta final (prefijo `citation_refs_gen_*`). Total 14 campos en `retrieval_metadata` cuando `has_kg_data AND kg_synthesis_enabled`. Senales rojas definidas: `out_of_range > 0` (alucinacion de referencias a chunks truncados o inventadas), `malformed > 0` (problema de prompt/modelo con el formato estricto), `gen_valid > synth_valid` (generador invento citas que no estaban en la narrativa). **Validacion empirica (run `mteb_hotpotqa_20260419_181645`, 35q, hybrid)**: los 14 campos poblados en 35/35 queries. Narrativa synth cito correctamente: `synth_valid=62 total`, `in_range=62`, `out_of_range=0`, `malformed=0`, `coverage_ratio=0.927` (alta diversidad), 33/35 queries con al menos una cita. **Hallazgo nuevo**: `citation_refs_gen_*=0` en 35/35 — el generador final NO propaga el formato `[ref:N]` de la narrativa synth a la respuesta que ve el usuario. Faithfulness no se ve afectada (se evalua contra el contexto estructurado, no contra la respuesta citada, y `judge.default_return_rate=0%`), pero se pierde trazabilidad inline para el usuario. Oportunidad de mejora pendiente antes de P0/P2 segun objetivo: instrumentar el prompt de `_execute_generation_async` para preservar `[ref:N]` si se quiere respuesta anclada. Requiere A/B porque cambiar el prompt del generador modifica comportamiento mas alla de las citas. **Nota de acoplamiento al prompt**: el formato `[ref:N]` se instruye en `KG_SYNTHESIS_SYSTEM_PROMPT` (`sandbox_mteb/config.py:309`); cualquier cambio a ese prompt exige revisar el parser y la interpretacion de sus metricas. |
| 8 | Chunks obtenidos via `source_doc_ids` del KG | **Resuelta + Validada end-to-end** | `retrieval_metadata.kg_fallback=null` en 35/35 queries (el KG produjo doc_ids en todas, sin fallback al vector search) |
| 9 | Enriquecimiento con vecinos 1-hop y endpoints | **Resuelta en codigo + observable per-query** | Tests `test_knowledge_graph.py` (`get_neighbors_ranked`) + 5 tests nuevos en `test_lightrag_fusion.py` para `_neighbor_coverage_stats` y su emision en los 4 return paths de `_retrieve_via_kg`. Observables `retrieval_metadata.kg_entities_with_neighbors` y `retrieval_metadata.kg_mean_neighbors_per_entity` disponibles en runs post-auditoria (JSON + CSV detail) |
| 10 | High-level keywords por chunk durante indexacion | **Resuelta en implementacion; calidad del output pendiente de verificacion empirica** | Canal arquitectonicamente presente: `retrieval_metadata.kg_chunk_keyword_matches > 0` en 35/35 queries (mean=34.8, 100% > 30% umbral). **Lo que NO esta validado**: la calidad de las keywords emitidas por la extraccion piggyback vs la llamada dedicada del paper. Ver advertencia detallada en la fila de divergencia #10 en "Divergencias de indexacion" — el observable actual `kg_chunk_keyword_matches` mide presencia del canal, no calidad de la senal. Bloqueante antes de P2 (catalogo especializado), no de P0 (HotpotQA). |

Detalles tecnicos de cada item:

### Divergencias arquitectonicas

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 8 | **Chunks via KG (opcion A)** | **Resuelta** | `_retrieve_via_kg()` en `retriever.py` obtiene chunks a traves de `source_doc_ids` de entidades/relaciones resueltas via Entity/Relationship VDB, con reference-count scoring. Fallback a vector search cuando KG devuelve vacio. Codigo muerto eliminado: `query_entities`, `query_by_keywords`, `get_entities_for_docs`, `get_relations_for_docs`, keyword index infrastructure, `_resolve_entity_names`, `_resolve_relationships_via_vdb`. |

### Divergencias de indexacion

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 10 | **High-level keywords por chunk durante indexacion** | **Resuelta en implementacion; calidad del output pendiente de verificacion empirica** | Piggyback en la extraccion de tripletas (misma llamada LLM). El LLM emite ademas de `entities` y `relations` un campo `high_level_keywords` con los temas abstractos del chunk. Pipeline: (1) `TRIPLET_EXTRACTION_PROMPT`/`TRIPLET_EXTRACTION_BATCH_PROMPT` en `triplet_extractor.py` instruyen al LLM a emitir el campo; `_build_entities_relations` aplica dedup case-insensitive, cap de 10 por doc y longitud in [2, 80]. (2) `KnowledgeGraph` persiste `_doc_to_keywords: Dict[doc_id, List[str]]` via `add_doc_keywords()` con serializacion v3 (v2 invalidado explicitamente con `ValueError` en `from_dict`). (3) `LightRAGRetriever._build_chunk_keywords_vdb()` construye una tercera VDB `{base}_chunk_keywords` (cosine, una entrada por doc_id, `page_content` = keywords joined por coma). (4) `_retrieve_via_kg` en modos `global`/`hybrid` llama `_resolve_chunks_via_keywords_vdb(high_level, top_k=KG_CHUNK_KEYWORDS_TOP_K)` y acumula `1/(1+rank) * similarity` por doc matched al mismo `doc_scores` de entidades/relaciones. Toggled por `KG_CHUNK_KEYWORDS_ENABLED` (default true). Metadata por query: `kg_chunk_keyword_matches` (conteo). **Validacion de presencia end-to-end** (run `mteb_hotpotqa_20260419_032905`, hybrid, 1000 docs): el LLM poblo `high_level_keywords` en 988/1000 docs indexados; Chunk Keywords VDB se construyo sin errores (988 docs indexados en 6s); `retrieval_metadata.kg_chunk_keyword_matches > 0` en **35/35 queries** (100%, mean=34.8 matches/query). Tests: `test_chunk_keywords_retrieval.py` (12 casos) + extensiones en `test_triplet_extractor.py` (7 casos) y `test_knowledge_graph.py` (10 casos, incluye rechazo de caches v2/v1). **⚠️ ADVERTENCIA DE RIESGO CONOCIDO — piggyback vs llamada dedicada del paper**: el paper original hace la extraccion de high-level keywords en una **llamada LLM dedicada** por chunk; nuestra implementacion las emite en la misma llamada que entities/relations. Motivacion del piggyback: ahorrar ~50% de llamadas LLM en indexacion (medido: indexacion 1000 docs ~30 min con piggyback vs estimado ~45-60 min con llamada separada). **Coste teorico**: cuando un mismo prompt pide tres outputs simultaneos (entities + relations + high_level_keywords), el LLM reparte atencion y tokens de output entre ellos. Los outputs concretos (entities, relations) son los mas naturales de extraer y tienden a dominar; los temas abstractos (keywords) son el output mas removido y el primero en degradarse — puede emitirse generico ("event", "person", "organization") en vez de especifico al tema del chunk. **Lo que el observable actual no captura**: `kg_chunk_keyword_matches > 0` solo confirma que la VDB tuvo matches semanticos, NO que las keywords representen los temas reales del chunk. Un LLM que emita `["article", "document", "text"]` como keywords en cada doc pasaria trivialmente este observable. **Cuando importa**: en HotpotQA (Wikipedia en pre-entrenamiento + DEV_MODE con gold docs garantizados) el canal vector directo satura el retrieval; la calidad del canal high-level no discrimina la metrica agregada — por eso esta implementacion paso Pre-P0. En el escenario objetivo del producto a largo plazo (P2, catalogo de 10-50 PDFs especializados donde las entidades son desconocidas para el embedding), el canal high-level es el unico que opera sobre conceptos que el embedding SI conoce; si piggyback lo degrada silenciosamente, rompemos la pata diferencial de LightRAG justo en su caso de uso objetivo. **Accion pendiente antes de P2** (no bloquea P0): (1) anadir observable de calidad de keywords — `mean_keywords_per_doc`, diversidad (Jaccard entre sets de keywords de docs del mismo tema), ratio de keywords genericas vs especificas (heuristica: inverse document frequency dentro del corpus); (2) si el observable muestra degradacion, exponer toggle `KG_CHUNK_KEYWORDS_DEDICATED_CALL=true` y aceptar el coste de tiempo adicional en indexacion. |

### Divergencias menores (cosmeticas / no funcionales)

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 3 | Entity cap 100K | **3/10** | Eviction mejorada con score compuesto. Para HotpotQA (66K docs, ~24K entidades) no se alcanza. |
| 11 | **Chunks en coleccion principal de ChromaDB, no en `text_chunks_vdb` dedicado** | **2/10** | El paper mantiene 3 VDBs: `entities_vdb`, `relationships_vdb` y `text_chunks_vdb` (coleccion separada para chunks con su id y metadata). Aqui los chunks se indexan en la coleccion principal de `ChromaVectorStore` (heredada de `SimpleVectorRetriever`), no en un VDB dedicado. Las Entity VDB y Relationship VDB si son colecciones separadas, construidas en `LightRAGRetriever._build_entities_vdb()` y `._build_relationships_vdb()` de `retriever.py`. **Consecuencia practica post-#8 resuelta**: con #8 cerrada (opcion A), los chunks se obtienen via `source_doc_ids` del KG en el path principal; la coleccion principal de `ChromaVectorStore` actua de facto como `text_chunks_vdb` (contiene los chunks con su id y metadata) y ademas sirve al fallback de vector search directo cuando el KG no produce doc_ids. Separar formalmente en colecciones dedicadas seria un cambio de naming/organizacion sin efecto funcional. **Criterio para re-evaluar**: si en P3 el schema de export hacia el administrador requiere distinguir VDBs por rol (p.ej. versionado independiente de chunks vs. VDBs del KG), o si aparece un segundo consumidor del `text_chunks_vdb` que no quiera la dependencia con `ChromaVectorStore`. |
| 12 | **Formato de contexto JSON-lines, no CSV con headers del paper** | **2/10** | El paper original presenta el contexto al LLM como tablas CSV con headers `Entities \| Relationships \| Sources`. Aqui `format_structured_context()` en `sandbox_mteb/retrieval_executor.py` usa bloques JSON-lines etiquetados como `"Knowledge Graph Data (Entity)"`, `"Knowledge Graph Data (Relationship)"` y `"Document Chunks"` con `reference_id` numerico (usado por la capa de synthesis para citas `[ref:N]`). **Consecuencia practica**: ambos son formatos estructurados que el LLM parsea sin problema. El JSON-lines tiene la ventaja de que la capa de synthesis (divergencia #2 resuelta) usa los `reference_id` para anclar citas. No requiere accion — cambiar a CSV romperia el esquema de citas sin beneficio funcional. |

Contexto historico sobre HotpotQA (por que satura, implicacion para la eleccion de benchmark de P0) en "Proximos pasos · Resultado F.5".

## Deuda tecnica vigente

| # | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB: colecciones huerfanas si el proceso se interrumpe | **BAJO** | `evaluator.py:_cleanup()` ahora elimina la coleccion correctamente via `delete_all_documents()` (que llama `delete_collection()` + recrea). Sin embargo, cada run crea `eval_{run_id}` — si el proceso se interrumpe antes de cleanup, la coleccion queda huerfana. Con `PersistentClient`, se acumulan en disco | Criterio de accion: auditar `VECTOR_DB_DIR` entre campanas (p.ej. pre/post P0) y purgar colecciones `eval_*` que no correspondan a runs vivos. Si el tamano del directorio supera el presupuesto de disco del host evaluador, automatizar la purga en `preflight.py` |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO (`head_bucket` + `list_objects MaxKeys=1`). No descarga ni parsea Parquet, no valida schema contra `DATASET_CONFIG`, no verifica espacio en disco. El riesgo principal no es infra (MinIO ya es compartido con el administrador) sino **schema drift del contrato upstream**: cuando el administrador produzca un catalogo nuevo con columnas/tipos/ids diferentes, el fallo ocurre horas despues del start en `_populate_from_dataframes()`, quemando compute | `--dry-run` primero y verificar que el dataset carga |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, usa fallback de 4000 chars (~1000 tokens). **Aclaracion importante**: este valor es el **presupuesto total de contexto** pasado al LLM, que `format_context()`/`format_structured_context()` usan para seleccionar cuantos fragmentos (chunks) caben. No es "el LLM recibe un documento truncado a 4000 chars". Con chunks tipicos de 500-1000 chars, 4000 = ~4-8 chunks relevantes. El riesgo es dejar senal en la mesa cuando el modelo soporta mas (p.ej. 192K chars): no se rompe nada, simplemente no se aprovecha toda la ventana. Se logea a nivel INFO (no WARNING) por lo que puede pasar desapercibido en logs filtrados; el run continua sin detectar la degradacion. Criterio de accion: para los runs de P0 el valor efectivo debe verificarse en `config_snapshot._runtime.max_context_chars` y ser consistente con la capacidad declarada del modelo LLM usado | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicitamente en `.env`. Antes de cada run P0, auditar `_runtime.max_context_chars` en el JSON — si muestra 4000 sin intencion explicita, descartar el run |
| 8 | Infraestructura pesada para el scope | **BAJO** | Para 1 dataset y 2 estrategias, la infraestructura (checkpoint, preflight, JSONL, export dual, subset selection, DEV_MODE) es considerable. El run F.5 ejercito cada componente sin incidentes. Criterio de accion: revisar al final de P0 que componentes tuvieron al menos una invocacion efectiva en los runs reales (checkpoint usado en `--resume`, preflight bloqueando runs mal configurados, etc.); si alguno queda sin uso despues de tres runs P0 consecutivos, candidato a eliminacion | Pendiente de revision post-P0 con el criterio anterior. No actuar antes de tener los datos |
| 9 | Lock-in a NVIDIA NIM | **MEDIO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |
| 10 | Sin indexacion incremental del KG | **MEDIO** | `LightRAGRetriever.index_documents()` en `retriever.py` siempre hace build completo del KG o carga desde cache. No hay `append_documents()` que anada docs nuevos a un KG existente sin reconstruir desde cero. `KnowledgeGraph` tiene `add_triplets()` y `add_entity_metadata()` como primitivas, pero falta el path de alto nivel que: (1) extraiga tripletas solo de docs nuevos, (2) las integre en el KG existente, (3) reconstruya solo los VDBs afectados. El paper (HKUDS/LightRAG) soporta ingestion incremental via `insert()`. Para el escenario del sistema administrador (PDFs que llegan en tandas), la falta de append obliga a re-indexar todo el corpus ante cada doc nuevo. Workaround: usar cache de disco y reconstruir | Cache de disco mitiga la re-extraccion LLM pero no el rebuild del grafo ni de VDBs. P3 (embedibilidad) requiere resolver esto |
| 11 | Duplicacion parcial de logica de iteracion de vecinos en KG | **BAJO** | `_get_neighbors_weighted()` en `knowledge_graph.py` itera `self._graph.incident(vid)` para obtener `(neighbor_name, edge_weight)`. `get_neighbors_ranked()` (divergencia #9) necesita ademas `degree_centrality` y la etiqueta de relacion del edge, asi que hace su propia iteracion sobre los mismos edges en vez de reutilizar `_get_neighbors_weighted`. La duplicacion es deliberada: `get_neighbors_ranked` accede a campos distintos del edge (`relations[0]`, `graph.degree()`) que `_get_neighbors_weighted` no expone, y extender la firma privada para cubrir ambos casos complicaria la API interna sin beneficio claro | Aceptable mientras solo haya dos consumidores. Si aparece un tercero, refactorizar ambos en un iterador comun sobre edges |
| 12 | Tests acoplados a mensajes de error de `AsyncLLMService` | **BAJO** | `tests/test_llm.py:145` (empty-response path) y `tests/test_llm.py:252` (retry exhaustion) usan regex laxa sobre el mensaje de `RuntimeError` porque hoy no hay excepciones custom que discriminen los casos. Mejor solucion: definir excepciones especializadas en `shared/llm.py` (`EmptyResponseError`, `RetriesExhaustedError`) y asertar por tipo + `call_count` del mock. **Coste real**: si el texto de `raise RuntimeError(...)` en `shared/llm.py` cambia, los tests fallan por regex, no por cambio de comportamiento — falso positivo del que solo se detecta editando el fichero | Criterio de accion: la proxima modificacion funcional a `shared/llm.py` incluye el refactor de excepciones como parte del PR. Mientras tanto, cualquier cambio a los mensajes de `RuntimeError` requiere sincronizar los regex de los dos tests |
| 13 | Reranker se inicializa innecesariamente para LIGHT_RAG | **RESUELTA** | Guard movido a `_init_components()` en `evaluator.py`: si `strategy == LIGHT_RAG` y `reranker.enabled=True`, se logea WARNING explicando que se omite por divergencia #6 (el paper no usa reranker) y `self._reranker` queda en `None`. `retrieval_executor.py` ya tenia auto-desactivacion en runtime; ahora tampoco se instancia la conexion al NIM al arranque. Tests: `test_evaluator.py::test_init_components_skips_reranker_for_lightrag` + `test_init_components_initializes_reranker_for_simple_vector` (regresion guard — SIMPLE_VECTOR sigue instanciando como antes). | — |
| 14 | Acceso a `_vector_store.collection_name` desde `LightRAGRetriever` | **BAJO** | Tres consumidores en `retriever.py` acceden a `self._vector_retriever._vector_store.collection_name` para derivar nombres de colecciones: `_build_entities_vdb()`, `_build_relationships_vdb()` y `_build_chunk_keywords_vdb()` (este ultimo anadido con divergencia #10). Es acceso a atributo privado de otro objeto, solo para naming (no para datos). **Coste real**: si alguien renombra `collection_name` en `SimpleVectorRetriever._vector_store` sin buscar referencias externas, los 3 builders rompen en runtime al indexar. Sin test que lo detecte aisladamente | Criterio de accion: exponer property `collection_name` en `SimpleVectorRetriever` en el primer cambio que toque esa clase, o cuando aparezca un cuarto consumidor externo del nombre de coleccion |
| 15 | **`retrieval_metadata` por query — exportacion JSON/CSV** | **RESUELTA** | Fix unificado sobre ambos exports. (1) **JSON**: `QueryEvaluationResult.to_dict()` en `shared/types.py` ahora emite un bloque `retrieval_metadata` per-query con el subset relevante — `lightrag_mode`, `kg_fallback` (`no_keywords`/`no_doc_ids`/`docs_not_in_store`/ausente), `kg_chunk_keyword_matches`, `kg_entities_count`, `kg_relations_count`, `kg_synthesis_used`, `kg_synthesis_error` (`timeout`/`error`/`empty`/ausente), `graph_active`, y — anadidos en la auditoria de observabilidad — `kg_entities_with_neighbors`, `kg_mean_neighbors_per_entity` (divergencia #9), `kg_budget_cap_triggered` (divergencia #4+5), y los 14 campos `citation_refs_{synth,gen}_{total,valid,malformed,in_range,out_of_range,distinct,coverage_ratio}` (divergencia #7). Listas completas de entidades/relaciones se colapsan a conteos para no inflar el JSON; queries SIMPLE_VECTOR no emiten el bloque (omitted si subset vacio); los 14 campos de citaciones solo se emiten cuando `has_kg_data AND kg_synthesis_enabled` para que SIMPLE_VECTOR y LIGHT_RAG sin synthesis no inflen el JSON con 14 ceros por query. (2) **CSV**: `shared/report.py:to_detail_csv()` tenia un guard muerto que buscaba `graph_candidates` (clave eliminada con la resolucion de divergencia #8), por lo que el bloque `if has_lightrag:` nunca ejecutaba y el detail.csv no tenia columnas KG en runs reales (auditado en `mteb_hotpotqa_20260418_223530_detail.csv`, cabecera sin `graph_*`). El guard y las columnas se reescribieron usando `extract_retrieval_metadata_subset()` como unica fuente de verdad compartida con el JSON: columnas `lightrag_mode`, `kg_fallback`, `kg_entities_count`, `kg_relations_count`, `kg_chunk_keyword_matches`, `kg_entities_with_neighbors`, `kg_mean_neighbors_per_entity`, `kg_budget_cap_triggered`, `kg_synthesis_used`, `kg_synthesis_error`, mas las 14 columnas `citation_refs_*`. Queries sin los campos obtienen celda vacia (no `0`) para discriminar "no aplica" de "cero validos". (3) **Synthesis outcome per-query**: `_synthesize_kg_context_async` ahora retorna `Tuple[str, Optional[str]]` con el `error_code`, y `_process_single_async` escribe `kg_synthesis_used`/`kg_synthesis_error` en `retrieval_detail.retrieval_metadata` para que ambos exports lo consuman. Tests: `test_report.py` (4 casos iniciales JSON + CSV + regresion guard SIMPLE_VECTOR; +2 casos para observables de #9/#4+5; +4 casos para observables de #7), `test_kg_synthesis.py::TestPerQuerySynthesisMetadata` (5 casos cubriendo success/timeout/empty/error/no-kg-data), `test_citation_parser.py` (13 casos del parser) y `TestCitationRefsIntegration` (5 casos end-to-end). Listo para usar en el run diagnostico del paso 2 del HANDOFF. | — |
| 16 | **Synthesis timeouts en corpus pequeno** | **RESUELTA** | Diagnostico secuencial con la instrumentacion de timing (`p50/p95/max_total_ms`, `p50/p95/max_queue_ms`, `p50/p95/max_llm_ms`): (a) run `mteb_hotpotqa_20260418_223530` (16 conc, `max_chars=0`, timeout 90s) → `fallback_rate=0.57`, `p50_queue_ms=...`, `p50_llm_ms=...` insuficientes para discriminar causa; (b) run `mteb_hotpotqa_20260419_004640` con instrumentacion landed → `fallback_rate=0.31` con `p50_queue_ms=28.9s`, `p50_llm_ms=41.9s`, 11 timeouts clusterizados temporalmente → **saturacion de cola (slot del semaforo)** como causa primaria; (c) run `mteb_hotpotqa_20260419_015230` con 32 conc + `max_chars=50000` → `fallback_rate=0.20`, `p50_queue_ms=0`, `p50_llm_ms=56.9s` → cola eliminada pero latencia LLM subio por saturacion GPU NIM, 7 timeouts por tail largo; (d) run `mteb_hotpotqa_20260419_032905` con 32 conc + `max_chars=50000` + timeout=180s → `fallback_rate=2.86%` (1/35), **umbral del 10% cumplido**. Config validada: `NIM_MAX_CONCURRENT_REQUESTS=32`, `KG_SYNTHESIS_MAX_CHARS=50000`, `KG_SYNTHESIS_TIMEOUT_S=180`. Conclusion: reducir `max_chars` **no** aceleraba el LLM (contraintuitivo, probable saturacion GPU bajo carga); el bottleneck real era la capacidad del NIM con concurrencia alta, que requiere timeout calibrado. | — |
| 17 | **Parametros fijos del canal de chunk keywords (divergencia #10)** | **BAJO (observacion)** | La resolucion de #10 expone solo `KG_CHUNK_KEYWORDS_ENABLED` y `KG_CHUNK_KEYWORDS_TOP_K`. Hay tres parametros fijados en `triplet_extractor.py`/`retriever.py` que podrian necesitar exposicion si el comportamiento empirico lo exige: (a) `MAX_CHUNK_KEYWORDS_PER_DOC=10` (cap de keywords por doc en el parseo), (b) `MIN_CHUNK_KEYWORD_LEN=2` / `MAX_CHUNK_KEYWORD_LEN=80` (validacion de longitud), (c) `_CHUNK_KEYWORDS_VDB_MAX_DISTANCE=0.8` (threshold de cosine distance, simetrico con Entity/Relationship VDB). Criterio de accion: exponer al `.env` solo si en un run end-to-end se observa que un caso real (p.ej. corpus con terminologia muy larga) cae fuera de estos defaults. No actuar antes de tener esa evidencia — exponer parametros sin uso concreto es over-engineering | Sin accion hasta que un run real demuestre necesidad |
| 18 | **Observable de citaciones #7 acoplado al prompt `KG_SYNTHESIS_SYSTEM_PROMPT`** | **BAJO** | El parser `shared/citation_parser.py::parse_citation_refs` usa regex estricto `\[ref:(\d+)\]` alineado con el formato que el prompt `KG_SYNTHESIS_SYSTEM_PROMPT` (`sandbox_mteb/config.py:309`) instruye al LLM a emitir. El parser captura tambien variantes (`[Ref:3]`, `[REF:4]`, `[ref: 5]`, `[ref:abc]`) como `malformed` para detectar derivas. **Coste real**: si alguien modifica el prompt de synthesis (p.ej. cambia formato de citas a `(ref N)`, o pide multiples refs por corchete `[ref:3,4,5]`), los 14 campos `citation_refs_*` dejan de medir lo que dicen medir — el parser reportara `malformed` inflado o `total=0` enganoso. Sin test que lo detecte porque es acoplamiento semantico, no sintactico | Criterio de accion: cualquier PR que toque `KG_SYNTHESIS_SYSTEM_PROMPT` debe incluir revision del parser y, si cambia el formato de citas, actualizacion de los regex `_VALID_RE`/`_CANDIDATE_RE` en sync. En la descripcion del PR mencionar explicitamente "toca observable #7" para forzar revision cruzada |

## Observabilidad de runs

Los `EvaluationRun` exportados a JSON incluyen en `config_snapshot._runtime` dos bloques de stats para auditoria post-run:

**`judge_fallback_stats`**: solo aparecen las `MetricType` del judge **efectivamente invocadas en el run**. Para datasets `HYBRID` (p.ej. HotpotQA) el judge corre `faithfulness` como secundaria → solo esa clave aparece. `answer_relevance` aparece cuando el dataset usa `MetricType.ANSWER_RELEVANCE` como primary o secondary (hoy, datasets tipo `ADAPTED` sin expected answer o con `secondary_metrics=[ANSWER_RELEVANCE]`). Por metrica se reporta `invocations`, `parse_failures` (JSON no parseable), `default_returns` (0.5 por defecto), `parse_failure_rate`, `default_return_rate`. Si `default_return_rate > JUDGE_FALLBACK_THRESHOLD` (default 2%) en cualquier metrica invocada, el run falla con `RuntimeError`. Que una metrica no aparezca no es un bug — significa que no se invoco; auditar via `primary_metric_type` / `secondary_metrics` en `query_results`.

```bash
jq '.config_snapshot._runtime.judge_fallback_stats' data/results/<run_id>.json
```

**`kg_synthesis_stats`**: cuando `KG_SYNTHESIS_ENABLED=true` y la estrategia es `LIGHT_RAG`, reporta contadores (`invocations`, `successes`, `errors`, `empty_returns`, `truncations`, `timeouts`, `fallback_rate`) **y timing per-invocacion** para discriminar causas de fallback alto (deuda #16): `p50_total_ms`/`p95_total_ms`/`max_total_ms` (wall-clock completo de `_synthesize_kg_context_async` desde `wait_for` hasta outcome), y desglose via hook `timing_out` de `AsyncLLMService.invoke_async` en `p50_queue_ms`/`p95_queue_ms`/`max_queue_ms` (espera del semaforo de concurrencia `NIM_MAX_CONCURRENT_REQUESTS`) y `p50_llm_ms`/`p95_llm_ms`/`max_llm_ms` (llamada al NIM tras acquire). `n_total_samples`/`n_queue_samples`/`n_llm_samples` exponen cuantas invocaciones contribuyeron a cada percentil — queue/llm pueden tener `n_samples < n_total_samples` cuando un timeout canceló la coroutine antes de que se poblara la clave correspondiente. `fallback_rate > 10%` sigue siendo el umbral de bloqueo; el timing sirve para accionar el ajuste correcto: `p50_queue_ms` alto → subir `NIM_MAX_CONCURRENT_REQUESTS` o bajar paralelismo de queries; `p50_llm_ms` alto → reducir `KG_SYNTHESIS_MAX_CHARS` o subir `KG_SYNTHESIS_TIMEOUT_S`. **Observacion empirica (abril 2026)**: smoke-test `mteb_hotpotqa_20260418_183018` (800 docs, 25q, hybrid, timeout 90s) reporto `fallback_rate=0.24` con 6 timeouts / 0 errores — inusual en corpus pequeno, ver deuda #16; la instrumentacion de timing se añadió después (commit Fase C del HANDOFF) y estara disponible en el siguiente run diagnostico.

```bash
jq '.config_snapshot._runtime.kg_synthesis_stats' data/results/<run_id>.json
```

Ambos tambien se emiten en el evento estructurado `run_complete` del JSONL y en logs INFO al final de cada run.

**`citation_refs_{synth,gen}_*` (divergencia #7, 14 campos per-query)**: aplican solo a queries LIGHT_RAG con synthesis activa. El parser `shared/citation_parser.py::parse_citation_refs` se invoca dos veces en `_process_single_async`: sobre la narrativa synthesized (`synth_*`) y sobre la respuesta final del LLM (`gen_*`), cada una con `n_chunks_emitted` (devuelto por `format_structured_context_with_stats`) como rango valido. Los 7 contadores por prefijo son: `total` (valid + malformed), `valid` (formato estricto `[ref:N]`), `malformed` (variantes tipo `[Ref:3]`), `in_range` (N en `[1, n_chunks_emitted]`), `out_of_range` (N fuera de rango — **senal roja**), `distinct` (unique N in_range), `coverage_ratio` (distinct / in_range).

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

| Metrica | Valor (abril 2026) |
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

Patron de fallo observado empiricamente durante el desarrollo: **claude_code tiende a presentar trabajo como completo antes de tiempo, y a categorizar problemas detectados como "menores" o "aceptables" cuando arreglarlos implicaria mas trabajo**. Es un sesgo hacia cerrar la tarea en vez de dejarla verdaderamente terminada.

**Por que importa aqui**: el ciclo de validacion de este proyecto es caro. Un run LIGHT_RAG sobre HotpotQA (125 queries, DEV_MODE) tarda ~1h30min y consume presupuesto de NIM. Un parametro mal capeado, un timeout hardcoded sin exponer, un test que no estresa el edge case real, o una alineacion parcial con el paper (secciones si, total no) no se detectan en `pytest`; se descubren cuando el run ya se ejecuto y las metricas no son interpretables, o cuando la comparacion con el paper se vuelve inverificable. Cada iteracion fallida son horas perdidas y datos no publicables.

**Manifestaciones concretas a vigilar**:
- Auto-evaluacion reactiva (solo tras peticion explicita) en vez de integrada al entregable.
- Adjetivos como "aceptable", "pendiente menor", "por ahora" aplicados a deuda tecnica sin medir su coste real.
- Tests que pasan pero no estresan el caso que motivo el cambio (p.ej. capping proporcional probado solo con budgets holgados).
- Alineacion con el paper descrita como completa cuando solo cubre un subconjunto de parametros (p.ej. budgets por seccion sin el budget total).
- Parametros operacionales hardcoded en `constants.py` en vez de expuestos en `.env` con la excusa de que "el default es razonable".

**Contramedidas para claude_code (al trabajar en este repo)**:
1. Antes de decir "listo para validar", simular mentalmente el run con los parametros puestos: con los valores tipicos, ¿el contexto real se parece al del paper? ¿el timeout da margen? ¿el test cubre el caso que motivo el cambio, o solo un caso feliz?
2. Enumerar las limitaciones de la solucion **antes** de entregarla, no despues. La auto-evaluacion debe ser parte del entregable.
3. Distinguir "menor" de "conveniente de no arreglar": si la razon para clasificarlo como menor es que arreglarlo requiere mas trabajo, es deuda real, no severidad baja.
4. Todo parametro que dependa del LLM usado (timeouts, tamaños de contexto, concurrencia) debe ir al `.env`, no a `constants.py`. Si se pone en `constants.py` es porque nunca deberia tocarse (p.ej. `CHARS_PER_TOKEN`).
5. Cualquier cambio que afecte el contexto que ve el LLM debe tener test que estrese el budget, no solo el caso con budget holgado.

**Validacion a ciegas del pipeline completo**: claude_code no ejecuta en el entorno con NIM+MinIO (ver deuda #9 para el contexto infra; TESTS.md F23 para el e2e actual mockeado). Implicaciones operativas:
- Cambios que afectan al flujo retrieval→synthesis→generation solo se validan cuando el usuario lanza el run.
- Antes de declarar un cambio como completo, claude_code debe enumerar los **criterios observables en el run**: que variable de `config_snapshot._runtime` esperar, que valor en `kg_synthesis_stats` o `judge_fallback_stats`, que rango en las metricas agregadas.
- "Falta test e2e de X" NO es deuda pendiente que claude_code deba resolver — es consecuencia estructural del entorno. Listarla como deuda induce iteraciones futuras intentando construir algo imposible de validar en sesion.

Esta seccion se actualiza con nuevas manifestaciones del patron segun se detecten. No borrar sin consenso.

## Proximos pasos

### Orden de prioridades

```
Pre-P0 (completitud arquitectural + ejecucion estable)  <-- GATE CERRADO (2026-04-19)
  |
P0 (replicacion empirica del paper)                     <-- FASE ACTUAL
  |
  +-- P1 (sanity on/off synthesis sobre HotpotQA)       <-- barato, en paralelo a P0
  |
  +-- P2 (experimento 3: catalogo especializado)        <-- SOLO si P0 pasa + riesgo piggyback #10 resuelto
  |
  +-- P3 (embedibilidad + export KG + integracion)      <-- SOLO si P2 pasa
```

**⚠️ Alerta sobre Pre-P0 cerrado con matiz**: la condicion 1 del gate (completitud arquitectural) se cumplio con una salvedad explicita sobre la divergencia #10 (ver su fila). El piggyback en la extraccion de keywords high-level puede degradar la calidad del canal **exactamente en el escenario donde LightRAG deberia brillar** (catalogo especializado, dominio fuera del pre-entrenamiento del embedding). HotpotQA no discrimina este riesgo por saturacion del vector directo. **Antes de lanzar P2 — no P0 — hace falta anadir observable de calidad de keywords y/o toggle a llamada dedicada.**

### Pre-P0 — Completitud arquitectural de LIGHT_RAG · **GATE CERRADO (2026-04-19)**

Las tres condiciones verificables se cumplieron simultaneamente sobre el run `mteb_hotpotqa_20260419_032905`:

1. **Arquitectonicamente completa frente al paper**: las 7 divergencias arquitectonicas (#2, #4+5, #6, #7, #8, #9, #10) resueltas en codigo + tests. **Validacion empirica directa de correctitud** para #2, #6, #8 via observables que discriminan calidad del comportamiento; **ejercitadas implicitamente** para #4+5, #7, #9 (corren en el pipeline pero sin observable per-query dedicado). **#10 con matiz especial**: canal arquitectonicamente presente (35/35 queries con matches en la Chunk Keywords VDB) pero **la calidad del output de la extraccion piggyback vs la llamada dedicada del paper no esta verificada empiricamente**. Bloqueante antes de P2, no de P0 (ver fila de divergencia #10 para la advertencia completa).
2. **Funcionando en ejecucion**: `kg_synthesis_stats.fallback_rate = 2.86%` (< 10% umbral); `judge_fallback_stats.default_return_rate = 0%` en faithfulness; `retrieval_metadata.kg_fallback=null` en 35/35 queries (KG produjo doc_ids en todas, sin fallback al vector search); `queries_failed = 0`.
3. **Funcionalidades extra alineadas con el entorno**: cache de KG, fallbacks ante errores del LLM/igraph, instrumentacion de timing (queue/LLM split) documentadas como adaptaciones operativas (deuda #16 cerrada).

**Historial de los 3 runs diagnosticos que cerraron la fase**:

| Run ID | Config | fallback_rate | Hallazgo clave |
|---|---|---|---|
| `20260419_004640` | 16 conc, max_chars=0, 90s | 31.4% | Queue saturation (p50_queue=29s, timeouts clusterizados) |
| `20260419_015230` | 32 conc, max_chars=50K, 90s | 20.0% | Queue eliminada (p50_queue=0), pero p50_llm subio a 57s (GPU saturada con 32 concurrentes) |
| `20260419_032905` | 32 conc, max_chars=50K, 180s | **2.86%** | Gate cerrado. Timeout calibrado al p95_llm real (~114s) + margen |

**Config validada** para runs LIGHT_RAG en infra actual (NIM nvidia/nemotron-3-nano):
- `NIM_MAX_CONCURRENT_REQUESTS=32`
- `KG_SYNTHESIS_MAX_CHARS=50000`
- `KG_SYNTHESIS_TIMEOUT_S=180`

Estos valores ya son los defaults de `sandbox_mteb/env.example` (desde PR #46 — antes los comentarios los citaban pero los defaults seguian anclados a `0` / `90.0`, contradiccion interna corregida). Marcador `[PRE-P0 VALIDATED]` en el template discrimina estos 3 valores de los defaults `[DEFAULT]` genericos.

**Run de validacion post-observabilidad (`mteb_hotpotqa_20260419_181645`)**: sobre la misma config (35q DEV_MODE, hybrid, 1000 docs) reproduce Pre-P0 baseline: `fallback_rate=2.857%`, `judge.default_return_rate=0%`, `kg_fallback=null` en 35/35. Adicionalmente valida end-to-end los 17 observables nuevos (ver columna "Evidencia empirica" de la tabla "Status de validacion" para #4+5, #7, #9). Confirma que el pipeline de observabilidad funciona en produccion sin regresar el comportamiento del run de referencia `032905`.

**Palancas post-Pre-P0 expuestas en `env.example`** para corpus o infraestructuras donde la extraccion inicial pierde senal: `KG_GLEANING_ROUNDS` (default `0`; expuesto en PR #48) ejecuta una pasada extra tras la inicial cuyo prompt lista las entidades ya vistas y pide al LLM que busque las que faltan. No re-extrae keywords de chunk (divergencia #10); solo recupera entidades/relaciones perdidas cuando hay muchos warnings `Batch parse failed` o `empty content after stripping reasoning tags` en el log. Coste: ~2x llamadas LLM durante indexacion. Usar solo si la cobertura del KG (`num_docs_with_entities / total_docs` en las stats finales) baja de ~95%; subirlo en Pre-P0 donde la cobertura ya es del 99.2% seria over-engineering.

### Resultado F.5 (referencia historica)

F.5 ejecuto dos rondas comparativas SIMPLE_VECTOR vs LIGHT_RAG sobre HotpotQA (125q, DEV_MODE, seed=42). **Conclusion central, todavia vigente**: HotpotQA no discrimina entre estrategias en generacion — Wikipedia en el pre-entrenamiento del embedding + DEV_MODE con gold docs garantizados + ventana de 192K chars → el embedding resuelve por si solo sin necesitar el KG. Los deltas observados (<0.01 puntos de gen score) cayeron dentro del ruido del LLM judge. **Implicacion operativa**: cualquier medicion comparativa de la hipotesis del paper (`LIGHT_RAG > SIMPLE_VECTOR`) debe hacerse sobre un benchmark donde el embedding NO sature (objetivo de P0 — ver "Proximos pasos · P0").

Las tablas numericas detalladas de F.5 (pre-refactor con divergencias #4+5/#6/#7 abiertas, post-refactor con esas cerradas) se omiten intencionadamente: **ambas rondas son anteriores a la arquitectura actual** (synthesis #2, chunk keywords VDB #10, chunks via KG #8, guard de reranker #6 post-fix, observabilidad per-query). Los 3 runs diagnosticos que cerraron Pre-P0 (`20260419_004640/015230/032905`) midieron solo LIGHT_RAG, no la comparativa. **Cualquier delta `LIGHT_RAG vs SIMPLE_VECTOR` sobre arquitectura actual requiere un run nuevo** — la comparativa es trabajo de P0, no referencia historica reutilizable. Git history preserva las tablas F.5 originales si en algun momento hacen falta.

**Nota estructural sobre divergencia #8**: las metricas de retrieval identicas query-por-query en F.5 eran consecuencia directa de #8 (entonces abierta) — con #8 opcion A resuelta, los chunks de LIGHT_RAG vienen del KG via `source_doc_ids` en vez de solaparse con los del vector directo, asi que las metricas de retrieval pueden diverger en cualquier dataset post-#8.

### P0 — Replicacion empirica del paper · **FASE ACTUAL**

**Objetivo**: demostrar que sobre al menos un benchmark donde el paper reporta `LIGHT_RAG > baseline vector`, nuestra implementacion reproduce la **direccion** del delta (magnitudes exactas son secundarias; el signo y su significancia sobre el ruido es lo que importa).

**Estado**: desbloqueado tras cierre de Pre-P0 el 2026-04-19. La arquitectura esta completa y ejecuta estable; el siguiente trabajo es seleccionar benchmark y correr la comparativa.

**Prerequisitos arquitectonicos**: todos cumplidos para P0 sobre HotpotQA o benchmark de contra-referencia similar. Divergencias #2, #6, #8 validadas directamente; #4+5, #7, #9 ejercitadas implicitamente; #10 presente pero con riesgo conocido de calidad por piggyback (ver su fila). **Para avanzar a P2 (catalogo especializado) hace falta adicionalmente cerrar el riesgo de piggyback en #10** — anadir observable de calidad de keywords y/o exponer toggle a llamada dedicada.

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
