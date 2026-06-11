# CLAUDE.md

## Que es este proyecto

**Motor de ejecucion para construccion y consulta de grafos de conocimiento sobre corpus arbitrarios.** Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB, baseline) y `LIGHT_RAG` (vector + KG via LLM, motor objetivo). Implementa la arquitectura de [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)).

El harness en `sandbox_mteb/` (datasets MTEB/BeIR) es **instrumento de verificacion temporal** - No es el producto (en desarrollo).

## Contexto del producto

**Vision a largo plazo**: este motor se integra en un sistema administrador de colecciones de datos (**LI_AD**: versionado de KGs, multi-tenant, ciclo de vida de corpus). Contrato compartido: MinIO + Parquet — la integracion es apuntar a otro prefijo, no rediseñar la ingesta.

**Hito (2026-05-20)**: la **construccion de grafos esta probada** — el motor LIGHT_RAG construye KGs de forma estable sobre corpus arbitrarios (evidencia en `./results/`). A partir de aqui **se abandona la comparacion empirica LIGHT_RAG vs SIMPLE_VECTOR** como objetivo activo (en HotpotQA el vector satura el retrieval; la pregunta no es respondible en este harness) y el foco pasa al **contrato de ingesta** con LI_AD.

**Escenario objetivo**: colecciones de tamaño moderado de dominio especializado con terminologia/entidades fuera del pre-entrenamiento del embedding. Es el caso donde se espera que el KG aporta diferencial. (HotpotQA NO es ese escenario — es harness de verificacion previa).

**Implicacion de diseño**: las decisiones favorecen la embedibilidad (config declarativa, interfaces claras, sin side-effects globales, corpus arbitrarios, muy alta calidad de generacion). Con la construccion de grafos probada, la embedibilidad y la ingesta dejan de ser solo objetivo de diseño y pasan a ser **trabajo activo** (ver fase actual).

**Fase actual**: definir y cerrar el **contrato de ingesta** con LI_AD — el esquema MinIO/Parquet (chunks por documento + manifest) que LI_AD produce y el motor consume. Contrato vivo en [`INGESTION_CONTRACT.md`](INGESTION_CONTRACT.md) (negociado con la sesion de LI_AD). Detalles en ["Proximos pasos"](#proximos-pasos).

**Export de KG (reciproco, futuro)**: LI_AD necesita leer el grafo para su UI (su Fase 2). El schema de export propuesto vive en [`KG_CONTRACT.md`](KG_CONTRACT.md). Hoy el KG es efimero (igraph + ChromaDB en memoria, descartado en `_cleanup()`); la implementacion del export es trabajo P3 nuestro, aun no construido.

**Auditoria iterativa**: ver [`audit.md`](audit.md). Es el estado cross-session de la auditoria de codigo/comentarios/estructura. Antes de abrir hallazgos nuevos, leer el protocolo y la fase activa en ese archivo.

## Archivos clave

Estructura completa del repo en [`README.md`](README.md). Archivos que concentran la logica del motor y donde aterrizan la mayoria de cambios:

- **Motor LIGHT_RAG**:
  - `shared/retrieval/lightrag/retriever.py` — `LightRAGRetriever`: retrieval vector + KG dual-level, agregacion de los 3 canales (entity/relationship/chunk-keywords VDBs)
  - `shared/retrieval/lightrag/knowledge_graph.py` — KG in-memory (igraph), BFS, `get_neighbors_ranked`
  - `shared/retrieval/lightrag/triplet_extractor.py` — extraccion LLM de tripletas + high-level keywords por chunk (piggyback) + query keywords
- **Motor comun**:
  - `shared/retrieval/__init__.py` — factory `get_retriever()`, unico punto de entrada
  - `shared/retrieval/core.py` — `RetrievalStrategy` enum, `RetrievalConfig`, `SimpleVectorRetriever`
  - `shared/llm.py` — `AsyncLLMService` (NIM client, async/sync bridge, `_PersistentLoop`)
  - `shared/citation_parser.py` — parser `[ref:N]` para observable de citacione
- **Harness / evaluacion**:
  - `sandbox_mteb/evaluator.py` — orquestador principal
  - `sandbox_mteb/generation_executor.py` — generacion async + `_synthesize_kg_context_async + metricas
  - `sandbox_mteb/config.py` — `MTEBConfig`, `KG_SYNTHESIS_SYSTEM_PROMPT` (acoplado al parser de citas)

## Comandos

Ver [`README.md`](README.md) para setup, ejecucion y tests.

## Convenciones

- **Config via .env**: toda la parametrizacion en `sandbox_mteb/.env`, leida por `MTEBConfig.from_env()` una sola vez. Sub-configs delegadas a `InfraConfig`, `RerankerConfig`, `RetrievalConfig` en shared/. `MTEBConfig.validate()` propaga validacion a sub-configs
- **Factory pattern**: `get_retriever(config, embedding_model)` en `shared/retrieval/__init__.py` crea el retriever correcto
- **2 estrategias**: `SIMPLE_VECTOR` y `LIGHT_RAG` — no hay mas
- **Enum en core.py**: `RetrievalStrategy` define las estrategias validas. `VALID_STRATEGIES` en `sandbox_mteb/config.py` debe coincidir
- **Tests**: mocks siempre a nivel de funcion, nunca modulos enteros. Ver seccion "Test coverage"
- **Logging**: JSONL estructurado via `shared/structured_logging.py`. Bare excepts tienen `logger.debug(...)` — no hay excepts silenciosos
- **Idioma**: identificadores (clases, funciones, variables) en ingles. Comentarios y docstrings mezclados ES/EN, aceptado como historico

## Estrategia LIGHT_RAG — como funciona

Adaptaciones operativas propias del entorno: cache de KG a disco, fallbacks ante errores del LLM/igraph, instrumentacion de observabilidad.

**Indexacion**: LLM extrae tripletas `(entidad, relacion, entidad)` + `high_level_keywords` tematicas por chunk (piggyback en la misma llamada, divergencia #10) → KnowledgeGraph in-memory (igraph) + 3 VDBs ChromaDB: Entity VDB, Relationship VDB, Chunk Keywords VDB. Gleaning opcional via `KG_GLEANING_ROUNDS` (solo entidades/relaciones perdidas; no re-extrae keywords).

**Retrieval (modos `local`/`global`/`hybrid`)**: los chunks se obtienen **via KG** con tres canales que suman al mismo `doc_scores` con `1/(1+rank) × similarity [× edge_weight]`:
- **Entidades** (local/hybrid): low-level keywords → Entity VDB → `source_doc_ids`
- **Relaciones** (global/hybrid): high-level keywords → Relationship VDB → `source_doc_ids` de endpoints
- **Chunk keywords** (global/hybrid, div. #10): high-level keywords → Chunk Keywords VDB → doc_ids directos

Contenido real fetcheado por `get_documents_by_ids`. Fallback a vector directo si KG no produce doc_ids (logueado en `retrieval_metadata.kg_fallback`). Resultados intermedios anotados en `retrieval_metadata.kg_entities`/`kg_relations`/`kg_chunk_keyword_matches`. Scoring usa `1/(1+rank)` (paper usa decay lineal — misma intencion, curva distinta).

**Enriquecimiento KG**: cada entidad resuelta incluye vecinos 1-hop rankeados por `edge_weight + degree_centrality` (`get_neighbors_ranked`), configurable via `KG_MAX_NEIGHBORS_PER_ENTITY` (default 5, 0=off). Cada relacion incluye descripcion+tipo de endpoints. Degradacion graceful en lookup fallido.

**Modos** (`LIGHTRAG_MODE`): `naive` (solo chunks), `local` (+entidades), `global` (+relaciones), `hybrid` (default, todo).

**Synthesis del contexto (value-add del proyecto)**: `GenerationExecutor._synthesize_kg_context_async()` reescribe las 3 secciones como narrativa coherente via LLM antes de la generacion final, con citas `[ref:N]` inline (prompt en `sandbox_mteb/config.py:KG_SYNTHESIS_SYSTEM_PROMPT`). **Faithfulness se evalua contra el contexto estructurado original**, no contra la narrativa — control anti-fabricacion para penalizar alucinacion de la propia synthesis. Degradacion graceful: error/vacio/timeout → fallback al contexto estructurado.

**Fallback global**: sin igraph o sin LLM → SimpleVectorRetriever puro y el run nunca se rompe, **NO DESEADO** (el KG deberia funcionar; este fallback enmascara fallos del motor).

<a id="divergencias"></a>
## Divergencias con el paper original

Diferencias entre esta implementacion y el [LightRAG original](https://github.com/HKUDS/LightRAG) (HKUDS, EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)). IDs (`div-N`) son navegables desde codigo: el comentario `# divergencia #10` resuelve a `<a id="div-10">`.

> **Upstream pin**: las referencias en codigo (`HKUDS/LightRAG operate.py`, `_merge_nodes_then_upsert`, etc.) apuntan al repo `main` sin commit SHA. Riesgo: si upstream renombra `operate.py` o refactoriza, las referencias se rompen silenciosamente (R13). Pendiente: fijar SHA reproducido (validable unicamente contrastando con el repo real, fuera del alcance de claude_code). Al pinearlo, sustituir `github.com/HKUDS/LightRAG` por `github.com/HKUDS/LightRAG/blob/<SHA>/lightrag/operate.py#L<line>` en los 6 sites citados (`shared/retrieval/lightrag/retriever.py:417,523,606,870`, `shared/retrieval/lightrag/knowledge_graph.py:780`, `shared/retrieval/lightrag/triplet_extractor.py:375`).

| # | Divergencias abiertas | Status | Evidencia empirica |
|---|---|---|---|
| <a id="div-10"></a>10 | High-level keywords por chunk durante indexacion (piggyback) | **Presencia validada; calidad NO validada** ⚠️ | Canal arquitectonicamente presente: `retrieval_metadata.kg_chunk_keyword_matches > 0` en 35/35 queries. **Riesgo**: el paper hace llamada LLM dedicada por chunk; aqui las keywords se emiten en la misma llamada que entities/relations para ahorrar ~50% del coste. Coste teorico: el LLM puede emitir keywords genericas ("event", "person", "document") en vez de temas reales del chunk — HotpotQA es ciego a esta degradacion porque el canal vector directo satura el retrieval. **Cuando importa**: P2 (catalogo especializado 10-50 PDFs) es el caso donde el canal high-level es el unico que opera sobre conceptos que el embedding SI conoce; si piggyback lo degrada silenciosamente, se rompe la pata diferencial de LightRAG. **Bloqueante antes de P2, no de P0.** Accion pendiente: (1) observable de calidad de keywords (diversidad Jaccard intra-tema, ratio genericas/especificas via IDF intra-corpus); (2) si muestra degradacion, exponer toggle `KG_CHUNK_KEYWORDS_DEDICATED_CALL=true` |

<a id="divergencias-menores"></a>
### Divergencias menores (cosmeticas / no funcionales)

- <a id="div-3"></a>**#3 — Entity cap 100K**: eviction con score compuesto. Para HotpotQA (66K docs, ~24K entidades) no se alcanza.
- <a id="div-4-5"></a>**#4+5 — Cap proporcional del budget KG**: reparto de `KG_SYNTHESIS_MAX_CHARS` entre las 3 secciones (entities/relations/chunks) cuando el contexto excede presupuesto, disparado por `kg_budget_cap_triggered` en `retrieval_metadata`. El paper no especifica cap; aqui se agrupa bajo un ID compuesto para reconciliar dos esquemas previos fusionados.
- <a id="div-7"></a><a id="div-12"></a>**#7 / #12 — Formato de contexto: JSON-lines con reference_id en vez de CSV con headers:** Permite que la capa de synthesis ancle citas [ref:N] al reference_id de cada chunk. Cambiar a CSV romperia el esquema de citas sin beneficio. Instrumentacion de citas (`citation_refs_{synth,gen}_*`, hallazgo `gen_*=0`, acoplamiento al prompt) documentada en "Observabilidad de runs".
- <a id="div-9"></a>**#9 — Enriquecimiento 1-hop per-entity / relation**: cada entidad resuelta incluye vecinos rankeados por `edge_weight + degree_centrality` (`get_neighbors_ranked`, `KG_MAX_NEIGHBORS_PER_ENTITY`); cada relacion incluye descripcion+tipo de endpoints. Observable per-query: `kg_neighbor_coverage_rate` en `retrieval_metadata`. El paper no describe enriquecimiento explicito — en el original las propiedades se integran dentro del contexto sin canal dedicado.
- <a id="div-11"></a>**#11 — Chunks en coleccion principal de ChromaDB, no en `text_chunks_vdb` dedicado**: Entity VDB y Relationship VDB si son colecciones separadas; los chunks residen en la coleccion principal de `ChromaVectorStore`. Post-#8 actua de facto como `text_chunks_vdb`. Separar formalmente seria naming sin efecto funcional. Re-evaluar si el schema de export hacia el administrador (P3) requiere distinguir VDBs por rol.

<a id="deuda-tecnica"></a>
## Deuda tecnica vigente

IDs (`dt-N`) son navegables desde codigo: `# ver deuda #5` resuelve a `<a id="dt-5">`. Numeros ausentes (#1, #2, #4, #6, #7, #8, #11, #13, #14, #15, #16) correspondieron a items cerrados en iteraciones anteriores; los IDs no se reasignan para mantener estabilidad de referencias historicas.

| # ID | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| <a id="dt-3"></a>3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| <a id="dt-5"></a>5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, fallback 4000 chars (presupuesto TOTAL de contexto, no truncado del doc; ~4-8 chunks). Riesgo: dejar senal en la mesa si el modelo soporta mucho mas (p.ej. 192K). Loguea INFO (no WARNING) → puede pasar desapercibido | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicito en `.env`. Auditar `config_snapshot._runtime.max_context_chars` antes de aceptar un run |
| <a id="dt-9"></a>9 | Lock-in a NVIDIA NIM | **BAJO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |
| <a id="dt-10"></a>10 | Sin indexacion incremental del KG | **MEDIO** | `LightRAGRetriever.index_documents()` siempre rebuild completo o carga de cache. No hay `append_documents()` para integrar docs nuevos sin reconstruir. El paper soporta `insert()` incremental. Para PDFs en tandas (escenario administrador), obliga a re-indexar todo | Cache de disco mitiga re-extraccion LLM pero no rebuild del grafo ni VDBs. El contrato de ingesta aporta la señal de invalidacion (`generation`/`chunking_fingerprint`, ver `INGESTION_CONTRACT.md`), pero el rebuild sigue siendo completo por generacion; el append incremental sigue pendiente (P3) |
| <a id="dt-12"></a>12 | Tests acoplados a mensajes de error de `AsyncLLMService` | **BAJO** | Tests en `test_llm.py` usan regex laxa sobre `RuntimeError` porque no hay excepciones custom. Si cambia el texto, los tests fallan por regex, no por comportamiento | Refactor de excepciones en el proximo PR funcional a `shared/llm.py`. Sincronizar regex al tocar mensajes |
| <a id="dt-17"></a>17 | Parametros fijos del canal de chunk keywords ([div #10](#div-10)) | **BAJO** | 3 parametros hardcoded en `triplet_extractor.py`/`retriever.py`: `MAX_CHUNK_KEYWORDS_PER_DOC=10`, `MIN/MAX_CHUNK_KEYWORD_LEN=2/80`, `_CHUNK_KEYWORDS_VDB_MAX_DISTANCE=0.8`. Solo `KG_CHUNK_KEYWORDS_ENABLED`/`_TOP_K` expuestos | Observables instrumentados: `chunk_keywords_{rejected_len,dropped_by_cap}` y `docs_chunk_keywords_capped` en stats del extractor; `kg_chunk_keywords_distance_filtered` per-query en `retrieval_metadata`. Thresholds para exponer al `.env`: `rejected_len / total_chunk_keywords > 5%`, `docs_chunk_keywords_capped / docs_with_keywords > 5%`, o mediana de `distance_filtered` per-query > 20% del total inspeccionado |
| <a id="dt-18"></a>18 | Observable de citaciones [#7](#div-7) acoplado al prompt de synthesis | **BAJO** | Parser usa regex `\[ref:(\d+)\]` alineado con `KG_SYNTHESIS_SYSTEM_PROMPT`. Si alguien cambia formato (p.ej. `(ref N)` o `[ref:3,4,5]`), los 14 campos dejan de medir lo que dicen. Acoplamiento semantico, sin test automatico | Al tocar `KG_SYNTHESIS_SYSTEM_PROMPT`, revisar parser y actualizar regex `_VALID_RE`/`_CANDIDATE_RE` en sync |
| <a id="dt-19"></a>19 | Modulos con cohesion media >500 lineas | **BAJO** | `retriever.py` (1224, KG build + 3 VDBs + retrieval orquestacion), `metrics.py` (985, text + judge tracker + embedding), `generation_executor.py` (678, `_KGSynthesisTracker` mezclado con `GenerationExecutor` — paralelo a `operational_tracker.py` que ya esta extraido). Layering correcto, sin violaciones; el tamano dificulta navegacion pero no introduce bugs | Sin accion inmediata. Disparadores que justificarian split: (a) dificultad demostrada para testear un path, (b) P3 embedibilidad pide reuso parcial del modulo, (c) refactor por requisito externo. Hacer split ahora es sobre-ingenieria sin disparador |
| <a id="dt-20"></a>20 | Workaround para timeout default 300s del `aiohttp.ClientSession` en langchain-nvidia-ai-endpoints 0.3.19 | **BAJO** | `shared/llm.py::AsyncLLMService.__init__` sobreescribe `self._client.get_async_session_fn` con una factory que inyecta `aiohttp.ClientTimeout(total=self.timeout)`. La factory llama al metodo "publico-privado" `_build_ssl_context()` del paquete upstream. Si upstream renombra `_build_ssl_context`, `get_async_session_fn`, o cambia el patron `model_post_init` que setea los factories, el patch deja de operar **silenciosamente** (vuelve al default 300s sin error visible). Origen: el cap exacto de 300 000 ms en `kg_synthesis.queue_ms` del run `mteb_hotpotqa_20260513_091337` (200x800 Qwen3-32B en SPARK donde p50_llm de synth alcanza 172s y p95 280s — por debajo de `KG_SYNTHESIS_TIMEOUT_S=600` pero por encima del default invisible) | Pin de version (`langchain-nvidia-ai-endpoints==0.3.19` en `requirements.lock`) mitiga. Al subir la version revisar manualmente que `_create_async_session`/`_build_ssl_context`/`get_async_session_fn` siguen igual y los factories se siguen seteando via `model_post_init`. Sin test automatico (conftest mockea el paquete entero, la factory nunca se ejecuta) |

## Observabilidad de runs

Los `EvaluationRun` exportados a JSON incluyen en `config_snapshot._runtime` tres bloques de stats para auditoria post-run:

**`judge_fallback_stats`**: solo aparecen las `MetricType` del judge **efectivamente invocadas**. Por metrica: `invocations`, `parse_failures`, `default_returns` (0.5), `parse_failure_rate`, `default_return_rate`. Si `default_return_rate > JUDGE_FALLBACK_THRESHOLD` (default 2%) en cualquier metrica invocada, el run falla con `RuntimeError`. Que una metrica no aparezca no es bug — significa no invocada; auditar `primary_metric_type`/`secondary_metrics` en `query_results`.

```bash
jq '.config_snapshot._runtime.judge_fallback_stats' <run_export.json>
```

**`kg_synthesis_stats`** (solo `LIGHT_RAG` con `KG_SYNTHESIS_ENABLED=true`): contadores (`invocations`, `successes`, `errors`, `empty_returns`, `truncations`, `timeouts`, `fallback_rate`) + timing per-invocacion: `p50/p95/max_total_ms` (wall-clock completo), `p50/p95/max_queue_ms` (espera del semaforo `NIM_MAX_CONCURRENT_REQUESTS`), `p50/p95/max_llm_ms` (llamada al NIM tras acquire). `n_total_samples`/`n_queue_samples`/`n_llm_samples` reportan cuantas invocaciones contribuyeron — queue/llm pueden tener menos cuando un timeout cancelo la coroutine antes. **Umbrales**: `fallback_rate > 10%` es senal roja (warning en logs, sin bloqueo activo; guardrail pendiente de implementacion — a dia de hoy solo `judge_fallback_stats` bloquea el run). **Diagnostico**: `p50_queue_ms` alto → subir `NIM_MAX_CONCURRENT_REQUESTS` o bajar paralelismo; `p50_llm_ms` alto → reducir `KG_SYNTHESIS_MAX_CHARS` o subir `KG_SYNTHESIS_TIMEOUT_S`.

```bash
jq '.config_snapshot._runtime.kg_synthesis_stats' <run_export.json>
```

Ambos tambien se emiten en el evento estructurado `run_complete` del JSONL y en logs INFO al final de cada run.

**`operational_stats`** (R14): contador per-evento de degradaciones silenciosas en 7 puntos del pipeline donde un bare-except traga la excepcion y continua con fallback. Siempre presente con los 7 tipos (valor 0 si no ocurrio):

| Evento | Sitio | Fallback |
|---|---|---|
| `neighbor_lookup_failure` | `retriever.py` enrichment 1-hop (div #9) | entidad sin vecinos |
| `chunk_keywords_vdb_error` | `retriever.py` canal div #10 | keyword omitido |
| `description_synthesis_error` | `retriever.py` merge LLM de descripciones | descripcion concatenada |
| `gleaning_error` | `triplet_extractor.py` `KG_GLEANING_ROUNDS` | sin tripletas extra |
| `keywords_parse_failure` | `triplet_extractor.py` parse JSON keywords | keywords vacias |
| `retrieval_error` | `retriever.py` build del KG en `index_documents` | vector puro sin KG |
| `generation_error` | `generation_executor.py` llamada LLM final | respuesta con `[ERROR: ...]` |

Valores altos indican degradacion del canal aunque el run termine sin error fatal. No bloquean el run — son senal para inspeccion post-hoc.

```bash
jq '.config_snapshot._runtime.operational_stats' <run_export.json>
```

**`citation_refs_{synth,gen}_*` (div #7, 14 campos per-query)**: solo LIGHT_RAG con synthesis activa. Parser `shared/citation_parser.py::parse_citation_refs` invocado dos veces en `_process_single_async`: sobre narrativa synth y sobre respuesta final, cada una con `n_chunks_emitted` como rango valido. 7 contadores por prefijo: `total`, `valid` (formato estricto `[ref:N]`), `malformed` (variantes), `in_range`, `out_of_range` (**senal roja**), `distinct`, `coverage_ratio`.

**Interpretacion cruzada faithfulness × citations** (casos criticos; el espacio completo es 2×4):

| Faithfulness | Citations | Diagnostico |
|---|---|---|
| Alta | `out_of_range = 0`, `valid > 0` | **Narrativa ideal**: respuesta consistente con contexto + anclaje trazable per-claim |
| Alta | `out_of_range > 0` | Respuesta OK pero alucinacion de referencias — revisar prompt del synth |
| Baja | `out_of_range > 0` | **Peor caso**: contenido fabricado Y citas fabricadas juntas. Alarma maxima |
| Cualquiera | `gen_valid > synth_valid` | Generador invento citas sobre la narrativa — separar del riesgo #2 (synth) |
| Cualquiera | `malformed > 0` | Problema de prompt/modelo con formato — revisar `KG_SYNTHESIS_SYSTEM_PROMPT` o cambiar modelo |

`out_of_range > 0` o `malformed > 0` agregados a nivel de run son **senales rojas equivalentes a `judge.default_return_rate > 2%`**: el run sigue ejecutando pero el observable discriminativo esta comprometido antes de interpretar deltas.

## Bare excepts tolerados (con criterio)

Estos `except Exception as e:` logean el error y devuelven un fallback en vez de re-lanzar. El criterio para tolerarlos: estan en wrappers de infraestructura donde el run debe continuar ante errores operacionales puntuales (ChromaDB transitorio, NIM latencia), cada uno **contabiliza el evento en stats** (`kg_synthesis_stats`, `operational_stats`, `rerank_failures`) y el fallback es observable desde el JSON del run. Si un bare except no cumple esas condiciones, es candidato a reclasificar.

La tabla lista los sites con **contador observable**. Se referencian por funcion en lugar de numero de linea para evitar drift en cada refactor (los numeros driftan en cuanto se mueve codigo; los nombres de funcion son estables).

| Ubicacion | Contexto | Contador |
|---|---|---|
| `reranker.py::CrossEncoderReranker.rerank` | Reranking error — retorna fallback sin rerank | `rerank_failures` |
| `generation_executor.py::_synthesize_kg_context_async` | `asyncio.TimeoutError` + `Exception` genericos durante synthesis KG — fallback al contexto estructurado | `kg_synthesis_stats.errors/timeouts` |
| `retriever.py::LightRAGRetriever.index_documents` (KG build) | Fallo construyendo el KG — fallback a vector puro | `operational_stats.retrieval_error` |
| `retriever.py` synthesis de descripciones multi-doc | LLM merge de descripciones fallo — concatenacion plana | `operational_stats.description_synthesis_error` |
| `retriever.py` query a Chunk Keywords VDB (div #10) | Keyword omitido en el canal | `operational_stats.chunk_keywords_vdb_error` |
| `retriever.py` enrichment 1-hop (div #9) | Entidad sin vecinos | `operational_stats.neighbor_lookup_failure` |
| `triplet_extractor.py` gleaning round | Sin tripletas extra | `operational_stats.gleaning_error` |
| `triplet_extractor.py` parse JSON keywords | Keywords vacias para el doc | `operational_stats.keywords_parse_failure` |
| `generation_executor.py` generacion LLM final | Respuesta con `[ERROR: ...]` | `operational_stats.generation_error` |

**Bare excepts log-only** (sin contador, no listados): `vector_store.py` (5 sites de operaciones ChromaDB), `metrics.py` (LLM-judge fallback ya queda visible via `judge_fallback_stats.default_returns`), `core.py::SimpleVectorRetriever` (3 sites), `loader.py`, `evaluator.py` (cleanup, init reranker), `retrieval_executor.py`, `embedding_service.py` (retry + GET /v1/models), cleanup de VDBs en `retriever.py`, fallos de metricas primaria/secundaria en `generation_executor.py`. Todos cumplen log + fallback + run continua, pero degradan localmente sin senalizar al `config_snapshot._runtime`. Si en un debug post-run aparece comportamiento inesperado y nada en `operational_stats` lo explica, revisar logs de estos sites.

## Test coverage

`conftest.py` mockea modulos de infra (dotenv, boto3, langchain, chromadb) si no estan instalados. Tests de integracion requieren NIM + MinIO reales. Dependencias opcionales para suite completa: `python-igraph`, `snowballstemmer`. Referencia completa en `TESTS.md`.

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio. Parece complejo pero es necesario

## Limitaciones de claude_code sobre este proyecto

**Claude_code no tiene acceso a la infraestructura del usuario.** NIM + MinIO viven en el entorno privado del usuario (deuda #9). Solo el usuario puede replicar la infra y ejecutar runs reales. Claude_code nunca ve `kg_synthesis_stats`, `judge_fallback_stats` ni `retrieval_metadata` reales antes de declarar un cambio como completo — todo lo que entrega es **hipotesis pendiente de verificacion humana**. Un run LIGHT_RAG completo consume ~1h30min y presupuesto de NIM; un parametro mal capeado o un test que no estresa el edge case real solo se descubre cuando las metricas ya no son interpretables.

**Implicaciones operativas**:
- Cambios al flujo retrieval→synthesis→generation solo se validan cuando el usuario lanza el run.
- Antes de declarar completo, claude_code debe enumerar los **criterios observables** que el usuario comprobara: que variable de `config_snapshot._runtime` esperar, que rango en `kg_synthesis_stats`/`judge_fallback_stats`, que metrica agregada.
- "Falta test e2e de X" NO es deuda pendiente — es consecuencia estructural. Listarla induce intentos de construir algo imposible de validar en sesion.

**Contramedidas (vehiculos pre-run; ninguno equivale a validar)**:
1. **Auto-evaluacion como parte del entregable**: enumerar limitaciones y criterios observables ANTES de entregar.
2. **Distinguir "menor" de "conveniente de no arreglar"**: si la razon para clasificarlo como menor es el coste de arreglarlo, es deuda real.
3. **Parametros que dependen del LLM** (timeouts, contexto, concurrencia) van al `.env`, no a `constants.py`. `constants.py` solo para cosas que nunca deberian tocarse (p.ej. `CHARS_PER_TOKEN`).
4. **Tests unitarios adversariales** para todo cambio que afecte el contexto que ve el LLM: estresar el budget, no solo el caso holgado.

## Despliegue alternativo: JupyterLab + OpenWebUI

Ademas del entorno principal (Linux Zen5+H100 con NIM internos), el motor se
despliega como modulo experimental en un pod **JupyterLab** con disco persistente
(NFS bajo `/home/jovyan`) y **B200 MIG 3g.90gb** (89 GB, CUDA 13.2).

**LLM y embeddings via OpenWebUI sin cambios de codigo**. El dialecto
OpenAI-compatible de OpenWebUI es suficiente para que `ChatNVIDIA` y
`NVIDIAEmbeddings` (`langchain-nvidia-ai-endpoints==0.3.19`) funcionen apuntados
a `https://open-webui.ia.labia.tics/api`. Verificado por smoke test: chat sync
+ async + embeddings + fallback `langchain-openai`, los cuatro caminos verde.
La api_key va en `NVIDIA_API_KEY` del `.env`; `langchain-nvidia` la lee
automaticamente.

- **Chat**: `coding-qwen3-coder-next` (32K ctx, sin thinking-mode, ~1 s por
  extraccion de tripletas sobre chunks de ~5K chars).
- **Embeddings**: `demo-bge-m3` (1024 dims, ~60 ms, vectores bit-identicos
  entre `NVIDIAEmbeddings` y `OpenAIEmbeddings`).

**Particularidades del entorno** (relevantes para tunear `.env`):

- `/opt/conda` esta en overlay (**no persiste**); venv, datos MinIO, cache KG
  y repo van en `/home/jovyan` (NFS 1 TB).
- **GitHub bloqueado** por proxy corporativo → traer el repo via GitLab interno
  (mirror) o tarball por la UI de JupyterLab.
- **`dl.min.io` bloqueado** → el binario de MinIO se sube una vez por la UI
  (≈110 MB, queda en NFS).
- `tmux`/`screen` no disponibles → procesos en background con `nohup`. Los
  datos persisten en NFS pero **los procesos mueren al reiniciar el pod**
  (relanzar MinIO manualmente).
- **OpenWebUI no expone `/v1/models` en formato NIM** → la autodeteccion de
  context window (deuda #5) cae al fallback silencioso de 4000 chars. Fijar
  `GENERATION_MAX_CONTEXT_CHARS` explicito en el `.env`.
- **Sin NIM de rerank** en el pod → `RERANKER_ENABLED=false` (LIGHT_RAG ya lo
  tiene gated off para indexacion; lo apagamos tambien para que la validacion
  de config no exija la URL).

**Plantilla `.env`**: [`sandbox_mteb/env.example.jupyter`](sandbox_mteb/env.example.jupyter).
**Scripts de bootstrap**: [`scripts/jupyter/`](scripts/jupyter/) (venv, MinIO,
bucket). Ver su `README.md` para el orden exacto.
**Estado**: smoke test de provider verde; despliegue MinIO + carga de
coleccion + indexacion siguen pendientes (ver `FAST_NOTES.md`).

## Proximos pasos

### Orden de prioridades

```
Pre-P0 (completitud arquitectural + ejecucion estable)   <-- GATE CERRADO (2026-04-19)
  |
Construccion de KG: PROBADA                              <-- HITO (2026-05-20)
  |
INTEGRACION (contrato de ingesta con LI_AD)              <-- FASE ACTUAL
  |   (rebanada de ingesta de P3, adelantada)
  |
  +-- P0  comparacion empirica LIGHT_RAG vs SIMPLE_VECTOR   <-- EN PAUSA (no se persigue)
  +-- P2  experimento 3: catalogo especializado             <-- EN PAUSA
  +-- P3  embedibilidad + export de KG (resto)              <-- futuro
```

### Pre-P0 — GATE CERRADO (2026-04-19)

Tres condiciones cumplidas simultaneamente:

1. **Arquitectonicamente completa**: Divergencias resueltas en codigo + tests. Detalle por divergencia en la tabla "Status de validacion".
2. **Ejecucion estable**: `kg_synthesis_stats.fallback_rate = 2.86%` (< 10% umbral), `judge.default_return_rate = 0%`, `retrieval_metadata.kg_fallback = null` en 35/35 queries, `queries_failed = 0`.
3. **Funcionalidades extra documentadas**: cache de KG, fallbacks ante errores LLM/igraph, instrumentacion de timing (queue/LLM split) — adaptaciones operativas, no sustitutos de piezas del paper.

**Config recomendada** para runs LIGHT_RAG en infra actual (defaults en `sandbox_mteb/env.example`):
- `NIM_MAX_CONCURRENT_REQUESTS=32`
- `KG_SYNTHESIS_MAX_CHARS=50000`
- `KG_SYNTHESIS_TIMEOUT_S=180`

**Palanca post-Pre-P0**: `KG_GLEANING_ROUNDS` (default `0`) ejecuta una pasada extra de extraccion para recuperar entidades/relaciones perdidas (no re-extrae keywords de chunk). Coste: ~2x llamadas LLM en indexacion. Usar solo si la cobertura del KG (`num_docs_with_entities / total_docs`) baja de ~95%.

** Matiz sobre #10 (piggyback)**: la calidad del canal high-level **NO** esta validada empiricamente. Es **bloqueante antes de P2**, no de P0.

### Construccion de KG — PROBADA (HITO 2026-05-20)

La construccion de grafos esta probada: el motor LIGHT_RAG construye KGs de forma estable y completa sobre corpus arbitrarios. A partir de este hito **se abandona la comparacion empirica LIGHT_RAG vs SIMPLE_VECTOR** (P0, ahora en pausa — ver mas abajo) y el foco pasa a la integracion con LI_AD.

**Evidencia — runs en `./results/`** (clasificados por experimento; estado factual, la salud per-run vive en cada JSON para inspeccion humana):

| Carpeta | Run | Infra / config | Status |
|---|---|---|---|
| `01_..._pre-parallelization_baseline` | `light_rag_20260510_172854` | nemotron-3-nano | baseline pre-paralelizacion |
| `02_..._post-parallelization_pr71` | `light_rag_20260511_103950` | nemotron-3-nano + gleaning paralelo (PR #71) | OK |
| `03_..._tracker-active_pr72` | `light_rag_20260511_125710` | nemotron-3-nano + tracker per-fase (PR #72) | OK |
| `04_qwen3-32b-nvfp4_dgx-spark_sanity-check` | `light_rag_20260512_084530` | Qwen3-32B NVFP4 en DGX Spark | sanity-check OK |
| `05_..._INVALID-endpoint-died-during-queries` | `light_rag_20260513_091337` | Qwen3-32B NVFP4 en DGX Spark | **INVALID** (endpoint cayo durante queries) |

Todos los runs conservados son `LIGHT_RAG`; el baseline `SIMPLE_VECTOR` previo (`20260510_155134`) se elimino del repo al dejar de perseguirse la comparacion.

**Lecciones operativas confirmadas (aplican a futuros runs de indexacion sobre nemotron-3-nano)**:
- Thinking-mode burn: `KG_EXTRACTION_MAX_TOKENS=8192` + `KG_BATCH_DOCS_PER_CALL=1` reduce drasticamente los warnings de empty content vs los defaults (4096 + 5).
- `KG_GLEANING_ROUNDS=1` con config limpia es viable sin las catastrofes de tiempo de runs antiguos.
- Estimaciones de tiempo por claude_code para runs LIGHT_RAG con thinking-mode son poco fiables (error 13x registrado). Asumir bandas amplias.

### INTEGRACION — Contrato de ingesta con LI_AD · FASE ACTUAL

**Objetivo**: cerrar el **contrato de ingesta** — el esquema MinIO/Parquet que LI_AD (productor) escribe y el motor (consumidor) lee — y adaptar el loader para consumirlo. El transporte no cambia (MinIO + Parquet leido por `sandbox_mteb/loader.py`); cambia el **esquema**: de los pasajes HotpotQA (`doc_id`/`title`/`text`) a chunks de PDFs con procedencia.

**Contrato vivo**: [`INGESTION_CONTRACT.md`](INGESTION_CONTRACT.md). Negociado con la sesion de LI_AD (v0 nuestro -> respuesta de LI_AD -> v1). Puntos acordados:
- **Layout**: `{prefix}/collections/{collection_id}/` con `chunks/{stem}.parquet` (un Parquet por documento) + `collection.json` (manifest).
- **Manifest-as-entrypoint**: el loader entra por `collection.json` (lista de parts + filas por part + `generation` + `chunking_fingerprint`), no por glob de directorio.
- **Esquema chunks**: `chunk_id`, `collection_id`, `text` (requisito); `document_id`, `chunk_index` como columnas (no se parsea el id); `source_file`/`page_*`/`token_count` (procedencia).
- **Clave de indexacion**: `(collection_id, chunk_id)`; el motor carga una coleccion por indice.
- **Char cap (Refinamiento B, opcion A cerrada 2026-06-11)**: el manifest emite `max_chunk_chars = max(5000, max observado)` (cota verdadera, sin hard-split en el chunker de LI_AD). El motor valida `text <= max_chunk_chars` al cargar y **rechaza la coleccion** si `max_chunk_chars > KG_MAX_TEXT_CHARS` (fail-fast en `06_index_collection.py`; nunca truncado silencioso).
- **Invalidacion de cache**: `(collection_id, generation, chunking_fingerprint)` + params de extraccion del motor reemplazan el hash de contenido de `_corpus_fingerprint`; rebuild solo si avanza `generation` (completo por generacion; append incremental = deuda #10).
- **Consistencia (Refinamiento A, prioridad baja)**: validar filas-por-part contra el manifest para detectar lectura a mitad de re-chunk; blindaje para concurrencia futura, no bloquea v1.

**Trabajo de codigo**: hecho — `loader.py::MinIOLoader.load_collection(collection_id)` entra por el manifest, lee las parts, mapea cada chunk a `NormalizedDocument` (`chunk_id`->`doc_id`, `text`->`content`, procedencia->`metadata`) y valida el contrato pre-carga (columnas requisito, unicidad de `chunk_id`, `text` <= `max_chunk_chars`, filas-por-part, `num_chunks`), fallando temprano con `ValueError`. Lee del prefijo dedicado `S3_COLLECTIONS_PREFIX` (default `admin/collections`), independiente de `S3_DATASETS_PREFIX` (eval) para que ingesta y HotpotQA convivan. Cubierto por tests unitarios en `tests/test_loader.py` (fixtures sinteticos). **Pendiente**: (1) validar contra un `collection.json` + `chunks/*.parquet` reales de LI_AD; (2) cablear `load_collection` en el flujo run/evaluator (selector de coleccion + modo ingesta vs eval); (3) integrar `(collection_id, generation, chunking_fingerprint)` en la invalidacion de cache del KG (`_corpus_fingerprint` en `retriever.py`). (2) y (3) tocan paths que solo se validan con runs reales (ver "Limitaciones de claude_code").

**Export reciproco del KG**: LI_AD depende de leer el grafo para su UI (su Fase 2). Schema propuesto en [`KG_CONTRACT.md`](KG_CONTRACT.md); implementacion P3 (KG hoy efimero).

### P0 — Comparacion empirica LIGHT_RAG vs SIMPLE_VECTOR · EN PAUSA

**Estado**: en pausa, no se persigue. Se conserva el contexto por si se retoma.

**Por que no es respondible en este harness**: en HotpotQA-DEV el vector satura el retrieval (Hit@5 ~ 1.0 con cosine puro sobre embedding moderno), por lo que LIGHT pierde por sustituir vector -> KG-mediated, no por mala calidad del motor. Ademas el reranker esta **gated off para LIGHT_RAG** (`sandbox_mteb/retrieval_executor.py`; justificacion: el cross-encoder single-hop penalizaria los docs multi-hop del KG), lo que hace la comparacion **asimetrica** (SIMPLE con reranker, LIGHT sin el) e infla deltas a favor de SIMPLE donde el vector satura. La pregunta directa "LIGHT > SIMPLE en HotpotQA" no es respondible; quedaba como argumento deductivo hacia P2.

### P2 — Experimento 3: catalogo especializado · EN PAUSA

**Estado**: en pausa (decision de producto, posterior a la integracion). Sobre un catalogo privado de 10-50 PDFs especializados (upstream de LI_AD), LIGHT_RAG vs SIMPLE_VECTOR en dos ejes:

- **Eje 1 — Precision/calidad**: delta esperado LIGHT_RAG > SIMPLE_VECTOR (>3-5pp gen score, >5-10pp Recall@K) porque el embedding no ha visto el dominio y el KG se construye del propio corpus.
- **Eje 2 — Resistencia a alucinacion**: el KG aporta grounding explicito que deberia reducir la fabricacion cuando el retrieval devuelve chunks poco relevantes. Se mide via `faithfulness` (LLM-judge); fiable solo si `judge_fallback_stats` reporta tasas sanas.

**Bloqueado ademas por** el riesgo piggyback [#10](#div-10) (calidad del canal high-level no validada).

### P3 — Embedibilidad + export de KG · futuro

La **rebanada de ingesta/contrato** de P3 se adelanto a la fase activa (ver "INTEGRACION", arriba). El resto queda como futuro:

- **Embedibilidad**: configuracion via dict inyectado (no solo `.env` global), corpus en memoria, separar "cargar/indexar" de "evaluar" para reusar indices entre runs, sin asunciones sobre el filesystem excepto `EVALUATION_RESULTS_DIR` explicito.
- **Export de KG a MinIO/Parquet**: serializador `KnowledgeGraph` -> Parquet (nodos + aristas + pesos + metadatos de co-ocurrencia) + VDBs. Schema reciproco propuesto en [`KG_CONTRACT.md`](KG_CONTRACT.md) (lo necesita LI_AD para su Fase 2). Hoy el KG es efimero (igraph + ChromaDB en memoria, descartado en `_cleanup()`). Sin export, multi-tenant y versionado son imposibles.

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`; deuda #3)
- Lock-in a NVIDIA NIM (deuda #9) — solo reproducible con acceso a NIM
