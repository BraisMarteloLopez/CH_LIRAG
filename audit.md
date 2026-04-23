# audit.md

Auditoria iterativa del codebase en fases, diseñada para avanzar entre sesiones de Claude Code sin perdida de contexto. Este archivo **es la fuente unica** del estado de la auditoria; no hay memoria en otro sitio.

---

## Protocolo cross-session

**Al abrir una sesion para continuar la auditoria**:

1. Leer la tabla "Fases" abajo. Localizar la fase con `status: active`.
2. Leer el bloque "Fase activa" para el scope y el metodo.
3. Leer los hallazgos `open` de esa fase.
4. **NO empezar otra fase** hasta que la activa cierre (todos sus hallazgos en estado `accepted/rejected/merged-*`). Multi-fase en paralelo rompe la trazabilidad.
5. **NO generar PRs sin registrar el hallazgo antes** — el log es append-only y el id del hallazgo va en el commit message del PR que lo cierra.
6. **NO reclasificar un hallazgo `rejected` a `accepted` sin dejar una linea en "Decisiones revisadas"** abajo (con fecha + razon). Rechazos son decisiones del usuario, no heuristicas.

**Cierre de fase**: cuando todos los hallazgos de la fase estan resueltos, marcar la fase como `done`, mover hallazgos cerrados a "Historial" (opcional si el log crece), y activar la siguiente fase del orden. Dejar al usuario el OK antes de activar la siguiente.

**Regla de conservacion**: si una sesion termina sin cerrar hallazgos, actualizar "Fase activa" → "Avance de la sesion" con lo explorado para que la proxima sesion no repita trabajo.

---

## Fases

Orden de menor a mayor invasividad. Una fase solo se activa cuando la anterior cierra (`status=done`).

| Id | Nombre | Scope | Status | Exit criteria |
|---|---|---|---|---|
| F1 | Dead code y exports innecesarios | Simbolos sin callers (funciones, clases, atributos de dataclass, items en `__all__`, imports) en `shared/`, `sandbox_mteb/` | pending | Todos los hallazgos cerrados; `grep` confirma 0 simbolos sin uso externo marcados como publicos |
| F2 | Comentarios obsoletos / redundantes / engañosos | Referencias a codigo eliminado, comentarios que restate el nombre del identificador, TODOs sin dueño, docstrings que divergen del comportamiento, referencias a `HKUDS/LightRAG` rotas | active | Todos los hallazgos cerrados; pasada completa por `shared/` y `sandbox_mteb/` |
| F3 | Naming y consistencia ES/EN | Mezcla ES-EN en identificadores + docstrings, prefijos/sufijos inconsistentes, `_privado` sin razon, nombres que cambian entre modulos para el mismo concepto | pending | Todos los hallazgos cerrados; glosario de terminos canonicos documentado si emerge |
| F4 | Tests vs produccion (paridad) | Tests que cubren features eliminadas, helpers fragmentados entre archivos, tests que mockean mas de lo que testean, coverage gaps declarados en TESTS.md vigentes | pending | Todos los hallazgos cerrados; `TESTS.md::Modulos sin tests dedicados` y `Gaps de cobertura conocidos` reconciliados con el estado real |
| F5 | Estructura de carpetas y modulos | Modulos sobredimensionados (>500 lineas), files que deberian fusionarse o partirse, layering `shared/` vs `sandbox_mteb/` (violaciones de dependencia), agrupacion por dominio vs por tipo | pending | Todos los hallazgos cerrados; arbol final documentado en README.md reconcilia con realidad |
| F6 | Configuracion y constantes | `.env` vs `constants.py` vs hardcoded, defaults razonables, env vars no documentadas en `env.example`, validacion faltante en `MTEBConfig.validate()` | pending | Todos los hallazgos cerrados; `env.example` es inventario exhaustivo de lo que el codigo lee |
| F7 | Drift de documentacion | CLAUDE.md, README.md, TESTS.md referenciando cosas que ya no existen o con contadores estancados; divergencias numericas (tablas de hallazgos, tablas de tests) | pending | Todos los hallazgos cerrados; cada referencia cross-doc verificada |

---

## Fase activa

### F2: Comentarios obsoletos / redundantes / engañosos

**Scope (literal)**: `shared/*.py`, `shared/retrieval/*.py`, `shared/retrieval/lightrag/*.py`, `sandbox_mteb/*.py`. Excluye `tests/`, docs markdown y `env.example`.

**Metodo**:
  1. Lectura completa de los archivos en scope; identificar comentarios/docstrings que violen las reglas de CLAUDE.md §"Doing tasks" (default no comments; solo WHY no-obvio).
  2. Criterios para clasificar hallazgo (NO falso positivo):
     - **WHAT**: restate lo que el codigo o el identificador ya dice.
     - **BANNER**: `# ===== SECCION =====` decorativos sin contenido.
     - **HISTORICAL/TASK-REF**: "FIX:", "extraido de X", "la version anterior", referencias a PR/fase/refactor historico.
     - **VERBOSE**: bloques >3 lineas que duplican informacion ya en CLAUDE.md (divergencias, deuda tecnica, observabilidad) y cuya referencia via `div-N`/`dt-N` es suficiente.
     - **ATTR-DOC**: comentarios inline en campos de dataclass que repiten el nombre/tipo.
  3. NO candidato (conservar):
     - Anchors de trazabilidad `# divergencia #N`, `# deuda #N`.
     - Referencias a paper/HKUDS con linea especifica (ver §divergencias "Upstream pin").
     - Comentarios que documenten workaround, invariante oculta, contrato con sistema externo.
     - Contratos externos en docstrings de modulo (env vars, schema MinIO, endpoints NIM).
  4. Registrar hallazgo en tabla "Hallazgos" con archivo + rango de lineas + categoria.

**Anti-scope**:
  - NO renombrar identificadores (F3).
  - NO mover codigo entre modulos (F5).
  - NO reescribir docstrings publicos con contrato con tests (F4).
  - NO tocar comentarios en `tests/` (F4).
  - NO alterar informacion contenida en `CLAUDE.md`, `README.md`, `TESTS.md` ni en `audit.md` salvo para cerrar hallazgos.

**Avance de la sesion** (append-only):
  - 2026-04-23: pasada completa por scope. Registrados H-1..H-11 agrupados por archivo (cada H-N agrupa comentarios del mismo fichero para evitar inflar el log). Aplicacion y merge al cierre de la sesion.

---

## Hallazgos

Log append-only. **No reescribir entradas cerradas**; si una decision se revisa, dejar la entrada original y agregar nueva en "Decisiones revisadas".

Formato: `| H-N | F{fase} | {ubicacion} | {recomendacion} | {status} | {nota corta} |`

Status:
- `open` — pendiente de decision del usuario
- `accepted` — usuario OK, pendiente de PR
- `rejected` — usuario NO, NO volver a proponer sin razon nueva
- `merged-#{PR}` — cerrado en PR identificado

| Id | Fase | Ubicacion | Recomendacion | Status | Nota |
|---|---|---|---|---|---|
| H-1 | F2 | `shared/vector_store.py` L263-290 | Eliminar prefijo `FIX:` y comentario historico del `finally`; conservar docstring minimo. | accepted | HISTORICAL/TASK-REF + WHAT |
| H-2 | F2 | `shared/retrieval/core.py` L49-51, L66-83 | Colapsar comentarios de `hnsw_num_threads`, `lightrag_generation_top_n` y `kg_chunk_keywords_enabled`: referenciar dt-3 / div-10 en CLAUDE.md. | accepted | VERBOSE duplicado con CLAUDE.md |
| H-3 | F2 | `shared/retrieval/lightrag/retriever.py` L1-22, L107-146, L193-220, L263-341, L349-361, L480-484, L591-597, L849-853 | Resumir docstring principal, quitar comentarios WHAT sobre VDBs (info en CLAUDE.md), quitar banners, dejar anchors `div-N`. | accepted | Mix VERBOSE/WHAT/BANNER |
| H-4 | F2 | `shared/retrieval/lightrag/knowledge_graph.py` L117-139, L166-168, L226-229, L244-247, L336-338, L395-413, L551-553, L601-604, L628-632, L656-700, L705-707, L767-770, L793-815 | Quitar banners decorativos, attr-doc inline triviales, comentarios WHAT de secciones; conservar anchors `div-10` y la restriccion de versionado de cache como docstring breve. | accepted | BANNER + WHAT + VERBOSE |
| H-5 | F2 | `shared/retrieval/lightrag/triplet_extractor.py` L122-127, L140, L154-163, L203-224, L284-308, L401-408, L431-586, L618-665, L698-700, L773-812 | Condensar comentarios WHAT del flujo batch/dedup, quitar banners, referenciar dt-17 una sola vez. | accepted | BANNER + WHAT + VERBOSE |
| H-6 | F2 | `shared/metrics.py` L354-370 | Sustituir banner largo por 1 linea; anchors a CLAUDE.md ya cubren los detalles. | accepted | VERBOSE + BANNER |
| H-7 | F2 | `sandbox_mteb/generation_executor.py` L100-115, L252-294, L377-395 | Eliminar banner del tracker, quitar explicaciones linea-a-linea de los resolvers de metrica primaria. | accepted | BANNER + WHAT |
| H-8 | F2 | `sandbox_mteb/evaluator.py` L73-75, L103-111, L156-158, L342-344, L426-428, L449-451, L458-459, L508-510, L707-709 | Quitar banners decorativos y comentarios WHAT repetidos sobre resets/inicializacion. | accepted | BANNER + WHAT |
| H-9 | F2 | `sandbox_mteb/result_builder.py` L167-196 | Quitar bloques de comentarios que duplican CLAUDE.md §"Observabilidad de runs"; el nombre del campo + typeddict `RuntimeSnapshot` ya documentan. | accepted | VERBOSE duplicado con CLAUDE.md |
| H-10 | F2 | `sandbox_mteb/retrieval_executor.py` L1-5, L79-162 | Borrar nota historica "Fase B descomposicion"; quitar comentarios WHAT del flujo retrieval/rerank; conservar el comentario WHY sobre "LightRAG no usa reranker". | accepted | HISTORICAL + WHAT |
| H-11 | F2 | `sandbox_mteb/embedding_service.py` L1-5, L72-100, L140-181 | Borrar nota historica "Fase B descomposicion"; quitar enumeracion "# 1. / # 2. / # 3." que duplica el docstring; reducir comentarios WHAT del batch loop. | accepted | HISTORICAL + WHAT |

---

## Decisiones revisadas

Si un hallazgo `rejected` pasa a `accepted` (o viceversa), registrar aqui:

- `YYYY-MM-DD` — H-N: {rejected → accepted | accepted → rejected}. Razon: {texto breve}.

---

## Historial (hallazgos cerrados, opcional)

Cuando el log principal supere ~30 entradas cerradas, mover aqui las de fases `done` para mantener legibilidad.
