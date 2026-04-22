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
| F2 | Comentarios obsoletos / redundantes / engañosos | Referencias a codigo eliminado, comentarios que restate el nombre del identificador, TODOs sin dueño, docstrings que divergen del comportamiento, referencias a `HKUDS/LightRAG` rotas | pending | Todos los hallazgos cerrados; pasada completa por `shared/` y `sandbox_mteb/` |
| F3 | Naming y consistencia ES/EN | Mezcla ES-EN en identificadores + docstrings, prefijos/sufijos inconsistentes, `_privado` sin razon, nombres que cambian entre modulos para el mismo concepto | pending | Todos los hallazgos cerrados; glosario de terminos canonicos documentado si emerge |
| F4 | Tests vs produccion (paridad) | Tests que cubren features eliminadas, helpers fragmentados entre archivos, tests que mockean mas de lo que testean, coverage gaps declarados en TESTS.md vigentes | pending | Todos los hallazgos cerrados; `TESTS.md::Modulos sin tests dedicados` y `Gaps de cobertura conocidos` reconciliados con el estado real |
| F5 | Estructura de carpetas y modulos | Modulos sobredimensionados (>500 lineas), files que deberian fusionarse o partirse, layering `shared/` vs `sandbox_mteb/` (violaciones de dependencia), agrupacion por dominio vs por tipo | pending | Todos los hallazgos cerrados; arbol final documentado en README.md reconcilia con realidad |
| F6 | Configuracion y constantes | `.env` vs `constants.py` vs hardcoded, defaults razonables, env vars no documentadas en `env.example`, validacion faltante en `MTEBConfig.validate()` | pending | Todos los hallazgos cerrados; `env.example` es inventario exhaustivo de lo que el codigo lee |
| F7 | Drift de documentacion | CLAUDE.md, README.md, TESTS.md referenciando cosas que ya no existen o con contadores estancados; divergencias numericas (tablas de hallazgos, tablas de tests) | pending | Todos los hallazgos cerrados; cada referencia cross-doc verificada |

---

## Fase activa

**Ninguna**. Al activar una fase, reemplazar este bloque con el detalle:

```
### F{N}: {nombre}

**Scope (literal)**: {que archivos/subdirs entran, que queda fuera}
**Metodo**:
  1. {comando concreto para enumerar candidatos, p.ej. grep, ruff, wc -l}
  2. {criterio para discriminar hallazgo vs falso positivo}
  3. {forma de registrar el hallazgo — append a tabla "Hallazgos" abajo}
**Anti-scope** (cosas que la fase NO hace):
  - {lista explicita para evitar scope creep}

**Avance de la sesion** (append-only):
  - YYYY-MM-DD: explorado {X}, hallazgos registrados {H-N..H-M}, pendiente {Y}
```

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

---

## Referencia externa: HKUDS/LightRAG vs esta implementacion

Baseline para detectar sobre-ingenieria. Upstream: https://github.com/HKUDS/LightRAG (EMNLP 2025, [arxiv](https://arxiv.org/abs/2410.05779)). **No esta pineado a SHA** (ver `CLAUDE.md::Upstream pin` — deuda R13); al contrastar, clonar upstream y buscar por nombre de funcion, no por linea.

**Uso en las fases**: F1 (dead code), F2 (comentarios que refieren a comportamiento upstream que ya no aplica), F5 (estructura de modulos), F6 (config — flags que upstream no expone).

### Mapeo de modulos a contrastar

| Nuestro | Upstream (referencia) | Lineas propias | Ratio |
|---|---|---|---|
| `shared/retrieval/lightrag/retriever.py` | `lightrag/operate.py` (funciones de retrieval + query) | 1.224 | — |
| `shared/retrieval/lightrag/knowledge_graph.py` | `lightrag/kg/*` (igraph/networkx storage) | 904 | — |
| `shared/retrieval/lightrag/triplet_extractor.py` | `lightrag/operate.py` (extract_entities + keywords) + prompts | 836 | — |
| **Total motor** | | **2.984** | a calcular vs upstream |

Completar "Ratio" con `wc -l` sobre upstream cuando se active F5. **Regla heuristica**: ratio ≤ 2× se justifica con adaptaciones operativas documentadas; ratio > 2× es señal roja de sobre-ingenieria y merece hallazgo individual por cada bloque engordado.

### Categorias extra que existen aqui y no en upstream

Referencias del codebase que ya declaran el extra. Una fase que encuentre codigo aqui descrito debe confirmar que **la adaptacion sigue siendo necesaria** contra el upstream actual:

1. **Adaptaciones operativas justificadas** (CLAUDE.md::Estrategia LIGHT_RAG, seccion "Funcionalidades extra documentadas"):
   - Cache de KG a disco (re-indexacion incremental impedida, ver dt-10)
   - Fallbacks ante errores LLM/igraph (el paper no los describe; los nuestros no deberian disparar en P0 verde)
   - Instrumentacion de timing queue/LLM split, `operational_stats`, `kg_synthesis_stats`, `judge_fallback_stats`
2. **Divergencias declaradas** (`CLAUDE.md#divergencias`, items #3, #4+5, #7, #9, #10, #11, #12):
   - Reevaluar si la divergencia sigue aportando valor o si upstream ya incorporo la funcionalidad (especialmente tras pinear SHA — R13)
3. **Capa synthesis LLM del contexto** (sandbox_mteb/generation_executor.py::`_synthesize_kg_context_async`):
   - Value-add del proyecto, no existe upstream. Validar que: (a) el prompt sigue alineado con el parser de citas (dt-18), (b) el codigo defensivo alrededor de la llamada no es redundante con el propio LLM service

### Heuristica para clasificar un hallazgo como "sobre-ingenieria"

Un bloque de codigo es candidato a hallazgo de sobre-ingenieria si **las tres** condiciones se cumplen:

1. Existe aqui pero NO en upstream (modulo, funcion o metodo equivalente).
2. No esta listado en "Adaptaciones operativas justificadas" ni en la tabla de divergencias de CLAUDE.md.
3. No tiene evidencia empirica en `config_snapshot._runtime.*_stats` que demuestre que el bloque se dispara en runs reales (o la evidencia muestra que se dispara ≈0% de las veces).

Casos especiales que **no** son sobre-ingenieria aunque cumplan (1):
- Infraestructura del harness de evaluacion (`sandbox_mteb/`), que upstream no provee por diseño.
- Codigo ligado a lock-in NIM (dt-9) — se retira cuando se abstraiga el provider, no antes.

---



Si un hallazgo `rejected` pasa a `accepted` (o viceversa), registrar aqui:

- `YYYY-MM-DD` — H-N: {rejected → accepted | accepted → rejected}. Razon: {texto breve}.

---

## Historial (hallazgos cerrados, opcional)

Cuando el log principal supere ~30 entradas cerradas, mover aqui las de fases `done` para mantener legibilidad.
