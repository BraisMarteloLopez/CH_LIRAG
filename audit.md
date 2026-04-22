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

## Decisiones revisadas

Si un hallazgo `rejected` pasa a `accepted` (o viceversa), registrar aqui:

- `YYYY-MM-DD` — H-N: {rejected → accepted | accepted → rejected}. Razon: {texto breve}.

---

## Historial (hallazgos cerrados, opcional)

Cuando el log principal supere ~30 entradas cerradas, mover aqui las de fases `done` para mantener legibilidad.
