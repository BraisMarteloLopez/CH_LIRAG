# HANDOFF.md

> **Archivo efimero**. Plan de trabajo para la siguiente sesion de
> claude_code. En cuanto se complete o se pivote de forma justificada,
> **eliminar este fichero** en el commit final.
>
> **NO referenciar desde `CLAUDE.md`, `README.md`, `TESTS.md` ni
> `PENDING_AUDIT.md`** — su existencia es transitoria por diseno.

## Contexto

La sesion anterior cerro la implementacion de la divergencia #10
(chunk keywords VDB, tercer canal del path high-level) y audito el
run `mteb_hotpotqa_20260418_223530` que la ejercitaba. Conclusiones:

- **#10 validada arquitectonicamente**: criterios 1 (LLM emite
  `high_level_keywords`, 987/1000 docs) y 2 (Chunk Keywords VDB se
  construye sin errores) pasan. Criterio 3 (matches per-query) **no
  auditable** desde los exports actuales porque el JSON no serializa
  `retrieval_metadata` per-query y el CSV tiene un guard de codigo
  muerto (ver `PENDING_AUDIT.md` item D).
- **Pre-P0 sigue bloqueado**: `kg_synthesis_stats.fallback_rate=0.57`
  (57%, > 10% umbral de CLAUDE.md). La causa raiz (`deuda #16`) no
  es discriminable con los datos actuales entre "saturacion de cola
  por semaforo" y "llamadas LLM genuinamente lentas".

## Entrada

Antes de actuar, leer en este orden:

1. `CLAUDE.md` — contexto del producto, divergencias con el paper,
   deuda tecnica, gate Pre-P0 con sus tres condiciones.
2. `PENDING_AUDIT.md` — 4 items menores documentados con evidencia,
   impacto, y accion propuesta (A, B, C, D).
3. `data/results/mteb_hotpotqa_20260418_223530.json` + 
   `data/results/console_log_light_rag_20260418_223530.txt` — el run
   que motivo este plan.

## Plan

Tres fixes de codigo (paso 1) + run diagnostica (paso 2) + analisis
(paso 3) + run de verificacion con ajustes (paso 4).

### Paso 1 — Paquete de fixes

Los tres son pequenos pero **sin ellos la siguiente run no es
diagnostica**.

**(1.1) Deuda #15 + PENDING item D**, en un solo fix coherente:
- Extender `QueryEvaluationResult.to_dict()` en `shared/types.py`
  para serializar un bloque filtrado de `retrieval_metadata` por
  query: `kg_fallback`, `kg_chunk_keyword_matches`, conteos de
  `kg_entities`/`kg_relations`, `kg_synthesis_used`,
  `kg_synthesis_error`.
- Reemplazar el guard `has_lightrag` en `shared/report.py:201-204`
  (hoy busca `graph_candidates`, clave eliminada con la resolucion
  de #8). Usar claves actuales (`kg_entities` o
  `kg_chunk_keyword_matches`). Actualizar la lista de columnas CSV
  para que coincidan con las claves actuales (no las legacy
  `graph_candidates` / `vector_candidates` / etc.).
- **Al cerrar, reescribir el row de deuda #15 en `CLAUDE.md`**: hoy
  describe solo el gap del JSON. Debe cubrir tambien el gap del CSV
  o marcarse como cerrada si queda todo.
- Tests: un case con `kg_chunk_keyword_matches > 0` en JSON export,
  uno con `kg_fallback != None`, uno donde `has_lightrag` se activa
  con metadata real del retriever actual.

**(1.2) Deuda #16 (instrumentacion)**:
- En `_kg_synthesis_tracker` (`sandbox_mteb/generation_executor.py`),
  capturar `start_time` antes del `asyncio.wait_for` y `end_time` al
  resolver (exito, timeout, error, empty). Acumular listas de
  duraciones por categoria.
- Exponer `p50_ms`, `p95_ms`, `max_ms` en
  `config_snapshot._runtime.kg_synthesis_stats`.
- **Opcional pero recomendado** (discrimina las hipotesis): separar
  "tiempo-en-cola" (creacion del task → semaforo acquire) de
  "tiempo-en-LLM" (acquire → respuesta). Requiere hook ligero en
  `AsyncLLMService.invoke_async` que reporte esos dos timestamps al
  caller. Exponer `p50_queue_ms` y `p50_llm_ms` separados.
- Tests: mock del LLM con delays conocidos (p.ej. `asyncio.sleep(2)`
  dentro del mock), verificar que p50 refleja los delays.

**(1.3) PENDING item A** (error_message vacio):
- En el except handler del evaluator async (buscar
  `Error async query` en `sandbox_mteb/evaluator.py`), escribir
  `f"{type(exc).__name__}: {str(exc) or repr(exc)}"` en
  `QueryEvaluationResult.error_message`.
- Test: simular excepcion con `str()` vacio (p.ej. `TimeoutError()`),
  verificar que `error_message` contiene al menos `"TimeoutError"`.

Al cerrar cada fix actualizar el estado en `PENDING_AUDIT.md` segun
el procedimiento que el propio archivo describe
(`open → closed pending doc review → closed`).

### Paso 2 — Run diagnostica

Objetivo: medir con la instrumentacion nueva sin cambiar nada mas.
Comparacion apples-to-apples con `mteb_hotpotqa_20260418_223530`.

- Config: **exactamente la misma** que la anterior — 35 queries,
  1000 docs DEV_MODE, seed=42, `LIGHTRAG_MODE=hybrid`,
  `KG_SYNTHESIS_TIMEOUT_S=90`, `NIM_MAX_CONCURRENT_REQUESTS=16`,
  `KG_CHUNK_KEYWORDS_ENABLED=true`.
- Antes de lanzar: verificar `KG_CACHE_DIR` vacio o sin caches v2
  (si los hay, la serializacion v3 lanza `ValueError` explicito).

### Paso 3 — Analisis de la run diagnostica

**Para #10 criterio 3**: contar cuantas queries tienen
`retrieval_metadata.kg_chunk_keyword_matches > 0` en el JSON
exportado. Si la fraccion es razonable (>30% en hybrid), promover
#10 a **"Resuelta"** en la tabla de divergencias de `CLAUDE.md`.

**Para #16**: con `p50_queue_ms` y `p50_llm_ms` separados:
- `p50_queue_ms > 60s` → saturacion de cola → subir
  `NIM_MAX_CONCURRENT_REQUESTS` (o reducir batch size de queries).
- `p50_llm_ms > 60s` → llamadas LLM genuinamente lentas → reducir
  `KG_SYNTHESIS_MAX_CHARS` (recorta el prompt) o subir
  `KG_SYNTHESIS_TIMEOUT_S`.
- Si ambos estan altos: ajuste combinado.

Si la instrumentacion con separacion no se implemento, solo se
tendra `p50_ms` total → el diagnostico es parcial; puede requerir
otra iteracion.

### Paso 4 — Run de verificacion

Aplicar el ajuste decidido en el paso 3 y re-ejecutar con la misma
config que el paso 2 excepto los parametros ajustados.

**Criterio de exito**: `kg_synthesis_stats.fallback_rate < 0.10`.

- Si se cumple + criterio 3 de #10 ya resuelto: **Pre-P0 cerrado**.
  Actualizar `CLAUDE.md` seccion "Contexto del producto" (fase
  actual pasa a P0) y seccion "Proximos pasos · Pre-P0" (gate
  cerrado). Eliminar este archivo en el commit final.
- Si no se cumple: analizar los nuevos datos, iterar en paso 3 o
  consultar con el usuario antes de seguir.

## Puntos de decision que requieren consultar al usuario

No improvisar en estos casos:

- Si los fixes del paquete (paso 1) generan regresiones en tests
  existentes y la causa no es trivial.
- Si la run diagnostica (paso 2) muestra `fallback_rate` igual o
  mayor al anterior (podria ser que la instrumentacion anadiera
  coste, o que el LLM tenga dia malo).
- Si los datos p50_queue/p50_llm no convergen a una accion obvia
  (p.ej. ambos <30s pero fallback_rate alto).
- Si la separacion cola/LLM es significativamente costosa de
  implementar (el hook en `AsyncLLMService` podria requerir
  cambios mas amplios).

## Avisos sobre patrones de fallo de la sesion que abrio este plan

Para que la proxima sesion los evite:

1. **No inventar "patrones del proyecto"** para justificar
   priorizaciones o criterios. Si una decision requiere apoyo
   doctrinal, citar `CLAUDE.md` literalmente o no reclamarlo.
2. **No marcar divergencias como "Resuelta"** hasta que los
   criterios observables esten verificados end-to-end con NIM +
   MinIO reales. El gate Pre-P0 lo exige explicitamente.
3. **Al auditar un run, leer los 4 ficheros completos** (JSON,
   summary.csv, detail.csv, console log entero). No usar grep
   parcial como sustituto del log completo — se pierden patrones
   temporales (p.ej. clustering de timeouts).
4. **Desconfiar de afirmaciones propias en docs** cuando no estan
   verificadas contra el codigo actual. La deuda #15 describia un
   comportamiento del CSV que no ocurre (guard muerto) — el autor
   del texto era un agente anterior, y la afirmacion no se
   reverifico hasta que fallo.

## Procedimiento de cierre

Cuando el plan termine (exito o pivote justificado):

1. Revisar estado de los 4 items en `PENDING_AUDIT.md`. El item D
   cerrara en paso 1.1; A cerrara en paso 1.3. B y C quedan `open`
   para sesiones posteriores si no se tocan.
2. Si #10 se promueve a "Resuelta": actualizar la tabla en
   `CLAUDE.md` + el historial en `README.md` (seccion Post-refactor
   abril 2026).
3. Si Pre-P0 cierra: actualizar `CLAUDE.md` seccion "Contexto del
   producto" (fase actual → P0) y `CLAUDE.md` seccion
   "Proximos pasos · Pre-P0" (marcar gate cerrado).
4. **Eliminar `HANDOFF.md`** en el commit final del plan.
