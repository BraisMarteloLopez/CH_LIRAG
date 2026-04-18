# PENDING_AUDIT.md

> **Archivo efimero**. Registra tareas pendientes detectadas al auditar el
> run `mteb_hotpotqa_20260418_223530` (2026-04-18) que no subieron a
> `CLAUDE.md#deuda-tecnica-vigente` porque se consideran menores u
> operativas. **En cuanto se aborden**, revisar si alguna deja rastro en
> `CLAUDE.md`/`README.md` (p.ej. cerrar una deuda existente, actualizar
> una seccion, anadir una convencion) y **eliminar este archivo**.
>
> **NO referenciar desde `CLAUDE.md`, `README.md` ni `TESTS.md`** — un
> enlace hacia este fichero sobrevive a su borrado y queda como
> referencia rota. Su existencia es transitoria por diseno.

## Resumen de items

Estado por item. Valores posibles:
- `open` — detectado y documentado, sin trabajo iniciado.
- `in progress` — hay cambio en curso pero aun no cerrado.
- `closed pending doc review` — cambio aplicado, falta validar si
  deja rastro en `CLAUDE.md` / `README.md` / otros docs permanentes.
- `closed` — cambio aplicado y contextualizacion completada. Listo
  para eliminar del fichero junto con los demas cuando todos cierren.

| # | Item | Estado |
|---|---|---|
| A | `error_message` vacio en queries fallidas | `closed` |
| B | Empty-content retries del LLM como patron operativo recurrente | `open` |
| C | Estimacion de tiempo de indexacion obsoleta tras #10 | `open` |
| D | Codigo muerto en `shared/report.py:201-212` (columnas LightRAG del detail.csv) | `closed` |

Cuando todos los items esten `closed`, seguir los pasos de la seccion
"Procedimiento de cierre" al final del archivo.

## Contexto

Run auditado: `data/results/mteb_hotpotqa_20260418_223530` (800 docs /
35q DEV_MODE, LIGHT_RAG hybrid, seed=42, primer run post-#10). 4
ficheros examinados (JSON + summary.csv + detail.csv + console_log). La
auditoria apoyo la promocion de la divergencia #10 a "Resuelta" sobre
los criterios 1 y 2 (extraccion LLM produce keywords, Chunk Keywords
VDB se construye), pero destapo los items abajo — no bloqueantes para
Pre-P0 pero pendientes de decidir.

---

## A. `error_message` vacio en queries fallidas

**Estado**: `closed` (doc review completo: no procede cambio en CLAUDE.md
porque el item lo declara explicitamente — bugfix puntual).

**Cambios aplicados** (Fase A del HANDOFF):
- `sandbox_mteb/evaluator.py`: nueva helper `_format_query_exc(exc)` que emite
  `"{type.__name__}: {str(exc) or repr(exc)}"`. El loop async captura la
  excepcion, la pasa a `_assemble_results(gen_errors=...)`, y esta la escribe
  en `QueryEvaluationResult.error_message` del FAILED. El log WARNING tambien
  usa el mismo helper (ya no emite cadena vacia tras los dos puntos).
- `tests/test_evaluator.py`: EV10 (exception preserva tipo+msg), EV11
  (`_format_query_exc` con str() vacio cae a repr). `test_assemble_results_failed_without_exception_falls_back_to_default_message`
  mantiene compat con la ruta actual sin excepcion.

**Doc review pendiente**: ningun cambio esperado en `CLAUDE.md` (item A
explicitamente marcaba "ninguna entrada nueva en CLAUDE.md").

**Evidencia**
- `query_results[*]` en JSON: `q_978` reporta `status="failed"`,
  `error_message: null`.
- Console log a `23:22:30,296`:
  `WARNING sandbox_mteb.evaluator:   Error async query q_978:` seguido
  de cadena vacia tras los dos puntos.
- La excepcion fue capturada por el except handler del evaluator pero
  su `str()` era vacio. El handler no hace fallback a `repr(exc)` ni a
  `type(exc).__name__`, asi que se pierde toda la traza.

**Impacto**
- Medio-bajo. La query se marca fallida correctamente y el run
  continua; la metrica `queries_failed` la cuenta. Pero no hay
  diagnostico: `error_message=null` obliga a correlacionar por
  timestamp con los WARNINGs del log para inferir la causa.
- En este caso concreto (q_978): timing (308s desde dispatch a fail)
  + tres `LLM returned empty content` consecutivos antes del fail
  apuntan a que la query agoto `GENERATION_QUERY_TIMEOUT_S=300s` por
  cascada de retries. Pero el diagnostico es inferencia, no
  observacion directa.

**Accion propuesta**
- En el except handler donde el evaluator captura excepciones async
  (buscar por `Error async query` en `sandbox_mteb/evaluator.py`):
  registrar `type(exc).__name__ + ": " + (str(exc) or repr(exc))` en
  `QueryEvaluationResult.error_message`.
- Test: simular excepcion con `str()` vacio (p.ej. `TimeoutError()`),
  verificar que `error_message` contiene al menos `"TimeoutError"`.

**Ubicacion esperada tras cierre**
- Ninguna entrada nueva en CLAUDE.md — es un bugfix puntual. El diff
  habla por si mismo en `git log`.

---

## B. Empty-content retries como patron operativo recurrente

**Estado**: `open`

**Evidencia**
- Console log del run:
  - ~10 WARNINGs `Intento N/4 fallo: LLM returned empty content
    after stripping reasoning tags` durante la indexacion. El retry
    los salvo en todos los casos visibles (ninguna tripleta perdida
    por esta causa).
  - 3 WARNINGs consecutivos inmediatamente antes de que q_978
    agote `GENERATION_QUERY_TIMEOUT_S` (23:20:20, 23:20:20, 23:21:12).
- Causa mecanica: el LLM configurado (`nvidia/nemotron-3-nano`) es
  thinking-mode; a veces toda la respuesta cabe en los tags de
  razonamiento y el stripping devuelve cadena vacia. El retry con
  backoff exponencial en `shared/llm.py` lo mitiga casi siempre.

**Impacto**
- Indexacion: observacion visible pero inocua (retries exitosos).
- Generacion: el retry se encadena al timeout por query (300s) y
  puede consumirlo si la ventana se llena. En este run, contribuyo
  al unico fallo (q_978). Si la tasa de empty-responses subiera en
  runs con mas queries, la tasa de fallos de query subiria con ella.

**Accion propuesta (secuencial, no mutuamente excluyente)**

Dos acciones con orden fijo porque la segunda requiere datos de la
primera para ser decidible sin ciegas:

1. **Instrumentar primero (barato, no-invasivo)**. Anadir una stat
   `llm_empty_responses` segmentada por fase (indexacion / synthesis
   / generation) en `shared/llm.py` y exponerla en
   `config_snapshot._runtime` como ya se hace con `judge_fallback_stats`
   y `kg_synthesis_stats`. Permite medir la tasa real y correlar con
   fallos de query en los ficheros de resultado sin abrir el log.

2. **Ajustar solo si los datos de (1) lo justifican**. Si tras uno o
   dos runs con (1) se observa que la tasa de empty-responses en fase
   generation + el numero medio de retries consumidos **pueden**
   exceder el presupuesto actual (300s) de forma recurrente, entonces
   subir `GENERATION_QUERY_TIMEOUT_S` a un valor calculado desde los
   datos (p.ej. `3 × p95_llm_call_s + margen`). Riesgo de subir a
   ciegas: runs globalmente mas lentos si el LLM se degrada por otra
   causa no relacionada con empty-content.

**Ubicacion esperada tras cierre**
- Tras (1): la stat se documenta en `CLAUDE.md#observabilidad-de-runs`
  junto a `judge_fallback_stats` y `kg_synthesis_stats`, con el
  comando `jq` correspondiente.
- Tras (2), si se aplica: basta con ajustar `.env.example` + comentario
  enlazando la justificacion (run_id del que salieron los datos). No
  requiere cambio de doc permanente mas alla de eso.

---

## C. Estimacion de tiempo de indexacion obsoleta tras #10

**Estado**: `open`

**Evidencia**
- Banner en `sandbox_mteb/evaluator.py` (mensaje WARNING al inicio de
  un run LIGHT_RAG):
  `LIGHT_RAG: indexacion hara ~1000 llamadas LLM... Estimacion: ~2 min
  (1000 docs, 16 concurrentes, ~2s/llamada).`
- Real medido en el run: 41 minutos (2330s para
  `TripletExtractor.extract_batch` sobre 1000 docs).
- Gap: ~20x respecto a la estimacion.

**Hipotesis (no verificadas) sobre el gap**
- El prompt post-#10 incluye el campo `high_level_keywords` adicional,
  lo que aumenta los tokens de output generados por el LLM. A 5
  docs/call con 10 keywords/doc de output extra, son ~50 tokens extra
  por llamada; con thinking-mode del modelo, el coste real en
  latencia puede ser mayor que esos tokens sugieren.
- Posibles factores adicionales: batch_parse_failures (6 en este run)
  que fuerzan fallback a single-doc extraction (mas caro), empty-
  content retries (punto B).
- No hay forma de discriminar entre factores con los datos actuales.

**Impacto**
- Operativo: la estimacion del banner es desinformativa. Un usuario
  que dispara el run confiando en "2 min" puede interrumpirlo pensando
  que algo va mal cuando en realidad va bien pero es lento.
- Senal no cuantificada sobre el coste real de #10. Si se confirmara
  via instrumentacion que el prompt #10 es la causa principal,
  tendriamos que decidir si aceptar el coste o permitir desactivar
  `KG_CHUNK_KEYWORDS_ENABLED` como optimizacion (ya existe el flag).

**Accion propuesta**
- Inmediata y barata: recalibrar el banner con el factor observado
  (p.ej. "~30-45 min" en lugar de "~2 min") y/o eliminar la estimacion
  (solo aviso cualitativo "indexacion puede tardar >30 min en corpus
  ~1000 docs").
- Opcional, si queremos datos: instrumentar tiempo por llamada LLM de
  extraccion (p50/p95) en `TripletExtractor.extract_batch_async` y
  calcular la estimacion dinamica tras el primer run.

**Ubicacion esperada tras cierre**
- Corrigiendo el banner: cambio puntual en `sandbox_mteb/evaluator.py`.
  No deja rastro documental.
- Si tras instrumentar se confirma que el coste viene de #10,
  actualizar el row de divergencia #10 en CLAUDE.md con una nota sobre
  el orden de magnitud del impacto en indexacion.

---

## D. Codigo muerto en `shared/report.py:201-212` (columnas LightRAG del detail.csv)

**Estado**: `closed` (doc review completo: deuda #15 en CLAUDE.md
reescrita a "RESUELTA" con alcance completo JSON+CSV+synthesis outcome).

**Cambios aplicados** (Fase B del HANDOFF, unificado con deuda #15):
- `shared/types.py`: nueva helper publica
  `extract_retrieval_metadata_subset()` + constantes
  `_RETRIEVAL_METADATA_PASSTHROUGH_KEYS` / `_RETRIEVAL_METADATA_COUNT_KEYS`
  como unica fuente de verdad del subset per-query LIGHT_RAG.
  `QueryEvaluationResult.to_dict()` la usa para emitir
  `retrieval_metadata` en el JSON (omitido si subset vacio).
- `shared/report.py`: guard `has_lightrag` reescrito — antes buscaba la
  clave legacy `graph_candidates` (eliminada con divergencia #8) y era
  siempre `False` en runs reales; ahora reusa
  `extract_retrieval_metadata_subset()`. Columnas CSV renombradas a las
  claves actuales: `lightrag_mode`, `kg_fallback`, `kg_entities_count`,
  `kg_relations_count`, `kg_chunk_keyword_matches`, `kg_synthesis_used`,
  `kg_synthesis_error`.
- `sandbox_mteb/generation_executor.py`:
  `_synthesize_kg_context_async` retorna `Tuple[str, Optional[str]]`
  (narrativa_o_fallback, error_code). `_process_single_async` escribe
  `kg_synthesis_used`/`kg_synthesis_error` en
  `retrieval_detail.retrieval_metadata` para que ambos exports los lean.
- `tests/test_report.py`: `test_lightrag_columns_when_graph_meta`
  reescrito con claves actuales; +3 tests (SIMPLE_VECTOR sin columnas,
  fallback/error serializados, JSON subset por-query).
- `tests/test_kg_synthesis.py::TestPerQuerySynthesisMetadata`: 5 casos
  cubriendo success/timeout/empty/error/no-kg-data.

**Doc review pendiente**: reescrito el row de deuda #15 en CLAUDE.md
como "RESUELTA" con descripcion del scope completo (JSON + CSV +
per-query synthesis outcome). No se esperan cambios en `README.md`
(la historia de runs F.5 no se ve afectada por un fix del exportador).

**Evidencia**
- `shared/report.py:201-204`:
  ```python
  has_lightrag = any(
      qr.retrieval.retrieval_metadata.get("graph_candidates")
      for qr in run.query_results
  )
  ```
- La resolucion de la divergencia #8 elimino el campo
  `graph_candidates` del metadata emitido por `_retrieve_via_kg`.
  Verificado via `git grep graph_candidates shared/retrieval/` → 0
  coincidencias en codigo de produccion.
- Consecuencia: `has_lightrag` es siempre `False` en runs actuales; el
  bloque `if has_lightrag:` (`report.py:205-212`) nunca ejecuta; las
  columnas `graph_candidates / vector_candidates /
  graph_only_candidates / graph_resolved / query_keywords` que
  deberian aparecer en `detail.csv` para runs LIGHT_RAG nunca se
  anaden. Confirmado empiricamente: la cabecera del `detail.csv`
  auditado no tiene ninguna de esas columnas.

**Relacion con deuda #15 existente**
- Cuando documente la deuda #15, afirme que "`shared/report.py:267`
  ya lee `retrieval_metadata` para el CSV de detalle, asi que la
  informacion llega al builder; el gap es solo el serializador JSON".
  Esa afirmacion era literalmente cierta (la linea 267 existe y lee
  retrieval_metadata) pero omite que el guard en linea 201 bloquea la
  ejecucion. **La deuda #15 describe el problema de forma incompleta
  — el CSV tiene el mismo gap, via otro mecanismo.**

**Impacto**
- Hoy, cero runs LIGHT_RAG exportan diagnostico KG a CSV (ademas de
  no exportarlo a JSON). El criterio 3 de la divergencia #10
  (`kg_chunk_keyword_matches` por-query) no es auditable desde ninguno
  de los exports post-run.

**Accion propuesta**
- Unificar con la resolucion de deuda #15. Al cerrar #15:
  1. Reemplazar el guard por uno basado en claves actuales
     (`kg_entities` / `kg_fallback` / `kg_chunk_keyword_matches`).
  2. Actualizar la lista de columnas CSV a las claves actuales
     (`kg_fallback`, `kg_chunk_keyword_matches`, conteos de entidades
     y relaciones).
  3. Mantener coherencia: las mismas claves en JSON
     (`QueryEvaluationResult.to_dict()`) y en CSV.
- Al hacer este trabajo, **reescribir la descripcion de deuda #15 en
  CLAUDE.md** para reflejar que cubre ambos exports.

**Ubicacion esperada tras cierre**
- El row de deuda #15 en `CLAUDE.md#deuda-tecnica-vigente` queda
  reescrito o cerrado (segun el alcance del fix) y este item D
  desaparece de aqui.

---

## Procedimiento de cierre

### Por cada item, al trabajarlo

Checklist comun (aplica a A, B, C, D):

- [ ] Implementar el cambio propuesto o descartarlo con justificacion
      explicita en el commit.
- [ ] Tests unitarios anadidos si el cambio lo merece; pasan.
- [ ] Marcar `**Estado**: open` → `closed pending doc review` en el
      bloque del item y actualizar la fila en la tabla "Resumen de
      items".
- [ ] Evaluar "Ubicacion esperada tras cierre" del item: aplicar los
      cambios en `CLAUDE.md` / `README.md` / `.env.example` / banner
      que correspondan. Si no procede ningun cambio permanente, anotarlo
      explicitamente en el commit.
- [ ] Marcar `closed pending doc review` → `closed` solo cuando los
      cambios permanentes esten aplicados y revisados.

### Al cerrar este archivo

Cuando los 4 items esten en `closed`:

1. Verificar por ultima vez que ningun `CLAUDE.md` / `README.md` /
   `TESTS.md` referencia a `PENDING_AUDIT.md` (ver nota de la cabecera).
2. Confirmar via `git log` que cada item tiene commit asociado.
3. Eliminar el archivo: `rm PENDING_AUDIT.md` en el mismo PR o commit
   final.

### Norma para agentes / sesiones futuras

**Si en una sesion se trabaja sobre codigo relacionado con alguno de
los items de este fichero, quien ejecute esa sesion debe tocar este
archivo en la misma PR**: actualizar estado, eliminar seccion si ha
cerrado, y (cuando todas cierren) borrar el fichero. Dejar un item
con trabajo hecho pero este registro sin sincronizar convierte el
fichero en referencia falsa.
