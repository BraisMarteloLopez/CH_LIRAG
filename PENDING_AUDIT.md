# PENDING_AUDIT.md

> **Archivo efimero**. Registra tareas pendientes detectadas al auditar el
> run `mteb_hotpotqa_20260418_223530` (2026-04-18) que no subieron a
> `CLAUDE.md#deuda-tecnica-vigente` porque se consideran menores u
> operativas. **En cuanto se aborden**, revisar si alguna deja rastro en
> `CLAUDE.md`/`README.md` (p.ej. cerrar una deuda existente, actualizar
> una seccion, anadir una convencion) y **eliminar este archivo**. No
> referenciar desde otros documentos.

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

**Accion propuesta**
- No es un bug, es una caracteristica del modelo + interaccion con
  timeouts globales. Dos opciones:
  1. Instrumentar una stat `llm_empty_responses` por fase
     (indexacion / synthesis / generation) y anadirla al
     `config_snapshot._runtime`. Criterio de accion: si supera
     threshold a definir, auditar el prompt o el modelo.
  2. Subir `GENERATION_QUERY_TIMEOUT_S` para absorber hasta 3 retries
     en el peor caso (300s → 480s p.ej.). Riesgo: runs mas lentos si
     el LLM se degrada globalmente.
- **Requiere decision sobre cual priorizar** antes de actuar.

**Ubicacion esperada tras cierre**
- Si se implementa (1): la stat se documenta en
  `CLAUDE.md#observabilidad-de-runs` junto a `judge_fallback_stats`
  y `kg_synthesis_stats`.
- Si se implementa (2): basta con ajustar `.env.example` +
  comentario; no requiere cambio de doc.

---

## C. Estimacion de tiempo de indexacion obsoleta tras #10

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

## Estado de referencia al cerrar este archivo

Cuando los 4 items esten resueltos (o descartados con justificacion):

1. Verificar que los cambios en doc de cada item estan aplicados donde
   corresponde (CLAUDE.md, README.md, `.env.example`, banner del
   evaluator).
2. Confirmar que ningun item deja hilos sueltos (tests, deuda residual,
   comportamiento no validado).
3. **Eliminar este archivo** (`rm PENDING_AUDIT.md`) y borrar la
   referencia del commit si alguien la anadio.

No enlazar este archivo desde README, CLAUDE ni TESTS — su existencia
es transitoria.
