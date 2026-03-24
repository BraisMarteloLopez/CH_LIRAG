# Plan: Fix thinking-mode exhaustion en nemotron-3-nano

## Analisis del problema

Los logs de produccion confirman que nemotron-3-nano gasta TODOS los tokens
en razonamiento `<think>` y no produce contenido util fuera de los tags.
Tras strippear tags → vacio → 4 retries identicos → fallo permanente.

### Que esta mal en la implementacion actual

Cada fix anterior fue un parche sobre el anterior sin resolver la causa raiz:

1. **Strip `<think>` tags** — Necesario pero insuficiente: si TODO el contenido
   es thinking, strippear deja vacio.
2. **"Do NOT think" en prompts** — Poco fiable, el modelo puede ignorarlo.
3. **`_ThinkingExhaustedError` + max_tokens adaptativo** — Contraproducente:
   con mas tokens el modelo simplemente piensa mas. Ademas, sobreingenieria
   (clase de excepcion custom para un workaround).
4. **Problema critico no detectado**: el JSON util puede estar DENTRO de los
   `<think>` tags (el modelo razona y genera el JSON como parte del
   razonamiento). Al strippear ciegamente, tiramos el contenido util.

### Solucion real disponible en el API

`ChatNVIDIA` (langchain-nvidia-ai-endpoints v0.3.19) soporta:

- `thinking_mode=False` en `ainvoke()` → Envia system message
  `"detailed thinking off"`. Desactiva el thinking a nivel de API.
- Kwargs extra en `ainvoke()` → `_get_payload()` hace `payload.update(kwargs)`,
  asi que cualquier parametro OpenAI-compatible llega al endpoint NIM.
- NIM soporta `response_format={"type": "json_object"}` para forzar output
  JSON valido.

## Plan de implementacion

### Paso 1: Extender `AsyncLLMService.invoke_async()` con `**kwargs`

**Archivo:** `shared/llm.py`

Anadir `**kwargs` passthrough en la cadena:
`invoke_async(**kwargs)` → `_invoke_with_retry(**kwargs)` → `ainvoke(**kwargs)`

Esto permite a los callers pasar parametros API como `thinking_mode`,
`response_format`, etc. sin que llm.py necesite conocerlos.

### Paso 2: Desactivar thinking mode por defecto

**Archivo:** `shared/llm.py`

En `_invoke_with_retry`, inyectar `thinking_mode=False` como default en kwargs
(si el caller no lo especifica). Razonamiento: el thinking mode no aporta valor
en nuestros casos de uso (extraccion JSON, generacion de respuestas) y causa
fallos sistematicos.

### Paso 3: Usar JSON mode en extraccion de tripletas y keywords

**Archivo:** `shared/retrieval/triplet_extractor.py`

En `extract_from_doc_async()` y `extract_query_keywords_async()`, pasar
`response_format={"type": "json_object"}` via los nuevos kwargs. Esto fuerza al
modelo a emitir JSON valido a nivel de API, eliminando la necesidad de parseo
con fallbacks.

### Paso 4: Hacer el strip de `<think>` inteligente (safety net)

**Archivo:** `shared/llm.py`

Cambiar la logica de strip para que, si tras eliminar tags el contenido queda
vacio, extraiga el contenido INTERIOR de los `<think>` tags en vez de descartarlo.
El modelo puede haber producido JSON valido dentro de su razonamiento.

Logica:
1. Buscar contenido DESPUES de `</think>` → si existe, usarlo (caso ideal)
2. Si vacio, extraer contenido DENTRO de `<think>...</think>` → usarlo
3. Para `<think>` truncado (sin cerrar), extraer lo que hay dentro
4. Solo fallar si no hay contenido en ningun sitio

### Paso 5: Limpiar sobreingenieria

**Archivo:** `shared/llm.py`

- Eliminar `_ThinkingExhaustedError` (clase innecesaria)
- Eliminar logica de max_tokens adaptativo en retries
- El ValueError generico vuelve a ser suficiente para el caso vacio

### Paso 6: Tests

- Verificar que los tests existentes siguen pasando (295 tests)
- Anadir test para el nuevo comportamiento: extraer contenido de dentro de
  `<think>` tags cuando el exterior esta vacio
- Anadir test para kwargs passthrough en invoke_async

## Orden de ejecucion

1. Paso 4 (smart strip) — Corrige el bug critico de tirar contenido util
2. Paso 5 (limpiar) — Elimina sobreingenieria antes de anadir cosas
3. Paso 1 (kwargs) — Infraestructura necesaria para pasos 2 y 3
4. Paso 2 (thinking off) — Prevencion a nivel API
5. Paso 3 (JSON mode) — Garantia de output JSON valido
6. Paso 6 (tests) — Validacion
