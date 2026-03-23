# Plan Revisado: Robustez para Modelos de Razonamiento (nemotron-3-nano)

## Diagnóstico Raíz

nemotron-3-nano con thinking mode activo genera respuestas con formato:
```
<think>...500+ tokens de razonamiento...</think>{"entities": [...], "relations": [...]}
```

**Esto causa DOS problemas simultáneos:**
1. **Presupuesto de tokens agotado**: Con `max_tokens=1024` para extracción de tripletas, el modelo gasta 500+ tokens en `<think>` → el JSON queda truncado → `json.loads()` falla.
2. **Parsing directo falla**: Incluso cuando no se trunca, `json.loads()` recibe `<think>...</think>{...}` que no es JSON válido.

**Bug latente en llm.py:300-310**: Si `content` viene vacío y `reasoning_content` existe, el fallback devuelve el **razonamiento** como respuesta, no la respuesta real.

---

## Fase 1: Defensa en la fuente — Strip `<think>` en llm.py

**Archivo**: `shared/llm.py` → `_invoke_with_retry()`

**Cambio**: Después de obtener `content` (línea 289-296) y ANTES del fallback a `reasoning_content`, strip tags de thinking:

```python
# --- Strip reasoning tags (nemotron-3-nano thinking mode) ---
content = str(content)
content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
# Handle unclosed <think> (modelo truncado por max_tokens)
content = re.sub(r'<think>[\s\S]*$', '', content).strip()
```

**Por qué `[\s\S]*?` y no `.*?`**: Más explícito sobre multiline. `re.DOTALL` también funciona pero `[\s\S]` es más legible en este contexto.

**Por qué el segundo regex**: Si el modelo alcanza `max_tokens` durante el bloque `<think>`, el tag de cierre `</think>` nunca aparece. Sin este segundo regex, todo el contenido se conservaría incluyendo el thinking parcial.

**Impacto**: Global — afecta generación, extracción de tripletas, keywords, y judge. Esto es correcto porque en ningún caso queremos thinking tags en la respuesta final.

---

## Fase 2: Aumentar `max_tokens` para extracción de tripletas

**Archivo**: `shared/retrieval/triplet_extractor.py`

**Cambio**:
- Línea ~213: `max_tokens=1024` → `max_tokens=2048`
- Línea ~336: `max_tokens=256` → `max_tokens=512`

**Justificación**: Incluso con Fase 1 stripping tags, el servidor NIM puede contar los tokens de thinking contra el límite. Si el modelo genera 600 tokens de thinking + 800 de JSON, con `max_tokens=1024` el JSON se trunca a 424 tokens ANTES de que lleguemos a stripear. Aumentar a 2048 da margen suficiente.

**Riesgo**: Mayor latencia y consumo. Aceptable porque:
- La extracción de tripletas ya es la operación más lenta del pipeline
- Solo se ejecuta durante indexación (una vez), no durante queries
- El costo adicional es marginal vs. 89% de fallos

---

## Fase 3: Extracción robusta de JSON en `_parse_triplet_json`

**Archivo**: `shared/retrieval/triplet_extractor.py` → `_parse_triplet_json()`

**Cambio**: Reemplazar el parsing actual (strip markdown + `json.loads`) con:

```python
import json

def _parse_triplet_json(self, raw: str, doc_id: str) -> Tuple[List[KGEntity], List[KGRelation]]:
    text = raw.strip()

    # 1. Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # 2. Intentar json.loads directo (fast path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 3. Usar raw_decode para encontrar primer objeto JSON válido
        decoder = json.JSONDecoder()
        data = None
        for i, ch in enumerate(text):
            if ch == '{':
                try:
                    data, _ = decoder.raw_decode(text, i)
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            raise ValueError(f"No JSON object found in response for doc {doc_id}")

    # 4. Validación (existente)
    ...
```

**Por qué `raw_decode` y no regex/llaves balanceadas**: Es la solución estándar de Python para "encontrar JSON embebido en texto arbitrario". Maneja correctamente strings con llaves, escapes, nested objects, etc. No reinventamos nada.

**Aplicar también a**: `_parse_keywords_json()` con la misma lógica.

---

## Fase 4: Fix del fallback `reasoning_content` en llm.py

**Archivo**: `shared/llm.py` líneas 298-313

**Problema actual**: Si `content` está vacío post-strip y `reasoning_content` existe, se devuelve el razonamiento como respuesta. Esto es incorrecto — si después de stripear `<think>` tags el content queda vacío, significa que el modelo NO generó respuesta útil, solo razonó.

**Cambio**: Mover el strip de `<think>` tags ANTES del check de vacío, y eliminar el fallback a `reasoning_content`:

```python
content = str(content)
content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
content = re.sub(r'<think>[\s\S]*$', '', content).strip()

if not content:
    raise ValueError("LLM returned empty content after stripping reasoning tags")

return content
```

**Justificación**: Devolver `reasoning_content` como respuesta causa respuestas incoherentes en generación ("Let me think about Scott Derrickson..." en vez de "Scott Derrickson") y JSON inválido en extracción. Es mejor fallar y reintentar.

---

## Fase 5: Observabilidad

**Archivo**: `shared/retrieval/triplet_extractor.py`

**Cambios**:
1. Log `raw[:200]` antes de parsear (truncado para no saturar logs)
2. Añadir counter `docs_json_recovered` a `_stats` para trackear cuántas veces el `raw_decode` fallback salva un parse
3. Log warning cuando se usa el fallback `raw_decode`

---

## Orden de Implementación

```
Fase 1 → Fase 4 → Fase 2 → Fase 3 → Fase 5
```

Fases 1 y 4 van juntas porque ambas modifican `llm.py:_invoke_with_retry()`.
Fase 2 es trivial (cambiar dos números).
Fase 3 es el cambio más extenso pero más seguro (solo parsing local, sin efectos externos).
Fase 5 es incremental y no cambia comportamiento.

---

## Archivos Modificados

| Archivo | Fases | Tipo de cambio |
|---------|-------|----------------|
| `shared/llm.py` | 1, 4 | Strip `<think>` tags, eliminar fallback `reasoning_content` |
| `shared/retrieval/triplet_extractor.py` | 2, 3, 5 | `max_tokens`, `raw_decode` parsing, observabilidad |

---

## Lo que NO hacemos (y por qué)

1. **No creamos dos instancias de ChatNVIDIA**: Sería más "limpio" pero añade complejidad innecesaria. Strip tags es suficiente y más robusto.
2. **No usamos `with_thinking_mode(enabled=False)`**: Depende de que el servidor NIM lo soporte. No podemos verificarlo. Strip tags funciona independientemente.
3. **No prepend `<think></think>` vacío**: Técnica documentada pero frágil — depende de la versión del modelo y del servidor. No la necesitamos si stripeamos.
4. **No cambiamos prompts**: Los prompts actuales ya dicen "Return ONLY valid JSON". El problema no es el prompt — es el thinking mode del modelo.
5. **No añadimos feature flags**: Dos archivos, cambios mínimos, sin configuración adicional.
