"""
Constantes tunables centralizadas.

Todas las constantes que afectan comportamiento del pipeline y que
podrian necesitar ajuste estan aqui. Cada constante tiene un comentario
explicando su proposito y contexto.

Modulos individuales importan desde aqui en lugar de definir sus propios
valores hardcoded.
"""

# =============================================================================
# METRICS (shared/metrics.py)
# =============================================================================

# Limite de chars de la respuesta generada enviada al LLM Judge.
# Guarda defensiva contra respuestas degeneradas (repeticiones, overflow).
# Para HotpotQA las respuestas reales rara vez superan 200 chars;
# 2000 chars (~500 tokens) es holgado sin desperdiciar contexto del judge.
MAX_RESPONSE_CHARS_FOR_JUDGE: int = 2000

# =============================================================================
# EMBEDDING (sandbox_mteb/embedding_service.py)
# =============================================================================

# Ratio chars/token para estimar limite de contexto desde model context window.
# 4.0 es conservador para ingles; idiomas densos (CJK) pueden necesitar ~2.0.
CHARS_PER_TOKEN: float = 4.0

# Tokens reservados para system prompt + user template + max_output.
OVERHEAD_TOKENS: int = 1024

# =============================================================================
# KNOWLEDGE GRAPH (shared/retrieval/lightrag/)
# =============================================================================

# Max chars para descripcion de entidad/relacion en el KG.
# Descripciones mas largas se truncan. 200 chars es suficiente para
# capturar la esencia sin inflar el grafo.
KG_MAX_DESCRIPTION_CHARS: int = 200

# Longitud minima de nombre de entidad para aceptarla.
# 1 permite entidades de un solo caracter (e.g., variables, siglas).
KG_MIN_ENTITY_NAME_LEN: int = 1

# Capacidad maxima del grafo en entidades. Eviction con score compuesto
# cuando se excede. Para HotpotQA (66K docs) no se alcanza.
KG_DEFAULT_MAX_ENTITIES: int = 100_000

# =============================================================================
# VECTOR STORE (shared/vector_store.py)
# =============================================================================

# Limite de IDs por consulta $in a ChromaDB para evitar queries SQLite
# excesivamente largas.
CHROMA_IN_BATCH_SIZE: int = 100

# =============================================================================
# CHECKPOINT (sandbox_mteb/checkpoint.py)
# =============================================================================

# Queries procesadas entre cada checkpoint. Bajar si los queries son
# costosos y se quiere granularidad fina de resume.
CHECKPOINT_CHUNK_SIZE: int = 50

# =============================================================================
# GENERATION (sandbox_mteb/generation_executor.py)
# =============================================================================

# Timeout en segundos para generacion+metricas de una query individual.
# Incluye synthesis KG + generacion + calculo de metricas (faithfulness LLM).
# Valor laxo para evitar falsos fallos; el budget de tokens del contexto
# es el verdadero limitador de carga al LLM, no el timeout.
GENERATION_QUERY_TIMEOUT_S: float = 300.0

# =============================================================================
# CONTEXT BUDGETS — paper LightRAG (HKUDS, EMNLP 2025)
# =============================================================================
# Presupuestos de tokens por seccion del contexto estructurado.
# El paper usa MAX_TOTAL_TOKENS=30000, MAX_ENTITY_TOKENS=6000,
# MAX_RELATION_TOKENS=8000. Convertimos a chars (×4.0 chars/token).

# Max chars para la seccion de entidades en el contexto estructurado.
# Paper: 6000 tokens. Ajustable via KG_MAX_ENTITY_CONTEXT_CHARS.
KG_MAX_ENTITY_CONTEXT_CHARS: int = 24_000  # ~6000 tokens

# Max chars para la seccion de relaciones en el contexto estructurado.
# Paper: 8000 tokens. Ajustable via KG_MAX_RELATION_CONTEXT_CHARS.
KG_MAX_RELATION_CONTEXT_CHARS: int = 32_000  # ~8000 tokens
