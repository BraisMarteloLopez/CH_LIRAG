"""
Configuracion para sandbox MTEB.

Toda la parametrizacion viene del .env. El entry point (run.py)
construye MTEBConfig.from_env() una sola vez.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

from shared.config_base import (
    InfraConfig,
    RerankerConfig,
    _env,
    _env_int,
    _env_float,
    _env_bool,
    _env_path,
    load_dotenv_file,
)
from shared.retrieval.core import RetrievalConfig


# =========================================================================
# STORAGE (MinIO)
# =========================================================================

@dataclass
class MinIOStorageConfig:
    """Config de almacenamiento MinIO para datasets MTEB pre-descargados."""
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = ""
    s3_datasets_prefix: str = "datasets/evaluation"
    datasets_cache_dir: Path = Path("./data/datasets_cache")
    evaluation_results_dir: Path = Path("./data/results")
    vector_db_dir: Path = Path("./data/vector_db")

    @classmethod
    def from_env(cls) -> "MinIOStorageConfig":
        return cls(
            minio_endpoint=_env("MINIO_ENDPOINT", ""),
            minio_access_key=_env("MINIO_ACCESS_KEY", ""),
            minio_secret_key=_env("MINIO_SECRET_KEY", ""),
            minio_bucket=_env("MINIO_BUCKET_NAME", ""),
            s3_datasets_prefix=_env("S3_DATASETS_PREFIX", "datasets/evaluation"),
            datasets_cache_dir=_env_path("DATASETS_CACHE_DIR", "./data/datasets_cache"),
            evaluation_results_dir=_env_path("EVALUATION_RESULTS_DIR", "./data/results"),
            vector_db_dir=_env_path("VECTOR_DB_DIR", "./data/vector_db"),
        )

    def validate(self) -> List[str]:
        errors = []
        if not self.minio_endpoint:
            errors.append("MINIO_ENDPOINT no configurado")
        if not self.minio_bucket:
            errors.append("MINIO_BUCKET_NAME no configurado")
        return errors


# =========================================================================
# CONFIG PRINCIPAL
# =========================================================================

@dataclass
class MTEBConfig:
    """
    Configuracion completa para un run de evaluacion sobre datasets MTEB/BeIR.

    Composicion de sub-configs de shared/ + config especifico del sandbox.
    Se construye una sola vez en el entry point via from_env().
    """
    infra: InfraConfig = field(default_factory=InfraConfig)
    storage: MinIOStorageConfig = field(default_factory=MinIOStorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    # Especifico MTEB
    dataset_name: str = "hotpotqa"
    max_queries: int = 50
    max_corpus: int = 1000

    # Generacion (opcional, desactivable)
    generation_enabled: bool = True

    # Limite de caracteres para contexto de generacion.
    # Si 0: se deriva automaticamente del context window del modelo LLM
    # via GET /v1/models. Si el query falla, se usa 4000 como fallback.
    # Si >0: se usa directamente como override manual.
    generation_max_context_chars: int = 0

    # Shuffle del corpus para evitar sesgo de orden (None = no shuffle)
    corpus_shuffle_seed: Optional[int] = 42

    # Modo desarrollo: subset con gold docs garantizados.
    # Ignora max_queries y max_corpus cuando esta activo.
    dev_mode: bool = False
    dev_queries: int = 200
    dev_corpus_size: int = 4000

    # Umbral de tasa de fallback del LLM judge.
    # Si en el run la proporcion de scores devueltos como default 0.5
    # (el judge no pudo producir JSON parseable ni score via regex) supera
    # este umbral para CUALQUIER metrica del judge, el run se marca como
    # fallido al final. Protege metricas como faithfulness del sesgo
    # silencioso hacia el centro. 0.0 desactiva la validacion (no fallar).
    # Default 0.02 (2%) — razonable para judge modernos bien prompteados;
    # subir durante iteracion, bajar antes del experimento 3.
    judge_fallback_threshold: float = 0.02

    # Synthesis de contexto KG en generacion (value-add propio del proyecto).
    # Cuando hay datos KG presentes (entidades o relaciones del KG) y esta
    # flag esta activa, el contexto multi-seccion (entidades + relaciones +
    # chunks) se reescribe como narrativa coherente via LLM ANTES de la
    # generacion final. Faithfulness se sigue evaluando contra el contexto
    # estructurado original (no contra la narrativa), para que cualquier
    # alucinacion introducida por la propia synthesis sea penalizada.
    # Solo aplica a LIGHT_RAG (SIMPLE_VECTOR no produce datos KG).
    kg_synthesis_enabled: bool = True
    # Limite de output de la synthesis en caracteres. Si 0, usa el mismo
    # limite que la generacion (max_context_chars resuelto en runtime).
    kg_synthesis_max_chars: int = 0
    # Timeout dedicado para la llamada de synthesis (1 LLM call extra).
    # Si se supera, se hace fallback al contexto estructurado original.
    kg_synthesis_timeout_s: float = 30.0

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "MTEBConfig":
        """Construye config completa desde .env.

        Llama validate() automaticamente. Si hay errores de configuracion,
        lanza ValueError con todos los errores concatenados.
        """
        load_dotenv_file(env_path)

        config = cls(
            infra=InfraConfig.from_env(),
            storage=MinIOStorageConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            reranker=RerankerConfig.from_env(),
            dataset_name=_env("MTEB_DATASET_NAME", "hotpotqa"),
            max_queries=_env_int("EVAL_MAX_QUERIES", 50),
            max_corpus=_env_int("EVAL_MAX_CORPUS", 1000),
            generation_enabled=_env_bool("GENERATION_ENABLED", True),
            generation_max_context_chars=_env_int("GENERATION_MAX_CONTEXT_CHARS", 0),
            corpus_shuffle_seed=_env_int("CORPUS_SHUFFLE_SEED", 42) if _env_int("CORPUS_SHUFFLE_SEED", 42) >= 0 else None,
            dev_mode=_env_bool("DEV_MODE", False),
            dev_queries=_env_int("DEV_QUERIES", 200),
            dev_corpus_size=_env_int("DEV_CORPUS_SIZE", 4000),
            judge_fallback_threshold=_env_float("JUDGE_FALLBACK_THRESHOLD", 0.02),
            kg_synthesis_enabled=_env_bool("KG_SYNTHESIS_ENABLED", True),
            kg_synthesis_max_chars=_env_int("KG_SYNTHESIS_MAX_CHARS", 0),
            kg_synthesis_timeout_s=_env_float("KG_SYNTHESIS_TIMEOUT_S", 30.0),
        )

        errors = config.validate()
        if errors:
            raise ValueError(
                "Errores de configuracion:\n  - " + "\n  - ".join(errors)
            )

        return config

    def validate(self) -> List[str]:
        """Valida la configuracion. Retorna lista de errores (vacia = OK)."""
        errors = []
        errors.extend(self.storage.validate())
        errors.extend(self.reranker.validate())

        if not self.infra.embedding_base_url:
            errors.append("EMBEDDING_BASE_URL no configurado")
        if not self.infra.embedding_model_name:
            errors.append("EMBEDDING_MODEL_NAME no configurado")

        # Estrategias validas para este sandbox
        from shared.retrieval.core import RetrievalStrategy
        VALID_STRATEGIES = (
            RetrievalStrategy.SIMPLE_VECTOR,
            RetrievalStrategy.LIGHT_RAG,
        )
        if self.retrieval.strategy not in VALID_STRATEGIES:
            valid_names = ", ".join(s.name for s in VALID_STRATEGIES)
            errors.append(
                f"RETRIEVAL_STRATEGY={self.retrieval.strategy.name} no soportada "
                f"en sandbox_mteb. Valores validos: {valid_names}"
            )

        # LLM requerido si generacion activa O si LIGHT_RAG
        _needs_llm = (
            self.generation_enabled
            or self.retrieval.strategy == RetrievalStrategy.LIGHT_RAG
        )
        if _needs_llm:
            _reason = "LIGHT_RAG" if self.retrieval.strategy == RetrievalStrategy.LIGHT_RAG else "GENERATION_ENABLED=true"
            if not self.infra.llm_base_url:
                errors.append(f"LLM_BASE_URL requerido ({_reason})")
            if not self.infra.llm_model_name:
                errors.append(f"LLM_MODEL_NAME requerido ({_reason})")

        if self.max_queries < 0:
            errors.append(f"EVAL_MAX_QUERIES={self.max_queries} debe ser >= 0 (0=all)")
        if self.max_corpus < 0:
            errors.append(f"EVAL_MAX_CORPUS={self.max_corpus} debe ser >= 0 (0=all)")

        if self.dev_mode:
            if self.dev_queries <= 0:
                errors.append(f"DEV_QUERIES={self.dev_queries} debe ser > 0 cuando DEV_MODE=true")
            if self.dev_corpus_size <= 0:
                errors.append(f"DEV_CORPUS_SIZE={self.dev_corpus_size} debe ser > 0 cuando DEV_MODE=true")

        if not 0.0 <= self.judge_fallback_threshold <= 1.0:
            errors.append(
                f"JUDGE_FALLBACK_THRESHOLD={self.judge_fallback_threshold} "
                "debe estar en [0.0, 1.0] (0.0 desactiva la validacion)"
            )

        if self.kg_synthesis_max_chars < 0:
            errors.append(
                f"KG_SYNTHESIS_MAX_CHARS={self.kg_synthesis_max_chars} "
                "debe ser >= 0 (0 = usar max_context_chars del run)"
            )
        if self.kg_synthesis_timeout_s <= 0:
            errors.append(
                f"KG_SYNTHESIS_TIMEOUT_S={self.kg_synthesis_timeout_s} "
                "debe ser > 0"
            )

        return errors

    def ensure_directories(self) -> None:
        """Crea directorios necesarios."""
        self.storage.datasets_cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage.evaluation_results_dir.mkdir(parents=True, exist_ok=True)
        self.storage.vector_db_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Resumen legible de la configuracion."""
        lines = [
            "=== MTEB Sandbox Config ===",
            f"  Dataset:    {self.dataset_name}",
            f"  Embedding:  {self.infra.embedding_model_name} ({self.infra.embedding_model_type})",
            f"  Strategy:   {self.retrieval.strategy.name}",
            f"  Reranker:   {'ON' if self.reranker.enabled else 'OFF'}",
            f"  Generation: {'ON' if self.generation_enabled else 'OFF'}",
        ]
        if self.dev_mode:
            lines.append(f"  DEV_MODE:   ON ({self.dev_queries} queries, {self.dev_corpus_size} corpus, gold docs garantizados)")
        else:
            lines.append(f"  Queries:    {self.max_queries if self.max_queries > 0 else 'ALL'}")
            lines.append(f"  Corpus:     {self.max_corpus if self.max_corpus > 0 else 'ALL'}")
        from shared.retrieval.core import RetrievalStrategy
        if self.retrieval.strategy == RetrievalStrategy.LIGHT_RAG:
            lines.extend([
                f"  KG mode:    {self.retrieval.lightrag_mode}",
                f"  KG tokens:  extraction={self.retrieval.kg_extraction_max_tokens}, keyword={self.retrieval.kg_keyword_max_tokens}",
                f"  KG batch:   {self.retrieval.kg_batch_docs_per_call} docs/call",
                f"  KG synth:   {'ON' if self.kg_synthesis_enabled else 'OFF'}"
                + (
                    f" (max_chars={self.kg_synthesis_max_chars or 'auto'}, "
                    f"timeout={self.kg_synthesis_timeout_s}s)"
                    if self.kg_synthesis_enabled
                    else ""
                ),
                f"  WARNING:    LIGHT_RAG requiere ~1 llamada LLM por documento "
                f"para construir el knowledge graph (concurrencia={self.infra.nim_max_concurrent})",
            ])
        lines.extend([
            f"  Shuffle:    seed={self.corpus_shuffle_seed}" if self.corpus_shuffle_seed is not None else "  Shuffle:    OFF (WARNING: ordering bias risk)",
            f"  MinIO:      {self.storage.minio_endpoint}/{self.storage.minio_bucket}",
            f"  Results:    {self.storage.evaluation_results_dir}",
        ])
        return "\n".join(lines)


# =========================================================================
# PROMPTS DE GENERACION POR DATASET
# =========================================================================

GENERATION_PROMPTS: Dict[str, Dict[str, str]] = {
    "hotpotqa": {
        "system": (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "Answer in as few words as possible — ideally a single entity, name, date, or number. "
            "For yes/no questions, answer only 'yes' or 'no'. "
            "Do not explain, elaborate, or add extra context."
        ),
        "user_template": "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:",
    },
    # Datasets adicionales: agregar prompt cuando tengan ETL y datos en MinIO.
    "default": {
        "system": "You are a helpful assistant. Use the provided context to answer the question.",
        "user_template": "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:",
    },
}


# =========================================================================
# PROMPT DE SYNTHESIS DE CONTEXTO KG
# =========================================================================
# El system prompt acepta un placeholder {max_chars} que se rellena en
# runtime con kg_synthesis_max_chars (o max_context_chars si es 0).
# El user prompt recibe la pregunta y el contexto multi-seccion ya
# formateado por format_structured_context().

KG_SYNTHESIS_SYSTEM_PROMPT = """You are a context-synthesis assistant for a Retrieval-Augmented Generation \
(RAG) system. A downstream model will answer the user's QUESTION using ONLY \
the narrative you produce, so your output must contain every piece of \
evidence relevant to the QUESTION and nothing else.

INPUT FORMAT
You will receive:
- QUESTION: the user's question.
- Knowledge Graph Data (Entity): JSON list of entities with descriptions.
- Knowledge Graph Data (Relationship): JSON list of (source, target, \
relation) triples with descriptions.
- Document Chunks: JSON list of {{"reference_id": N, "content": "..."}} \
passages from the corpus.
Any of the KG sections may be empty.

YOUR TASK
Rewrite the input as a coherent narrative (one or a few paragraphs) that:
- Connects the entities and relationships to the textual evidence.
- Focuses on information relevant to the QUESTION.
- Preserves chunk provenance: when you state a fact taken from a chunk, \
cite it inline as [ref:N] using the chunk's reference_id.

STRICT RULES
1. Use ONLY information present in the input sections. Do NOT introduce \
facts, entities, dates, or relationships that are not explicitly stated. \
Direct inferences from the input are allowed.
2. Do NOT answer the QUESTION. Synthesize the supporting context only.
3. If the KG data and the chunks disagree, prefer the chunks (the chunks \
are the source; the KG is derived from them by an extractor that may have \
errored).
4. If a section is empty, omit it from the narrative.
5. Keep your output under {max_chars} characters. Prefer clarity over \
completeness when the limit is tight.

OUTPUT
Plain prose. No headings, no bullet lists, no JSON. Use [ref:N] inline \
to cite chunk content."""


KG_SYNTHESIS_USER_TEMPLATE = """QUESTION:
{query}

{structured_context}

Now produce the synthesis narrative."""


__all__ = [
    "MTEBConfig",
    "MinIOStorageConfig",
    "GENERATION_PROMPTS",
    "KG_SYNTHESIS_SYSTEM_PROMPT",
    "KG_SYNTHESIS_USER_TEMPLATE",
]
