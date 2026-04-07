"""
Embedding service: batch embedding de queries y context window detection.

Extraido de evaluator.py para reducir su tamano (Fase B descomposicion).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Dict, List, Optional

from .config import MTEBConfig

logger = logging.getLogger(__name__)


from shared.constants import CHARS_PER_TOKEN as _CHARS_PER_TOKEN
from shared.constants import OVERHEAD_TOKENS as _OVERHEAD_TOKENS


def query_model_context_window(llm_base_url: str) -> Optional[int]:
    """
    Consulta GET /v1/models del LLM NIM para obtener max_model_len.

    Returns:
        max_model_len en tokens, o None si no se puede obtener.
    """
    base_url = llm_base_url.rstrip("/")
    url = f"{base_url}/models"

    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        models = data.get("data", [])
        if not models:
            logger.warning("GET /v1/models: respuesta vacia")
            return None

        max_model_len = models[0].get("max_model_len")
        if max_model_len and isinstance(max_model_len, (int, float)):
            logger.info(
                f"  LLM context window: {int(max_model_len)} tokens "
                f"(modelo: {models[0].get('id', 'unknown')})"
            )
            return int(max_model_len)

        logger.warning("GET /v1/models: max_model_len no encontrado en respuesta")
        return None

    except Exception as e:
        logger.warning(f"No se pudo consultar context window del LLM: {e}")
        return None


def resolve_max_context_chars(config: MTEBConfig) -> int:
    """
    Determina el limite de caracteres para contexto de generacion.

    Prioridad:
      1. Override manual via GENERATION_MAX_CONTEXT_CHARS > 0
      2. Derivado del context window del modelo via /v1/models
      3. Fallback hardcodeado: 4000 chars
    """
    fallback = 4000

    # 1. Override manual
    if config.generation_max_context_chars > 0:
        logger.info(
            f"  Context chars: {config.generation_max_context_chars} "
            "(override manual via GENERATION_MAX_CONTEXT_CHARS)"
        )
        return config.generation_max_context_chars

    # 2. Derivar del modelo
    max_model_len = query_model_context_window(config.infra.llm_base_url)
    if max_model_len is not None:
        available_tokens = max_model_len - _OVERHEAD_TOKENS
        if available_tokens <= 0:
            logger.warning(
                f"  Context window ({max_model_len}) menor que overhead "
                f"({_OVERHEAD_TOKENS}). Usando fallback={fallback}"
            )
            return fallback

        derived_chars = int(available_tokens * _CHARS_PER_TOKEN)
        logger.info(
            f"  Context chars: {derived_chars} "
            f"(derivado: {max_model_len} tokens - {_OVERHEAD_TOKENS} overhead "
            f"= {available_tokens} tokens * {_CHARS_PER_TOKEN} chars/token)"
        )
        return derived_chars

    # 3. Fallback
    logger.info(f"  Context chars: {fallback} (fallback por defecto)")
    return fallback


def batch_embed_queries(
    query_texts: List[str],
    config: MTEBConfig,
) -> List[List[float]]:
    """
    Embebe todas las queries en batch via REST al NIM de embeddings.

    Usa input_type=query para modelos asimetricos (el NIM distingue
    entre query y passage). Para modelos simetricos, input_type se omite.

    Returns:
        Lista de vectores, uno por query. Si falla, retorna lista vacia
        y el caller debe hacer fallback a retrieval sin pre-embed.
    """
    n = len(query_texts)
    batch_size = config.infra.embedding_batch_size or 5
    base_url = config.infra.embedding_base_url.rstrip("/")
    model_name = config.infra.embedding_model_name
    model_type = config.infra.embedding_model_type
    url = f"{base_url}/embeddings"

    all_vectors: List[List[float]] = []

    logger.info(
        f"  Pre-embedding {n} queries (batch={batch_size}, "
        f"type={model_type})..."
    )
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch = query_texts[batch_start : batch_start + batch_size]

        payload: Dict[str, Any] = {
            "input": batch,
            "model": model_name,
        }
        # Modelos asimetricos requieren input_type
        if model_type == "asymmetric":
            payload["input_type"] = "query"

        # FIX DTm-6: retry por batch antes de abandonar todo el pre-embed.
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, data=body, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                # Ordenar por index (la API puede devolver desordenado)
                items = sorted(data["data"], key=lambda x: x["index"])
                for item in items:
                    all_vectors.append(item["embedding"])
                break  # batch OK

            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"  Batch embed retry {attempt + 1}/{max_retries} "
                        f"(offset={batch_start}): {e}. "
                        f"Reintentando en {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"  Error en batch embed (offset={batch_start}) "
                        f"tras {max_retries + 1} intentos: {e}. "
                        "Descartando todos los vectores (pre-embed incompleto)."
                    )
                    # Retornar los vectores parciales acumulados hasta aqui.
                    # El caller detecta len(vectors) < n_queries y hace
                    # fallback a retrieval sin pre-embed para TODAS las queries,
                    # lo cual es consistente (no mezcla queries con/sin vector).
                    return all_vectors

        batch_end = batch_start + len(batch)
        if batch_end % 500 == 0 or batch_end == n:
            logger.info(f"  Queries embebidas: {batch_end}/{n}")

    elapsed = time.time() - t0
    logger.info(
        f"  Pre-embedding completado: {n} queries en {elapsed:.1f}s "
        f"({n / elapsed:.0f} queries/s)"
    )
    return all_vectors


__all__ = ["batch_embed_queries", "resolve_max_context_chars", "query_model_context_window"]
