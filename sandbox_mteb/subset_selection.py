"""
Subset selection: selecciona queries y corpus para evaluacion.

Extraido de evaluator.py para reducir su tamano (Fase B descomposicion).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple

from shared.types import LoadedDataset, NormalizedQuery

from .config import MTEBConfig

logger = logging.getLogger(__name__)


def select_subset_dev(
    dataset: LoadedDataset,
    config: MTEBConfig,
) -> Tuple[List[NormalizedQuery], Dict[str, Any]]:
    """
    Subset para DEV_MODE: gold docs garantizados en corpus.

    1. Shuffle queries con seed, tomar dev_queries
    2. Recopilar gold doc_ids de queries seleccionadas
    3. Incluir gold docs en corpus
    4. Rellenar con distractores aleatorios hasta dev_corpus_size
    """
    seed = config.corpus_shuffle_seed or 42
    rng = random.Random(seed)
    dev_queries = config.dev_queries
    dev_corpus_size = config.dev_corpus_size

    # 1. Seleccionar queries
    all_queries = list(dataset.queries)
    rng.shuffle(all_queries)
    if dev_queries >= len(all_queries):
        logger.warning(
            f"  DEV_MODE: dev_queries ({dev_queries}) >= total queries "
            f"({len(all_queries)}). Usando todas."
        )
    queries = all_queries[:dev_queries]

    # 2. Recopilar gold docs
    gold_ids = set()
    for q in queries:
        gold_ids.update(q.relevant_doc_ids)

    available_gold = gold_ids & set(dataset.corpus.keys())
    missing = gold_ids - available_gold
    if missing:
        logger.warning(
            f"  DEV_MODE: {len(missing)} gold docs ausentes en corpus"
        )

    if len(available_gold) > dev_corpus_size:
        raise ValueError(
            f"DEV_MODE: gold docs ({len(available_gold)}) > "
            f"DEV_CORPUS_SIZE ({dev_corpus_size}). Aumentar DEV_CORPUS_SIZE."
        )

    # 3. Corpus: gold obligatorios + distractores aleatorios
    corpus = {k: dataset.corpus[k] for k in available_gold}

    non_gold = [k for k in dataset.corpus if k not in gold_ids]
    rng.shuffle(non_gold)
    n_distractors = dev_corpus_size - len(corpus)
    for doc_id in non_gold[:n_distractors]:
        corpus[doc_id] = dataset.corpus[doc_id]

    n_gold = len(available_gold)
    logger.info(
        f"  DEV_MODE: {len(queries)} queries, "
        f"{n_gold} gold docs, "
        f"{len(corpus) - n_gold} distractores, "
        f"{len(corpus)} corpus total"
    )

    return queries, corpus


def select_subset_standard(
    dataset: LoadedDataset,
    config: MTEBConfig,
) -> Tuple[List[NormalizedQuery], Dict[str, Any]]:
    """
    Subset para modo estandar: max_queries y max_corpus con shuffle.

    max_queries=0 / max_corpus=0 significa "usar todo".
    """
    # Queries
    if config.max_queries > 0:
        queries = dataset.queries[: config.max_queries]
    else:
        queries = dataset.queries

    # Shuffle corpus antes de slice para evitar sesgo de orden.
    # Sin shuffle, corpus[0:N] puede contener artificialmente
    # todos los docs relevantes para queries[0:M] (alineacion
    # por posicion detectada en HotpotQA Parquet).
    corpus_ids = list(dataset.corpus.keys())
    corpus_seed = config.corpus_shuffle_seed
    if corpus_seed is not None:
        # Instancia aislada para no contaminar RNG global.
        rng_corpus = random.Random(corpus_seed)
        rng_corpus.shuffle(corpus_ids)
        logger.info(f"  Corpus shuffled con seed={corpus_seed}")
    else:
        logger.warning(
            "  CORPUS_SHUFFLE_SEED no configurado. "
            "Corpus NO shuffled (riesgo de sesgo de orden)."
        )

    if config.max_corpus > 0:
        corpus_ids = corpus_ids[: config.max_corpus]
    # else: usar todo el corpus (max_corpus=0)

    corpus = {k: dataset.corpus[k] for k in corpus_ids}

    return queries, corpus


__all__ = ["select_subset_dev", "select_subset_standard"]
