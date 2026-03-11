"""
Modulo: Triplet Extractor
Descripcion: Extraccion de tripletas (entidad, relacion, entidad) y
             keywords de query usando LLM (AsyncLLMService via NIM).

Ubicacion: shared/retrieval/triplet_extractor.py

Uso: LIGHT_RAG usa este modulo durante indexacion (extraccion de
tripletas del corpus) y durante retrieval (analisis de query).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from shared.llm import AsyncLLMService, run_sync

from .knowledge_graph import KGEntity, KGRelation

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES DE VALIDACION (DTm-16)
# =============================================================================

VALID_ENTITY_TYPES = {"PERSON", "ORG", "PLACE", "CONCEPT", "EVENT", "OTHER"}
MAX_DESCRIPTION_CHARS = 200
MIN_ENTITY_NAME_LEN = 2


# =============================================================================
# PROMPTS
# =============================================================================

TRIPLET_EXTRACTION_SYSTEM = """You are a knowledge graph extraction system.
Extract entities and their relationships from the given text.
Return ONLY valid JSON with no extra text."""

TRIPLET_EXTRACTION_PROMPT = """Extract entities and relationships from this text.

Rules:
- Extract the most important entities (people, organizations, places, concepts, events)
- Extract relationships between entities that are explicitly stated or strongly implied
- Keep entity names concise but unambiguous
- Keep relation types short (2-4 words)
- Return at most 10 entities and 10 relations

Return JSON in this exact format:
{{"entities": [{{"name": "Entity Name", "type": "PERSON|ORG|PLACE|CONCEPT|EVENT|OTHER", "description": "brief description"}}], "relations": [{{"source": "Entity A", "target": "Entity B", "relation": "relation type", "description": "brief description"}}]}}

Text:
{text}"""

QUERY_KEYWORDS_SYSTEM = """You are a query analysis system for knowledge graph retrieval.
Extract specific entities and abstract themes from the query.
Return ONLY valid JSON with no extra text."""

QUERY_KEYWORDS_PROMPT = """Analyze this search query and extract two types of keywords:

1. low_level: Specific entity names mentioned or implied (people, places, organizations, products)
2. high_level: Abstract themes, topics, or concepts the query is about

Return JSON in this exact format:
{{"low_level": ["entity1", "entity2"], "high_level": ["theme1", "theme2"]}}

Query: {query}"""


# =============================================================================
# TRIPLET EXTRACTOR
# =============================================================================

class TripletExtractor:
    """Extrae tripletas y keywords usando LLM.

    Dos funciones principales:
      1. extract_from_doc(): tripletas de un documento (indexacion)
      2. extract_query_keywords(): keywords de una query (retrieval)

    Batch processing con semaphore para concurrencia controlada.
    Coroutines se crean en chunks para evitar presion de memoria
    con corpus grandes (DTm-22).

    Estadisticas de extraccion accesibles via get_stats() (DTm-33).
    """

    # Maximo de coroutines creadas simultaneamente en extract_batch_async.
    # El semaphore del LLM limita HTTP concurrente, pero las coroutines
    # en si retienen referencias a doc content mientras esperan turno.
    _COROUTINE_BATCH_SIZE = 500

    def __init__(
        self,
        llm_service: AsyncLLMService,
        max_text_chars: int = 3000,
    ) -> None:
        self._llm = llm_service
        self._max_text_chars = max_text_chars
        # Estadisticas de extraccion (DTm-33)
        self._stats: Dict[str, int] = {
            "docs_processed": 0,
            "docs_success": 0,
            "docs_failed": 0,
            "docs_empty_input": 0,
            "docs_empty_result": 0,
            "total_entities": 0,
            "total_relations": 0,
        }

    def get_stats(self) -> Dict[str, int]:
        """Devuelve estadisticas acumuladas de extraccion (DTm-33)."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reinicia contadores de extraccion."""
        for k in self._stats:
            self._stats[k] = 0

    def _parse_extraction_json(
        self, raw: str, doc_id: str
    ) -> Tuple[List[KGEntity], List[KGRelation]]:
        """Parsea JSON de extraccion de tripletas con fallback robusto."""
        entities: List[KGEntity] = []
        relations: List[KGRelation] = []

        try:
            # Limpiar markdown code blocks si los hay
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Quitar primera y ultima linea (```json y ```)
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)

            rejected = 0
            for e in data.get("entities", []):
                if not isinstance(e, dict):
                    continue
                name = (e.get("name") or "").strip()
                if not name or len(name) < MIN_ENTITY_NAME_LEN:
                    rejected += 1
                    continue
                etype = (e.get("type") or "OTHER").upper()
                if etype not in VALID_ENTITY_TYPES:
                    etype = "OTHER"
                desc = (e.get("description") or "")[:MAX_DESCRIPTION_CHARS]
                entities.append(KGEntity(
                    name=name,
                    entity_type=etype,
                    description=desc,
                    source_doc_ids={doc_id},
                ))

            if rejected:
                logger.debug(
                    f"Doc {doc_id}: {rejected} entidades rechazadas por validacion"
                )

            for r in data.get("relations", []):
                if isinstance(r, dict) and r.get("source") and r.get("target"):
                    rel_desc = (r.get("description") or "")[:MAX_DESCRIPTION_CHARS]
                    relations.append(KGRelation(
                        source=r["source"],
                        target=r["target"],
                        relation=r.get("relation", "related to"),
                        description=rel_desc,
                        source_doc_id=doc_id,
                    ))

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Error parseando JSON de doc {doc_id}: {e}")

        return entities, relations

    async def extract_from_doc_async(
        self, doc_id: str, text: str,
    ) -> Tuple[List[KGEntity], List[KGRelation]]:
        """Extrae entidades y relaciones de un documento via LLM.

        Args:
            doc_id: ID del documento.
            text: Contenido del documento.

        Returns:
            Tupla (entidades, relaciones).
        """
        self._stats["docs_processed"] += 1

        if not text.strip():
            self._stats["docs_empty_input"] += 1
            return [], []

        truncated = text[:self._max_text_chars]
        prompt = TRIPLET_EXTRACTION_PROMPT.format(text=truncated)

        try:
            raw = await self._llm.invoke_async(
                prompt,
                system_prompt=TRIPLET_EXTRACTION_SYSTEM,
                max_tokens=1024,
            )
            entities, relations = self._parse_extraction_json(raw, doc_id)
            self._stats["docs_success"] += 1
            self._stats["total_entities"] += len(entities)
            self._stats["total_relations"] += len(relations)
            if not entities and not relations:
                self._stats["docs_empty_result"] += 1
            return entities, relations
        except Exception as e:
            self._stats["docs_failed"] += 1
            logger.warning(f"Error extrayendo tripletas de {doc_id}: {e}")
            return [], []

    def extract_from_doc(
        self, doc_id: str, text: str,
    ) -> Tuple[List[KGEntity], List[KGRelation]]:
        """Wrapper sincrono de extract_from_doc_async."""
        return run_sync(self.extract_from_doc_async(doc_id, text))

    async def extract_batch_async(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[List[KGEntity], List[KGRelation]]]:
        """Extraccion en batch con concurrencia controlada por el LLM service.

        Args:
            documents: Lista de dicts con keys: doc_id, content.

        Returns:
            Dict doc_id -> (entidades, relaciones).
        """
        t0 = time.perf_counter()
        results: Dict[str, Tuple[List[KGEntity], List[KGRelation]]] = {}

        async def _extract_one(doc: Dict[str, Any]) -> None:
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            entities, relations = await self.extract_from_doc_async(doc_id, content)
            results[doc_id] = (entities, relations)

        # Procesar en chunks para limitar coroutines vivas en memoria
        # (DTm-22). El semaphore del LLM controla HTTP, pero cada
        # coroutine retiene su doc content mientras espera turno.
        batch_sz = self._COROUTINE_BATCH_SIZE
        for start in range(0, len(documents), batch_sz):
            chunk = documents[start:start + batch_sz]
            tasks = [_extract_one(doc) for doc in chunk]
            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_entities = sum(len(e) for e, _ in results.values())
        total_relations = sum(len(r) for _, r in results.values())
        # DTm-33: reportar fallos en el log del batch
        stats = self.get_stats()
        failed = stats["docs_failed"]
        empty_input = stats["docs_empty_input"]
        empty_result = stats["docs_empty_result"]
        fail_info = ""
        if failed or empty_input or empty_result:
            fail_info = (
                f" | {failed} fallidos, {empty_input} input vacio, "
                f"{empty_result} resultado vacio"
            )
        logger.info(
            f"TripletExtractor: batch {len(documents)} docs en {elapsed_ms:.0f}ms. "
            f"{total_entities} entidades, {total_relations} relaciones extraidas."
            f"{fail_info}"
        )
        if failed:
            logger.warning(
                f"TripletExtractor: {failed}/{len(documents)} documentos "
                f"fallaron en extraccion de tripletas."
            )

        return results

    def extract_batch(
        self, documents: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[List[KGEntity], List[KGRelation]]]:
        """Wrapper sincrono de extract_batch_async."""
        return run_sync(self.extract_batch_async(documents))

    # -------------------------------------------------------------------------
    # QUERY ANALYSIS
    # -------------------------------------------------------------------------

    def _parse_keywords_json(
        self, raw: str,
    ) -> Tuple[List[str], List[str]]:
        """Parsea JSON de keywords con fallback."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)
            low = [str(k) for k in data.get("low_level", []) if k]
            high = [str(k) for k in data.get("high_level", []) if k]
            return low, high
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Error parseando keywords JSON: {e}")
            return [], []

    async def extract_query_keywords_async(
        self, query: str,
    ) -> Tuple[List[str], List[str]]:
        """Extrae keywords de bajo y alto nivel de una query.

        Args:
            query: Texto de la query.

        Returns:
            Tupla (low_level_keywords, high_level_keywords).
        """
        prompt = QUERY_KEYWORDS_PROMPT.format(query=query)

        try:
            raw = await self._llm.invoke_async(
                prompt,
                system_prompt=QUERY_KEYWORDS_SYSTEM,
                max_tokens=256,
            )
            return self._parse_keywords_json(raw)
        except Exception as e:
            logger.warning(f"Error extrayendo keywords de query: {e}")
            return [], []

    def extract_query_keywords(
        self, query: str,
    ) -> Tuple[List[str], List[str]]:
        """Wrapper sincrono de extract_query_keywords_async."""
        return run_sync(self.extract_query_keywords_async(query))

    async def extract_query_keywords_batch_async(
        self,
        queries: List[str],
    ) -> List[Tuple[List[str], List[str]]]:
        """Extrae keywords de multiples queries en paralelo.

        Returns:
            Lista de (low_level, high_level) en mismo orden que queries.
        """
        t0 = time.perf_counter()
        results: List[Optional[Tuple[List[str], List[str]]]] = [None] * len(queries)

        async def _extract_one(idx: int, query: str) -> None:
            low, high = await self.extract_query_keywords_async(query)
            results[idx] = (low, high)

        batch_sz = self._COROUTINE_BATCH_SIZE
        for start in range(0, len(queries), batch_sz):
            chunk_tasks = [
                _extract_one(i, q)
                for i, q in enumerate(queries[start:start + batch_sz], start=start)
            ]
            await asyncio.gather(*chunk_tasks, return_exceptions=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"TripletExtractor: query keywords batch {len(queries)} queries "
            f"en {elapsed_ms:.0f}ms"
        )

        # Rellenar None con vacios
        return [(r[0], r[1]) if r else ([], []) for r in results]

    def extract_query_keywords_batch(
        self, queries: List[str],
    ) -> List[Tuple[List[str], List[str]]]:
        """Wrapper sincrono."""
        return run_sync(self.extract_query_keywords_batch_async(queries))


__all__ = [
    "TripletExtractor",
    "TRIPLET_EXTRACTION_PROMPT",
    "QUERY_KEYWORDS_PROMPT",
]
