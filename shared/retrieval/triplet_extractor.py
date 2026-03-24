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
MIN_ENTITY_NAME_LEN = 1


# =============================================================================
# PROMPTS
# =============================================================================

TRIPLET_EXTRACTION_SYSTEM = """You are a knowledge graph extraction system.
Extract entities and their relationships from the given text.
Do NOT reason or think step-by-step. Do NOT use <think> tags.
Respond with ONLY the JSON object, nothing else."""

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

TRIPLET_EXTRACTION_BATCH_PROMPT = """Extract entities and relationships from EACH of these documents separately.

Rules:
- Extract the most important entities (people, organizations, places, concepts, events)
- Extract relationships between entities that are explicitly stated or strongly implied
- Keep entity names concise but unambiguous
- Keep relation types short (2-4 words)
- Return at most 10 entities and 10 relations per document

{doc_blocks}

Return JSON in this exact format (one entry per document, preserve doc_id):
{{"documents": [{{"doc_id": "id1", "entities": [{{"name": "Entity Name", "type": "PERSON|ORG|PLACE|CONCEPT|EVENT|OTHER", "description": "brief description"}}], "relations": [{{"source": "Entity A", "target": "Entity B", "relation": "relation type", "description": "brief description"}}]}}]}}"""

QUERY_KEYWORDS_SYSTEM = """You are a query analysis system for knowledge graph retrieval.
Extract specific entities and abstract themes from the query.
Do NOT reason or think step-by-step. Do NOT use <think> tags.
Respond with ONLY the JSON object, nothing else."""

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

    # Multiplicador para calcular batch de coroutines a partir del semaforo
    # HTTP del LLM. Con semaforo=32 y mult=4, batch=128 coroutines vivas
    # simultaneamente (DTm-25). Reduce presion de memoria sin afectar
    # throughput (el semaforo HTTP es el cuello de botella real).
    _CONCURRENCY_MULTIPLIER = 4
    _MIN_BATCH_SIZE = 64

    def __init__(
        self,
        llm_service: AsyncLLMService,
        max_text_chars: int = 3000,
    ) -> None:
        self._llm = llm_service
        self._max_text_chars = max_text_chars
        # DTm-25: batch size adaptativo al semaforo HTTP
        self._batch_size = max(
            self._MIN_BATCH_SIZE,
            llm_service._max_concurrent * self._CONCURRENCY_MULTIPLIER,
        )
        # Estadisticas de extraccion (DTm-33)
        self._stats: Dict[str, int] = {
            "docs_processed": 0,
            "docs_success": 0,
            "docs_failed": 0,
            "docs_empty_input": 0,
            "docs_empty_result": 0,
            "docs_json_recovered": 0,
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

    @staticmethod
    def _find_json_object(text: str) -> Any:
        """Encuentra el primer objeto JSON valido en texto arbitrario.

        Usa json.JSONDecoder.raw_decode() — la solucion estandar de Python
        para extraer JSON embebido en texto mixto.
        """
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == '{':
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    return obj
                except json.JSONDecodeError:
                    continue
        return None

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
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            # Fast path: texto es JSON puro
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: buscar primer objeto JSON en texto mixto
                data = self._find_json_object(text)
                if data is None:
                    raise ValueError(
                        f"No JSON object found in response for doc {doc_id}"
                    )
                self._stats["docs_json_recovered"] += 1
                logger.debug(
                    f"Doc {doc_id}: JSON recuperado via raw_decode "
                    f"(respuesta comenzaba con: {text[:80]!r})"
                )

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

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.debug(
                f"Error parseando JSON de doc {doc_id}: {e} "
                f"| raw (first 200 chars): {raw[:200]!r}"
            )

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
                max_tokens=2048,
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

    # -------------------------------------------------------------------------
    # MULTI-DOC BATCH EXTRACTION
    # -------------------------------------------------------------------------

    # Default max documents per LLM call in batch mode.
    _DEFAULT_BATCH_DOCS_PER_CALL = 5

    def _group_docs_for_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_docs_per_call: int,
    ) -> List[List[Dict[str, Any]]]:
        """Agrupa documentos en mini-batches respetando presupuesto de chars.

        Cada mini-batch tiene a lo sumo batch_docs_per_call documentos y
        su texto total truncado no excede max_text_chars * batch_docs_per_call.
        Documentos muy largos (>= max_text_chars) se envian solos.
        """
        budget = self._max_text_chars * batch_docs_per_call
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0

        for doc in documents:
            content = doc.get("content", "")
            doc_chars = min(len(content), self._max_text_chars)

            # Si añadir este doc excede presupuesto o cantidad, flush
            if current and (
                current_chars + doc_chars > budget
                or len(current) >= batch_docs_per_call
            ):
                groups.append(current)
                current = []
                current_chars = 0

            current.append(doc)
            current_chars += doc_chars

        if current:
            groups.append(current)

        return groups

    def _build_batch_prompt(self, docs: List[Dict[str, Any]]) -> str:
        """Construye prompt multi-documento con marcadores [DOC N]."""
        blocks: List[str] = []
        for i, doc in enumerate(docs, 1):
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")[:self._max_text_chars]
            blocks.append(
                f'[DOC {i} id="{doc_id}"]\n{content}\n[/DOC {i}]'
            )
        doc_blocks = "\n\n".join(blocks)
        return TRIPLET_EXTRACTION_BATCH_PROMPT.format(doc_blocks=doc_blocks)

    def _parse_batch_extraction_json(
        self,
        raw: str,
        docs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Tuple[List[KGEntity], List[KGRelation]]]]:
        """Parsea JSON de extraccion batch multi-documento.

        Returns:
            Dict doc_id -> (entities, relations), o None si el formato
            batch no se reconoce (el caller hara fallback a single-doc).
        """
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = self._find_json_object(text)
                if data is None:
                    return None

            # Debe tener key "documents" con lista
            doc_list = data.get("documents")
            if not isinstance(doc_list, list):
                return None

            results: Dict[str, Tuple[List[KGEntity], List[KGRelation]]] = {}
            for entry in doc_list:
                if not isinstance(entry, dict):
                    continue
                doc_id = entry.get("doc_id", "")
                if not doc_id:
                    continue
                # Reusar el parser single-doc con JSON del entry
                entry_json = json.dumps(entry)
                entities, relations = self._parse_extraction_json(entry_json, doc_id)
                results[doc_id] = (entities, relations)

            return results

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Error parseando batch JSON: {e}")
            return None

    async def _extract_multi_doc_async(
        self,
        docs: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[List[KGEntity], List[KGRelation]]]:
        """Extrae tripletas de multiples documentos en una sola llamada LLM.

        Si el parsing batch falla, hace fallback a extraccion individual.
        """
        # Filtrar docs vacios antes del batch
        non_empty = [d for d in docs if d.get("content", "").strip()]
        empty = [d for d in docs if not d.get("content", "").strip()]

        results: Dict[str, Tuple[List[KGEntity], List[KGRelation]]] = {}

        # Registrar vacios
        for d in empty:
            doc_id = d.get("doc_id", "")
            self._stats["docs_processed"] += 1
            self._stats["docs_empty_input"] += 1
            results[doc_id] = ([], [])

        if not non_empty:
            return results

        # Si solo un doc, usar path single-doc directamente
        if len(non_empty) == 1:
            doc = non_empty[0]
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            entities, relations = await self.extract_from_doc_async(doc_id, content)
            results[doc_id] = (entities, relations)
            return results

        # Multi-doc batch call
        prompt = self._build_batch_prompt(non_empty)
        # Scale max_tokens: 2048 per doc (same as single-doc)
        max_tokens = min(2048 * len(non_empty), 8192)

        try:
            raw = await self._llm.invoke_async(
                prompt,
                system_prompt=TRIPLET_EXTRACTION_SYSTEM,
                max_tokens=max_tokens,
            )

            batch_results = self._parse_batch_extraction_json(raw, non_empty)
            if batch_results is not None:
                # Update stats for each doc that came back
                for doc in non_empty:
                    doc_id = doc.get("doc_id", "")
                    self._stats["docs_processed"] += 1
                    if doc_id in batch_results:
                        entities, relations = batch_results[doc_id]
                        self._stats["docs_success"] += 1
                        self._stats["total_entities"] += len(entities)
                        self._stats["total_relations"] += len(relations)
                        if not entities and not relations:
                            self._stats["docs_empty_result"] += 1
                        results[doc_id] = (entities, relations)
                    else:
                        # Doc not in batch response — count as empty result
                        self._stats["docs_empty_result"] += 1
                        results[doc_id] = ([], [])
                return results

            # DTm-54: batch parsing failed — fallback to single-doc (WARNING, not debug)
            logger.warning(
                f"Batch parse failed for {len(non_empty)} docs, "
                f"falling back to single-doc extraction"
            )
        except Exception as e:
            logger.warning(f"Batch LLM call failed: {e}, falling back to single-doc")

        # Fallback: extract each doc individually
        for doc in non_empty:
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            entities, relations = await self.extract_from_doc_async(doc_id, content)
            results[doc_id] = (entities, relations)

        return results

    async def extract_batch_async(
        self,
        documents: List[Dict[str, Any]],
        batch_docs_per_call: int = 0,
    ) -> Dict[str, Tuple[List[KGEntity], List[KGRelation]]]:
        """Extraccion en batch con concurrencia controlada por el LLM service.

        Args:
            documents: Lista de dicts con keys: doc_id, content.
            batch_docs_per_call: Docs por llamada LLM (0 = default 5).
                Con batch_docs_per_call=1, se desactiva el batching
                multi-documento y cada doc se envia en su propia llamada.

        Returns:
            Dict doc_id -> (entidades, relaciones).
        """
        if batch_docs_per_call <= 0:
            batch_docs_per_call = self._DEFAULT_BATCH_DOCS_PER_CALL

        t0 = time.perf_counter()
        results: Dict[str, Tuple[List[KGEntity], List[KGRelation]]] = {}

        if batch_docs_per_call == 1:
            # Legacy mode: one LLM call per doc
            async def _extract_one(
                doc: Dict[str, Any],
            ) -> Tuple[str, List[KGEntity], List[KGRelation]]:
                doc_id = doc.get("doc_id", "")
                content = doc.get("content", "")
                entities, relations = await self.extract_from_doc_async(doc_id, content)
                return (doc_id, entities, relations)

            batch_sz = self._batch_size
            for start in range(0, len(documents), batch_sz):
                chunk = documents[start:start + batch_sz]
                tasks = [_extract_one(doc) for doc in chunk]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in chunk_results:
                    if isinstance(r, BaseException):
                        logger.warning(f"TripletExtractor: doc extraction failed: {r}")
                        continue
                    doc_id, entities, relations = r
                    results[doc_id] = (entities, relations)
        else:
            # Multi-doc batch mode: group docs and send N per LLM call
            groups = self._group_docs_for_batch(documents, batch_docs_per_call)

            # Process groups with concurrency control via chunks
            batch_sz = self._batch_size
            for start in range(0, len(groups), batch_sz):
                chunk = groups[start:start + batch_sz]
                tasks = [self._extract_multi_doc_async(g) for g in chunk]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in chunk_results:
                    if isinstance(r, BaseException):
                        logger.warning(f"TripletExtractor: batch extraction failed: {r}")
                        continue
                    results.update(r)

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
        """Parsea JSON de keywords con fallback raw_decode."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = self._find_json_object(text)
                if data is None:
                    raise ValueError("No JSON object found in keywords response")
                logger.debug(
                    f"Keywords JSON recuperado via raw_decode "
                    f"(respuesta comenzaba con: {text[:80]!r})"
                )

            low = [str(k) for k in data.get("low_level", []) if k]
            high = [str(k) for k in data.get("high_level", []) if k]
            return low, high
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
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
                max_tokens=512,
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
        # DTm-48: list comprehension (no mutable default sharing)
        results: List[Tuple[List[str], List[str]]] = [([], []) for _ in range(len(queries))]

        async def _extract_one(
            idx: int, query: str,
        ) -> Tuple[int, List[str], List[str]]:
            low, high = await self.extract_query_keywords_async(query)
            return (idx, low, high)

        batch_sz = self._batch_size
        for start in range(0, len(queries), batch_sz):
            chunk_tasks = [
                _extract_one(i, q)
                for i, q in enumerate(queries[start:start + batch_sz], start=start)
            ]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            for r in chunk_results:
                if isinstance(r, BaseException):
                    logger.warning(f"TripletExtractor: keyword extraction failed: {r}")
                    continue
                idx, low, high = r
                results[idx] = (low, high)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"TripletExtractor: query keywords batch {len(queries)} queries "
            f"en {elapsed_ms:.0f}ms"
        )

        return results

    def extract_query_keywords_batch(
        self, queries: List[str],
    ) -> List[Tuple[List[str], List[str]]]:
        """Wrapper sincrono."""
        return run_sync(self.extract_query_keywords_batch_async(queries))


__all__ = [
    "TripletExtractor",
    "TRIPLET_EXTRACTION_PROMPT",
    "TRIPLET_EXTRACTION_BATCH_PROMPT",
    "QUERY_KEYWORDS_PROMPT",
]
