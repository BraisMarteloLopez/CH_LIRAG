"""Carga una coleccion del contrato (INGESTION_CONTRACT.md) y construye el KG.

CORRE EN JUPYTER (o en cualquier entorno con .env valido), con el venv activo,
DESDE LA RAIZ DEL REPO (los paths del .env son relativos).

Flujo: load_collection (valida contrato) -> embedding + LLM -> LightRAGRetriever
-> index_documents (extraccion de tripletas + 3 VDBs + cache de KG a disco)
-> stats. Con --query hace ademas un retrieval de sanidad.

Es el sustituto temporal del cableado run/evaluator para modo ingesta
(pendiente C2 en FAST_NOTES.md): permite indexar sin tocar el harness de eval.

Uso:
    python scripts/jupyter/06_index_collection.py col_XXXX_yyyy \
        [--env sandbox_mteb/.env] [--query "..."] [--top-k 5]
"""
import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("collection_id")
    ap.add_argument("--env", default="sandbox_mteb/.env")
    ap.add_argument("--query", default="", help="query opcional de sanidad post-indexacion")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    from sandbox_mteb.config import MTEBConfig
    from sandbox_mteb.loader import MinIOLoader
    from shared.llm import AsyncLLMService, load_embedding_model
    from shared.retrieval import get_retriever, RetrievalStrategy

    config = MTEBConfig.from_env(args.env)
    errors = config.validate()
    if errors:
        print("CONFIG INVALIDA:")
        for e in errors:
            print(f"  - {e}")
        return 1
    if config.retrieval.strategy != RetrievalStrategy.LIGHT_RAG:
        print(f"AVISO: RETRIEVAL_STRATEGY={config.retrieval.strategy.name}; "
              f"este script esta pensado para LIGHT_RAG (construccion de KG).")

    # --- 1) Cargar coleccion (valida el contrato, falla temprano) ---
    loader = MinIOLoader(config.storage)
    if not loader.check_connection():
        print(f"ERROR: MinIO no alcanzable en {config.storage.minio_endpoint}")
        return 1
    try:
        ds = loader.load_collection(args.collection_id)
    except ValueError as e:
        print(f"CONTRATO INCUMPLIDO: {e}")
        return 2
    md = ds.metadata
    print(f"coleccion cargada: {len(ds.corpus)} chunks | "
          f"generation={md.get('generation')} | "
          f"fingerprint={str(md.get('chunking_fingerprint'))[:24]}... | "
          f"max_chunk_chars={md.get('max_chunk_chars')}")
    if config.retrieval.kg_max_text_chars < int(md.get("max_chunk_chars") or 0):
        print(f"AVISO: KG_MAX_TEXT_CHARS={config.retrieval.kg_max_text_chars} < "
              f"max_chunk_chars={md.get('max_chunk_chars')} — la extraccion "
              f"truncara chunks en silencio. Sube KG_MAX_TEXT_CHARS en el .env.")

    # --- 2) Providers ---
    embedding = load_embedding_model(
        base_url=config.infra.embedding_base_url,
        model_name=config.infra.embedding_model_name,
        model_type=config.infra.embedding_model_type,
    )
    llm = AsyncLLMService(
        base_url=config.infra.llm_base_url,
        model_name=config.infra.llm_model_name,
        max_concurrent=config.infra.nim_max_concurrent,
        timeout_seconds=config.infra.nim_timeout,
        max_retries=config.infra.nim_max_retries,
    )

    # --- 3) Retriever + indexacion (mismo wiring que evaluator._index_documents) ---
    collection_name = f"ingest_{args.collection_id}"
    retriever = get_retriever(
        config=config.retrieval,
        embedding_model=embedding,
        collection_name=collection_name,
        embedding_batch_size=config.infra.embedding_batch_size,
        llm_service=llm,
    )
    documents = [
        {"doc_id": d.doc_id, "content": d.get_full_text(), "title": d.title or ""}
        for d in ds.corpus.values()
    ]
    n = len(documents)
    print(f"indexando {n} chunks (concurrencia={config.infra.nim_max_concurrent}, "
          f"batch={config.retrieval.kg_batch_docs_per_call} docs/llamada)...")
    if config.retrieval.kg_cache_dir:
        print(f"  KG cache: {config.retrieval.kg_cache_dir} "
              f"(si existe cache valido, salta la extraccion LLM)")

    t0 = time.time()
    ok = retriever.index_documents(documents, collection_name=collection_name)
    dt = time.time() - t0
    if not ok:
        print(f"INDEXACION FALLIDA tras {dt:.1f}s — revisar logs")
        return 3
    print(f"indexacion OK en {dt:.1f}s ({dt / max(n, 1):.2f}s/chunk amortizado)")

    # --- 4) Stats del KG y del extractor (acceso defensivo a internals) ---
    kg = getattr(retriever, "_kg", None)
    if kg is not None:
        try:
            s = kg.get_stats()
            print("KG stats:")
            for k, v in dict(s).items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  (no se pudieron leer KG stats: {e})")
    else:
        print("AVISO: retriever sin KG (_kg=None) — fallback a vector puro, "
              "NO DESEADO. Revisar igraph/LLM en logs.")
    extractor = getattr(retriever, "_extractor", None)
    if extractor is not None:
        try:
            print("extractor stats:")
            for k, v in extractor.get_stats().items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  (no se pudieron leer extractor stats: {e})")

    # --- 5) Retrieval de sanidad (opcional) ---
    if args.query:
        print(f"\nretrieval de sanidad: {args.query!r} (top_k={args.top_k})")
        res = retriever.retrieve(args.query, top_k=args.top_k)
        print(f"  strategy_used: {res.strategy_used.name} | "
              f"{res.retrieval_time_ms:.0f} ms | {len(res.doc_ids)} docs")
        meta = res.metadata or {}
        for key in ("kg_fallback", "kg_entities", "kg_relations",
                    "kg_chunk_keyword_matches", "kg_neighbor_coverage_rate"):
            if key in meta:
                print(f"  {key}: {meta[key]}")
        for doc_id, score in zip(res.doc_ids, res.scores):
            print(f"    {score:.4f}  {doc_id}")

    # NO llamamos a clear_index(): el KG cache (KG_CACHE_DIR) y las
    # colecciones Chroma quedan reutilizables para la proxima sesion.
    return 0


if __name__ == "__main__":
    sys.exit(main())
