"""
Preflight check para runs LIGHT_RAG.

Verifica dependencias, conectividad NIM, y config antes de lanzar un run
que puede tardar horas. Evita descubrir problemas en la query 3000.

Uso:
    python -m sandbox_mteb.preflight                # Usa .env por defecto
    python -m sandbox_mteb.preflight --env /path    # .env alternativo

Checks:
    1. Dependencias criticas (igraph, chromadb, langchain, etc.)
    2. Config .env valida para LIGHT_RAG
    3. Conectividad NIM (embedding, LLM, reranker si habilitado)
    4. Conectividad MinIO + dataset disponible
    5. Smoke test: una llamada LLM real (triplet extraction)
    6. requirements.lock presente y actualizado
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import List, Tuple


def _check(name: str, fn) -> Tuple[bool, str]:
    """Ejecuta un check y retorna (passed, message)."""
    try:
        result = fn()
        if result is True or result is None:
            return True, f"  [OK] {name}"
        return True, f"  [OK] {name}: {result}"
    except Exception as e:
        return False, f"  [FAIL] {name}: {e}"


def check_dependencies() -> List[Tuple[bool, str]]:
    """Verifica que todos los paquetes criticos estan instalados."""
    results = []

    critical = [
        ("igraph", "Knowledge Graph (LIGHT_RAG)"),
        ("chromadb", "Vector store"),
        ("langchain_nvidia_ai_endpoints", "NIM access"),
        ("langchain_core", "LangChain core"),
        ("pandas", "Data processing"),
        ("pyarrow", "Parquet I/O"),
        ("boto3", "MinIO/S3 client"),
    ]
    optional = []

    for pkg, desc in critical:
        def _check_pkg(p=pkg):
            mod = importlib.import_module(p)
            version = getattr(mod, "__version__", "?")
            return f"v{version}"
        ok, msg = _check(f"{pkg} ({desc})", _check_pkg)
        results.append((ok, msg))

    for pkg, desc in optional:
        def _check_pkg(p=pkg):
            mod = importlib.import_module(p)
            version = getattr(mod, "__version__", "?")
            return f"v{version}"
        ok, msg = _check(f"{pkg} ({desc})", _check_pkg)
        if not ok:
            # Optional: downgrade to warning
            results.append((True, msg.replace("[FAIL]", "[WARN]")))
        else:
            results.append((ok, msg))

    return results


def check_config(env_path: str) -> List[Tuple[bool, str]]:
    """Valida config .env para LIGHT_RAG."""
    results = []

    def _load():
        from sandbox_mteb.config import MTEBConfig
        config = MTEBConfig.from_env(env_path)
        errors = config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return config.retrieval.strategy.name

    ok, msg = _check(f"Config .env ({env_path})", _load)
    results.append((ok, msg))

    if ok:
        from sandbox_mteb.config import MTEBConfig
        config = MTEBConfig.from_env(env_path)

        # Verificar que la estrategia es LIGHT_RAG
        from shared.retrieval.core import RetrievalStrategy
        strategy = config.retrieval.strategy
        if strategy != RetrievalStrategy.LIGHT_RAG:
            results.append((False,
                f"  [FAIL] RETRIEVAL_STRATEGY={strategy.name}, "
                f"esperado LIGHT_RAG. Cambiar en .env"
            ))
        else:
            results.append((True, "  [OK] RETRIEVAL_STRATEGY=LIGHT_RAG"))

        # KG cache configurado?
        if config.retrieval.kg_cache_dir:
            results.append((True,
                f"  [OK] KG_CACHE_DIR={config.retrieval.kg_cache_dir} "
                f"(KG se persistira entre runs)"
            ))
        else:
            results.append((True,
                "  [WARN] KG_CACHE_DIR vacio — KG se reconstruira en cada run. "
                "Configurar para evitar re-extraccion"
            ))

        # Estimar llamadas LLM
        if config.dev_mode:
            corpus_est = config.dev_corpus_size
            queries_est = config.dev_queries
        else:
            corpus_est = config.max_corpus if config.max_corpus > 0 else 66576
            queries_est = config.max_queries if config.max_queries > 0 else 7405

        llm_calls_index = corpus_est  # ~1 LLM call per doc for triplet extraction
        llm_calls_query = queries_est  # ~1 LLM call per query for keyword extraction
        llm_calls_gen = queries_est if config.generation_enabled else 0
        llm_calls_faith = queries_est if config.generation_enabled else 0
        total_llm = llm_calls_index + llm_calls_query + llm_calls_gen + llm_calls_faith

        results.append((True,
            f"  [INFO] Estimacion LLM calls: "
            f"{llm_calls_index} (triplets) + {llm_calls_query} (keywords) + "
            f"{llm_calls_gen} (generation) + {llm_calls_faith} (faithfulness) "
            f"= ~{total_llm} total"
        ))

        concurrent = config.infra.nim_max_concurrent
        est_seconds = total_llm / concurrent * 2  # ~2s per LLM call avg
        est_minutes = est_seconds / 60
        results.append((True,
            f"  [INFO] Tiempo estimado: ~{est_minutes:.0f} min "
            f"(con concurrencia={concurrent}, ~2s/call)"
        ))

    return results


def check_connectivity(env_path: str) -> List[Tuple[bool, str]]:
    """Verifica conectividad a NIM y MinIO."""
    results = []

    try:
        from sandbox_mteb.config import MTEBConfig
        config = MTEBConfig.from_env(env_path)
    except Exception as e:
        results.append((False, f"  [FAIL] No se pudo cargar config: {e}"))
        return results

    # NIM Embedding
    def _check_embedding():
        import requests
        url = config.infra.embedding_base_url.rstrip("/").replace("/v1", "") + "/v1/models"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return f"{config.infra.embedding_model_name} @ {config.infra.embedding_base_url}"

    ok, msg = _check("NIM Embedding", _check_embedding)
    results.append((ok, msg))

    # NIM LLM
    def _check_llm():
        import requests
        url = config.infra.llm_base_url.rstrip("/").replace("/v1", "") + "/v1/models"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return f"{config.infra.llm_model_name} @ {config.infra.llm_base_url}"

    ok, msg = _check("NIM LLM", _check_llm)
    results.append((ok, msg))

    # NIM Reranker (if enabled)
    if config.reranker.enabled:
        def _check_reranker():
            import requests
            url = config.reranker.base_url.rstrip("/").replace("/v1", "") + "/v1/models"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return f"{config.reranker.model_name} @ {config.reranker.base_url}"

        ok, msg = _check("NIM Reranker", _check_reranker)
        results.append((ok, msg))

    # MinIO
    def _check_minio():
        import boto3
        from botocore.client import Config as BotoConfig
        client = boto3.client(
            "s3",
            endpoint_url=config.storage.minio_endpoint,
            aws_access_key_id=config.storage.minio_access_key,
            aws_secret_access_key=config.storage.minio_secret_key,
            config=BotoConfig(signature_version="s3v4"),
        )
        # Check bucket exists
        client.head_bucket(Bucket=config.storage.minio_bucket)
        # Check dataset prefix exists
        prefix = f"{config.storage.s3_datasets_prefix}/{config.dataset_name}/"
        response = client.list_objects_v2(
            Bucket=config.storage.minio_bucket,
            Prefix=prefix,
            MaxKeys=1,
        )
        count = response.get("KeyCount", 0)
        if count == 0:
            raise ValueError(f"No hay objetos en {prefix}")
        return f"{config.storage.minio_endpoint}/{config.storage.minio_bucket}/{prefix}"

    ok, msg = _check("MinIO dataset", _check_minio)
    results.append((ok, msg))

    return results


def check_smoke_llm(env_path: str) -> List[Tuple[bool, str]]:
    """Smoke test: una llamada LLM real para triplet extraction."""
    results = []

    try:
        from sandbox_mteb.config import MTEBConfig
        config = MTEBConfig.from_env(env_path)
    except Exception:
        results.append((False, "  [SKIP] Config no cargada"))
        return results

    def _smoke():
        from shared.llm import AsyncLLMService
        service = AsyncLLMService(
            base_url=config.infra.llm_base_url,
            model_name=config.infra.llm_model_name,
            max_concurrent=2,
            timeout_seconds=30,
            max_retries=1,
        )
        t0 = time.time()
        response = service.invoke(
            "Extract entities and relations from: "
            "'Albert Einstein was born in Ulm, Germany in 1879.' "
            "Return JSON with entities and relations.",
            max_tokens=256,
        )
        elapsed = time.time() - t0
        # Verify non-empty
        if not response or len(response) < 10:
            raise ValueError(f"Respuesta demasiado corta: '{response}'")
        return f"{elapsed:.1f}s, {len(response)} chars"

    ok, msg = _check("Smoke test LLM (triplet extraction)", _smoke)
    results.append((ok, msg))

    return results


def check_lock_file() -> List[Tuple[bool, str]]:
    """Verifica que requirements.lock existe y tiene paquetes criticos."""
    results = []
    lock_path = Path(__file__).parent.parent / "requirements.lock"

    if not lock_path.exists():
        results.append((False,
            "  [FAIL] requirements.lock no existe. "
            "Ejecutar: pip freeze > requirements.lock"
        ))
        return results

    content = lock_path.read_text()
    has_pending = "Pendientes de pin" in content or "Regenerar" in content
    if has_pending:
        results.append((False,
            "  [FAIL] requirements.lock incompleto (tiene paquetes pendientes de pin). "
            "Regenerar con: pip freeze > requirements.lock"
        ))
    else:
        # Count pinned packages
        pinned = [l for l in content.splitlines()
                  if l.strip() and not l.startswith("#") and "==" in l]
        results.append((True, f"  [OK] requirements.lock: {len(pinned)} paquetes pinneados"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Preflight check para LIGHT_RAG")
    default_env = str(Path(__file__).parent / ".env")
    parser.add_argument(
        "--env", default=default_env,
        help=f"Ruta al archivo .env (default: {default_env})",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Omitir smoke test LLM (mas rapido)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PREFLIGHT CHECK — LIGHT_RAG Baseline Run")
    print("=" * 60)

    all_results = []
    sections = [
        ("1. Dependencias", check_dependencies),
        ("2. Lock file", check_lock_file),
        ("3. Configuracion", lambda: check_config(args.env)),
        ("4. Conectividad", lambda: check_connectivity(args.env)),
    ]
    if not args.skip_smoke:
        sections.append(
            ("5. Smoke test LLM", lambda: check_smoke_llm(args.env))
        )

    for section_name, check_fn in sections:
        print(f"\n{section_name}")
        print("-" * 40)
        try:
            results = check_fn()
        except Exception as e:
            results = [(False, f"  [FAIL] Error inesperado: {e}")]
        all_results.extend(results)
        for _, msg in results:
            print(msg)

    # Summary
    passed = sum(1 for ok, _ in all_results if ok)
    failed = sum(1 for ok, _ in all_results if not ok)
    print(f"\n{'=' * 60}")
    print(f"RESULTADO: {passed} passed, {failed} failed")

    if failed > 0:
        print("\nCorregir los [FAIL] antes de lanzar el run.")
        print("Ejecutar run: python -m sandbox_mteb.run")
        return 1
    else:
        print("\nTodos los checks pasaron. Listo para lanzar:")
        print("  python -m sandbox_mteb.run")
        return 0


if __name__ == "__main__":
    sys.exit(main())
