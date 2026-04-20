"""
Sin reranker -> reranked_status=None, rerank_failures=None.
"""
from shared.retrieval.core import RetrievalResult, RetrievalStrategy, RetrievalConfig
from shared.config_base import InfraConfig, RerankerConfig
from shared.types import LoadedDataset, NormalizedDocument
from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
from sandbox_mteb.retrieval_executor import RetrievalExecutor
from sandbox_mteb.result_builder import build_run


class MockRetriever:
    def retrieve(self, query_text, top_k=None):
        k = top_k or 20
        return RetrievalResult(
            doc_ids=[f"doc_{i}" for i in range(k)],
            contents=[f"content_{i}" for i in range(k)],
            scores=[1.0 - i * 0.01 for i in range(k)],
            strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        )

    def retrieve_by_vector(self, query_text, query_vector, top_k=None):
        return self.retrieve(query_text, top_k)


def test_no_reranker():
    """Sin reranker: reranked_status=None, generation_doc_ids vacio, config_snapshot rerank_failures=None."""
    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(retrieval_k=20),
        reranker=RerankerConfig(enabled=False),
    )
    executor = RetrievalExecutor(
        retriever=MockRetriever(),
        reranker=None,
        config=config,
    )

    detail, reranked_status = executor.execute("test query", ["doc_0"])

    assert reranked_status is None
    assert len(detail.generation_doc_ids) == 0

    # config_snapshot tambien refleja None
    dataset = LoadedDataset(name="test", corpus={"doc_0": NormalizedDocument("doc_0", "c")})
    run = build_run(
        config=config,
        run_id="test_run",
        dataset=dataset,
        query_results=[],
        elapsed_seconds=1.0,
        indexed_corpus_size=1,
        max_context_chars=4000,
        rerank_failures=executor.rerank_failures,
        strategy_mismatches=executor.strategy_mismatches,
    )
    assert run.config_snapshot["_runtime"]["rerank_failures"] is None
