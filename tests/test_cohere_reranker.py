from types import SimpleNamespace

import src.rag_core.retrieval as retrieval
from src.rag_core import RetrievedChunk


def _hit(cid, rank, text):
    return RetrievedChunk(
        rank=rank, chunk_id=cid, text=text, source="satcom-ngp.pdf",
        start_offset=0, end_offset=len(text), start_page=1, end_page=1,
        article=None, section="2.5.6" if "2.5.6" in cid else None, heading=None, distance=0.5, metadata={},
    )


def test_cohere_rerank_reorders_candidates_with_mock(monkeypatch):
    candidates = [
        _hit("c1", 1, "irrelevant text"),
        _hit("c2", 2, "licenses to offer a service in a territory"),
        _hit("c3", 3, "other text"),
    ]

    class FakeClient:
        def rerank(self, model, query, documents, top_n):
            # return c2 best, then c1
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.99),
                    SimpleNamespace(index=0, relevance_score=0.20),
                    SimpleNamespace(index=2, relevance_score=0.10),
                ][:top_n],
                meta=SimpleNamespace(billed_units={"search_units": 1}),
            )

    monkeypatch.setattr(retrieval, "_get_cohere_client", lambda: FakeClient())
    reranked, debug = retrieval.cohere_rerank("licenses territory", candidates, top_n=2, model="rerank-english-v3.0")
    assert debug["applied"] is True
    assert debug["model"] == "rerank-english-v3.0"
    assert [h.chunk_id for h in reranked] == ["c2", "c1"]
    assert reranked[0].rank == 1
    assert reranked[0].metadata.get("rerank_score") == 0.99


def test_cohere_rerank_reports_error_without_key(monkeypatch):
    candidates = [_hit("c1", 1, "text")]
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    try:
        retrieval.cohere_rerank("q", candidates, top_n=1)
        assert False, "expected error"
    except RuntimeError as e:
        assert "Cohere API key not configured" in str(e)
