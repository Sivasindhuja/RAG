from src.rag_core import RetrievedChunk
from src.rag_core.retrieval import rrf_fuse, tokenize


def _hit(cid, rank, distance=0.1):
    return RetrievedChunk(
        rank=rank, chunk_id=cid, text=f"text {cid}", source="s.pdf",
        start_offset=0, end_offset=10, start_page=1, end_page=1,
        article=None, section=None, heading=None, distance=distance, metadata={},
    )


def test_tokenize_preserves_clause_like_tokens():
    toks = tokenize("Clause 2.5.6 applies to GMPCS services.")
    assert "2.5.6" in toks
    assert "gmpcs" in toks


def test_rrf_fuse_combines_dense_and_bm25_ranks():
    dense = [_hit("a", 1), _hit("b", 2), _hit("c", 3)]
    bm25 = [("b", 5.0, 1), ("d", 4.0, 2), ("a", 3.0, 3)]
    fused = rrf_fuse(dense, bm25, rrf_k=60)
    ids = [x[0] for x in fused]
    # b appears rank1 in bm25 and rank2 in dense -> should be top
    assert ids[0] == "b"
    assert "a" in ids and "d" in ids
    details = {x[0]: x[2] for x in fused}
    assert details["b"]["bm25_rank"] == 1
    assert details["b"]["dense_rank"] == 2
