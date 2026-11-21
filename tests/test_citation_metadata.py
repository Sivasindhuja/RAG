from types import SimpleNamespace

from src.rag_core import FixedCharacterChunker, SourceDocument
from src.rag_core.document import Page
from src.rag_core.pipeline import _citation_for, _governing_metadata_for_offset


def _doc_from_text(text: str, source: str = "synthetic.pdf") -> SourceDocument:
    return SourceDocument(source=source, pages=[Page(page_number=1, text=text, start_char=0, end_char=len(text))])


def test_fixed_chunker_uses_start_governing_clause_for_mid_clause_window():
    # chunk 0: starts at 4.1, includes 4.2 later
    # chunk 1: starts inside 4.2 body (no 4.2 heading in chunk) and ends with 4.3 heading
    text = (
        "4.1 SERVICE CLASSIFICATION\n\n"
        + ("alpha " * 40)
        + "\n\n4.2 INTERNATIONAL COMMUNICATION SERVICES\n\n"
        + ("beta " * 180)
        + "\n\n4.3 DOMESTIC COMMUNICATION  SERVICES CLASSIFICATION\n\n"
        + ("gamma " * 40)
    )
    doc = _doc_from_text(text)
    chunker = FixedCharacterChunker(chunk_size=1000, overlap=200)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    c0, c1 = chunks[0], chunks[1]
    assert c0.section == "4.1"
    assert c1.start_offset > text.index("4.2 INTERNATIONAL")
    assert c1.start_offset < text.index("4.3 DOMESTIC")
    # governing at chunk1 start is 4.2, even though 4.3 appears later in chunk1
    assert c1.section == "4.2"
    assert c1.heading.startswith("4.2 INTERNATIONAL COMMUNICATION SERVICES")


def _make_hit(chunk_id, text, start_offset, end_offset, article="ARTICLE-4", section="4.3", heading="4.3 DOMESTIC"):
    return SimpleNamespace(
        chunk_id=chunk_id,
        source="satcom-ngp.pdf",
        article=article,
        section=section,
        heading=heading,
        start_page=13,
        end_page=13,
        start_offset=start_offset,
        end_offset=end_offset,
        text=text,
        distance=0.5,
    )


def test_citation_uses_statement_governing_clause_inside_chunk():
    # chunk contains 4.2 heading then text then 4.3 heading; statement in 4.2 region
    text = (
        "4.1 SERVICE CLASSIFICATION\n\nintro\n\n"
        "4.2 INTERNATIONAL COMMUNICATION SERVICES\n\n"
        "Definition sentence in 4.2. "
        "More 4.2 text.\n\n"
        "4.3 DOMESTIC COMMUNICATION  SERVICES CLASSIFICATION\n\n"
        "Domestic text."
    )
    hit = _make_hit("c1", text, 1000, 1000 + len(text), section="4.1", heading="4.1 SERVICE CLASSIFICATION")
    stmt_local = text.index("Definition sentence")
    article, section, heading = _governing_metadata_for_offset(hit, 1000 + stmt_local)
    assert section == "4.2"
    assert heading.startswith("4.2 INTERNATIONAL COMMUNICATION SERVICES")
    cit = _citation_for(hit, "[1]", 200, 1000 + stmt_local)
    assert cit.section == "4.2"
    assert cit.start_offset == 1000 and cit.end_offset == 1000 + len(text)
    assert cit.start_page == 13


def test_citation_keeps_start_governing_clause_when_statement_before_first_in_chunk_heading():
    # chunk starts mid-4.2 (no 4.2 heading in text) and contains 4.3 later
    text = (
        "continuation of 4.2 definition text. "
        "More details.\n\n"
        "4.3 DOMESTIC COMMUNICATION  SERVICES CLASSIFICATION\n\n"
        "Domestic text."
    )
    # chunk metadata already start-governing (after chunker fix / correct index)
    hit = _make_hit("c2", text, 30978, 30978 + len(text), section="4.2", heading="4.2 INTERNATIONAL COMMUNICATION SERVICES")
    stmt_local = text.index("continuation")
    article, section, heading = _governing_metadata_for_offset(hit, 30978 + stmt_local)
    assert section == "4.2"
    assert "4.3" not in (heading or "")
    cit = _citation_for(hit, "[1]", 200, 30978 + stmt_local)
    assert cit.section == "4.2"
    assert cit.heading.startswith("4.2 INTERNATIONAL COMMUNICATION SERVICES")


def test_q12_definition_statements_resolve_to_clause_4_2():
    # integration-style check against real PDF/chunker/indexed semantics
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    from src.rag_app import AppConfig
    from src.rag_core import DenseIndex, DenseIndexConfig, GroundedPipeline, load_questions

    root = Path(__file__).resolve().parent.parent
    cfg = AppConfig.from_env()
    model = SentenceTransformer(cfg.embedding_model)
    index = DenseIndex(
        DenseIndexConfig(
            collection_name=cfg.collection_name,
            persist_dir=cfg.persist_dir,
            embedding_model=cfg.embedding_model,
        ),
        model=model,
    )
    assert index.count() > 0, (
        f"Configured index is empty (collection={cfg.collection_name}, persist_dir={cfg.persist_dir}); "
        "build the production index before running this integration test."
    )

    questions = {q.question_id: q for q in load_questions(root / "experiments" / "exp001_questions.csv")}
    q = questions["EXP001-Q12"]
    hits = index.search(q.question, k=5)
    ans = GroundedPipeline(index, sibling_score_ratio=0.5, definition_intent_boost=True).answer_from_hits(q.question, hits)

    assert ans.statements
    for s in ans.statements:
        assert s.start_offset >= 0 and s.end_offset > s.start_offset
    # all Q12 definition statements selected in EXP-008 are in 4.2
    by_chunk = {c.chunk_id: c for c in ans.citations}
    for s in ans.statements:
        assert by_chunk[s.chunk_id].section == "4.2"
        assert by_chunk[s.chunk_id].start_page == 13
    assert {31006, 31255} & {s.start_offset for s in ans.statements}
