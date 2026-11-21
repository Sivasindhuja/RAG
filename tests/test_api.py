"""WP0/WP1 API and RAG service tests (extractive-first, Gemini optional/controlled)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_app import AppConfig, RAGService  # noqa: E402
from src.rag_core import (  # noqa: E402
    ANSWERED,
    AnswerStatement,
    Citation,
    GeneratedAnswer,
    GeneratedClaim,
    RetrievedChunk,
)
import backend.main as backend_main  # noqa: E402


class FakeIndex:
    def __init__(self, hits):
        self._hits = list(hits)

    def count(self):
        return 49

    def search(self, query, k=5):
        return list(self._hits[:k])


class FakePipeline:
    def __init__(self, grounded):
        self._grounded = grounded

    def answer_from_hits(self, question, hits):
        return self._grounded


class FakeGrounded:
    def __init__(
        self,
        status="answered",
        question="q",
        text="A grounded answer [1]",
        reason=None,
        token_support=0.8,
        best_distance=0.5,
        statements=None,
        citations=None,
        retrieved=None,
    ):
        self.status = status
        self.question = question
        self.text = text
        self.reason = reason
        self.token_support = token_support
        self.best_distance = best_distance
        self.statements = statements or []
        self.citations = citations or []
        self.retrieved = retrieved or []


def _hit(chunk_id="c1", section="4.2", rank=1):
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        text="Sample evidence text.",
        source="satcom-ngp.pdf",
        start_offset=100,
        end_offset=140,
        start_page=13,
        end_page=13,
        article="ARTICLE-4",
        section=section,
        heading=section,
        distance=0.4,
        metadata={},
    )


def _stmt(text="Sample evidence text.", marker="[1]", chunk_id="c1"):
    return AnswerStatement(
        text=text,
        marker=marker,
        chunk_id=chunk_id,
        start_offset=100,
        end_offset=100 + len(text),
    )


def _cit(marker="[1]", chunk_id="c1", section="4.2"):
    return Citation(
        marker=marker,
        chunk_id=chunk_id,
        source="satcom-ngp.pdf",
        article="ARTICLE-4",
        section=section,
        heading=section,
        start_page=13,
        end_page=13,
        start_offset=100,
        end_offset=140,
        quote="Sample evidence text.",
        distance=0.4,
    )


def test_service_extractive_contract_and_observability():
    grounded = FakeGrounded(
        status="answered",
        statements=[_stmt()],
        citations=[_cit()],
        retrieved=[_hit()],
    )
    service = RAGService(
        config=AppConfig(allow_gemini=False),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
    )
    resp = service.query("What is sample?", mode="extractive")
    assert resp.status == "answered"
    assert resp.mode == "extractive"
    assert resp.abstained is False
    assert resp.request_id
    assert resp.answer == "A grounded answer [1]"
    assert len(resp.citations) == 1
    assert resp.citations[0].section == "4.2"
    assert resp.citations[0].chunk_id == "c1"
    assert resp.selected_evidence_ids == ["c1"]
    assert "retrieval" in resp.timings_ms
    assert "selection_and_extractive" in resp.timings_ms
    assert resp.generation.attempted is False
    assert resp.citation_validation.app_resolved is True


def test_service_gemini_uses_app_resolved_citations_and_falls_back_on_error():
    grounded = FakeGrounded(
        status="answered",
        statements=[_stmt()],
        citations=[_cit()],
        retrieved=[_hit()],
        token_support=0.9,
    )

    class ErrorGemini:
        model = "gemini-test"

        def generate(self, question, statements, citations, evidence_sufficient=True):
            return GeneratedAnswer(
                status="generation_error",
                question=question,
                text="",
                claims=[],
                citations=[],
                evidence_statements=list(statements),
                raw_response="",
                model="gemini-test",
                latency_ms=5.0,
                errors=["boom"],
                metadata={"prompt_version": "m1-v1"},
            )

    service = RAGService(
        config=AppConfig(allow_gemini=True, gemini_sufficiency_threshold=0.5),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
        gemini_generator=ErrorGemini(),
    )
    resp = service.query("What is sample?", mode="gemini")
    assert resp.generation.attempted is True
    assert resp.generation.mode == "gemini"
    assert resp.generation.fallback_to_extractive is True
    assert resp.status == "answered"
    assert resp.answer == "A grounded answer [1]"  # safe fallback to extractive
    assert resp.citations[0].section == "4.2"
    assert "boom" in " ".join(resp.generation.errors)


def test_service_gemini_abstains_on_insufficient_selected_evidence():
    grounded = FakeGrounded(
        status="answered",
        statements=[_stmt(text="unrelated words only", chunk_id="c1")],
        citations=[_cit()],
        retrieved=[_hit()],
        token_support=0.2,
    )
    service = RAGService(
        config=AppConfig(allow_gemini=True, gemini_sufficiency_threshold=0.5),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
    )
    resp = service.query("completely different question tokens here", mode="gemini")
    # Conservative answerability guard applies before generation and blocks unsafe answering.
    assert resp.status == "abstained"
    assert resp.abstained is True
    assert "insufficient_selected_evidence" in (resp.abstention_reason or "")
    assert resp.generation.attempted is True
    assert resp.generation.status == "abstained_insufficient_evidence"
    assert resp.generation.fallback_to_extractive is True
    assert resp.debug.get("answerability_guard_applied") is True


def test_service_extractive_abstains_on_insufficient_selected_evidence():
    grounded = FakeGrounded(
        status="answered",
        statements=[_stmt(text="unrelated words only", chunk_id="c1")],
        citations=[_cit()],
        retrieved=[_hit()],
        token_support=0.2,
    )
    service = RAGService(
        config=AppConfig(allow_gemini=False, gemini_sufficiency_threshold=0.5),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
    )
    resp = service.query("completely different question tokens here", mode="extractive")
    assert resp.status == "abstained"
    assert resp.abstained is True
    assert resp.abstention_reason and "insufficient_selected_evidence" in resp.abstention_reason
    assert resp.answer
    # Preserve traceable selected evidence for inspection even when abstaining.
    assert resp.statements and resp.citations and resp.retrieved
    assert resp.debug.get("answerability_guard_applied") is True


def test_service_extractive_keeps_supported_answer():
    grounded = FakeGrounded(
        status="answered",
        text="License required. [1]",
        statements=[_stmt(text="License is required for the service.", marker="[1]", chunk_id="c1")],
        citations=[_cit()],
        retrieved=[_hit()],
        token_support=0.9,
    )
    service = RAGService(
        config=AppConfig(allow_gemini=False, gemini_sufficiency_threshold=0.5),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
    )
    resp = service.query("Is license required for service?", mode="extractive")
    assert resp.status == "answered"
    assert resp.abstained is False
    assert resp.abstention_reason is None
    assert resp.citations and resp.statements
    assert resp.debug.get("answerability_guard_applied") is False


def test_api_endpoints_with_mocked_service(monkeypatch):
    grounded = FakeGrounded(
        status="answered",
        statements=[_stmt()],
        citations=[_cit()],
        retrieved=[_hit()],
    )
    fake_service = RAGService(
        config=AppConfig(allow_gemini=False),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
    )
    if hasattr(backend_main.get_service, "cache_clear"):
        backend_main.get_service.cache_clear()
    monkeypatch.setattr(backend_main, "get_service", lambda: fake_service)
    client = TestClient(backend_main.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["collection_name"] == "satcom_ngp_production_fixed_char_800_160"

    resp = client.post("/query", json={"question": "What is sample?", "mode": "extractive"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "answered"
    assert data["request_id"]
    assert data["citations"][0]["chunk_id"] == "c1"
    assert "timings_ms" in data


@pytest.mark.skipif(
    not (ROOT / "chroma_db_production" / "chroma.sqlite3").exists(),
    reason="production chroma index not present",
)
def test_real_index_extractive_smoke_q05_and_q12_representative():
    """Representative smoke tests against current selected production index/config.

    Uses measured 800/160 fixed-character chunking and app defaults. Validates
    end-to-end behavior, traceability, and app-resolved citation metadata.
    """
    service = RAGService(config=AppConfig.from_env())

    q05 = service.query(
        "Who is responsible for obtaining licenses to offer services in a territory?",
        mode="extractive",
        retrieval_mode="hybrid",
    )
    assert q05.request_id
    assert q05.mode == "extractive"
    assert q05.status == "answered"
    assert q05.abstained is False
    assert q05.citations and q05.statements and q05.retrieved
    assert "retrieval" in q05.timings_ms and "selection_and_extractive" in q05.timings_ms
    assert q05.citation_validation.app_resolved is True
    assert any("2.5.6" in (s.text or "") or "territory" in (s.text or "").lower() for s in q05.statements)

    q12 = service.query(
        "What satellite communication services are classified as domestic services?",
        mode="extractive",
        retrieval_mode="hybrid",
    )
    assert q12.request_id
    assert q12.status == "answered"
    assert q12.abstained is False
    assert q12.statements, "Q12 should have selected extractive evidence statements"
    assert q12.citations, "Q12 should have app-resolved citations"
    assert q12.citation_validation.app_resolved is True
    cited_chunks = {c.chunk_id for c in q12.citations}
    assert set(q12.selected_evidence_ids).issubset(cited_chunks | {s.chunk_id for s in q12.statements})
    for stmt in q12.statements:
        assert stmt.start_offset < stmt.end_offset
        assert stmt.chunk_id
        assert stmt.marker
    for cit in q12.citations:
        assert cit.source == "satcom-ngp.pdf"
        assert cit.start_page >= 1 and cit.end_page >= cit.start_page
        assert cit.start_offset < cit.end_offset
        assert cit.marker and cit.chunk_id
    assert "retrieval" in q12.timings_ms and "selection_and_extractive" in q12.timings_ms

def test_metrics_endpoint(monkeypatch):
    grounded = FakeGrounded(
        status="answered",
        statements=[_stmt()],
        citations=[_cit()],
        retrieved=[_hit()],
    )
    fake_service = RAGService(
        config=AppConfig(allow_gemini=False),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
    )
    monkeypatch.setattr(backend_main, "get_service", lambda: fake_service)
    client = TestClient(backend_main.app)
    client.post("/query", json={"question": "What is sample?", "mode": "extractive"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "totals" in data and "rates" in data and "latency_ms" in data
    assert data["totals"]["requests_total"] >= 1


def test_dashboard_endpoint(monkeypatch):
    grounded = FakeGrounded(status="answered", statements=[_stmt()], citations=[_cit()], retrieved=[_hit()])
    fake_service = RAGService(config=AppConfig(allow_gemini=False, retrieval_mode="dense"), index=FakeIndex([_hit()]), pipeline=FakePipeline(grounded))
    monkeypatch.setattr(backend_main, "get_service", lambda: fake_service)
    client = TestClient(backend_main.app)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "SATCOM NGP RAG" in resp.text
    assert "Live operational metrics" in resp.text


def test_ingest_disabled_by_default_fails_closed(monkeypatch):
    calls = {"build": 0}

    class NoBuildIndex(FakeIndex):
        def build(self, chunks, reset=True):
            calls["build"] += 1
            return {"count": 1}

    fake_service = RAGService(
        config=AppConfig(allow_gemini=False),
        index=NoBuildIndex([_hit()]),
        pipeline=FakePipeline(FakeGrounded(status="answered", statements=[_stmt()], citations=[_cit()], retrieved=[_hit()])),
    )
    if hasattr(backend_main.get_service, "cache_clear"):
        backend_main.get_service.cache_clear()
    monkeypatch.setattr(backend_main, "get_service", lambda: fake_service)
    monkeypatch.delenv("RAG_INGEST_TOKEN", raising=False)

    client = TestClient(backend_main.app)
    resp = client.post("/ingest")
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()
    assert calls["build"] == 0


def test_ingest_rejects_missing_or_wrong_token(monkeypatch):
    calls = {"build": 0}

    class NoBuildIndex(FakeIndex):
        def build(self, chunks, reset=True):
            calls["build"] += 1
            return {"count": 1}

    fake_service = RAGService(
        config=AppConfig(allow_gemini=False),
        index=NoBuildIndex([_hit()]),
        pipeline=FakePipeline(FakeGrounded(status="answered", statements=[_stmt()], citations=[_cit()], retrieved=[_hit()])),
    )
    if hasattr(backend_main.get_service, "cache_clear"):
        backend_main.get_service.cache_clear()
    monkeypatch.setattr(backend_main, "get_service", lambda: fake_service)
    monkeypatch.setenv("RAG_INGEST_TOKEN", "secret-token")

    client = TestClient(backend_main.app)
    resp_missing = client.post("/ingest")
    resp_wrong = client.post("/ingest", headers={"X-Ingest-Token": "wrong"})
    assert resp_missing.status_code == 403
    assert resp_wrong.status_code == 403
    assert "unauthorized" in resp_missing.json()["detail"].lower()
    assert "unauthorized" in resp_wrong.json()["detail"].lower()
    assert calls["build"] == 0


def test_ingest_authorized_request_triggers_rebuild(monkeypatch):
    import src.rag_core as rag_core

    calls = {"build": 0, "reset": None}

    class BuildIndex(FakeIndex):
        def build(self, chunks, reset=True):
            calls["build"] += 1
            calls["reset"] = reset
            return {"count": len(chunks), "status": "ok"}

    fake_service = RAGService(
        config=AppConfig(allow_gemini=False),
        index=BuildIndex([_hit()]),
        pipeline=FakePipeline(FakeGrounded(status="answered", statements=[_stmt()], citations=[_cit()], retrieved=[_hit()])),
    )
    fake_service._retriever = object()
    if hasattr(backend_main.get_service, "cache_clear"):
        backend_main.get_service.cache_clear()
    monkeypatch.setattr(backend_main, "get_service", lambda: fake_service)
    monkeypatch.setenv("RAG_INGEST_TOKEN", "secret-token")

    class DummyDoc:
        total_chars = 10

    class DummyChunker:
        def __init__(self, *args, **kwargs):
            pass

        def chunk(self, doc):
            return [SimpleNamespace(text="chunk")]

    monkeypatch.setattr(rag_core, "extract_source_document", lambda path: DummyDoc())
    monkeypatch.setattr(rag_core, "FixedCharacterChunker", DummyChunker)
    monkeypatch.setattr(rag_core, "StructureAwareChunker", DummyChunker)
    monkeypatch.setattr(
        rag_core,
        "audit_source_coverage",
        lambda doc, chunks, max_chars, allow_overlap: {"ok": True, "coverage_percent": 100.0, "overlap_chars": 0, "uncovered_non_whitespace_chars": 0, "total_chars": 10, "covered_chars": 10, "issues": []},
    )

    client = TestClient(backend_main.app)
    resp = client.post("/ingest", headers={"X-Ingest-Token": "secret-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunk_count"] == 1
    assert calls["build"] == 1
    assert calls["reset"] is True
    assert fake_service._retriever is None


def test_ingest_protection_does_not_affect_health_or_query(monkeypatch):
    grounded = FakeGrounded(status="answered", statements=[_stmt()], citations=[_cit()], retrieved=[_hit()])
    fake_service = RAGService(
        config=AppConfig(allow_gemini=False),
        index=FakeIndex([_hit()]),
        pipeline=FakePipeline(grounded),
    )
    if hasattr(backend_main.get_service, "cache_clear"):
        backend_main.get_service.cache_clear()
    monkeypatch.setattr(backend_main, "get_service", lambda: fake_service)
    monkeypatch.delenv("RAG_INGEST_TOKEN", raising=False)

    client = TestClient(backend_main.app)
    ingest_resp = client.post("/ingest")
    assert ingest_resp.status_code == 403

    health = client.get("/health")
    assert health.status_code == 200
    query = client.post("/query", json={"question": "What is sample?", "mode": "extractive"})
    assert query.status_code == 200
    assert query.json()["status"] == "answered"
