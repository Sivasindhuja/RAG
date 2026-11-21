from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_app.config import AppConfig  # noqa: E402


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", "TRUE", " Yes "])
def test_definition_intent_boost_env_true_values(monkeypatch, value):
    monkeypatch.setenv("RAG_DEFINITION_INTENT_BOOST", value)
    cfg = AppConfig.from_env()
    assert cfg.definition_intent_boost is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE", " Off "])
def test_definition_intent_boost_env_false_values(monkeypatch, value):
    monkeypatch.setenv("RAG_DEFINITION_INTENT_BOOST", value)
    cfg = AppConfig.from_env()
    assert cfg.definition_intent_boost is False


def test_definition_intent_boost_env_unset_defaults_false(monkeypatch):
    monkeypatch.delenv("RAG_DEFINITION_INTENT_BOOST", raising=False)
    cfg = AppConfig.from_env()
    assert cfg.definition_intent_boost is False


def test_definition_intent_boost_env_invalid_raises(monkeypatch):
    monkeypatch.setenv("RAG_DEFINITION_INTENT_BOOST", "maybe")
    with pytest.raises(ValueError, match="Invalid boolean value for RAG_DEFINITION_INTENT_BOOST"):
        AppConfig.from_env()


def test_definition_intent_boost_propagates_to_service_pipeline(monkeypatch):
    from src.rag_app import RAGService
    from src.rag_core import GroundedPipeline

    class FakeIndex:
        def count(self):
            return 1

        def search(self, question, k=5):
            return []

    monkeypatch.setenv("RAG_DEFINITION_INTENT_BOOST", "true")
    cfg = AppConfig.from_env()
    service = RAGService(config=cfg, index=FakeIndex())
    assert service.pipeline.definition_intent_boost is True

    monkeypatch.setenv("RAG_DEFINITION_INTENT_BOOST", "false")
    cfg2 = AppConfig.from_env()
    service2 = RAGService(config=cfg2, index=FakeIndex())
    assert service2.pipeline.definition_intent_boost is False
