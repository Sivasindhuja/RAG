import json
from pathlib import Path

import pytest

from src.rag_app.prompts import PromptRegistry
from src.rag_core.generation import build_gemini_evidence_prompt
from src.rag_core.pipeline import AnswerStatement


def test_prompt_registry_loads_generation_template():
    reg = PromptRegistry()
    tpl = reg.generation_grounded()
    assert tpl.version
    assert tpl.path.exists()
    text = tpl.load()
    assert "{{question}}" in text
    assert "{{evidence}}" in text
    assert "JSON" in text


def test_prompt_render_detects_unresolved_placeholders(tmp_path: Path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "manifest.json").write_text(json.dumps({
        "generation": {"bad": {"path": "bad.txt", "version": "bad_v1"}}
    }), encoding="utf-8")
    (prompts / "bad.txt").write_text("Hello {{name}} {{missing}}", encoding="utf-8")
    reg = PromptRegistry(prompts)
    tpl = reg.get("generation", "bad")
    with pytest.raises(ValueError):
        tpl.render(name="x")


def test_prompt_missing_file_raises(tmp_path: Path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "manifest.json").write_text(json.dumps({
        "generation": {"x": {"path": "missing.txt", "version": "v1"}}
    }), encoding="utf-8")
    reg = PromptRegistry(prompts)
    with pytest.raises(FileNotFoundError):
        reg.generation_grounded() if False else reg.get("generation", "x").load()


def test_build_gemini_prompt_uses_external_template():
    stmts = [AnswerStatement(text="Clause 2.5.6 says X.", marker="[1]", chunk_id="c1", start_offset=1, end_offset=2)]
    prompt = build_gemini_evidence_prompt("Who is responsible?", stmts)
    assert "Clause 2.5.6 says X." in prompt
    assert "Who is responsible?" in prompt
    assert "Prompt version:" in prompt
    assert "[1]" in prompt
