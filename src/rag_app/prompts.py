from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    path: Path
    description: str = ""

    def load(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.path}")
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Prompt file is empty: {self.path}")
        return text

    def render(self, **kwargs: Any) -> str:
        template = self.load()
        # Simple {{var}} replacement (no external templating dependency).
        rendered = template
        for key, value in kwargs.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
        # Detect unresolved placeholders to fail fast on malformed/mismatched usage.
        if "{{" in rendered or "}}" in rendered:
            # allow JSON braces in content? our templates use JSON braces but placeholders are {{name}}
            # unresolved placeholders would still contain {{name}}
            import re

            unresolved = re.findall(r"\{\{[^}]+\}\}", rendered)
            if unresolved:
                raise ValueError(f"Unresolved prompt placeholders in {self.path}: {sorted(set(unresolved))}")
        return rendered


class PromptRegistry:
    def __init__(self, prompts_dir: Path = PROMPTS_DIR):
        self.prompts_dir = prompts_dir
        self.manifest_path = prompts_dir / "manifest.json"
        self._manifest: Optional[Dict[str, Any]] = None

    @property
    def manifest(self) -> Dict[str, Any]:
        if self._manifest is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Prompt manifest not found: {self.manifest_path}")
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    def get(self, group: str, name: str) -> PromptTemplate:
        try:
            entry = self.manifest[group][name]
        except Exception as exc:  # noqa: BLE001
            raise KeyError(f"Prompt not found in manifest: {group}.{name}") from exc
        rel = entry.get("path")
        if not rel:
            raise ValueError(f"Prompt entry missing path: {group}.{name}")
        path = (self.prompts_dir / rel).resolve()
        # Ensure prompt path stays within prompts directory.
        if self.prompts_dir.resolve() not in path.parents and path != self.prompts_dir.resolve():
            raise ValueError(f"Prompt path escapes prompts directory: {path}")
        return PromptTemplate(
            name=name,
            version=str(entry.get("version") or name),
            path=path,
            description=str(entry.get("description") or ""),
        )

    def generation_grounded(self) -> PromptTemplate:
        return self.get("generation", "grounded_json")


def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry(PROMPTS_DIR)
