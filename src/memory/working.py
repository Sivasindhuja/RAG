"""Working memory: recent turns + compact summary for follow-up questions."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import SETTINGS, PROMPTS
from src.generation.models import CostTracker, generate


@dataclass
class WorkingMemory:
    turns: list[dict] = field(default_factory=list)
    summary: str = ""

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        max_turns = SETTINGS["memory"]["max_turns"]
        if len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]

    def recent_dialogue(self) -> str:
        if not self.turns:
            return ""
        lines = [f"{t['role'].upper()}: {t['content']}" for t in self.turns[-4:]]
        return "\n".join(lines)

    def as_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Session summary: {self.summary}")
        recent = self.recent_dialogue()
        if recent:
            parts.append(f"Recent dialogue:\n{recent}")
        return "\n\n".join(parts) if parts else ""

    def maybe_summarize(self, tracker: CostTracker | None = None):
        if len(self.turns) < 4:
            return
        prompt = PROMPTS["memory_summary"].format(
            dialogue=self.recent_dialogue(),
            previous_summary=self.summary or "None",
        )
        self.summary = generate("memory_summary", prompt, tracker)
