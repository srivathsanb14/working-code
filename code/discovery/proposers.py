"""Where candidates come from.  Every proposer has the same interface:

    propose(system: str, user: str, n: int, generation: int) -> list[dict(name, hypothesis, code)]

  SeedProposer       hand-written starting programs in seeds/ (no LLM)
  DirectoryProposer  *.py files dropped into an inbox folder (any LLM, a human, another tool)
  DryRunProposer     writes the exact prompt to disk and returns nothing (inspect before spending)
  ClaudeProposer     the Claude API (anthropic SDK); only used when explicitly selected
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from . import prompts

SEEDS = Path(__file__).resolve().parent / "seeds"


def _from_file(path: Path):
    src = path.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    return dict(name=path.stem, hypothesis=doc.split("\n\n")[0].replace("\n", " "), code=src)


class SeedProposer:
    def __init__(self, folder=SEEDS):
        self.files = sorted(Path(folder).glob("*.py"))
        self.used = False

    def propose(self, system, user, n, generation):
        if self.used:
            return []
        self.used = True
        return [_from_file(p) for p in self.files]


class DirectoryProposer:
    """Reads new *.py candidates from an inbox; moves them to inbox/consumed afterwards."""

    def __init__(self, inbox):
        self.inbox = Path(inbox); (self.inbox / "consumed").mkdir(parents=True, exist_ok=True)

    def propose(self, system, user, n, generation):
        (self.inbox / f"prompt_gen{generation:03d}.md").write_text(f"# SYSTEM\n\n{system}\n\n# USER\n\n{user}\n")
        out = []
        for p in sorted(self.inbox.glob("*.py"))[:n]:
            out.append(_from_file(p)); p.rename(self.inbox / "consumed" / p.name)
        return out


class DryRunProposer:
    def __init__(self, folder):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)

    def propose(self, system, user, n, generation):
        (self.folder / f"prompt_gen{generation:03d}.md").write_text(f"# SYSTEM\n\n{system}\n\n# USER\n\n{user}\n")
        return []


class ClaudeProposer:
    """Claude Opus 5.5 with adaptive thinking, streamed, structured JSON output.

    Requires `pip install anthropic` and credentials (ANTHROPIC_API_KEY or `ant auth login`).
    The system prompt is identical across generations and is marked for prompt caching.
    Server-side refusal fallbacks are enabled ("default" routing).
    """

    def __init__(self, model="claude-opus-5-5", effort="high", max_tokens=64000, log_dir=None):
        import anthropic  # imported lazily: nothing here runs unless this proposer is chosen
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.log_dir = Path(log_dir) if log_dir else None

    def propose(self, system, user, n, generation):
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": prompts.RESPONSE_SCHEMA}},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"response_gen{generation:03d}.json").write_text(msg.to_json())
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(msg.stop_details, 'category', None)}")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("response hit max_tokens; lower n or raise max_tokens")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return prompts.parse_response(text)[:n]
