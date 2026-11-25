"""Where candidates come from: the Claude API, and nothing else.

    ClaudeProposer.propose(system, user, n, generation, ctx) -> list[dict(name, hypothesis, code)]

Every candidate the search evaluates, generation 0 included, is written by the LLM through one API request.  Each
request is stateless: it carries the system prompt (identical every time, cached) and that one request's user prompt,
nothing else; no chat history, no earlier answers, no data files.  The data never leaves this machine: the prompt
holds only the summaries the scorer computed (see prompts.py and closedform.analysis).

The closed-form loop (evolve.py) passes ctx = {"operator", "parents": [records], "dir": request folder}; the raw
response is saved there as response.json.  `client` can be injected (tests use a stand-in with the same interface).
"""
from __future__ import annotations

import ast
from pathlib import Path

from . import prompts


def _from_file(path: Path):
    """A hand-written model file as {name, hypothesis, code} (reference baselines only; never a parent)."""
    src = path.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    return dict(name=path.stem, hypothesis=doc.split("\n\n")[0].replace("\n", " "), code=src)


class ClaudeProposer:
    """Claude Opus 5.5 with adaptive thinking, streamed, structured JSON output.

    Requires `pip install anthropic` and credentials (ANTHROPIC_API_KEY or `ant auth login`).
    The system prompt is identical across generations and is marked for prompt caching.
    Server-side refusal fallbacks are enabled ("default" routing).
    """

    def __init__(self, model="claude-opus-5-5", effort="high", max_tokens=64000, log_dir=None, max_usd=None, client=None):
        if client is None:
            import anthropic  # imported lazily, so the scorer and tests run without the SDK installed
            client = anthropic.Anthropic()
        self.client = client
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.log_dir = Path(log_dir) if log_dir else None
        self.usage = {}; self.last_usage = None; self.max_usd = max_usd

    PRICE_PER_MTOK = {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75}   # USD, edit to the current price list

    def _call(self, system, messages, ctx, generation, tag=""):
        if self.max_usd is not None and self.usage.get("usd", 0) >= self.max_usd:
            raise RuntimeError(f"budget reached: ${self.usage['usd']:.2f} spent of --max-usd {self.max_usd}")
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": prompts.RESPONSE_SCHEMA}},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()
        if ctx and ctx.get("dir"):                      # closed-form loop: next to the prompt of this request
            (Path(ctx["dir"]) / f"response{tag}.json").write_text(msg.to_json())
        elif self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"response_gen{generation:03d}{tag}.json").write_text(msg.to_json())
        u = getattr(msg, "usage", None)
        if u is not None:                               # tokens and cost per request
            tok = dict(input=getattr(u, "input_tokens", 0) or 0, output=getattr(u, "output_tokens", 0) or 0,
                       cache_read=getattr(u, "cache_read_input_tokens", 0) or 0, cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0)
            tok["usd"] = round(sum(tok[k] * self.PRICE_PER_MTOK[k] for k in self.PRICE_PER_MTOK) / 1e6, 4)
            for k, v in tok.items():
                self.usage[k] = self.usage.get(k, 0) + v
            self.last_usage = tok
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(msg.stop_details, 'category', None)}")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("response hit max_tokens; lower n or raise max_tokens")
        return next((b.text for b in msg.content if b.type == "text"), "")

    def propose(self, system, user, n, generation, ctx=None):
        messages = [{"role": "user", "content": user}]
        text = self._call(system, messages, ctx, generation)
        try:
            return prompts.parse_response(text)[:n]
        except prompts.ParseError as e:                 # one retry, telling the model what was wrong
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Your answer could not be used: {e}. Reply again with only the JSON object."}]
            return prompts.parse_response(self._call(system, messages, ctx, generation, tag="_retry"))[:n]
