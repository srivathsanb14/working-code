"""Where candidates come from.  Every proposer has the same interface:

    propose(system: str, user: str, n: int, generation: int) -> list[dict(name, hypothesis, code)]

  SeedProposer       hand-written starting programs in seeds/ (no LLM)
  DirectoryProposer  *.py files dropped into an inbox folder (any LLM, a human, another tool)
  DryRunProposer     writes the exact prompt to disk and returns nothing (inspect before spending)
  ClaudeProposer     the Claude API (anthropic SDK); only used when explicitly selected
  MockProposer       NO LLM: mechanical mutation (rescale numbers) and crossover (swap templates) of the
                     parents, so the island / lineage / logging machinery can be exercised and inspected offline

The closed-form loop (evolve.py) passes ctx = {"operator", "parents": [records], "dir": request folder}.
"""
from __future__ import annotations

import ast
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

    def propose(self, system, user, n, generation, ctx=None):
        if self.used:
            return []
        self.used = True
        return [_from_file(p) for p in self.files]


class DirectoryProposer:
    """Reads new *.py candidates from an inbox; moves them to inbox/consumed afterwards."""

    def __init__(self, inbox):
        self.inbox = Path(inbox); (self.inbox / "consumed").mkdir(parents=True, exist_ok=True)

    def propose(self, system, user, n, generation, ctx=None):
        (self.inbox / f"prompt_gen{generation:03d}.md").write_text(f"# SYSTEM\n\n{system}\n\n# USER\n\n{user}\n")
        out = []
        for p in sorted(self.inbox.glob("*.py"))[:n]:
            out.append(_from_file(p)); p.rename(self.inbox / "consumed" / p.name)
        return out


class DryRunProposer:
    def __init__(self, folder):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)

    def propose(self, system, user, n, generation, ctx=None):
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
        self.usage = {}; self.last_usage = None

    PRICE_PER_MTOK = {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75}   # USD, edit to the current price list

    def _call(self, system, messages, ctx, generation, tag=""):
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
        if u is not None:                               # backlog E4: tokens and cost per request
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
        except prompts.ParseError as e:                 # backlog E5: one retry, telling the model what was wrong
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Your answer could not be used: {e}. Reply again with only the JSON object."}]
            return prompts.parse_response(self._call(system, messages, ctx, generation, tag="_retry"))[:n]


class MockProposer:
    """Stands in for the LLM (clearly labelled "mock" in every record).  Mutation rescales one or two numbers of
    the parent; crossover takes each template (top-level function or UPPERCASE constant) present in both parents
    from A or B at random; explore rescales several numbers of a seed.  Children may fail the checks: that is
    useful too, the failures show up in the logs exactly as LLM failures would."""

    def __init__(self, seed=0):
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def _scale(self, src, k):
        tree = ast.parse(src)
        doc = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)}
        nums = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float) and id(n) not in doc and n.value != 0]
        if not nums:
            return src, "no numbers to change"
        what = []
        for i in self.rng.choice(len(nums), size=min(k, len(nums)), replace=False):
            f = float(self.rng.choice([0.6, 0.8, 1.25, 1.6]))
            what.append(f"{nums[i].value:g} -> {nums[i].value * f:g}")
            nums[i].value = nums[i].value * f
        return ast.unparse(tree) + "\n", ", ".join(what)

    def propose(self, system, user, n, generation, ctx=None):
        op, parents = ctx["operator"], ctx["parents"]
        out = []
        for j in range(n):
            if op == "mutation":
                code, what = self._scale(parents[0]["code"], 1 + int(self.rng.integers(2)))
                hyp = f"[mock mutation, no LLM] rescaled {what} in {parents[0]['name']}"
            elif op == "crossover":
                A, B = (ast.parse(p["code"]) for p in parents)
                key = lambda n: n.name if isinstance(n, ast.FunctionDef) else (
                    n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.isupper() else None)
                fb = {key(n): n for n in B.body if key(n)}
                took = []
                for i, nd in enumerate(A.body):
                    k = key(nd)
                    if k in fb and ast.unparse(nd) != ast.unparse(fb[k]) and self.rng.random() < 0.5:
                        A.body[i] = fb[k]; took.append(k)
                code = ast.unparse(A) + "\n"
                hyp = (f"[mock crossover, no LLM] {parents[0]['name']} with " + (", ".join(took) or "nothing")
                       + f" taken from {parents[1]['name']}")
            else:
                base = parents[int(self.rng.integers(len(parents)))] if parents else None
                if base is None:
                    return []
                code, what = self._scale(base["code"], 4)
                hyp = f"[mock explore, no LLM] rescaled {what} in {base['name']}"
            out.append(dict(name=f"mock_{op[:3]}_g{generation}_{j}", hypothesis=hyp, code=code))
        return out
