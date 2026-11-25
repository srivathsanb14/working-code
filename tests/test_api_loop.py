"""The whole discovery loop, with a stand-in for the Claude API client (same interface; no network, no key).

Checks what the user asked to be sure of: every request is stateless (system prompt + one user message), the system
prompt is identical in every request, a request carries summaries and a short excerpt, never the episodes, no
episode or building name appears in any prompt, and the split keeps whole episodes apart."""
import json
from pathlib import Path

import pytest

from discovery import evolve, proposers, splits
from discovery.episodes import _save_cached
from discovery.splits import split_episodes

D = Path(__file__).resolve().parents[1] / "code" / "discovery"
NET = (D / "seeds_cf" / "seed_cf_network.py").read_text()


class _Msg:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason, self.stop_details = "end_turn", None
        self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0,
                                    "cache_creation_input_tokens": 0})()

    def to_json(self):
        return json.dumps({"text": self.content[0].text})


class FakeClient:
    """Answers like the API would: one candidate per request (the network model, a constant changed per call)."""

    def __init__(self):
        self.calls = []
        outer = self

        class _Stream:
            def __init__(self, kw):
                self.kw = kw

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get_final_message(self):
                k = len(outer.calls)
                code = NET.replace("H_SLAB, A_SPREAD, K_AIR = 5.0", f"H_SLAB, A_SPREAD, K_AIR = {5.0 + 0.5 * k}")
                return _Msg(json.dumps({"candidates": [{"name": f"cand_{k}", "hypothesis": "test", "code": code}]}))

        class _Messages:
            def stream(self, **kw):
                outer.calls.append(kw)
                return _Stream(kw)

        self.beta = type("Beta", (), {"messages": _Messages()})()


@pytest.fixture()
def dataset(tmp_path, toy_episodes):
    ds = tmp_path / "dataset"; ds.mkdir()
    idx = {}
    for e in toy_episodes:
        _save_cached(e, ds / f"{e.name}.npz")
        idx[e.name] = dict(file=f"{e.name}.npz", status="ok", policy=e.policy, building=e.building)
    (ds / "index.json").write_text(json.dumps(idx))
    (ds / "split.json").write_text(json.dumps({"val": ["toy0"], "test_medium": ["toy1"]}))
    return ds


def test_loop_uses_only_stateless_api_requests(dataset, tmp_path, monkeypatch, toy_episodes):
    fake = FakeClient()
    monkeypatch.setattr(evolve, "OUTD", tmp_path / "runs")
    monkeypatch.setattr(evolve, "make_proposer", lambda a: proposers.ClaudeProposer(client=fake))
    evolve.main_args(["--run", "t", "--dataset", str(dataset), "--generations", "2", "--islands", "1", "--every", "24",
                      "--n-init", "1", "--n-mut", "1", "--n-cx", "1", "--n-exp", "1", "--test", "--test-front", "1"])
    run = tmp_path / "runs" / "t"
    recs = [json.loads(x) for x in (run / "database.jsonl").read_text().splitlines()]
    assert recs[0]["operator"] == "init" and all(r["operator"] != "seed" for r in recs)   # generation 0 from the API
    assert len(fake.calls) >= 2
    systems = {c["system"][0]["text"] for c in fake.calls}
    assert len(systems) == 1                                                          # identical system prompt
    data_numbers = sum(e.n * e.V for e in toy_episodes)
    for c in fake.calls:
        assert len(c["messages"]) == 1 and c["messages"][0]["role"] == "user"        # no chat history
        user = c["messages"][0]["content"]
        assert len(user) < 40_000                                                    # summaries, not the data
        assert "toy0" not in user and "toy1" not in user and "toy_0" not in user
    assert data_numbers > 1000
    summary = json.loads((run / "summary.json").read_text())
    assert "test_medium" in summary["best"]["test"]                                  # held-out tier scored at the end


def test_split_never_shares_an_episode(toy_episodes):
    S = split_episodes(toy_episodes, {"val": ["toy0"], "test_medium": ["toy1"]})
    assert splits.check_whole_episodes(S) == {"val": ["toy_0"], "test_medium": ["toy_1"]}
    S["test_medium"] = S["test_medium"] + S["val"]
    with pytest.raises(ValueError):
        splits.check_whole_episodes(S)


def test_worse_children_are_shown_to_the_llm(tmp_path):
    """A child that ran but scored worse than its parent is listed in the next request, with what it tried and the
    change in each error; a child that improved, and one that crashed, are not in that list."""
    from discovery import prompts
    db = evolve.DB(tmp_path)
    obj = lambda m: {"main": m, "fast": m / 2, "slow": m * 2}                       # noqa: E731
    db.add(dict(id="p", island=0, operator="init", parents=[], name="parent", hypothesis="h", status="ok", score=0.70, objectives=obj(0.70)))
    db.add(dict(id="w", island=0, operator="mutation", parents=["p"], name="stronger_slab", hypothesis="double the slab term in corner rooms",
                status="ok", score=0.80, objectives=obj(0.80)))
    db.add(dict(id="b", island=0, operator="mutation", parents=["p"], name="better", hypothesis="x", status="ok", score=0.60, objectives=obj(0.60)))
    db.add(dict(id="c", island=0, operator="mutation", parents=["p"], name="crashed", hypothesis="y", status="error", score=None))
    db.add(dict(id="o", island=1, operator="mutation", parents=["p"], name="other_island", hypothesis="z", status="ok", score=0.9, objectives=obj(0.9)))
    s = db.setbacks(0, 4)
    assert [w["id"] for w in s] == ["w"]
    assert abs(s[0]["deltas"]["slow"] - 0.2) < 1e-9
    parent = dict(db.get("p"), code=NET)
    text = prompts.cf_user_mutation(1, 0, parent, "analysis", "", [], 2, None, s)
    assert "scored WORSE" in text and "stronger_slab" in text and "double the slab term" in text and "+0.1000" in text
    assert "better" not in text.split("scored WORSE")[1]
