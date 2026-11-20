from pathlib import Path

from discovery import contract
from discovery.proposers import MockProposer

SEEDS = sorted((Path(__file__).resolve().parents[1] / "code" / "discovery" / "seeds_cf").glob("*.py"))


def test_mock_mutation_and_crossover_give_loadable_code():
    p = MockProposer(seed=1)
    parents = [dict(name=s.stem, code=s.read_text(), id=s.stem) for s in SEEDS]
    for op, par in (("mutation", parents[:1]), ("crossover", parents[:2]), ("explore", parents)):
        for c in p.propose("", "", 2, 1, {"operator": op, "parents": par}):
            compile(c["code"], "<c>", "exec")
            if op == "mutation":
                contract.load_closedform(c["code"])
