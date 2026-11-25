"""The data pipeline at full size: memory-light split and loading, stratified samples, chunked test scoring, the
audit's tier checks, a clear stop without the API library, and a dry run that writes every prompt without the API."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from discovery import evolve, splits
from discovery.episodes import _save_cached, load_light


@pytest.fixture()
def dataset(tmp_path, toy_episodes):
    ds = tmp_path / "dataset"; ds.mkdir()
    idx = {}
    for e in toy_episodes:
        _save_cached(e, ds / f"{e.name}.npz")
        idx[e.name] = dict(file=f"{e.name}.npz", status="ok", policy=e.policy, building=e.building, weather=e.weather,
                           scenario=e.scenario)
    (ds / "index.json").write_text(json.dumps(idx))
    (ds / "split.json").write_text(json.dumps({"val": ["toy0"], "test_medium": ["toy1"]}))
    return ds


def _ref(b, w, p):
    return SimpleNamespace(name=f"{b}__{w}__{p}", building=b, weather=w, policy=p, scenario=f"{b}/{w}")


def test_stratified_sample_covers_every_group_first():
    refs = [_ref(f"b{b}", f"{st}_{k}", p) for b in range(3) for st in ("atlanta", "fairbanks") for k in range(4)
            for p in ("pid", "all_off")]                                     # 12 groups of 4
    pick = splits.stratified_sample(refs, 12)
    groups = {(r.building, r.weather.rsplit("_", 1)[0], r.policy) for r in pick}
    assert len(pick) == 12 and len(groups) == 12                             # one per group
    assert splits.stratified_sample(refs, 0) == refs and len(splits.stratified_sample(refs, 30)) == 30
    assert splits.stratified_sample(refs, 12) == pick                        # deterministic


def test_light_episode_has_geometry_and_outside_temperature_only(tmp_path, toy_episodes):
    e = toy_episodes[0]; p = tmp_path / "e.npz"; _save_cached(e, p)
    L = load_light(p)
    assert L.scenario == e.scenario and set(L.inputs) == {"T_out"} and L.T_meas.size == 0
    assert np.array_equal(L.inputs["T_out"], e.inputs["T_out"]) and set(L.geo) == set(e.geo)


def test_loader_reads_only_the_validation_sample(dataset):
    a = SimpleNamespace(dataset=str(dataset), split=None, auto_split=False, policies=None, max_val=30, max_test=0)
    val, tests, split, n = evolve.load_split(a)
    assert n == 2 and [e.name for e in val] == ["toy_0"] and val[0].T_meas.size > 0      # loaded
    assert [r.path.name for r in tests["test_medium"]] == ["toy_1.npz"]                    # a file reference, not loaded


def test_chunked_test_scores_merge_as_episode_weighted_means():
    parts = [(16, {"status": "ok", "val_3h_mean": 1.0, "objectives": {"main": 1.0}}),
             (4, {"status": "ok", "val_3h_mean": 2.0, "objectives": {"main": 3.0}})]
    m = evolve._merge_chunks(parts)
    assert m["n_episodes"] == 20 and m["val_3h_mean"] == pytest.approx(1.2) and m["objectives"]["main"] == pytest.approx(1.4)
    assert evolve._merge_chunks(parts + [(3, {"status": "unstable"})])["status"] == "unstable"


def test_audit_flags_a_scenario_in_two_tiers():
    import audit_data
    split = {"val": ["b1/atlanta_1"], "test_easy": ["b1/atlanta_1", "b1/atlanta_2"]}
    assert audit_data.tiers_of("b1/atlanta_1", split) == ["val", "test_easy"]
    assert audit_data.tiers_of("b1/atlanta_2", split) == ["test_easy"]
    assert audit_data.tiers_of("b9/denver_1", split) == []


def test_no_api_client_without_generations_and_a_clear_stop_without_the_library(monkeypatch):
    a = SimpleNamespace(generations=0, dry_run=False)
    assert evolve.api_proposer(a) is None
    assert evolve.api_proposer(SimpleNamespace(generations=3, dry_run=True)) is None

    def missing(_):
        raise ImportError("anthropic")
    monkeypatch.setattr(evolve, "make_proposer", missing)
    with pytest.raises(SystemExit, match="requirements-llm"):
        evolve.api_proposer(SimpleNamespace(generations=3, dry_run=False))


def test_dry_run_writes_every_prompt_without_the_api(dataset, tmp_path, monkeypatch):
    def no_api(_):
        raise AssertionError("the dry run must not create an API client")
    monkeypatch.setattr(evolve, "make_proposer", no_api)
    monkeypatch.setattr(evolve, "OUTD", tmp_path / "runs")
    evolve.main_args(["--run", "dry", "--dataset", str(dataset), "--dry-run", "--generations", "5", "--islands", "2", "--every", "24"])
    run = tmp_path / "runs" / "dry"
    assert (run / "system_prompt.md").exists()
    assert len(list(run.glob("generations/g000/island_*/init/prompt.md"))) == 2
    written = {p.parent.name for p in run.glob("generations/g001/island_0/*/prompt.md")}
    assert {"mutation", "explore"} <= written
    assert "MUTATION" in (run / "generations/g001/island_0/mutation/prompt.md").read_text()


def test_split_refuses_a_missing_dataset(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "code" / "make_splits.py"
    p = subprocess.run([sys.executable, str(script), "--dataset", str(tmp_path / "none"), "--out", str(tmp_path / "s.json")],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "build_dataset.py" in (p.stderr + p.stdout) and not (tmp_path / "s.json").exists()
