# Using an LLM for symbolic discovery of the room dynamics

The LLM proposes **structure**: small Python programs that state a hypothesis about how room temperatures evolve. Numerical optimization fits every constant, and a simulation on held-out buildings decides which program is better. The LLM never supplies numbers or verdicts.

**Status.** The loop runs end to end with the `dry` proposer, but only the two hand-written seeds have been evaluated; no LLM generation has run yet. **Default deployment is offline** (`--deployment offline`): no fitting on a new building, only its floor plan, measured history and universal constants (STATE_SPACE.md §17.8).

Code: `code/discovery/` (`loop.py` orchestrates; `prompts.py` is what the LLM sees; `contract.py` is what it must write; `evaluate.py` fits and scores; `diagnostics.py` writes the feedback; `proposers.py` is where candidates come from). Entry point: `code/run_discovery.py`.

```
 data/dataset (build_dataset.py) ──► episodes: geometry G, inputs, measured temperatures
                                          │
 proposer ──► candidate programs ──► contract check ─► physics checks ─► fit on train (p + shared θ) ─► score on val ─► diagnostics
    ▲                                                                                                                         │
    └─────────────── next prompt: best programs + their diagnostics + recent failures ◄──────────────────────────────────────┘
```

## 1. What goes into the LLM

One request per generation. The exact text of every request is saved in `outputs/discovery/<run>/prompts/prompt_genNNN.md`; read it before spending money.

**System prompt** (identical every generation, so it is cached):

| Part | Content |
|---|---|
| Task | Discover the dynamics of rooms as interpretable simulation programs: what stores heat, what exchanges heat with what, how every coefficient follows from geometry plus a few constants. Prefer energy balances, sensible units and parsimony |
| Data dictionary | **Geometry** `geo`: `V`, `area`, `facade_faces`, `wall_cells` (V×V), `door_cells` (V×V), `n_doors` (V×V), `bbox_x/y`, `ahu`, `floor`. **Inputs** (n steps): `T_out`, `g` (W/K), `T_supply`, `Q_reheat`, `Q_gain`, `occupants` (building total), `comfort`, clock. **Target**: measured room temperatures after each step, from a uniform 294 K start. **Constants**: 300 s step, 0.2 m cell, 3 m height, air 1206 J/m³K, 70 W per occupant. These are exactly GUIDE.md B.3/B.5, all available at run time |
| Contract | `PARAMS` (global constants with bounds), `THETA` (material constants with bounds), `simulate(geo, inputs, theta, p, T0) -> (n, V)`, `network(...)` (required for offline scoring); numpy, math and `dlib` only; at most 16 constants; no per-room fitted coefficients; implicit integration recommended (`dlib.LinearNetwork`) |
| Scoring rules | The checks, the fit and the held-out protocol of the chosen `--deployment` (offline: shared θ, no fitting on val, 3 h filtered-forecast score; online: θ from day 1, open-loop score on later days), and the complexity price |
| Physics notes | Only with `--informed`: a short summary of the simulator audit (STATE_SPACE.md §14) |

**User message** (changes every generation):

| Part | Content |
|---|---|
| Elites | The best `--elites` programs so far, each with: full source, score, held-out and train RMSE, and **diagnostics**. The diagnostics give error by facade exposure, door count and room size (tertiles), the worst rooms with their features, HVAC on vs off, time of day, error growth day 1 → 3, and the spread of fitted θ across buildings (identifiability) |
| Failures | Recent rejected or crashed programs with the reason (failed physics check, exception and traceback, timeout) |
| Request | Write `--n` new candidates, at least `--explore` of them structurally new (new state, coupling law or coefficient formula); the rest may refine the elites |

## 2. What comes out of the LLM

Structured JSON (schema `RESPONSE_SCHEMA` in `prompts.py`):

```json
{"candidates": [{"name": "slab_two_layer", "hypothesis": "…why, citing the diagnostics…", "code": "import numpy as np\nimport dlib\nPARAMS = {…}\nTHETA = {…}\ndef network(geo, inputs, theta, p): …\ndef simulate(geo, inputs, theta, p, T0): …"}]}
```

Each `code` is a complete, deployable model: it reads only run-time quantities. The loop's outputs, in `outputs/discovery/<run>/`:

| File | Content |
|---|---|
| `summary.json` | leaderboard; best program's fitted `PARAMS`, its shared θ (`theta_shared`, offline) and the θ used per val scenario, its val errors (and test errors with `--test`) |
| `best.py` | the winning program, the discovered dynamics equation |
| `database.jsonl` | every candidate: status, score, fitted constants, errors, diagnostics, source |
| `candidates/*.py` | every program evaluated |
| `prompts/`, `responses/` | every request and raw response |
| `data_report.json`, `config.json` | which runs were used; the run settings |

## 3. Running it

**With Claude** (`claude-opus-5-5`, adaptive thinking, structured output, cached system prompt, server-side refusal fallbacks):

```bash
pip install -r requirements-llm.txt
```

Credentials come from `ANTHROPIC_API_KEY` or `ant auth login`. Then:

```bash
python code/run_discovery.py --run main --proposer claude --generations 20 --n 6 --explore 2
```

**Without spending anything first:** score the seeds and write the next prompt:

```bash
python code/run_discovery.py --run main --generations 0
```

```bash
python code/run_discovery.py --run main --proposer dry --generations 1
```

**With any other LLM, or by hand:** take `prompts/prompt_genNNN.md` to the LLM, save each returned program as a `.py` file in `outputs/discovery/main/inbox/`, then:

```bash
python code/run_discovery.py --run main --proposer dir --generations 1
```

The loop also writes the next prompt into the inbox, so this can be repeated generation by generation.

A run continues from its database when re-run with the same `--run`.

| Option | Default | Use |
|---|---|---|
| `--dataset` | `data/dataset` | the built dataset; its `split.json` is used unless `--split` is given |
| `--split` | dataset's | glob patterns, GUIDE.md A.3 |
| `--policies` | all | restrict which behaviour policies' runs are used |
| `--generations`, `--n`, `--explore`, `--elites`, `--failures` | 0, 6, 2, 4, 6 | loop size and prompt content |
| `--seeds` | all | `all` (naive and audited physics network), `naive` (blind runs), `none` |
| `--informed` | off | add the audit's physics notes to the prompt |
| `--deployment` | `offline` | `offline`: no fitting on new buildings, one shared θ, score = val 3 h filtered-forecast RMSE (needs `network()`). `online`: θ per building from day 1, score = open-loop RMSE after day 1. `both`: runs each into `<run>_offline` and `<run>_online` and compares (calls the LLM twice as often) |
| `--max-nfev`, `--stride` | 30, 2 | fitting effort (lower = faster, rougher) |
| `--complexity-weight` | 0.0001 K per AST node | parsimony price |
| `--timeout` | 1200 s | per candidate |
| `--model`, `--effort` | `claude-opus-5-5`, `high` | Claude settings |
| `--test` | off | at the end, also score `best.py` on test (prefer `test_model.py`, step 7) |

## 4. Choosing a mode

- **Blind** (`--seeds naive`, no `--informed`): the LLM knows nothing from the simulator's source. Use it for a scientific claim of discovery. Success looks like the search reaching the audited network's error (the `seed_slab_wall_network` score) or rediscovering its ingredients: a slab store, door-dominated coupling, a spreading resistance.
- **Informed** (default seeds, optionally `--informed`): start from the audited structure and let the LLM improve the coefficient formulas and the slow residual. Use it for the competition.

## 5. Steering it

- **Read the diagnostics of the elite before each generation.**
  - Error growing day 1 → 3 means a slow store or a steady-state gain is off.
  - Bias by exposure tertile means the facade law is off.
  - Bias by door count means the coupling law is off.
  - Wide θ spread across buildings (online mode) means a constant is not identifiable from one day, so merge or fix it.
- **Keep `--explore` ≥ 2**, so the search does not only re-tune.
- **Watch the complexity.** A much longer program has to earn its error reduction on val.
- **Train with all five policies.** `bang_bang` and `all_off` carry most of the information.
- **Never select on test.** Select on val (rotate the val station, RUNBOOK step 2), then run `test_model.py` once.

## 6. Writing a candidate by hand

Copy `code/discovery/seeds/seed_rc_one_node.py` (the smallest valid model) and change it. A candidate must:
1. define `PARAMS` and `THETA` as `{"name": (init, lo, hi)}` with finite bounds;
2. compute every per-room coefficient from `geo` and the constants;
3. return `(n, V)` temperatures from `simulate`;
4. pass the physics checks at the initial values and at both bounds:
   - no drift at equilibrium;
   - cooling bounded toward a colder outside;
   - heat into a room warms it and cools no neighbour;
   - stable under a 2400 W/K air stream.

Define `network()` as the seeds do: the default offline score is the filtered forecast, so candidates without it are rejected (online mode accepts them).

## 7. From the discovered model to the controller

`best.py` with the global `PARAMS` from `summary.json` is the model the policy ships (in a Kaggle Dataset). At run time the policy:
1. builds the geometry from the metadata arrays;
2. uses the shared θ from `summary.json` (offline, default), or fits θ on the first day (online, optional);
3. runs a Kalman filter over `network()` on the measured room temperatures, so the hidden state follows the measured history;
4. predicts with the same inputs it computes from its own actions (GUIDE.md B.5).

`test_model.py` measures exactly this protocol.
