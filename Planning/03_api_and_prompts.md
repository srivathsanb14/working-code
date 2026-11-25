# 03 · A free check of the data, a read of the exact prompts, then the API

**Needs:** 02 · **Produces:** baseline scores on the real split, reviewed prompts, a working API call, the budget ·
**Time:** half a day

## Steps

- [ ] **1. Free data check, no API, no key needed.** Score the hand-written models on the validation sample:
  ```bash
  cd $WC
  $PY code/run_discovery.py --dataset $D/dataset --run baselines --generations 0 --baselines --workers $J
  ```
  The start-up line prints how many buildings, stations and controllers the 30-episode validation sample covers (it is
  stratified: every group first). Expect the network model to be the best of the three and to beat persistence at 3 h
  (12-run set: 0.740 K vs 0.889 K). If not, stop and look at the data before spending on the API. Note the minutes per
  candidate (about 1 minute for 30 episodes on a 10-core laptop).
- [ ] **2. Read the prompts, free:**
  ```bash
  $PY code/run_discovery.py --dataset $D/dataset --run dry --dry-run --workers $J
  ```
  Writes `outputs/discovery/dry/system_prompt.md`, three init prompts and a mutation, crossover and explore prompt (the
  hand-written models stand in as parents). Read them for units, wording, missing columns, any building or station name
  (there must be none), anything that could read as permission to fit numbers. Fix and repeat. Commit two examples under
  `docs/prompt_examples/`.
- [ ] **3. Install the API client and key:** `.venv/bin/pip install -r requirements-llm.txt`; add `export
  ANTHROPIC_API_KEY=...` to `~/sb_env.sh` (never committed). Without either, discovery stops at once with a message.
- [ ] **4. One real request ($2 cap)**, to prove the request shape (beta header, server-side fallback, JSON-schema
  output) and the parser:
  ```bash
  $PY code/run_discovery.py --dataset $D/dataset --run probe --generations 1 --islands 1 --n-init 1 --max-usd 2
  ```
  `outputs/discovery/probe/generations/g000/island_0/init/response.json` exists; `log.md` shows one candidate scored.
  Pin the `anthropic` version that worked in `requirements-llm.txt`.
- [ ] **5. Prices and budget.** Compare the tokens in `log.md` with the dollars it reports; check
  `proposers.ClaudeProposer.PRICE_PER_MTOK` against the current price list and fix it if needed (otherwise `--max-usd`
  caps the wrong amount). One generation = 3 islands × 3 requests (generation 0: 3 init requests). Write the caps for 04
  and 06 in `docs/RUNBOOK.md`.

## Handoff to 04

Baseline scores on record (the bar to beat); prompts reviewed; one paid request worked; prices checked; budget written.
