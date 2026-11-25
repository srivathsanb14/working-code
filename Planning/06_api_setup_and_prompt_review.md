# 06 · API access, budget, and a review of the exact prompts

**Phase:** Discovery · **Priority:** P0 · **Size:** S–M · **Depends on:** 03, 05

## Goal

Make the Claude API callable from the data machine with a spending cap, and have a person read the exact prompts the
LLM will receive before any money is spent on a run.

## Steps

1. **Credentials.** Install `requirements-llm.txt`; set `ANTHROPIC_API_KEY` in the environment of the data machine
   (never committed). `run_pipeline.sh` defaults to `MODEL=claude-opus-5-5` and `MAX_USD=50`; `evolve.py` itself has no
   cap unless `--max-usd` is given, so always pass it.
2. **Budget.** Decide the cap per run and per generation. One mutation request is about 3,400–3,900 tokens plus the
   cached 11 KB system prompt; 3 islands make 9 requests per generation (3 init requests in generation 0).
3. **A prompts-only mode (code change).** There is no way today to see the prompts without calling the API. Add
   `--dry-run` to `code/discovery/evolve.py`: build every request of generation 0 and 1 exactly as a real run would
   (using the hand-written models in `seeds_cf/` as stand-in parents for generation 1), write `system_prompt.md` and each
   `prompt.md`, and stop before `proposer.propose`. Reuse the `FakeClient` pattern from `tests/test_api_loop.py`.
4. **Read them.** System prompt (task, fixed equation, data dictionary, constants, contract and rules, scoring) and one
   prompt per operator (init, mutation, crossover, explore): wrong units, unclear wording, missing columns, anything that
   names a building or station, anything the LLM could misread as permission to fit numbers.

## Done when

- A one-candidate API call succeeds from the data machine; the budget is written in `docs/RUNBOOK.md`.
- `--dry-run` exists, is tested, and its output for the stage-1 dataset has been read and fixed; example prompts are
  committed under `docs/prompt_examples/`.
