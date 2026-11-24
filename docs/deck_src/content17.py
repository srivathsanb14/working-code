"""Nov 24, 2026 (4): deck audited against the code after the API-only refactor.  The hand-written models are references,
not generation 0; nothing is trained; the data is made outside this pipeline; the search settings match evolve.py."""
import content16 as C16

B, S, SECTIONS = C16.B, C16.S, C16.SECTIONS
new, insert_after = C16.new, C16.insert_after
_sub = C16.C15._sub


def put(sid, obj):
    B[sid].clear(); B[sid].update(obj)


# ---- wording that changed everywhere: the hand-written models are references, not starting points
for s in S:
    o = dict(s)
    for a, b in [("hand-written starting models", "hand-written models"), ("hand-written starting model", "hand-written model"),
                 ("the starting models", "the hand-written models"), ("Starting models", "Hand-written models"),
                 ("starting models", "hand-written models"), ("A second starting model", "A second hand-written model"),
                 ("Network starting model", "Hand-written network model"), ("network starting model", "hand-written network model"),
                 ("starting model", "hand-written model")]:
        o = _sub(o, a, b)
    put(s["id"], o)

B["canfit"]["cards"][1] = ["Reference: the hand-written models", "The network and history models we wrote by hand, scored on the same real runs "
                           "(next slide). They are references only; the search starts from what the LLM writes.", "reference", "accent1"]
B["seedscompare"]["eyebrow"] = "Real data · the hand-written models, nothing fitted"
B["seedscompare"]["title"] = "The network model is the best hand-written model; all are far from the ceiling"
B["s8"]["question"] = "How do we measure how wrong a model is, and what do the six shipped buildings' 30-day runs say about the hand-written models?"
B["honestfit"]["cards"][0] = ["Hand-chosen constants", "The hand-written models' constants (U₀ = 2.0 W/m²K, shrink 5, slab coupling 4 W/m²K, 6 h store "
                              "time constant; mid-range materials) were set once by reasoning, never optimised. They are references, not parents.",
                              "hand-written models", "accent3"]

# ---- nothing is trained; data is made outside this pipeline
B["plant"]["notes"] = "The six shipped buildings have 3 to 9 AHUs and 30 to 85 rooms (kit configs; the website says 28 to 85). Generated buildings can be much larger (part 2)."
B["shipped"]["callout"] = ("Final scores come only from 6 different buildings at 5 different weather stations, which are never released. Nothing is "
                           "trained on any building: the split rule (later in this part) decides which of these runs the search sees and which are held out.")
B["generator"]["callout"] = ("Footprints: the generator draws 400–700 × 300–500 ft; the six shipped buildings are 318–430 × 188–301 ft. The organisers "
                             "say the test uses the same generator and ranges as the shipped buildings, so which size range the test uses is not certain.")
B["official"]["eyebrow"] = "Development and test buildings, from the organisers' pages"
B["official"]["title"] = "Which buildings are for development, which are for the test, and what we may generate"
B["official"]["rows"][4] = ["May we generate more buildings?", "Yes, as many buildings and scenarios as we like. We fit nothing, so they become validation and test data",
                            "[Kaggle] data; [website] scoring"]
B["change"]["callout"] = B["change"]["callout"].replace("every training building", "every generated building")
B["decdataset"]["title"] = "How simulator runs become episodes, and why"
B["decmap"]["rows"][1] = ["Data generation", "made beforehand with the kit's own tools, outside this pipeline (not yet run) · all 5 controllers · "
                          "results so far: the six shipped buildings", "2"]
B["decmap"]["rows"][2] = ["Dataset", "recover commands by replay · use only what the policy can see · heat flows as inputs · split whole episodes into "
                          "validation and three test tiers, computed from the data", "2"]
B["decmap"]["rows"][5] = ["Scoring", "physics checks · forecast horizons 1/3/6 h · test-time inputs · five objectives (Pareto) · a price per formula size", "7"]
B["decmap"]["rows"][6] = ["Search", "Claude API only, generation 0 included · islands · mutation, crossover, explore · analysis by place, time and term · retries and resume", "7"]
B["status"]["rows"][1] = ["Data generation with the kit (outside this pipeline)", "not run yet: results use the 6 shipped buildings"]

# ---- search settings as evolve.py has them
B["decsearch"]["rows"][2] = ["Requests", "generation 0: 1 init request per island (3 candidates); then per island per generation: 1 mutation (2 children), "
                             "1 crossover (2), 1 explore (1)", "balance refining, combining and new structure", "explore only: no refinement"]
B["decsearch"]["rows"][5] = ["LLM", "Claude API only: one stateless request per operator; a --max-usd budget", "the context is the prompt alone, never a chat history",
                             "one long chat: earlier answers leak into later ones"]
B["genloop"]["notes"] = "Generation 0: each island asks the API for 3 first candidates written from the system prompt alone."
B["discoutput"]["cards"][2] = ["Into the policy", "Built (part 9): the policy computes geometry from the metadata and tracks the model live; a controller "
                               "can then plan with it (part 10, a later step).", "parts 9–10", "accent3"]
B["discoutput"]["notes"] = "summary.json holds the scores and the test-tier results."
