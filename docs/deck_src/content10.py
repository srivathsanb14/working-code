"""Full audit pass: order of parts 2, 4, 6, 7 fixed so each slide builds on the previous one; one redundant slide
dropped; inaccuracies fixed (the current generator does not rebuild the shipped six; footprints are in feet; what the
local test can and cannot claim); stale notes removed."""
import content9 as C9

B, S, SECTIONS = C9.B, C9.S, C9.SECTIONS


def reorder(first, last, ids):
    i, j = [s["id"] for s in S].index(first), [s["id"] for s in S].index(last)
    block = {s["id"]: s for s in S[i + 1:j]}
    extra = set(block) - set(ids)
    missing = set(ids) - set(block) - set(B)
    assert not missing, missing
    S[i + 1:j] = [B[k] for k in ids]
    return extra


# ---------------------------------------------------------------- opening
B["questions"]["notes"] = "The source line on every slide says whether a claim comes from the official pages, the kit, simulator runs, our measurements or our code."
B["legend"]["cards"][1][0] = "From the simulator: its runs and its code"
B["decmap"]["rows"][1][1] = ("use only the kit's tools · new buildings from the kit's generator (planned, not yet run) · "
                             "hold out whole weather stations · hold out the six shipped buildings as our local test · all 5 controllers")

# ---------------------------------------------------------------- part 1
B["obsroles"]["title"] = "Most of those numbers echo our own commands; the new information is room temperatures and eight more"

# ---------------------------------------------------------------- part 2: what the kit ships, what the organisers allow, what we do
B["shipped"]["eyebrow"] = "What the kit ships · buildings"
B["shipped"]["rows"][0][3] = "Footprint (ft)"
B["scenarios30"]["eyebrow"] = "What the kit ships · scenarios"
B["windows"]["eyebrow"] = "What the kit ships · weather"
B["seed"]["eyebrow"] = "What the kit ships · the generator"
B["seed"]["cards"][2] = ["Same number, same building",
                         "The generator is deterministic: a number always gives the same layout. Any number not used before gives a building nobody has seen.",
                         "why it matters", "accent1"]
B["seed"]["callout"] = ("The six shipped buildings list the numbers they were made from (0, 1, 3001, 4007, 8, 9008), but they are smaller than the current "
                        "generator allows, so today's generator does not rebuild them exactly. They are best read as six examples from the same family.")
B["generator"]["title"] = "What the generator varies: the ranges it draws every new building from"
B["generator"]["callout"] = ("Footprints: the current generator draws 400–700 × 300–500 ft; the six shipped buildings are 318–430 × 188–301 ft. "
                             "The organisers say the test uses the same generator and ranges as the training buildings, so which size range the test uses is not certain.")
B["layouts"]["notes"] = "Shapes vary continuously; the model has to take any floor plan."
B["official"]["eyebrow"] = "So which buildings are for what? Checked with the organisers' pages"
B["official"]["notes"] = ("Honest limit: our hand-written starting models and the physics audit were developed while looking at these six buildings, so for those models "
                          "the six are not a perfectly clean test. They stay clean for anything the LLM discovers on generated buildings.")
B["change"]["eyebrow"] = "Making more data"
moved_out = reorder("s2", "s3", ["shipped", "scenarios30", "windows", "seed", "generator", "layouts", "official",
                                 "change", "datarecipe", "decdatagen", "decdataset"])
assert moved_out == {"climates"}, moved_out

# ---------------------------------------------------------------- part 3: open with the weather of the runs we look at
k = [s["id"] for s in S].index("s3")
S.insert(k + 1, B["climates"])
B["climates"]["eyebrow"] = "The runs in this part"
B["climates"]["title"] = "Six shipped buildings, one 30-day weather window each: from −40 °C in Fairbanks to 25–30 °C in Singapore"
B["climates"]["points"] = ["Each line is one 30-day window: fairbanks_1 (Jan), atlanta_3 (Apr), denver_2 (Feb), phoenix_1 (Jan), singapore_6 (Oct), atlanta_1 (Jan)",
                           "Each building is run twice in that window: with the best baseline (pid_water) and with the HVAC off: 12 runs in all",
                           "These 12 runs are the data behind parts 3, 4 and 8"]
B["climates"]["caption"] = "[sim data] outside air in the six 30-day windows; kit evaluate.py, 8,640 steps"

# ---------------------------------------------------------------- part 4: equations in the order they are derived; ssform was a duplicate of ssmatrix
B["ssmatrix"]["eyebrow"] = "Step 4 · the state-space model"
B["ssmatrix"]["title"] = "Stack every node: x(t+1) = A(g) x(t) + B(g) u(t) + E(g) d(t), one model for the whole building"
B["ssvars"]["callout"] = "State x = (T, M, W, P): 148 numbers for building_train_8. We measure the 30 room-air temperatures, and our commands act directly only on those."
B["ssvars"]["size"] = 20
dropped = reorder("s4", "s5", ["roadmap4", "cells", "balance", "linear", "layers", "sources", "commands", "roomeq", "coefftable",
                               "worked", "timescales", "fastslowreal", "euler", "sseqfull", "ssmatrix", "ssvars", "ssobs",
                               "matrix", "structok", "unknowns"])
assert dropped == {"ssform"}, dropped
B["euler"]["points"][0] = ("Heat flows are evaluated at the END of the step. x: all node temperatures; L: the conductance matrix "
                           "(L_ii = Σ_j G_ij + G_out,i, L_ij = −G_ij); Δt = 300 s. The next slide writes this out node by node")

# ---------------------------------------------------------------- part 5
B["tried"]["points"][2] = "But the hidden test gives no day 1 to fit on, and any fitting would have to run inside 0.125 s per step"

# ---------------------------------------------------------------- part 6: objective → idea → examples → whole method → why an LLM → what 'no fitting' means
B["goal"]["eyebrow"] = "What we are after"
reorder("s6", "s7", ["goal", "observation", "threeways", "histformula", "histnotfit", "observer", "bigpicture", "whympc",
                     "whyllm", "nofitdef", "nobpcompare", "nobackprop", "nofit", "honestfit", "enforce", "decmodel"])
B["whympc"]["eyebrow"] = "What the forecast is for"
B["nobackprop"]["eyebrow"] = "The alternatives, side by side"

# ---------------------------------------------------------------- part 7: problem → what the LLM reads → what it writes → how it is judged → the feedback → the search
reorder("s7", "s8", ["discflow", "discproblem", "episode", "geomscript", "templates", "assemble",
                     "sysprompt", "llmreads", "llmwrites", "netseed",
                     "checks", "scoreeq", "decscoring",
                     "analysis", "llmanalysis", "profiletab",
                     "operators", "islands", "genloop", "decsearch",
                     "pipeline", "discoutput", "bigmachine"])
B["checks"]["eyebrow"] = "How a candidate is judged · stage 0"
B["scoreeq"]["eyebrow"] = "How a candidate is judged · stages 1 and 2"
B["analysis"]["eyebrow"] = "The feedback the LLM gets"
B["operators"]["eyebrow"] = "The search · three kinds of request"
B["islands"]["points"][2] = "Every request, analysis, answer and score is written to disk (part 8: where to look)"
B["llmanalysis"]["notes"] = ("Computed by closedform.analysis on the 12 thirty-day runs of the six shipped buildings; 'held-out' in the text means "
                             "the scored episodes. In a real run they are validation buildings the search was not shown.")

# ---------------------------------------------------------------- part 8
B["s8"]["question"] = ("How do we measure how wrong a model is, and what do the six shipped buildings' 30-day runs say about the hand-written starting models?")
B["devmodel"]["notes"] = "In a real run all of these are computed on validation buildings the search was not shown."

# ---------------------------------------------------------------- part 12
B["decisions"]["rows"].insert(2, ["Footprint range for generated buildings", "the current generator's 400–700 × 300–500 ft, the shipped six's smaller sizes, or both"])
