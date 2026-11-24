"""Sentence-by-sentence check against the code (Nov 23, 2026).  Fixes: per-run evidence recomputed (ceiling beats
persistence in 12 of 12 runs; history model loses in 8 of 12), exact operator quotes, pipeline defaults, physics-check
count, U-value definition, matrix count, package contents, occupant split; 'seed' now means only the generator's
building number ('starting model' for the hand-written programs)."""
import re
import content10 as C10

B, S, SECTIONS = C10.B, C10.S, C10.SECTIONS
ids = [s["id"] for s in S]


def walk(x, f):
    if isinstance(x, str):
        return f(x)
    if isinstance(x, list):
        return [walk(v, f) for v in x]
    if isinstance(x, tuple):
        return tuple(walk(v, f) for v in x)
    if isinstance(x, dict):
        return {k: (v if k in ("id", "kind", "img", "src", "code") else walk(v, f)) for k, v in x.items()}
    return x


# ---------------------------------------------------------------- one meaning for 'seed'
def rename(t):
    for a, b in (("network observer seed", "network model"), ("history seed", "history model"), ("network seed", "network model"),
                 ("one-node seed", "one-node model"), ("History seed", "History model"), ("Network seed", "Network model")):
        t = t.replace(a, b)
    t = re.sub(r"\bthe seeds\b", "the starting models", t)
    t = re.sub(r"\bSeeds\b", "Starting models", t)
    t = re.sub(r"\bseeds\b", "starting models", t)
    t = re.sub(r"\bevery seed\b", "every starting model", t)
    t = re.sub(r"\ba seed\b", "a starting model", t)
    return t


for s in S[ids.index("s5"):]:
    if s["id"] in ("decdatagen", "datarecipe", "official", "seed", "change", "status", "decisions", "bigmachine"):
        continue
    for k in list(s):
        if k not in ("id", "kind", "img", "src", "code"):
            s[k] = walk(s[k], rename)
B["seedscompare"]["eyebrow"] = "Real data · the hand-written starting models, nothing fitted"
for k in ("seedscompare", "byepisode", "realprofile"):
    B[k]["notes"] = (B[k].get("notes", "") + " 'Seed' in the figure's labels means starting model.").strip()
B["figsources"]["rows"] = [[c.replace("seed vs persistence", "starting models vs persistence") for c in r] for r in B["figsources"]["rows"]]
B["llmwrites"]["eyebrow"] = "What the LLM writes back: shown with a hand-written starting model, the 'history model' (abridged)"
B["genloop"]["notes"] = "The hand-written starting models fill every island in generation 0."

# ---------------------------------------------------------------- evidence, recomputed from the 12 runs
B["byepisode"]["title"] = "Where the history model loses: 8 of 12 runs at 3 h, including all six with the HVAC off"
B["byepisode"]["points"] = ["The history model beats persistence in 4 of the 6 pid_water runs and loses in all 6 HVAC-off runs",
                            "The ceiling (per-room fitted, diagnostic) beats persistence in all 12 runs, on the second half of each run",
                            "The gap between the starting models and the ceiling is what the LLM search has to close"]
B["byepisode"]["src"] = "[measured] closedform.evaluate (every 12 steps) and closedform.ceiling on the 12 thirty-day runs, rechecked Nov 23, 2026"
B["seedscompare"]["points"][-1] = "Ceiling 0.31 / 0.46 / 0.62 K; the one-node model goes unstable on one run"
B["observer"]["points"][3] = ("Real data: its 3-hour forecasts are off by 0.93 K (root-mean-square) over the 12 thirty-day runs with mid-range materials, "
                              "0.959 K when the future weather is unknown; it still loses to persistence in 4 of the 12 runs")

# ---------------------------------------------------------------- definitions that must match the code
B["histformula"]["points"][1] = "x: how much that change would be with U = 1 W/m²K: 300 s × wall area × (T_out − T) ÷ C, divided by the same end-of-step denominator as the other terms"
B["matrix"]["points"][0] = "525 non-zero entries of the 148 × 148 matrix (the diagonal included); each off-diagonal dot is one conductance between two nodes"
c = list(B["ssobs"]["cards"][2]); c[1] = ("Each hidden node in use is tied to a room's air by a positive conductance, so it shows in how that air changes. "
                             "So it can be estimated from the measured history (part 6).")
B["ssobs"]["cards"][2] = c
B["checks"]["title"] = "Before any scoring: five checks over two simulated days on a real floor plan, plus a one-day shift warning"
B["checks"]["rows"][2][2] = "rooms stay within 274–294 K and every room ends cooler than it started"
B["checks"]["rows"][5][1] = "2,400 W/K of 290 K supply air in every room, outside 294 K"
B["checks"]["rows"][6] = ["Shift (warning only)", "one day with every temperature (start, outside, supply, water) raised by 5 K", "the forecast rises by 5 K too, within 0.05 K"]
B["checks"]["callout"] = "A failed check rejects the candidate before any data is used; the reason goes back to the LLM in the island's next requests."
B["operators"]["rows"][1][2] = "“Each must change ONE thing that the analysis points to (…) and keep the rest. Say in the hypothesis which diagnostic line motivates the change.”"
B["operators"]["rows"][2][2] = ("“Write n children that COMBINE the parents: take each template from the parent that is better where that template matters (…), "
                                "make the pieces consistent, and state in the hypothesis which template came from which parent.”")
B["operators"]["rows"][3][2] = ("“Write n candidates with a STRUCTURALLY different hypothesis from all of the above (a different slow-store summary, "
                                "a different way to infer per-building materials from history, different coupling laws), not a re-tuning.”")
B["scoreeq"]["points"][3] = "Any forecast leaving 230–350 K marks the candidate unstable: its score becomes infinite and it cannot survive"
B["decscoring"]["rows"][5][1] = "any forecast leaving 230–350 K: status 'unstable', score infinite"

# ---------------------------------------------------------------- pipeline and package, as the scripts are today
B["bigmachine"]["rows"][2] = ["2", "generate buildings, scenarios and runs with the kit's tools; the six shipped buildings go to the test split",
                              "code/generate_data.py --oversample 1.3 --include-shipped"]
B["bigmachine"]["rows"][5] = ["6", "discovery (PROPOSER=dry by default: writes the prompts; claude for a real run); best scored once on test",
                              "code/run_discovery.py --proposer $PROPOSER --test"]
B["datarecipe"]["steps"][0] = ("100 new buildings", "numbers 500–599; rejected ones replaced from 10,500 up")
B["upload"]["rows"][3] = ["dist/kaggle_dataset/best.py", "the model file: today the hand-written network model; the discovered formulas after LLM runs",
                          "MPC used when present; pid_water otherwise"]
c = list(B["discoutput"]["cards"][0]); c[1] = "The winning formulas: prepare() and step(), or the stateful trio. Readable code (the hand-written starting models are about 60 lines); under a millisecond per step."
B["discoutput"]["cards"][0] = c
B["decisions"]["rows"] = [r if r[0] != "Occupants per room" else
                          ["Occupants per room", "split equally (the starting models) or by floor area (the controller's cost); a time-of-day rule is untested"]
                          for r in B["decisions"]["rows"]]
c = list(B["generator"]["cards"][0]); c[1] = "Floors 2–4\nFootprint 400–700 × 300–500 ft\nSmallest room side 30–60 ft\nAir handlers 3–7 (more if needed: at most 10 rooms each)\n1–2 people per room"
B["generator"]["cards"][0] = c
