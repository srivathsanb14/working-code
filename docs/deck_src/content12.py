"""Language pass: no wording that implies an earlier version of the deck or the tools, and no stock phrases."""
import content11 as C11

B, S, SECTIONS = C11.B, C11.S, C11.SECTIONS


def card(sid, i, j, text):
    c = list(B[sid]["cards"][i]); c[j] = text; B[sid]["cards"][i] = c


B["legend"]["callout"] = "Every figure is simulator data, a kit image, or computed from a real building's floor plan."
B["objective"]["notes"] = "The rest of the deck works toward this."
B["shipped"]["callout"] = ("Final scores come only from 6 different buildings at 5 different weather stations, which are never released. "
                           "We also never train on these six: they are our own local test (next slides).")
B["seed"]["callout"] = ("The six shipped buildings list the numbers they were made from (0, 1, 3001, 4007, 8, 9008), but they are smaller than the "
                        "generator's footprint ranges, so running it with those numbers gives different, larger buildings. "
                        "Read the six as examples from the same family.")
B["generator"]["callout"] = ("Footprints: the generator draws 400–700 × 300–500 ft; the six shipped buildings are 318–430 × 188–301 ft. "
                             "The organisers say the test uses the same generator and ranges as the training buildings, so which size range the test uses is not certain.")
B["official"]["eyebrow"] = "Training and test buildings, from the organisers' pages"
B["official"]["notes"] = ("A limit: our hand-written starting models and the physics audit were developed while looking at these six buildings, "
                          "so for those models the six are not a perfectly clean test. They stay clean for anything the LLM discovers on generated buildings.")
B["change"]["notes"] = "The shipped buildings' footprints (318–430 × 188–301 ft) are below the generator's ranges (400–700 × 300–500 ft), so generated buildings can be larger."
B["timescales"]["callout"] = "Fast, then slow: the two speeds seen in the data on the next slide. (A rough single-room estimate: neighbours and the wall's own storage are ignored.)"
B["euler"]["title"] = "Step forward with backward Euler, the same scheme the simulator uses"
B["comparison"]["rows"][-1][2] = "yes: the formulas take any floor plan"
B["histnotfit"]["eyebrow"] = "A common question"
B["nofitdef"]["eyebrow"] = "What 'nothing fitted' means"
B["nobpcompare"]["callout"] = "The search learns which formulas work. It never adjusts the numbers inside a formula."
B["nobackprop"]["callout"] = "We use the last row. Whether to add online updates on top of a discovered formula is an open question (part 12)."
B["honestfit"]["eyebrow"] = "Choices that remain"
B["honestfit"]["title"] = "Where human or LLM choices still enter, and how they are checked"
B["honestfit"]["callout"] = "The fitted variants we tried first (part 5) stay in the code for comparison; closed-form discovery, with nothing fitted, is the default."
B["enforce"]["eyebrow"] = "Checked by the code"
B["s7"]["title"] = "LLM discovery"
B["pipeline"]["eyebrow"] = "The whole pipeline"
B["analysis"]["callout"] = "Each analysis is saved as analysis.md next to its prompt, so anyone can read what the LLM was told."
card("canfit", 2, 1, "Non-negative per-room weights fitted on half of each run; the non-negative weights keep its forecasts stable. Never deployed.")
B["canfit"]["notes"] = "A good one-step fit does not guarantee a good forecast."
B["realprofile"]["points"][-1] = "These are the errors the pre-mutation analysis points the LLM to"
B["parity"]["title"] = "Inside the policy, the forecasts match the offline scores"
B["ctrlres"]["caption"] = "[measured] code/rehearse_submission.py through the kit's harness, with the network model"
B["gate"]["callout"] = "Ready to submit today: the pid_water-only notebook. The MPC notebook waits for this gate."
B["status"]["eyebrow"] = "Status"
B["byepisode"]["src"] = "[measured] closedform.evaluate (every 12 steps) and closedform.ceiling on the 12 thirty-day runs"
B["s1"]["question"] = "What are we asked to do, what must we submit, and how is it scored?"
B["s7"]["question"] = "How does an LLM write the formulas, what does it see, how is each candidate judged, and how does the search keep the best?"
B["s11"]["question"] = "What gets uploaded, how is it checked, and when is it ready?"
B["tried"]["points"][3] = "The way forward: keep the structure, and compute the material numbers from the measurements with a formula (part 6)"
B["observation"]["eyebrow"] = "The observation behind our approach"
B["observation"]["callout"] = "Write the physics once (part 4), fill in the geometry from the floor plan, and compute the materials from the measured history."
B["decisions"]["rows"] = [[c.replace("the current generator's", "the generator's") for c in r] for r in B["decisions"]["rows"]]
