"""Code-vs-deck audit (Nov 24, 2026): sources point at the files that exist, the crossover analysis slide matches the
code (coefficient values compared term by term), package size, timings, and controller checks re-run with the
coefficient-form network model."""
import content13 as C13

B, S, SECTIONS = C13.B, C13.S, C13.SECTIONS


def card(sid, i, j, text):
    c = list(B[sid]["cards"][i]); c[j] = text; B[sid]["cards"][i] = c


B["roomeq"]["src"] = "[audit] STATE_SPACE.md §14.3 (room equation); [our code] seeds_cf_legacy/seed_cf_network_nodes.py (the node network)"
B["coefftable"]["src"] = "[audit] STATE_SPACE.md §14.3 (coefficient table, spreading resistance R_s(A, L)); [our code] seeds_cf_legacy/seed_cf_network_nodes.py"
B["worked"]["src"] = "[kit] building_train_8 floor_plan.npy, zone_map.npy; [our code] episodes.geometry, seeds_cf_legacy/seed_cf_network_nodes.py (same formulas as seeds_cf/seed_cf_network.py)"
B["histnotfit"]["src"] = "[our code] seeds_cf/seed_cf_history.py facade_u; prompts.CF_CONTRACT ('the only estimation allowed is a closed-form expression of the history buffer')"
B["nofit"]["left"] = ("Buys", ["Works from step 1: no identification period", "A bad score means the formula is wrong, not the optimizer",
                              "Cheap: about 0.1 ms per step for 30 rooms (measured inside the policy)"])
B["templates"]["notes"] = "One physical question per formula, so crossover can recombine them."
card("discoutput", 0, 1, "The winning coefficient formulas: coefficients(), and heat() if the LLM wrote one. Readable code (the starting models are 20–65 lines); about 0.1 ms per step.")
B["upload"]["rows"][2][1] = "our runtime, controllers, safety layer, vendored pid_water (26 files)"
B["upload"]["rows"][3][1] = "the model file: today the hand-written network model (seeds_cf/seed_cf_network.py); the discovered formulas after LLM runs"
B["decisions"]["callout"] = ("Settled: no fitting · universal constants · a fixed state-space function whose coefficients the LLM writes · hidden stores "
                             "corrected by measurements · test-time scoring · error in the first hours accepted.")
card("analysis", 1, 1, "Both parents side by side:\nwho wins where\nwhich functions differ\nhow far apart their coefficient values are, term by term, on a held-out building")
B["analysis"]["src"] = "[our code] closedform.analysis, crossover_analysis, coefficient_values"
B["operators"]["rows"][2][2] = ("“Write n children that COMBINE the parents: take each coefficient formula from the parent that is better where that "
                                "coefficient matters (…), make the formulas consistent, and state in the hypothesis which formula came from which parent.”")
B["ctrlres"].update(title="Model-based control: a small gain over pid_water in short checks, mostly in the cold",
    points=["Score change against pid_water (negative = better). MPC: −3.4% in fairbanks_1, +0.5% in atlanta_1; −1.4% on the mean",
            "Feed-forward (pid_water, but the model decides when to start pre-heating and when to coast in the evening): −1.8% in fairbanks_1, equal in atlanta_1",
            "Two buildings, two days each: a hint, not the benchmark",
            "Planning takes 35–45 ms every 3 steps on 30–39 rooms"],
    caption="[measured] code/rehearse_submission.py through the kit's harness, network model (seeds_cf/seed_cf_network.py)",
    src="[measured] rehearse_submission.py --controller pid_water|ff|mpc --steps 576 (outputs/rehearsal/check_*/report.json); make_figs10.py")

C13.C12.C11.C10.C9.C8.C7.C6.new("gaincalc", "eq", eyebrow="How the gains are calculated", title="Same building, same weather, same two days: only the controller changes",
    eq=("score = mean over the 576 steps of the competition's loss per step (0.6·comfort + 0.2·energy + 0.2·carbon)\n"
        "change = ( score_controller − score_pid_water ) / score_pid_water\n"
        "Fairbanks, MPC:  ( 0.02230 − 0.02309 ) / 0.02309 = −3.4%      (negative = better)"), eqsize=22, eqpt=11,
    points=["Each run is the first 2 days (576 five-minute steps) of the shipped scenario, through the kit's own harness; the simulator is deterministic, so a difference is due to the controller alone",
            "pid_water: the kit's best rule-based controller. Feed-forward: pid_water, but the model chooses when to start pre-heating and when to coast. MPC: every 15 minutes, for each of 4 modes (schedule, comfort, eco, off), the model predicts 2 h ahead with pid_water acting on the predicted rooms, prices it with the competition's loss, and the cheapest mode runs",
            "'Mean' is the average of the two buildings' scores. These are 2-day scores, so they are not comparable with the 30-day baseline score of 0.054"],
    src="[our code] rehearse_submission.py, control/policies.py (MPCPolicy, FeedForwardPolicy), control/planner.py, control/costs.py; [measured] outputs/rehearsal/check_*/report.json")
k = [s["id"] for s in S].index("ctrlres")
S.insert(k, B["gaincalc"])
