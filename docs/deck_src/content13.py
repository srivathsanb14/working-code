"""The LLM writes the coefficient formulas of the state-space model (nodes, capacities C, conductances L, outside links,
heat inputs); the framework builds A, B, E from them.  Starting models, contract text and evidence follow the code
(discovery/coefform.py, seeds_cf/*.py, prompts.CF_CONTRACT; make_figs9.py on the 12 thirty-day runs)."""
import content12 as C12

B, S, SECTIONS = C12.B, C12.S, C12.SECTIONS
new = C12.C11.C10.C9.C8.C7.C6.new


def card(sid, i, j, text):
    c = list(B[sid]["cards"][i]); c[j] = text; B[sid]["cards"][i] = c


def insert_after(after, *ids):
    k = [s["id"] for s in S].index(after)
    S[k + 1:k + 1] = [B[i] for i in ids]


# ---------------------------------------------------------------- part 4: node counts of the network model
B["ssvars"]["rows"][3] = ["State (hidden)", "W_i", "outside wall of room i (only rooms that have one)", "25 rooms", "never measured"]
B["ssvars"]["callout"] = "State x = (T, M, W, P): 143 numbers for building_train_8. We measure the 30 room-air temperatures, and our commands act directly only on those."
B["ssvars"]["src"] = "[audit] STATE_SPACE.md §5, §13.2, §14.3; [our code] seeds_cf_legacy/seed_cf_network_nodes.py nodes(); [kit] building_train_8 action spec (4 discrete + 71 continuous = 75)"
B["matrix"]["points"][0] = "525 non-zero entries of the 143 × 143 matrix (the diagonal included); each of the 382 off-diagonal dots is one conductance between two nodes"
B["matrix"]["points"][1] = "Air-to-air dots are doors; air-to-wall dots are shared walls; every room has a slab node, and the 25 rooms with an outside wall have an outside-wall node"
B["matrix"]["caption"] = "[our code] the node network of this part (seeds_cf_legacy/seed_cf_network_nodes.py) on building_train_8's metadata"
B["ssobs"]["cards"][2] = ["Hidden but recoverable", "Each hidden node is tied to a room's air by a positive conductance, so it shows in how that air changes. So it can be estimated from the measured history (part 6).", "observability", "accent3"]

# ---------------------------------------------------------------- part 6: file names, and the correction rule now belongs to the framework
B["threeways"]["src"] = "[our code] seeds_cf/seed_cf_network.py (second), seed_cf_history.py (third); [decision] no fitting"
B["histformula"]["src"] = "[our code] seeds_cf/seed_cf_history.py facade_u (U_NOMINAL 2.0, SHRINK 5.0, U_LO 0.3, U_HI 8.0, HISTORY 288, REFRESH 36)"
B["histformula"]["points"][3] = "A fixed formula, recomputed every 3 hours (REFRESH = 36 steps) on the latest 288 steps; no parameter is fitted, stored or tuned"
B["observer"]["src"] = "[our code] discovery/coefform.py (TermModel.correct), seeds_cf/seed_cf_network.py; [measured] closedform.evaluate"
B["observer"]["points"][0] = "Every 5 minutes the framework predicts all nodes with the LLM's coefficients, then overwrites the room-air nodes with what was measured"
card("honestfit", 0, 1, "The starting models' constants (U₀ = 2.0 W/m²K, shrink 5, slab coupling 4 W/m²K, 6 h store time constant; mid-range materials) were set once by reasoning, never optimised.")
B["enforce"]["rows"][5] = ["Same path live", "runtime/closedform_tracker.py, discovery/coefform.py",
                           "the policy evaluates the same coefficient formulas and builds the same model as the scorer: no estimation step exists"]
B["enforce"]["rows"].insert(5, ["Test-time data only", "tests/test_coefform.py",
                                "the coefficient formulas only receive geometry, universal constants and measured history up to now; changing later data changes nothing they saw"])
B["decmodel"]["rows"][3] = ["Form", "formulas for every coefficient of a linear state-space model; the framework builds A, B, E and steps it",
                            "the search is over the physics itself; every candidate is a valid, stable model", "free-form one-step maps (still loadable)"]
B["decmodel"]["rows"][4] = ["Memory", "hidden nodes (slab, walls, ...) whose coefficients may read up to one day of measured history",
                            "stands in for the unmeasured slab and wall temperatures without fitting", "one temperature per room: misses the slow part"]
B["decmodel"]["rows"][5] = ["Correction", "room-air nodes set to the measurement every step; hidden nodes keep their predictions",
                            "the hidden stores follow the measured history; nothing tuned", "tuned correction weights (a Kalman gain): fitted numbers"]

B["observer"]["points"][3] = ("Real data, network model: its 3-hour forecasts are off by 0.87 K (root-mean-square) over the 12 thirty-day runs, "
                              "0.902 K when the future weather is unknown; it still loses to persistence in 3 of the 12 runs")
B["decmodel"]["rows"][3] = ["Form", "one fixed linear state-space function; the LLM writes a formula for each of its coefficients",
                            "the search is over the physics itself; every candidate is a valid model, stable for any coefficients ≥ 0",
                            "free-form one-step maps (still loadable)"]
B["decmodel"]["rows"][4] = ["Memory", "three hidden stores per room (slab, outside wall, inner walls), coefficients may read up to one day of history",
                            "stands in for the unmeasured slab and wall temperatures without fitting", "one temperature per room: misses the slow part"]
B["decmodel"]["rows"][5] = ["Correction", "room-air temperatures set to the measurement every step; the stores keep their predictions",
                            "the stores follow the measured history; nothing tuned", "tuned correction weights (a Kalman gain): fitted numbers"]

# ---------------------------------------------------------------- part 7: the fixed function, and the coefficient formulas the LLM writes
B["s7"]["question"] = "How does an LLM write the coefficient formulas of the state-space function, what does it see, how is each candidate judged, and how does the search keep the best?"
B["discproblem"].update(title="Find a formula for every coefficient of one state-space function, that works in every building",
    eq=("coefficient_k = f_k( floor plan, universal constants, measured history to now )\n"
        "x(t + 5 min)  = fixed function( x(t), commands, weather; coefficients )"), eqsize=28, eqpt=14,
    points=["The function is fixed: a few terms per room (supply air, heat, outside air, neighbours, three heat stores). The LLM writes f_k for each term's coefficient",
            "Each f_k may read only what exists at test time: the floor plan, universal constants, the measured history. Never the building's configuration file",
            "The same formulas must work for a 30-room or an 85-room building, in Fairbanks or Phoenix",
            "Fitting is ruled out, so we search over formulas: the LLM proposes, a scorer on unseen buildings decides, evolution keeps the best"])
B["assemble"].update(eyebrow="The fixed state-space function", title="Per room, every 5 minutes: a weighted average of its heat sources, plus three heat stores",
    eq=("T_i⁺ = ( T_i + air·g_i·T_supply + outside·T_out + Σ_j nb_ij·T_j + slab·S_i + wall·W_i + inner·P_i + heat·Q_i )\n"
        "       / ( 1 + air·g_i + outside + Σ_j nb_ij + slab + wall + inner )\n"
        "S_i⁺ = ( S_i + slab_back·T_i⁺ ) / ( 1 + slab_back )                      slab store\n"
        "W_i⁺ = ( W_i + wall_back·T_i⁺ + wall_out·T_out ) / ( 1 + wall_back + wall_out )   outside-wall store\n"
        "P_i⁺ = ( P_i + inner_back·T_i⁺ ) / ( 1 + inner_back )                     inner-wall store"), eqsize=20, eqpt=10,
    points=["Each line is the backward-Euler (end-of-step) form of x⁺ = x + Σ_k a_k (source_k − x): stable for any coefficients ≥ 0",
            "Together it is the linear state-space model x(t+1) = A(g) x(t) + B(g) u(t) + E(g) d(t) with x = (T, S, W, P); the coefficients are the entries of A, B and E",
            "After each step the room air T is set to the measurement; the three stores keep their predictions and so follow the measured history"],
    src="[our code] discovery/coefform.py (TermModel), prompts.CF_CONTRACT")
new("terms", "table", eyebrow="The coefficients the LLM writes formulas for", title="Eleven coefficients per room, each with a physical meaning",
    rows=[["Coefficient", "Term it scales", "Physical meaning (per 5-minute step)", "In the network starting model"],
          ["air, heat", "supply air g·T_supply; direct heat Q", "300 s ÷ the room air's heat capacity", "300 s ÷ (1,206 J/m³K × 3 m × floor area)"],
          ["outside", "outside air T_out, directly", "300 s × conductance to outside ÷ air capacity", "0 (it goes through the outside-wall store)"],
          ["neighbours", "room j's air T_j", "300 s × conductance through shared walls and doors ÷ air capacity", "wall halves in series + doors, from the floor plan"],
          ["slab, slab_back", "the slab-and-furniture store S", "exchange with the slab, seen from the air and from the slab", "5 W/m²K × area, ÷ air or ÷ slab capacity"],
          ["wall, wall_back, wall_out", "the outside-wall store W", "air ↔ outside wall ↔ outside air", "facade faces × material, ÷ capacities"],
          ["inner, inner_back", "the inner-wall store P", "exchange with the room's half of its shared walls", "half-wall conductance and capacity"]],
    colw=[0.17, 0.22, 0.31, 0.30], size=21, font=10,
    callout="Every coefficient is 300 s × a conductance ÷ a heat capacity: a floor-plan count times a material property. The materials are what the formulas must read from the measured history.",
    src="[our code] discovery/coefform.py TERMS_ROOM; seeds_cf/seed_cf_network.py")
B["templates"].update(eyebrow="What the LLM writes", title="One function returning a formula for every coefficient; the rest is fixed",
    cards=[["coefficients(geo, hist, U)", "A closed-form formula for each coefficient of the fixed function: air, heat, outside, neighbours, slab, wall and inner stores", "the A, B, E entries", "accent1"],
           ["heat(geo, u, U)", "Optional: the direct heat into each room (reheat, lights and equipment, people). Default: people split evenly", "inputs", "accent3"],
           ["HISTORY, REFRESH", "How much measured history the formulas may read (up to 24 h), and how often the coefficients are recomputed", "settings", "accent3"],
           ["The fixed function", "Ours, not the LLM's: the state-space function the coefficients go into, stepped and corrected by the framework", "fixed", "accent2"]],
    callout="Each formula may read only test-time data: the floor plan, universal constants and the measured history up to now. Nothing is fitted.",
    src="[our code] discovery/coefform.py, prompts.CF_CONTRACT")
B["sysprompt"]["rows"][1][1] = "discover ONE set of closed-form formulas for the coefficients of a fixed state-space function that predicts every room's temperature in ANY building, for MPC; nothing fitted"
B["sysprompt"]["rows"][3] = ["Model form", "the fixed function, written out; one formula per coefficient, one physical question per formula"]
B["llmreads"].update(eyebrow="What the LLM reads (excerpt of the real prompt)", title="The contract: the fixed function, and the coefficients to write",
    code=("The state-space function is FIXED; you write the FORMULA for every\n"
          "coefficient in it.  Per room i, every 5-minute step:\n"
          "  T_i+ = ( T_i + air_i g_i T_supply_i + outside_i T_out + ... ) / ( 1 + ... )\n"
          "  S_i+ = ( S_i + slab_back_i T_i+ ) / ( 1 + slab_back_i )      ...\n\n"
          "HISTORY = 288     # steps of measured history your formulas may read\n"
          "REFRESH = 36      # coefficients recomputed every REFRESH steps\n\n"
          "def coefficients(geo, hist, U):\n"
          "    return {\"air\": ..., \"heat\": ..., \"outside\": ...,\n"
          "            \"neighbours\": ...,                       # (V, V)\n"
          "            \"slab\": ..., \"slab_back\": ...,\n"
          "            \"wall\": ..., \"wall_back\": ..., \"wall_out\": ...,\n"
          "            \"inner\": ..., \"inner_back\": ...}        # each >= 0\n\n"
          "  hist  ONLY data a policy has at that moment on an unseen building\n"
          "        ... the building's configuration file is never available"),
    points=["Around it: the task, physics hypotheses, data dictionary, universal constants, scoring rules",
            "Then the operator-specific part: parents, their analysis, recent failures"],
    src="[our code] prompts.CF_CONTRACT (the text sent to the LLM, generated by prompts.cf_system_prompt)")
B["llmwrites"].update(eyebrow="What the LLM writes back, shown with the hand-written history model", title="A candidate: a formula for every coefficient",
    code=("HISTORY = 288                              # one day of 5-minute steps\n"
          "REFRESH = 36                               # U recomputed every 3 hours\n\n"
          "def coefficients(geo, hist, U):\n"
          "    dt = U[\"DT\"]\n"
          "    C_air = U[\"RHO_C_AIR\"] * U[\"HEIGHT\"] * geo[\"area\"]\n"
          "    U_wall = facade_u(geo, hist, U)          # U-value FROM HISTORY\n"
          "    return {\"air\": dt / C_air, \"heat\": dt / C_air,\n"
          "            \"neighbours\": dt * G_nb / C_air[:, None],\n"
          "            \"slab\": dt * H_SLOW * geo[\"area\"] / C_air,\n"
          "            \"slab_back\": dt / TAU_SLOW,           # 6 h store\n"
          "            \"outside\": dt * facade_area * U_wall / C_air}"),
    points=["Every number is universal, geometric, or computed from the measured history",
            "Terms it leaves out (the wall and inner stores here) are zero",
            "The answer comes as JSON: name, one-paragraph hypothesis, code"],
    src="[our code] code/discovery/seeds_cf/seed_cf_history.py (abridged; a hand-written starting model in the format the LLM must use)")
B["netseed"].update(title="Part 4's physics, written as coefficient formulas",
    code=("K_EXT = (0.15 * 0.55) ** 0.5            # W/mK   middle of the generator's range\n"
          "H_FILM = 17.75                          # W/m2K  middle of the stations' range\n\n"
          "def coefficients(geo, hist, U):\n"
          "    C_air  = RHO_C_AIR * H * A                       # air capacity\n"
          "    G_slab = H_SLAB * A;  C_slab = RC_MASS * H * A   # slab\n"
          "    G_wall = 1 / (1/g_fac + spread(A, nf * c))       # air -> outside wall\n"
          "    C_wall = RC_EXT * nf * face * c / 2\n"
          "    G_wall_out = nf * face / (c/(4*K_EXT) + 1/H_FILM) # wall -> outside\n"
          "    ...                                              # shared walls, doors\n"
          "    return {\"air\": dt/C_air, \"heat\": dt/C_air,\n"
          "            \"neighbours\": dt * (G_wall_ij + G_door) / C_air,\n"
          "            \"slab\": dt*G_slab/C_air, \"slab_back\": dt*G_slab/C_slab,\n"
          "            \"wall\": dt*G_wall/C_air, \"wall_back\": dt*G_wall/C_wall,\n"
          "            \"wall_out\": dt*G_wall_out/C_wall,\n"
          "            \"inner\": dt*G_inner/C_air, \"inner_back\": dt*G_inner/C_inner}"),
    points=["Each coefficient is 300 s × a row of part 4's coefficient table ÷ a heat capacity",
            "The materials are fixed mid-range values; the LLM's job is to replace them with formulas that read the building's materials from its measured history",
            "Network starting model: 0.87 K at 3 h on the 12 runs"],
    src="[our code] code/discovery/seeds_cf/seed_cf_network.py (abridged)")
B["operators"]["rows"][1][2] = "“Each must change ONE thing that the analysis points to (one coefficient formula, a different history summary, a history-inferred material value) and keep the rest.”"
B["operators"]["rows"][2][2] = ("“Take each coefficient formula from the parent that is better where that coefficient matters (e.g. the outside-wall conductance "
                                "from the parent better on exterior rooms), make the formulas consistent, and state which formula came from which parent.”")
B["operators"]["rows"][3][2] = ("“Write n candidates with a STRUCTURALLY different hypothesis from all of the above (a different use of the stores, a different way "
                                "to infer per-building materials from history, different coupling laws), not a re-tuning.”")
card("analysis", 1, 1, "Both parents side by side:\nwho wins where\nwhich coefficient formulas differ\nso the child takes each formula from the parent that does it better")
insert_after("assemble", "terms")

# ---------------------------------------------------------------- part 8 and 9: numbers of the starting models
B["seedscompare"].update(title="The network model starts best; every starting model is far from the ceiling",
    points=["Network model: 0.68 / 0.87 / 1.11 K at 1 / 3 / 6 h; persistence 0.66 / 0.97 / 1.47",
            "History model: 1.17 / 1.40 / 1.64 K: one store, materials partly read from history",
            "Score with weather unknown, 3 h: network 0.902, history 1.262 K",
            "Ceiling 0.31 / 0.46 / 0.62 K; the one-node model goes unstable on one run"],
    caption="[sim data] 12 thirty-day runs, forecasts started every hour; nothing fitted except the diagnostic ceiling",
    src="[measured] closedform.evaluate and closedform.ceiling on the kit runs replayed by our starterkit_data (make_figs9.py)")
B["seedscompare"]["notes"] = ""
B["byepisode"].update(title="Where the history model loses: 9 of 12 runs at 3 h, including all six with the HVAC off",
    points=["The network model loses to persistence in 3 of 12 runs; the history model in 9, every HVAC-off run among them",
            "The ceiling (per-room fitted, diagnostic) beats persistence in all 12 runs, on the second half of each run",
            "The gap between the starting models and the ceiling is what the LLM search has to close"])
B["byepisode"]["notes"] = ""
B["realprofile"]["caption"] = "[sim data] building_train_2, Atlanta window 3, pid_water, days 7–14; network model (seed_cf_network), nothing fitted"
B["realprofile"]["title"] = "Week 2 of a 30-day run: what the network model gets right, and where it is wrong"
B["realprofile"]["notes"] = ""
B["parity"]["rows"][4] = ["History model: 5-min / 3 h error", "0.5467 / 2.3963 K", "0.5467 / 2.3963 K"]
B["parity"]["rows"][5] = ["Network model: 5-min / 3 h error", "0.1950 / 0.5775 K", "0.1950 / 0.5775 K"]
B["parity"]["rows"][6] = ["Model time per step", "0.1 ms (30 rooms)", "-"]
B["parity"]["src"] = "[measured] run_policy.py --check-replay on building_train_8 / atlanta_1 (seeds_cf/seed_cf_network.py, seed_cf_history.py); runtime/closedform_tracker.py"
B["status"]["rows"][4] = ["Coefficient contract: the LLM writes a formula for each coefficient of a fixed state-space function; no fitting", "built and tested"]
B["realprofile"]["points"] = ["Daily shape: right in both rooms",
                              "Under the baseline's rapid switching, the forecast swings lower than the room does",
                              "Exposed room: up to about 0.8 K too warm at night; interior room a little too cold",
                              "These are the errors the pre-mutation analysis points the LLM to"]
B["llmanalysis"].update(title="The pre-mutation analysis of the network model on 12 real 30-day runs",
    code=("held-out rollout RMSE by horizon (model / persistence), K:\n"
          "    1h 0.681/0.662, 3h 0.872/0.973, 6h 1.105/1.470\n"
          "one-step (5 min) RMSE 0.2953 K (persistence 0.3515)\n"
          "3h rollout error by group (rmse, bias), K:\n"
          "  exterior rooms                         1.010  -0.385\n"
          "  interior rooms                         0.911  -0.599\n"
          "  HVAC air on                            1.209  -0.641\n"
          "  HVAC air off                           0.929  -0.401\n"
          "  room being reheated                    1.542  -1.262\n"
          "  room not reheated                      0.959  -0.416\n"
          "  colder half of outside temps           1.112  -0.526\n"
          "  warmer half                            0.846  -0.366\n"
          "skill vs persistence (ratio <1 is better): 1h 1.03, 3h 0.90, 6h 0.75\n"
          "worst rooms: episode 1 room #27 (rmse 3.42, bias -3.16,\n"
          "  facade faces 94, area 177 m2, doors 2); ..."),
    points=["About equal to 'temperature stays' at 1 h, better at 3 and 6 h: the slow part is right, the fast part not yet",
            "Too cold almost everywhere, worst in rooms being reheated (−1.26 K): the heat formula or the air coefficient is off",
            "Obvious mutations: a different heat() formula, a smaller 'air'/'heat' coefficient (furniture adds capacity), a material read from history"],
    src="[measured] closedform.analysis on the network model, 12 thirty-day kit runs (6 buildings × pid_water / HVAC off)")
B["profiletab"].update(
    code=("hours after 06:00 | room #84 (most exposed) | room #14 (largest interior) | T_out | HVAC\n"
          "   0h             |  17.70 / 18.24 C        |  23.50 / 23.04 C            | 13.9  | on\n"
          "   4h             |  20.37 / 20.23 C        |  25.90 / 25.08 C            | 19.4  | on\n"
          "   8h             |  22.27 / 21.73 C        |  24.43 / 24.23 C            | 21.7  | on\n"
          "  12h             |  22.97 / 22.71 C        |  24.79 / 24.56 C            | 18.9  | on\n"
          "  14h             |  21.86 / 22.07 C        |  24.97 / 24.72 C            | 14.4  | off\n"
          "  16h             |  20.81 / 21.51 C        |  25.19 / 24.86 C            | 12.8  | off\n"
          "  20h             |  18.97 / 20.13 C        |  25.57 / 25.15 C            | 11.1  | off\n"
          "  22h             |  18.08 / 19.37 C        |  25.74 / 25.28 C            |  8.9  | off"),
    points=["After the HVAC stops, the exposed room is forecast up to 1.3 K too warm: it loses heat to the outside too slowly at night",
            "The interior room is within about 0.8 K all day, a little too cold",
            "The LLM reads this table next to the analysis"],
    src="[measured] evolve.profile_text, network model, building_train_2 atlanta_3 pid_water (day 2)")

# ---------------------------------------------------------------- what the formulas may read at deployment
new("available", "compare", eyebrow="What the formulas may read", title="Available on an unseen building at deployment, and never available",
    left=("Available: the formulas may use it", [
        "Metadata: floor plan, zone map, air-handler map, schedule → each room's area, walls, doors",
        "Every 5 min: room temperatures (one step late), outside air, people in the building, comfort flag, clock",
        "Our own commands → supply air, reheat heat, boiler water",
        "Universal constants: air and water properties, cell size, height, step, 70 W per person"]),
    right=("Never available: the formulas must not use it", [
        "The building's configuration file (YAML): materials, seed, generator settings",
        "Slab and wall temperatures (never measured)",
        "People per room; future weather and occupancy",
        "Which building or weather window it is; anything after 'now'"]),
    callout="The materials are hidden in the configuration file, so any formula that needs them must read them from the measured history.",
    src="[website] submission.html (building metadata); [Kaggle] data tab; [our code] closedform.EXOGENOUS, runtime/inputs.py, tests/test_coefform.py, tests/test_deployment.py")
insert_after("discproblem", "available")
