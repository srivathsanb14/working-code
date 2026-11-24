"""Undergraduate-lecture pass: nothing synthetic or drawn; the physics derived from one 20 cm cell up to the building's
state-space model, with a real room worked through; the LLM discovery told as 'formulas for the unknown numbers in
that state-space model', with real prompts, a real analysis and the exact scoring.  Every number cites code or data."""
import content5 as C5

B, S, SECTIONS = C5.B, C5.S, C5.SECTIONS
put = C5.put
for _s in S:
    B[_s["id"]] = _s          # the slides in the current order are the live ones


def new(sid, kind, **kw):
    B[sid] = dict(id=sid, kind=kind, **kw)


# ============================================================ opening: no synthetic or drawn material anywhere
put("legend", "cards", eyebrow="How to read this deck", title="Every slide says where its facts come from",
    cards=[["From the organisers", "[website]  [Kaggle]\nthe official competition pages, read Nov 19, 2026\n\n[kit]\nthe starter kit's code, files and images",
            "official", "accent2"],
           ["From real simulator runs", "[sim data]  runs made with the kit's evaluate.py\n\n[measured]  numbers we computed from those runs\n\n[audit]  our line-by-line reading of the simulator's code",
            "data", "accent1"],
           ["From our code and plans", "[our code]  working-code/code\n\n[decision]  a choice we made, with the reason\n\n[plan]  not built yet",
            "ours", "accent3"]],
    callout="Every figure is real simulator data, a kit image, or computed from a real building's floor plan. Nothing is made up or hand-drawn.",
    src="this deck")
B["questions"]["rows"][2][1] = "4 The physics, from cell to state space · 5 The usual approaches · 6 Our approach"
B["questions"]["rows"][2][2] = "How does heat move, how is it written as a state-space model, what is unknown, and how do we fill it?"
B["questions"]["rows"][3][2] = "How does an LLM find the formulas, how are they scored, and do they behave the same in a live policy?"

B["s2"]["question"] = "What does the kit ship, what is a seed, what is a weather window, and what may we change to make more data?"
B["s3"]["question"] = "What happens to every room over 30 days, and where does the score come from?"
B["s5"]["question"] = "How is building control normally done, and why does it not fit this competition?"
B["s8"]["question"] = "How do we measure how wrong a model is, and what do the real 30-day runs say about the starting models?"

B["decmap"]["rows"].insert(3, ["Physics", "derive from one cell to rooms · backward Euler like the simulator · one linear state-space model per building", "4"])

# ============================================================ part 4: from one cell to the building's state-space model
B["s4"].update(title="The physics, from one cell to a state-space model",
               question="How does heat move in the simulator, how do we write a whole building as one state-space model, and which numbers in it are unknown?")
new("roadmap4", "flow", eyebrow="The plan for this part", title="Five steps from the simulator's grid to the model we forecast with",
    steps=[("1 Cells", "the simulator's world: 20 cm squares"), ("2 Energy balance", "heat in − heat out = storage"),
           ("3 Rooms", "lump the cells of a room into a few nodes"), ("4 State space", "stack every room: x⁺ = A x + B u + E d"),
           ("5 Unknowns", "which numbers in A, B, E we do not know")],
    points=["Each step is checked against the simulator's code (our audit) or its data (part 3)",
            "A real room of building_train_8 is worked through with numbers",
            "The end point is the question the rest of the deck answers: how to fill the unknown numbers without fitting"],
    src="[audit] STATE_SPACE.md §8, §14 (read from building.py, finite_difference_solver.py, vav.py)")
new("cells", "image", eyebrow="Step 1 · the simulator's world", title="A building is a grid of 20 cm cells, 3 m tall",
    img="phys_cells.png", side="right",
    points=["Every cell is outside air, facade, partition wall, room air or door; floor_plan and zone_map in the metadata say which",
            "Each floor is a separate grid: no heat between floors, no roof, no ground, no sun",
            "Every 5 minutes the simulator updates every cell's temperature",
            "From the metadata alone we rebuild every cell exactly: 323 of 323 rooms and 0 mismatched cells over the six buildings"],
    caption="[kit] building_train_8 floor_plan.npy and zone_map.npy; cell classes computed by our room_model.grid_geometry",
    src="[audit] STATE_SPACE.md §8.1, §14.1-14.2; [our code] room_model.py, metadata_only_check.py")
new("balance", "eq", eyebrow="Step 2 · conservation of energy", title="One cell: the heat that flows in is stored as a temperature rise",
    eq="C_i · dT_i/dt  =  Σ_j G_ij · (T_j − T_i)  +  Q_i",
    points=["C_i, heat capacity (J/K) = density × specific heat × volume. One air cell: 1,206 J/m³K × 0.2 × 0.2 × 3 m = 145 J/K",
            "G_ij, conductance (W/K) = conductivity × contact area ÷ distance (Fourier's law). One cell face is 0.2 m × 3 m = 0.6 m²",
            "Q_i, heat put in directly (W): supply air, reheat, people, lights and equipment",
            "Every term is linear in temperature: twice the temperature gap, twice the flow. The next slide checks this on data"],
    src="[audit] STATE_SPACE.md §14.2; [kit] config.gin (interior_cv_density 1.2, heat_capacity 1005, control_volume_cm 20, floor_height_cm 300)")
B["linear"]["eyebrow"] = "Checked on data · linear in temperature"
new("layers", "table", eyebrow="Step 2 · walls are layers in series", title="Through a wall, resistances add: 1/G = Σ thickness ÷ (conductivity × area)",
    rows=[["Path, per 0.6 m² face", "Layers it crosses", "Conductance G (W/K)", "At mid-range materials"],
          ["room → outside", "half an air cell + facade cell + outside film", "0.6 / (0.002 + 0.1/k_ext + 1/h)", "1.48 W/K  (U ≈ 2.5 W/m²K)"],
          ["room → room through a wall", "half an air cell + partition cell + half an air cell", "0.6 / (0.004 + 0.2/k_iw)", "1.56 W/K"],
          ["room → room through a door", "two air cells (a door is open air)", "0.6 / (2 × 0.2/50)", "75 W/K"],
          ["room air → its slab and furniture", "a fixed coupling per m² of floor", "5 W/m²K × floor area", "297 W/K for a 59 m² room"]],
    colw=[0.22, 0.32, 0.26, 0.20], size=24,
    callout="Room air has an effective conductivity of 50 W/mK (0.1 m / 50 = 0.002). A door face carries as much heat as about 48 faces of wall, so a 2-cell door ≈ 19 m of partition: doors dominate room-to-room exchange.",
    src="[audit] STATE_SPACE.md §14.2 (per-face formulas); mid-range k_ext 0.29, k_iw 0.525 W/mK, h 17.75 W/m²K (network seed)")
new("sources", "cards", eyebrow="Step 2 · the heat sources", title="Four ways heat enters a room, each computed from commands or known constants",
    cards=[["Supply air", "g · (T_supply − T)\n\ng = air density × specific heat × airflow, in W/K. At the box maximum of 2.0 m³/s, g = 2,412 W/K", "HVAC", "accent2"],
           ["Reheat", "q, in W: hot water warms the supply air at the box; limited by airflow, water temperature and a 35 °C discharge cap", "HVAC", "accent2"],
           ["Lights and equipment", "10 W per m² of floor during comfort hours, 20% of that otherwise", "known schedule", "accent3"],
           ["People", "70 W per person in the room. Only the building total is observed (two steps late)", "measured", "accent1"]],
    callout="The commands turn into g, T_supply and q through the kit's own equipment equations (next slide); after that, everything is linear again.",
    src="[audit] STATE_SPACE.md §8.2 B, §14.2; [kit] config.gin (vav_max_air_flowrate 2.0), vav.py")
new("roomeq", "eq", eyebrow="Step 3 · from cells to rooms", title="Lump each room's cells into a few nodes: the same equations for every room",
    eq=("C_i   dT_i/dt  = G_m,i (M_i − T_i) + κW_i (W_i − T_i) + Σ_j κP_ij (P_ij − T_i) + Σ_j κD_ij (T_j − T_i)\n"
        "                 + g_i (T_s − T_i) + q_i + Q_gain,i + 70·n_i                       room air\n"
        "C_m,i dM_i/dt = G_m,i (T_i − M_i)                                  slab and furniture\n"
        "C_W,i dW_i/dt = κW_i (T_i − W_i) + γ_i (T_out − W_i)                outside wall\n"
        "C_P   dP_ij/dt = κP_ij (T_i − P_ij) + κP_ji (T_j − P_ij)            wall shared with room j"),
    eqsize=22, eqpt=11,
    points=["T_i: room air (measured). M_i, W_i, P_ij: slab, outside wall, shared walls (never measured)",
            "Doors are open air, so they link two rooms' air directly (κD)",
            "This is the cell equation summed over a room's cells: same physics, fewer temperatures"],
    src="[audit] STATE_SPACE.md §14.3 (room equation); [our code] seeds_cf/seed_cf_network_observer.py")
new("coefftable", "table", eyebrow="Step 3 · every coefficient", title="Every number in the room equations = a count from the floor plan × a material property",
    rows=[["Coefficient", "Formula", "From the floor plan", "Material (hidden)"],
          ["C_i, air", "1,206 J/m³K × 3 m × A_i", "floor area A_i", "none"],
          ["C_m,i, G_m,i, slab", "ρc_mass × 3 m × A_i;  5 W/m²K × A_i", "floor area", "ρc_mass"],
          ["κW_i, air → outside wall", "n_f × 0.6 / (0.002 + 0.05/k_ext), in series with a spreading resistance", "facade faces n_f", "k_ext"],
          ["γ_i, outside wall → outside air", "n_f × 0.6 / (0.05/k_ext + 1/h)", "facade faces n_f", "k_ext, h"],
          ["C_W,i, outside wall", "ρc_ext × facade cell volume", "facade cells", "ρc_ext"],
          ["κP_ij, C_P, shared wall", "n × 0.6 / (0.002 + 0.1/k_iw) + spreading;  ρc_iw × wall volume", "shared wall faces n", "k_iw, ρc_iw"],
          ["κD_ij, door", "n × 75 W/K, in series with spreading on both sides", "door faces n", "none"]],
    colw=[0.22, 0.43, 0.17, 0.18], size=22,
    callout="Spreading resistance: heat entering through a short opening must spread through the room's air. It is a universal formula in area and opening length, the same in every building.",
    src="[audit] STATE_SPACE.md §14.3 (coefficient table, spreading resistance R_s(A, L)); [our code] seed_cf_network_observer.prepare")
new("worked", "image", eyebrow="Step 3 · a worked example", title="Room 5 of building_train_8, with numbers",
    img="phys_room.png", side="right",
    points=["Air: C = 1,206 × 3 × 59.4 m² = 215 kJ/K. Slab: C_m = 3.56 MJ/K (17× the air), coupled by G_m = 297 W/K",
            "17 m of outside wall: air → wall 255 W/K, wall → outside 221 W/K; in series 118 W/K (≈ 2.3 W/m²K over 51 m²)",
            "To room 4: 9.8 m of wall (137 W/K to the wall node) and a 2-cell door (68 W/K after spreading, 150 before)",
            "To room 12: 5.2 m of wall (74 W/K to the wall node)"],
    caption="[our code] network seed formulas at mid-range materials, applied to the real floor plan",
    src="[kit] building_train_8 floor_plan.npy, zone_map.npy; [our code] episodes.geometry, seed_cf_network_observer.prepare")
new("timescales", "stats", eyebrow="Step 3 · what those numbers predict", title="Three time scales follow from capacity ÷ conductance",
    stats=[["89 s", "full supply air resets the room: 215 kJ/K ÷ 2,412 W/K, under one 5-minute step"],
           ["12 min", "room air and slab even out: 215 kJ/K ÷ 297 W/K"],
           ["≈ 9 h", "air + slab lose heat outside: 3.78 MJ/K ÷ 118 W/K"]],
    callout="Fast, then slow: exactly the two speeds seen in the data on the next slide. (A rough single-room estimate: neighbours and the wall's own storage are ignored.)",
    src="[measured] from the worked example's numbers; [kit] vav_max_air_flowrate 2.0 m³/s")
B["fastslowreal"]["eyebrow"] = "Checked on data · two speeds"
new("euler", "eq", eyebrow="Step 4 · stepping in time", title="Step forward with backward Euler, exactly as the simulator does",
    eq=("C (x⁺ − x) / Δt        =  −L x⁺ + G_out T_out + g ⊙ (T_s − x⁺) + Q\n"
        "(C/Δt + L + diag g) x⁺  =  (C/Δt) x + G_out T_out + g ⊙ T_s + Q"),
    eqsize=30, eqpt=15,
    points=["Heat flows are evaluated at the END of the step. x: all node temperatures; L: the conductance matrix (L_ii = Σ_j G_ij + G_out,i, L_ij = −G_ij); Δt = 300 s",
            "Why the end of the step: supply air can replace a room's heat in 89 s. A forward step multiplies the error by 1 − 300/89 = −2.4 each step and blows up",
            "For one room with only supply air this gives T⁺ = (T + a·T_s)/(1 + a), a = g·Δt/C = 3.4: always between T and T_s, stable for any airflow"],
    src="[audit] STATE_SPACE.md §8.1, §14.3 (backward Euler at 300 s); [our code] dlib.LinearNetwork, dlib.end_of_step")
new("ssform", "eq", eyebrow="Step 4 · the state-space model", title="Stack every room: one linear state-space model for the whole building",
    eq="x(t+1)  =  A(g) · x(t)  +  B(g) · u(t)  +  E(g) · d(t)\ny(t)    =  room-air part of x(t)   (what is measured)", eqsize=36, eqpt=18,
    points=["M(g) = C/Δt + L + diag g;   A(g) = M(g)⁻¹ C/Δt;   B(g) = M(g)⁻¹ applied to u = (g ⊙ T_s, q);   E(g) = M(g)⁻¹ applied to d = (T_out, gains, people)",
            "Linear in the temperatures; the commands change only g, so A changes with the airflow (a 'linear time-varying' model)",
            "Size: 148 nodes for building_train_8's 30 rooms in our network (air, slab, outside wall per room, plus 58 shared walls)",
            "C and L hold every coefficient of the coefficient table, so A, B and E are floor-plan counts × material properties"],
    src="[audit] STATE_SPACE.md §13.2, §14.3; [our code] dlib.LinearNetwork, dlib.network_predict")
new("matrix", "image", eyebrow="Step 4 · a real building's matrix", title="building_train_8's conductance matrix: sparse, and built from the floor plan",
    img="phys_matrix.png", side="right",
    points=["Each dot is one conductance between two nodes: 525 of 148 × 148 entries",
            "Air-to-air dots are doors; air-to-partition dots are shared walls; each room has one slab and one outside-wall node",
            "A bigger building gives a bigger matrix from the same formulas: nothing about the size is hard-coded"],
    caption="[our code] seed_cf_network_observer.prepare on building_train_8's metadata",
    src="[kit] building_train_8 floor plan; [our code] dlib.LinearNetwork")
new("structok", "table", eyebrow="Step 4 · is the structure right?", title="With the right materials the structure reproduces the simulator; with wrong ones it does not",
    rows=[["Model (3-day open-loop runs, no measurement correction)", "Error, K (min / median / max over 18 runs)"],
          ["Cell-level rebuild from the metadata, HVAC off", "0.0004 K"],
          ["Room network with spreading, true materials", "0.17 / 0.52 / 0.95"],
          ["Room network without spreading, true materials", "0.34 / 1.08 / 2.22"],
          ["Air node only (1 per room), true materials", "0.63 / 1.59 / 4.32"],
          ["Room network, materials at the middle of their ranges", "up to 9.6 K (building_train_1, Fairbanks)"]],
    colw=[0.62, 0.38], size=25,
    callout="So the equations are right, the slow stores matter, and the remaining difficulty is the material numbers.",
    src="[audit] STATE_SPACE.md §14.5 ([sim data] 6 building-weather pairs × HVAC off / random / pid_water, 864 steps)")
new("unknowns", "cards", eyebrow="Step 5 · what is unknown", title="Everything in A, B, E is known except six material numbers per building",
    cards=[["Drawn per building", "k_ext, ρc_ext: outside wall conductivity and heat capacity\nk_iw, ρc_iw: shared walls\nρc_mass: slab and furniture", "hidden", "accent1"],
           ["Set per weather station", "h: the outside film, how fast wind takes heat off the wall (8.8–26.7 W/m²K)", "hidden", "accent1"],
           ["Effectively 4–5 numbers", "k_ext and h act only together (the wall's U-value); ρc_iw barely shows in the data", "identifiability", "accent3"]],
    callout="This is the whole modelling problem: infer a handful of material numbers for a building we have never seen, from its first hours of measurements, with nothing fitted.",
    src="[audit] STATE_SPACE.md §14.2, §14.4; [kit] materials.py, weather/*.yaml")
B["obsgroups"]["eyebrow"] = "The interface · what the policy sees"
B["obsroles"]["eyebrow"] = "The interface · which fields matter"
B["actions"]["eyebrow"] = "The interface · what the policy sends"

# ============================================================ part 6: formulas for the unknown numbers
B["s6"].update(title="Our approach: formulas for the unknown numbers",
               question="The structure is known and only a few material numbers are missing. How do we fill them in for an unseen building without fitting?")
B["observation"]["callout"] = "So: write the physics once (part 4), fill in the geometry from the floor plan, and compute the materials from the measured history."
new("threeways", "table", eyebrow="Filling the gap", title="Three ways to get the material numbers for a new building",
    rows=[["Way", "How", "Problem / strength"],
          ["Fit them on the building", "adjust θ until the model matches the first day", "needs a day of data and fitting time inside 0.125 s per step (part 5)"],
          ["One universal value, state corrected by measurements", "mid-range materials; each step the room-air nodes are reset to the measured temperatures", "the hidden slab and wall nodes follow the measurements; wrong materials still bias the forecast"],
          ["A formula of the measured history", "compute the material number each step from the last hours of data, by a fixed closed-form expression", "works from the first hours; nothing is stored or learned"]],
    colw=[0.24, 0.40, 0.36], size=24,
    callout="Ours is the second and third, and their mixtures. Which formulas, which hidden nodes and which history summaries work best is what the LLM searches for.",
    src="[our code] seeds_cf/seed_cf_network_observer.py (second), seed_cf_history_slab.py (third); [decision] no fitting")
new("histformula", "eq", eyebrow="Example · a material number from history", title="The outside-wall U-value, computed from the last 24 hours of measurements",
    eq="Û  =  ( Σ x·r  +  s·U₀ )  /  ( Σ x²  +  s )\nclipped to 0.3 – 8 W/m²K", eqsize=40, eqpt=20,
    points=["r: how much each exterior room's measured 5-minute change differs from what every other term predicts",
            "x: how much that change would be with U = 1 W/m²K (300 s × wall area × (T_out − T) ÷ C)",
            "So Û is the least-squares ratio r/x over every exterior room and step in the buffer, pulled toward U₀ = 2 W/m²K (s = 5) while the buffer is short",
            "A fixed formula, re-evaluated every step on the latest 288 steps; no parameter is fitted, stored or tuned"],
    src="[our code] seeds_cf/seed_cf_history_slab.py facade_u (U_NOMINAL 2.0, SHRINK 5.0, U_LO 0.3, U_HI 8.0, HISTORY 288)")
new("observer", "eq", eyebrow="Example · hidden stores from measurements", title="The slab and wall temperatures, carried by the model and corrected by the measurements",
    eq="predict:  x⁺ = A(g) x + B(g) u + E(g) d\ncorrect:  room-air part of x ← measured T;   slab and wall parts unchanged",
    eqsize=30, eqpt=15,
    points=["Every 5 minutes the model predicts all nodes, then overwrites the room-air nodes with what was measured",
            "Hidden nodes keep their predicted values, so they are pulled toward the truth by the measured air over the following steps",
            "Like a Kalman filter with gain 1 on the air and 0 on the hidden nodes: a fixed rule, nothing tuned",
            "Real data: 0.93 K error at 3 h on 12 thirty-day runs with mid-range materials (0.959 K when the future weather is unknown)"],
    src="[our code] seed_cf_network_observer.py (init_state / predict / correct), dlib.correct_rooms; [measured] closedform.evaluate")
new("bigpicture", "flow", eyebrow="The whole method in one line", title="Floor plan + universal constants + measured history → formulas → A, B, E → forecast → control",
    steps=[("Floor plan", "areas, wall lengths, doors (metadata)"), ("History", "the last minutes to 24 h of measurements"),
           ("Formulas", "written by the LLM: coefficients and hidden stores"), ("State space", "A(g), B(g), E(g) for this building"),
           ("Forecast + MPC", "predict 2 h ahead, choose commands")],
    points=["The formulas are the only part that is searched for. They take geometry, universal constants and the measured history, and return every coefficient of the state-space model",
            "The LLM writes candidate formulas; generated data on unseen buildings decides which are better",
            "Learning happens once, offline, over the structure of the formulas; at test time nothing is fitted"],
    src="[our code] discovery/ (contract, closedform, evolve), runtime/closedform_tracker.py, control/")
put("whympc", "eq", eyebrow="Why a forecast", title="The controller plans with the forecast: model predictive control",
    eq="choose commands u(t … t+H) that minimise\n   Σ_k  0.6·comfort(T̂_k, people_k) + 0.2·energy(u_k) + 0.2·carbon(u_k)\nwith the temperatures T̂_k forecast by the model",
    eqsize=30, eqpt=15,
    points=["H = 2 h ahead (24 steps); the plan is recomputed every 15 minutes and only its first steps are applied",
            "The cost is the competition's own loss (part 1), with energy and carbon estimated from the commands",
            "Comfort depends on the forecast temperatures, so a forecast that is wrong at 1–3 h makes the plan wrong",
            "That is why candidates are scored on 1, 3 and 6 h forecasts, not on 5-minute steps"],
    src="[our code] control/planner.py, control/policies.py, control/costs.py (part 10)")

# ============================================================ part 7: LLM discovery, expanded
B["s7"]["question"] = "How does an LLM write the formulas, what exactly does it see, how is each candidate judged, and how does the search keep the best?"
new("discflow", "flow", eyebrow="Discovery in one picture", title="An LLM proposes formulas; real simulator data on unseen buildings judges them",
    steps=[("Generated data", "new buildings × weather × controllers (part 2)"), ("LLM proposes", "a Python module: the formulas"),
           ("Checks", "physics sanity on a real floor plan"), ("Score", "forecast error on unseen buildings"),
           ("Analysis", "where it is wrong, fed back to the LLM")],
    points=["The loop repeats for many generations; the best programs survive (evolutionary search)",
            "The LLM never sees the scorer's buildings by name and never supplies a number or a verdict",
            "Status: the loop, prompts, scorer and logs are built and tested on real data with hand-written models; no LLM generation has run yet"],
    src="[our code] discovery/evolve.py, prompts.py, closedform.py; docs/DISCOVERY_WALKTHROUGH.md")
B["discproblem"].update(eq="T_rooms(t + 5 min)  =  step( geometry, measured history,\n                             this step's inputs, universal constants )", eqsize=34, eqpt=17,
    points=["step() is the room equations of part 4 with every coefficient written as a formula: the same code for a 30-room or an 85-room building, Fairbanks or Phoenix",
            "Materials are never given: step() must compute what it needs from the measured history",
            "Fitting is ruled out, so we search over programs instead of over numbers",
            "The LLM proposes programs; a scorer on unseen buildings decides; an evolutionary loop keeps the best"])
new("episode", "table", eyebrow="What the LLM's formulas get to read", title="One training episode: geometry, inputs every 5 minutes, measured temperatures",
    rows=[["Part", "Contents (exactly as in the prompt)", "Source at test time"],
          ["Geometry, per room", "area, facade faces (total and by side), number of sides with outside wall, boundary faces, neighbours, bounding box, aspect, floor, air handler", "floor_plan, zone_map, AHU map"],
          ["Geometry, per pair", "shared wall cells, door cells, number of doors", "same"],
          ["Inputs, per room", "g (supply air, W/K, 0 to ~2,400), T_supply (K), Q_reheat (W), Q_gain (W)", "own commands + the kit's equipment laws"],
          ["Inputs, building", "T_out, occupants (total), comfort flag, hour and weekday (sin, cos), T_water", "observation"],
          ["Target", "every room's measured air temperature after the step; every episode starts at 294 K", "observation"]],
    colw=[0.18, 0.56, 0.26], size=22,
    callout="Train: 100 generated buildings × 4 stations × 5 controllers, 7 days each. Scored on 20 other buildings at an unseen station.",
    src="[our code] prompts.CF_DATA, episodes.geometry, starterkit_data.episode_inputs; generate_data.py defaults")
B["assemble"].update(title="The one-step form is backward Euler for one room, with its sources held for the step",
    eq="C (T⁺ − T)/Δt  =  Σ_k G_k (T_k − T⁺) + Q\n⇒   T⁺  =  ( T + Σ_k w_k T_k + b ) / ( 1 + Σ_k w_k )\n    with  w_k = Δt·G_k / C   and   b = Δt·Q / C",
    eqsize=30, eqpt=15,
    points=["Each source k (supply air, neighbours, outside, slab, wall) pulls the room toward its temperature T_k with weight w_k",
            "A weighted average plus a heat term: stays between the sources for any airflow, so it passes the stiff-air check",
            "Given to the LLM as dlib.assemble; the LLM chooses the sources, the G_k formulas and how to compute hidden T_k"])
new("sysprompt", "table", eyebrow="What the LLM reads first", title="The system prompt, part by part (identical in every request)",
    rows=[["Part", "What it says"],
          ["Task", "discover ONE set of closed-form formulas that predicts every room's temperature in ANY building, for MPC; nothing fitted"],
          ["Physics hypotheses", "heat flow is linear in temperatures; commands → heat is already done; one temperature per room is probably not enough (fast, then slow)"],
          ["Templates", "a room's equation is a sum of small formulas filled with that room's numbers: physics, HVAC, disturbances, bookkeeping"],
          ["Data dictionary", "the geometry, inputs and target of one training episode (above), with units and shapes"],
          ["Universal constants", "10 values with their kit sources: Δt, cell size, height, air and water properties, 294 K start, 70 W per person"],
          ["Contract and rules", "the module to write (next slide); no fitted numbers; conductance = conductivity × area ÷ distance; layers add"],
          ["Scoring", "the checks, the screen and the test-time 3 h score, with the size price"]],
    colw=[0.2, 0.8], size=23,
    src="[our code] prompts.CF_SYSTEM, CF_DATA, CF_CONTRACT, CF_SCORING; discovery/universal_constants.json")
new("netseed", "code", eyebrow="A second starting model: the network seed (excerpt)", title="Part 4's coefficient table, written as code",
    code=("K_EXT = (0.15 * 0.55) ** 0.5            # W/mK   middle of the generator's range\n"
          "H_FILM = 17.75                          # W/m2K  middle of the stations' range\n\n"
          "def prepare(geo, U):\n"
          "    A = geo[\"area\"]; face = U[\"HEIGHT\"] * U[\"CELL\"]          # 0.6 m2\n"
          "    C[air]  = U[\"RHO_C_AIR\"] * U[\"HEIGHT\"] * A                 # air capacity\n"
          "    C[slab] = RC_MASS * U[\"HEIGHT\"] * A                        # slab capacity\n"
          "    G[air, slab] = 5.0 * A                                     # slab coupling\n"
          "    nf = geo[\"facade_faces\"]\n"
          "    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))     # air -> wall\n"
          "    G[air, fac] = 1 / (1 / g_in + spread(A, nf * c))           # + spreading\n"
          "    G_amb[fac] = nf * face / (c / (4 * K_EXT) + 1 / H_FILM)    # wall -> outside\n"
          "    ...                                                       # shared walls, doors\n"
          "    return dict(net=dlib.LinearNetwork(C, G, G_amb, V))\n\n"
          "def correct(coef, geo, state, T_measured, U):\n"
          "    return dlib.correct_rooms(state, T_measured, V)           # air <- measurement"),
    points=["Each line is one row of part 4's table: geometry × a material number",
            "Here the materials are fixed mid-range values; the measurement correction carries the hidden stores",
            "An LLM candidate may replace any fixed material with a formula of the history, like the U-value formula of part 6"],
    src="[our code] code/discovery/seeds_cf/seed_cf_network_observer.py (abridged)")
new("operators", "table", eyebrow="What the LLM is asked to do", title="Three kinds of request, with the exact instruction each one ends with",
    rows=[["Operator", "What is sent", "The instruction (from prompts.py)"],
          ["Mutation", "one parent's code, its score, the pre-mutation analysis, a measured-vs-predicted table, recent failures",
           "“Each must change ONE thing that the analysis points to … and keep the rest. Say which diagnostic line motivates the change.”"],
          ["Crossover", "two parents' code and the pre-crossover analysis: who wins where, which functions differ",
           "“Take each template from the parent that is better where that template matters … state which template came from which parent.”"],
          ["Explore", "the island's leaderboard (names, scores, hypotheses)",
           "“A STRUCTURALLY different hypothesis … a different slow-store summary, a different way to infer materials from history … not a re-tuning.”"]],
    colw=[0.13, 0.4, 0.47], size=22,
    callout="Every answer is JSON: for each candidate a name, a one-paragraph hypothesis, and the complete module.",
    src="[our code] prompts.cf_user_mutation, cf_user_crossover, cf_user_explore, RESPONSE_SCHEMA")
put("llmanalysis", "code", eyebrow="What the LLM is told about a parent (real output)", title="The pre-mutation analysis of the history seed on 12 real 30-day runs",
    code=("held-out rollout RMSE by horizon (model / persistence), K:\n"
          "    1h 0.990/0.662, 3h 1.163/0.973, 6h 1.338/1.470\n"
          "one-step (5 min) RMSE 0.3108 K (persistence 0.3515)\n"
          "3h rollout error by group (rmse, bias), K:\n"
          "  exterior rooms                         1.308  +0.237\n"
          "  interior rooms                         0.900  -0.402\n"
          "  most exposed third of exterior rooms   1.592  +0.452\n"
          "  HVAC air on                            0.883  +0.132\n"
          "  HVAC air off                           1.269  +0.048\n"
          "  comfort hours                          1.228  +0.556\n"
          "  night/weekend                          1.199  -0.244\n"
          "skill vs persistence (ratio <1 is better): 1h 1.49, 3h 1.20, 6h 0.91\n"
          "worst rooms: episode 5 room #15 (rmse 2.29, bias +0.58,\n"
          "  facade faces 181, area 63 m2, doors 1); ..."),
    points=["Worse than 'temperature stays' at 1 and 3 h, better at 6 h: the slow part is roughly right, the fast part is not",
            "Exterior rooms too warm and interior rooms too cold; too warm in comfort hours, too cold at night; worst in the most exposed rooms: the outside-wall formula is off",
            "The obvious mutations: a different U-value formula, a faster short-term term, a different slow-store summary"],
    src="[measured] closedform.analysis on the history seed, 12 thirty-day kit runs (6 buildings × pid_water / HVAC off)")
new("profiletab", "code", eyebrow="Also in every mutation request (real output)", title="Measured vs predicted, 3 h ahead, for two rooms of a held-out run",
    code=("hours after 06:00 | room #84 (most exposed) | room #14 (largest interior) | T_out | HVAC\n"
          "   0h             |  17.70 / 18.83 C        |  23.50 / 23.08 C            | 13.9  | on\n"
          "   4h             |  20.37 / 21.44 C        |  25.90 / 25.38 C            | 19.4  | on\n"
          "   8h             |  22.27 / 22.31 C        |  24.43 / 23.84 C            | 21.7  | on\n"
          "  12h             |  22.97 / 21.16 C        |  24.79 / 24.50 C            | 18.9  | on\n"
          "  14h             |  21.86 / 18.20 C        |  24.97 / 24.64 C            | 14.4  | off\n"
          "  16h             |  20.81 / 17.48 C        |  25.19 / 24.73 C            | 12.8  | off\n"
          "  20h             |  18.97 / 18.17 C        |  25.57 / 25.07 C            | 11.1  | off\n"
          "  22h             |  18.08 / 18.24 C        |  25.74 / 25.23 C            |  8.9  | off"),
    points=["When the HVAC stops (14 h), the exposed room is forecast 3.7 K colder than measured: the model loses heat through the wall far too fast at night",
            "The interior room is within about 0.5 K all day",
            "The LLM reads this table next to the analysis"],
    src="[measured] evolve.profile_text, history seed, building_train_2 atlanta_3 pid_water (day 2)")
new("checks", "table", eyebrow="Stage 0 · physics checks", title="Before any scoring: six checks on a real floor plan, two simulated days each",
    rows=[["Check", "Set-up", "Must hold"],
          ["Equilibrium", "outside 294 K, rooms 294 K, no heat, no air", "no room drifts more than 0.05 K"],
          ["Bounded cooling", "outside 274 K", "rooms stay within 274–294 K and cool toward the outside"],
          ["Monotone", "outside 284 K instead of 274 K", "no room ends colder"],
          ["Heating", "2 kW of reheat into one room", "that room warms; no other room cools"],
          ["Stiff air", "2,400 W/K of 290 K supply air in every room", "rooms stay within 290–294 K and approach 290 K"],
          ["Shift (warning)", "every temperature raised by a constant", "the forecast shifts by the same constant"]],
    colw=[0.17, 0.43, 0.40], size=24,
    callout="A failure rejects the candidate before any data is used, and the reason goes back to the LLM in the next request.",
    src="[our code] closedform.physics_checks; DISCOVERY_WALKTHROUGH.md (shift warning)")
new("scoreeq", "eq", eyebrow="Stages 1 and 2 · the score", title="The score is the 3-hour forecast error on unseen buildings, plus a small price for size",
    eq="score  =  mean over held-out episodes of RMSE_3h\n        +  0.0001 K × (number of nodes in the code's syntax tree)",
    eqsize=32, eqpt=16,
    points=["Stage 1 screen: the 5-minute error from measured history; more than 5× persistence's error and the candidate is dropped",
            "Stage 2: from every 30 minutes after a 3 h warm-up, the model runs on its own predictions for 1, 3 and 6 h; RMSE_h = √(mean over starts and rooms of (forecast − measured)²)",
            "As at test time: the controller's own future commands are known; future outside temperature and occupancy are held at their last values",
            "Any forecast leaving 230–350 K marks the candidate unstable"],
    src="[our code] closedform.evaluate, episode_scores (every 6, warmup 36, screen 5.0), evolve.py defaults (main horizon 36 steps)")
put("islands", "flow", eyebrow="The search · evolution", title="Three islands of six programs; each generation every island asks for five children",
    steps=[("Pick", "2 random programs of the island's best 6; keep the better"), ("Mutate", "1 request, 2 children"),
           ("Cross", "1 request, 2 children"), ("Explore", "1 request, 1 child"), ("Keep", "the island's best 6 survive")],
    points=["3 islands × 5 children = 15 new programs per generation, each checked and scored as above",
            "Every 3 generations each island's best is copied to the next island (1 → 2 → 3 → 1), so good ideas spread but islands stay different",
            "Every request, analysis, answer and score is written to disk (next part: where to look)"],
    src="[our code] evolve.py defaults (--islands 3, --elites 6, --n-mut 2, --n-cx 2, --n-exp 1, --migrate-every 3), evolve.tournament")
B["genloop"]["title"] = "One generation on one island, step by step"
B["bigmachine"]["callout"] = "Before spending on the LLM: run_discovery.py --proposer dry writes every request and calls nothing, so the prompts can be read first."
B["pipeline"]["points"] = ["Built and tested with hand-written models on real data; real LLM runs have not started",
                           "One command: ./run_pipeline.sh"]

# ============================================================ part 8: no planted or synthetic evidence
B["canfit"].update(title="Is a bad score the formula's fault or the data's?",
    cards=[["Floor: persistence", "The forecast 'temperature stays where it is'. Every candidate must beat it.", "floor", "accent3"],
           ["Starting point: the seeds", "The hand-written network and history models, scored on the same real runs (next slide).", "seeds", "accent1"],
           ["Ceiling: fitted per room", "Non-negative per-room weights fitted on half of each episode; stable by construction. Never deployed.", "diagnostic only", "accent2"]],
    src="[our code] closedform.ceiling; [measured]")
B["figsources"]["rows"] = [r for r in B["figsources"]["rows"] if r[1] not in ("synthetic by design", "DRAWN")] + [
    ["Cell grid, one room, the network matrix", "real (geometry)", "building_train_8 floor plan; make_figs8.py"]]

B["legend"]["notes"] = "Every figure is data, a kit image, or computed from a real floor plan."
B["llmreads"]["src"] = "[our code] prompts.CF_CONTRACT (the text sent to the LLM, generated by prompts.cf_system_prompt)"
B["llmanalysis"]["notes"] = "Computed by closedform.analysis on the 12 real 30-day episodes; this is the exact text a mutation request would carry."
B["analysis"]["notes"] = "Both analyses are computed by code from the scores; the LLM only reads them."
B["status"]["rows"] = [[a, b] for a, b in B["status"]["rows"]]
for r in B["status"]["rows"]:
    r[0] = r[0].replace("; floor, ceiling, self-test", "; floor and ceiling")
    r[1] = r[1].replace("built and tested (mock LLM)", "built; exercised without an LLM")

for _k in ("episode", "layers", "operators", "threeways", "checks", "structok", "decmap"):
    B[_k]["font"] = 11

# ============================================================ new order
ORDER = """cover questions legend decmap
s1 why plant evaluation submission obsgroups obsroles actions budget scoring leaderboard objective
s2 shipped scenarios30 windows seed change datarecipe generator layouts decdatagen climates decdataset
s3 profiles week control devcontrol comfortdev geometry
s4 roadmap4 cells balance linear layers sources commands roomeq coefftable worked timescales fastslowreal euler ssform matrix structok unknowns
s5 rules smarter comparison tried
s6 observation threeways histformula observer bigpicture goal whympc nobackprop whyllm nofit decmodel
s7 discflow discproblem episode geomscript templates assemble sysprompt llmreads llmwrites netseed operators analysis llmanalysis profiletab checks scoreeq islands genloop decscoring decsearch pipeline discoutput bigmachine
s8 devmodel canfit seedscompare byepisode realprofile wheretolook
s9 parity deploy decruntime
s10 costval occfc ctrlres decctrl
s11 upload gate
s12 figsources status decisions roadmap close""".split()
missing = [k for k in ORDER if k not in B]
assert not missing, missing
S[:] = [B[k] for k in ORDER]
SECTIONS[:] = [(sid, d.replace("4 · Physics, checked on data", "4 · The physics, from cell to state space")
                .replace("6 · Our approach and the model form", "6 · Our approach: formulas for the unknowns")) for sid, d in SECTIONS]
