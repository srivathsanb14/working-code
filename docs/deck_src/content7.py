"""State-space model spelled out (state, control, disturbance, output; every equation; observability), and a full,
honest account of 'no backpropagation, no fitting' with how the code enforces it."""
import content6 as C6

B, S, SECTIONS = C6.B, C6.S, C6.SECTIONS
new = C6.new


def insert_after(after, *ids):
    k = [s["id"] for s in S].index(after)
    S[k + 1:k + 1] = [B[i] for i in ids]


# ============================================================ part 4: what is what, every equation
new("ssvars", "table", eyebrow="Step 4 · what is what", title="State, control, disturbance and output of the building model",
    rows=[["Role", "Symbol", "What it is", "Count (building_train_8)", "Known how?"],
          ["State (observed)", "T_i", "air temperature of room i", "V = 30", "measured every 5 min, one step late"],
          ["State (hidden)", "M_i", "slab and furniture of room i", "V = 30", "never measured"],
          ["State (hidden)", "W_i", "outside wall of room i (rooms without one: unused)", "V = 30", "never measured"],
          ["State (hidden)", "P_ij", "wall shared by rooms i and j", "58 walls", "never measured"],
          ["Control", "a", "the action: K + 1 on/off switches, 3K + 2 + 2V setpoints", "4 + 71", "we choose it"],
          ["Control, as heat", "g_i, T_s, q_i", "supply airflow (W/K), supply temperature, reheat (W)", "per room", "from the action, via the kit's equipment laws"],
          ["Disturbance", "T_out, Q_gain,i, n_i", "outside air; lights and equipment; people", "1 + V + V", "measured; schedule; building total"],
          ["Output", "y = T", "the room-air temperatures", "V = 30", "the measurement"]],
    colw=[0.15, 0.13, 0.36, 0.16, 0.20], size=21, font=10,
    callout="State x = (T, M, W, P): 148 numbers for building_train_8. We measure 30 of them and act directly on the same 30.",
    src="[audit] STATE_SPACE.md §5, §13.2, §14.3; [our code] seed_cf_network_observer.prepare; [kit] building_train_8 action spec (4 discrete + 71 continuous = 75)")
new("sseqfull", "eq", eyebrow="Step 4 · every equation, discretised", title="One backward-Euler step, written out node by node (+ = end of the step)",
    eq=("air i:    (C_i/Δt + G_m,i + κW_i + Σ_j κP_ij + Σ_j κD_ij + g_i) T_i⁺ − G_m,i M_i⁺ − κW_i W_i⁺\n"
        "                  − Σ_j κP_ij P_ij⁺ − Σ_j κD_ij T_j⁺  =  (C_i/Δt) T_i + g_i T_s + q_i + Q_gain,i + 70 n_i\n"
        "slab i:   (C_m,i/Δt + G_m,i) M_i⁺ − G_m,i T_i⁺            =  (C_m,i/Δt) M_i\n"
        "wall i:   (C_W,i/Δt + κW_i + γ_i) W_i⁺ − κW_i T_i⁺        =  (C_W,i/Δt) W_i + γ_i T_out\n"
        "wall ij:  (C_P/Δt + κP_ij + κP_ji) P_ij⁺ − κP_ij T_i⁺ − κP_ji T_j⁺  =  (C_P/Δt) P_ij"),
    eqsize=20, eqpt=10,
    points=["Left: every unknown temperature at the end of the step. Right: what is known now (the current state, the commands, the disturbances)",
            "One equation per node, N equations in N unknowns: a single linear solve per step, nothing iterative",
            "Each coefficient is one entry of the coefficient table: a floor-plan count × a material number"],
    src="[audit] STATE_SPACE.md §14.3 (room equations); [our code] dlib.LinearNetwork, dlib.network_predict (the same solve)")
new("ssmatrix", "eq", eyebrow="Step 4 · the same thing as matrices", title="Stacked: M(g) x(t+1) = D x(t) + B₀ u(t) + E₀ d(t)",
    eq=("x(t+1)  =  A(g) x(t)  +  B(g) u(t)  +  E(g) d(t)          y(t)  =  H x(t)\n\n"
        "M(g) = D + L + diag(g, 0)     D = diag(C)/Δt     A = M⁻¹ D     B = M⁻¹ B₀     E = M⁻¹ E₀     H = [ I_V  0 ]"),
    eqsize=24, eqpt=12,
    points=["x = (T, M, W, P), N numbers;  u = (g_i T_s, q_i), 2V numbers;  d = (T_out, Q_gain,i + 70 n_i);  y = the V room temperatures",
            "L: the conductance matrix (row sums on the diagonal, −G off it). B₀ puts u on the air rows; E₀ puts T_out on the wall rows (γ_i) and gains on the air rows",
            "The commands enter twice: through u, and through g on the diagonal of M. So the model is linear in the state but the matrices depend on the airflow"],
    src="[audit] STATE_SPACE.md §13.2, §14.3; [our code] dlib.LinearNetwork (A0inv = (D + L)⁻¹, Woodbury update for diag g)")
new("ssobs", "cards", eyebrow="Step 4 · observable and controllable", title="We see and push only the room air; the hidden stores are reached through it",
    cards=[["Observed", "Only the V room-air temperatures, one step late. Slab, wall and shared-wall temperatures are never measured.", "y = H x", "accent2"],
           ["Controlled", "Only the room-air rows: airflow g and reheat q act on the air nodes. The slab and walls change only by exchanging heat with the air.", "B₀ acts on air rows", "accent1"],
           ["Hidden but recoverable", "Every hidden node is linked to a room's air by a positive conductance, so its temperature shows up in how that air changes. That is why it can be estimated from the measured history (part 6).", "observability", "accent3"]],
    callout="The plant's other state, the boiler water temperature, is observed and enters through q (reheat), computed from our own water-setpoint commands.",
    src="[audit] STATE_SPACE.md §13.1-13.2, §14.3; [our code] runtime/inputs.py (T_water from commands)")
insert_after("ssform", "ssvars", "sseqfull", "ssmatrix", "ssobs")

# ============================================================ part 6: no backpropagation, no fitting
new("nofitdef", "table", eyebrow="What 'nothing fitted' means, exactly", title="No backpropagation and no fitting, anywhere the model is made or used",
    rows=[["Where fitting usually happens", "What we do instead"],
          ["Before submission: tune weights or constants on the training data (backpropagation, least squares)", "There is no parameter vector to tune. A candidate with any fitted-constant table is rejected"],
          ["On the test building: estimate constants from its first day (online fitting)", "No optimiser runs inside the policy. Each step evaluates a fixed formula"],
          ["Gradients: differentiate an error and step the parameters", "No error is ever differentiated. The code imports only numpy, math and our small dlib"],
          ["Where the model's numbers come from", "Only three kinds: universal constants cited from the kit, counts from the floor plan, and expressions of the current measured history"],
          ["What changes from one candidate to the next", "Only the program text, written by the LLM"]],
    colw=[0.45, 0.55], size=23, font=11,
    callout="So the model is the same code with the same numbers in every building. A new building changes only its floor plan and its measurements.",
    src="[our code] contract.load_closedform, contract._exec (sandbox), closedform.evaluate; prompts.CF_CONTRACT")
new("nobpcompare", "compare", eyebrow="Backpropagation vs our search", title="Backprop adjusts numbers inside a fixed model; we choose between whole programs",
    left=("Backpropagation", ["Fix an architecture with free weights",
                              "Define a loss on training data",
                              "Differentiate the loss with respect to every weight",
                              "Step every weight downhill, thousands of times",
                              "The trained weights are the model"]),
    right=("Our search", ["No free weights: the LLM writes a complete program",
                          "Run it on held-out buildings; the score is one number used only to rank",
                          "No derivative of anything is taken",
                          "Keep the better programs, discard the rest, ask for new ones",
                          "The program text is the model; its numbers never change after it is written"]),
    callout="Learning happens, but over the structure of the formulas and by selection, not by gradient steps on numbers.",
    src="[our code] evolve.py (tournament, elites), closedform.evaluate (score only)")
new("histnotfit", "points", eyebrow="A fair question", title="Isn't the history U-value formula a fit in disguise?",
    points=["It is a fixed function of the last 288 measurements, re-evaluated every step, like a moving average: Û = (Σ x·r + 5·2.0) / (Σ x² + 5)",
            "The formula and its constants (2.0, 5, 0.3, 8) are the same in every building; nothing is chosen to reduce an error",
            "Nothing is stored or carried between steps, buildings or episodes: it has no memory of the training buildings",
            "Fitting = search for parameters that minimise an error, then keep them. This = compute a statistic of the data in front of you",
            "A formula like this is allowed precisely because it uses only what the policy measures at test time"],
    src="[our code] seeds_cf/seed_cf_history_slab.py facade_u; prompts.CF_CONTRACT ('the only estimation allowed is a closed-form expression of the history buffer')")
new("honestfit", "cards", eyebrow="Being exact about it", title="Where choices still enter, and what keeps them honest",
    cards=[["Hand-chosen constants", "The seeds' constants (U₀ = 2.0 W/m²K, shrink 5, slab coupling 4 W/m²K, 6 h averaging; mid-range materials) were set once by reasoning, never optimised.", "seeds", "accent3"],
           ["Constants the LLM writes", "The LLM may add physically sensible constants. Selection could favour lucky values indirectly; guarded by the size price, held-out buildings and a separate test station.", "the search", "accent1"],
           ["The one fitted thing", "The 'ceiling': a per-room model fitted by non-negative least squares on half of each episode. A diagnostic only; never deployed or used in selection.", "diagnostic", "accent2"]],
    callout="The fitted modes we tried first (part 5) remain in the code only for comparison; closed-form discovery, with nothing fitted, is the default.",
    src="[our code] seeds_cf/*.py, closedform.ceiling (scipy nnls, diagnostic), discovery/loop.py --deployment offline|online")
new("enforce", "table", eyebrow="Enforced in code, not by promise", title="How 'no fitting' is checked automatically",
    rows=[["Check", "Where", "Effect"],
          ["No fitted-constant tables", "contract.load_closedform", "a candidate with non-empty PARAMS or THETA is rejected before it runs"],
          ["Sandbox imports", "contract._exec", "only numpy, math and dlib can be imported: no optimisers, no autodiff libraries, no files"],
          ["Scorer fits nothing", "closedform.evaluate", "only rollouts and error statistics; the single optimiser call in the module is the diagnostic ceiling"],
          ["No special-casing of buildings", "contract._reject_building_specific", "code comparing a building's size or geometry to a fixed number is rejected"],
          ["Same path live", "runtime/closedform_tracker.py", "the policy only calls prepare, step (or init_state, predict, correct): no estimation step exists"]],
    colw=[0.24, 0.3, 0.46], size=22, font=11,
    src="[our code] discovery/contract.py, closedform.py, runtime/closedform_tracker.py; tests/test_deployment.py")
insert_after("threeways", "nofitdef", "nobpcompare")
insert_after("histformula", "histnotfit")
insert_after("observer", "honestfit", "enforce")
B["decmap"]["rows"][4][1] = "state-space form · no backpropagation, no fitting (enforced in code) · universal constants · history formulas"
B["roadmap4"]["steps"][3] = ("4 State space", "state, control, disturbance, output; every equation")
