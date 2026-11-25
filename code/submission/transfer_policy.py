"""The policy core the Kaggle notebook wraps.

    core = TransferPolicyCore(building_meta_data, action_spec, model_src=None)
    core.get_action(obs) -> {"continuous_action": float32 (3K+2+2V,), "discrete_action": int32 (K+1,)}

Control comes from `controller` (default: the kit's pid_water baseline, vendored; RulePolicy is a simpler option).  If `model_src` (a closed-form best.py) is given, a
ClosedFormTracker follows the building from the metadata, observations and own actions; a model-based
controller plans from it (not built yet).  The tracker can never break control: its errors are caught.
Every action passes the safety layer.
"""
from __future__ import annotations

import time

from submission.layout import Layout
from submission.rule_policy import RulePolicy
from submission.safety import SafeAction
from submission.vendor.pid_plus import PIDWaterResetPolicy     # the kit's pid_water baseline (vendored, Apache-2.0)


class TransferPolicyCore:
    def __init__(self, building_meta_data, action_spec=None, model_src=None, controller_cls=PIDWaterResetPolicy):
        t0 = time.time()
        self.L = Layout(building_meta_data)
        try:
            self.controller = controller_cls(building_meta_data, action_spec)
        except Exception:  # noqa: BLE001  # a model-based controller that cannot start falls back to pid_water
            self.controller = PIDWaterResetPolicy(building_meta_data, action_spec)
        # safe default: mid-range setpoints, every AHU and the boiler on (ventilation floors apply anyway)
        self.safe = SafeAction(self.L.n_cont, self.L.n_disc)
        try:
            c, d = RulePolicy(building_meta_data).native_action(_zero_obs(building_meta_data))
            self.safe.default_c = self.L.to_normalized(c).astype("float32"); self.safe.default_d = d
        except Exception:  # noqa: BLE001
            pass
        self.tracker = None
        self.tracker_errors = 0
        if model_src:
            try:
                from runtime.closedform_tracker import ClosedFormTracker
                self.tracker = ClosedFormTracker(building_meta_data, model_src)
            except Exception:  # noqa: BLE001
                self.tracker = None; self.tracker_errors += 1
        self.init_seconds = time.time() - t0

    def name(self):
        return "transfer_policy_core"

    def get_action(self, obs):
        action = self.safe.call(self.controller.get_action, obs)
        if self.tracker is not None:
            try:
                self.tracker.observe(obs, action)
            except Exception:  # noqa: BLE001
                self.tracker_errors += 1
                if self.tracker_errors > 10:
                    self.tracker = None
        return action


def _zero_obs(meta):
    import numpy as np
    return np.zeros(len(meta["observation_field_names"]))
