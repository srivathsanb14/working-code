"""A policy with the competition interface that carries the state-space model.

    class X(Policy):
        def __init__(self, building_meta_data, action_spec): ...
        def get_action(self, obs): ... -> {"continuous_action", "discrete_action"}

This is what the Kaggle notebook submits.  Today the CONTROL decisions are delegated to a base
policy (locally the starter kit's pid_water; in the notebook a copy of it), and the model only
tracks the building: it is built from the metadata in __init__, and every get_action feeds it the
observation and the action returned, so `self.model.state` is the filtered state a model-based
controller would plan from.  Replacing the delegation with decisions computed from that state is
Controller 1 and 2 (STATE_SPACE.md section 18.1).  Because the actions are the base policy's, the
return must equal the base policy's return exactly: a built-in check that tracking changes nothing.
"""
from __future__ import annotations

import time

from runtime.online_model import OnlineBuildingModel


def make_model_policy(base_policy_class, model, params, theta, name="model_tracking"):
    """Return a policy CLASS (the harness instantiates it per building) closed over the learned constants."""

    class ModelPolicy:
        instances = []                                   # lets a local runner read the model after an episode

        def __init__(self, building_meta_data, action_spec):
            t0 = time.time()
            self.base = base_policy_class(building_meta_data, action_spec)
            self.model = OnlineBuildingModel(building_meta_data, model, params, theta)
            self.init_seconds = time.time() - t0
            self.step_seconds = []
            ModelPolicy.instances.append(self)

        def name(self):
            return name

        def get_action(self, obs):
            action = self.base.get_action(obs)
            t0 = time.time()
            self.model.observe(obs, action)
            self.step_seconds.append(time.time() - t0)
            return action

    return ModelPolicy


def make_closedform_policy(base_policy_class, src, name="closedform_tracking"):
    """Same as make_model_policy, for a closed-form candidate (discovery/closedform.py contract): nothing fitted."""
    from runtime.closedform_tracker import ClosedFormTracker

    class ClosedFormPolicy:
        instances = []

        def __init__(self, building_meta_data, action_spec):
            t0 = time.time()
            self.base = base_policy_class(building_meta_data, action_spec)
            self.model = ClosedFormTracker(building_meta_data, src)
            self.init_seconds = time.time() - t0
            self.step_seconds = []
            ClosedFormPolicy.instances.append(self)

        def name(self):
            return name

        def get_action(self, obs):
            action = self.base.get_action(obs)
            t0 = time.time()
            self.model.observe(obs, action)
            self.model.predict_next()                    # the forecast a controller would use, timed with the rest
            self.step_seconds.append(time.time() - t0)
            return action

    return ClosedFormPolicy
