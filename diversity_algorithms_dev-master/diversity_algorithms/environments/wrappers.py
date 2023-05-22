import jax.numpy as jnp
from brax.v1 import envs
from brax.v1.physics.system import System

# Written according to QDax implementation:
# https://github.com/adaptive-intelligent-robotics/QDax/blob/main/qdax/environments/locomotion_wrappers.py
# wrapper to get the feet contact with the ground for behavior descriptor

# MIT License

# Copyright (c) 2022 Adaptive and Intelligent Robotics Lab and InstaDeep Ltd

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


FEET_NAMES = {
    "ant": ["$ Body 4", "$ Body 7", "$ Body 10", "$ Body 13"],
    "halfcheetah": ["ffoot", "bfoot"],
    "walker2d": ["foot", "foot_left"],
    "hopper": ["foot"],
    "humanoid": ["left_shin", "right_shin"],
}


class QDSystem(System):
    """Inheritance of brax physic system.

    Work precisely the same but store some information from the physical
    simulation in the aux_info attribute.

    This is used in FeetContactWrapper to get the feet contact of the
    robot with the ground.
    """

    def __init__(
        self, config, resource_paths = None
    ):
        super().__init__(config, resource_paths=resource_paths)
        self.aux_info = None

    def step(self, qp, act) :
        qp, info = super().step(qp, act)
        self.aux_info = info
        return qp, info
    

class FeetContactWrapper(envs.Wrapper):
    def __init__(self, env, env_name):
        if env_name not in FEET_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        
        super().__init__(env)
        self._env_name = env_name
        
        if hasattr(self.env, "sys"):
            self.env.sys = QDSystem(self.env.sys.config)
            
        self._feet_contact_idx = jnp.array(
            [env.sys.body.index.get(name) for name in FEET_NAMES[env_name]]
        )
            
    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["feet_contact"] = self._get_feet_contact(
            self.env.sys.info(state.qp)
        )
        return state
    
    def step(self, state, action):
        state = self.env.step(state, action)
        state.info["feet_contact"] = self._get_feet_contact(self.env.sys.aux_info)
        return state

    def _get_feet_contact(self, info):
        contacts = info.contact.vel
        return jnp.any(contacts[self._feet_contact_idx], axis=1).astype(jnp.float32)
    

