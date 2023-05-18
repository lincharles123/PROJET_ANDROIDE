import jax.numpy as jnp
from brax.v1 import envs
from brax.v1.physics.system import System

# citer QDAX, licence

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
    
