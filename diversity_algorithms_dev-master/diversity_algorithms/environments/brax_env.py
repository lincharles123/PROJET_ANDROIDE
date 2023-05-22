# coding: utf-8

from jax import numpy as jp
import jax
from functools import partial
from diversity_algorithms.environments import wrappers
from brax.v1 import envs

def create(env_name,
           episode_length = 100,
           action_repeat = 1,
           auto_reset = True,
           **kwargs):
	"""Creates an Env with a specified brax system."""
	name = env_name.split('-')[0]
	env = envs._envs[name]()
	env = wrappers.FeetContactWrapper(env, name)
	if episode_length is not None:
		env = envs.wrappers.EpisodeWrapper(env, episode_length, action_repeat)
	if auto_reset:
		env = envs.wrappers.AutoResetWrapper(env)

	return env  # type: ignore
# Fitness/evaluation function

class EvaluationFunctor:
	def __init__(self, env_name=None, kwargs={}, controller=None, controller_type=None, controller_params=None,
				 output='total_reward', episode_length=100, bd_function=None):

		self.out = output
		self.episode_length = episode_length
		self.controller = controller
		self.controller_type = controller_type
		self.controller_params = controller_params
		if (env_name is not None):
			self.set_env(env_name, kwargs)
		else:
			self.env = None
		self.get_behavior_descriptor = bd_function

	def set_env(self, env_name, kwargs):
		self.env = create(env_name, episode_length=self.episode_length, kwargs=kwargs)
		self.env_name = env_name
		print("Environment set to", self.env_name)

		if self.controller is None:  # Build controller
			if (self.controller_type is None):
				raise RuntimeError("Please either give a controller or specify controller type")
			self.controller = self.controller_type(self.env.observation_size, self.env.action_size,
												   params=self.controller_params)

		else:
			if self.controller_type is not None or self.controller_params is not None:
				print(
					"WARNING: EvaluationFunctor built with both controller and controller_type/controller_params. controller_type/controller_params arguments  will be ignored")

		# jit functions
		self.step_fn = jax.vmap(jax.jit(self.env.step))
		self.inference_fn = jax.vmap(jax.jit(self.controller.predict))
		self.reset_fn = jax.vmap(jax.jit(self.env.reset))
		self.convert_fn = jax.vmap(jax.jit(self.controller.array_to_dict))

	def get_controller(self):
		return self.controller

	@partial(jax.jit, static_argnums=(0,))
	def eval(self, init_states, params):
		def eval_step(carry, _):
			states = carry
			actions = self.inference_fn(params, states.obs)
			next_states = self.step_fn(states, actions)
			return (next_states), (next_states.reward, next_states.metrics, next_states.info)
		return jax.lax.scan(eval_step, init_states, None, length=self.episode_length)

	def __call__(self, genotypes, random_key):
		# print("Eval functor CALL")
		# Load genotype
		# print("Load gen")
		if (type(genotypes) == tuple):
			gens, ngeneration, idx = jp.array(genotypes)
		# print("Start main eval loop -- #%d evals for this functor so far" % self.evals)
		# print("Evaluating indiv %d of gen %d" % (idx, ngeneration))
		# print('Eval thread: memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		else:
			gens = genotypes

		random_key, subkey = jax.random.split(random_key)
		params = self.convert_fn(gens)  # convert array type gen to fdict
		keys = jax.random.split(subkey, gens.shape[0])  # generate random keys for each individual
		init_states = self.reset_fn(keys) # get different initial states for each individual

		states, (rewards, metrics, info) = self.eval(init_states, params)  # evaluate

		# Reshape to have shape (n_indiv, _)
		metrics = jax.tree_map(lambda x: x.swapaxes(1, 0), metrics)
		info = jax.tree_map(lambda x: x.swapaxes(1, 0), info)
		rewards = rewards.swapaxes(1, 0)

		# Select fitness
		outdata = str(self.out)
		# Detect minus sign
		if (outdata[0] == '-'):
			outdata = outdata[1:]
			sign = -1
		else:
			sign = 1

		if (outdata == 'total_reward'):
			fitness = jp.sum(rewards, axis=1)
		elif (outdata == 'final_reward'):
			fitness = states.reward
		elif (outdata == None or self.out == 'none'):
			fitness = [[None] for _ in range(gens.shape[0])]
		elif (outdata in states.metrics):
			fitness = jp.sum(metrics[outdata], axis=1)
		else:
			print("ERROR: No known output %s" % outdata)
			return None

		# Change sign if needed
		fitness = jax.lax.map(lambda x: sign * x, fitness)
		fitness = [[f] for f in fitness.tolist()]

		if self.get_behavior_descriptor is None:
			raise RuntimeError("No behavior descriptor function defined")
		else:
			bd = jax.lax.map(self.get_behavior_descriptor, info)
			return fitness, bd, random_key
