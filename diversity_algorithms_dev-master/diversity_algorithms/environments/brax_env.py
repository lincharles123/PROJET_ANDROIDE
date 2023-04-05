
# coding: utf-8

# pyMaze expriments

from brax import envs
from brax import jumpy as jnp
import jax
from functools import partial
import time

#import resource

from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis.data_utils import listify
from brax.training.networks import MLP

# Fitness/evaluation function

default_max_step = 2000 # same as C++ sferes experiments
    

class EvaluationFunctor:
	def __init__(self, gym_env_name=None, gym_params={}, controller=None, controller_type=None, controller_params=None, output='distance_from_origin',max_step=default_max_step, bd_function=None):
		global current_serial
		#print("Eval functor created")
		#Env
		#Controller
		self.key = jax.random.PRNGKey(0)
		self.out = output
		self.max_step = max_step
		self.controller=controller
		self.controller_type=controller_type
		self.controller_params=controller_params
		if (gym_env_name is not None):
			self.set_env(gym_env_name, gym_params)
		else:
			self.env = None
		self.get_behavior_descriptor = bd_function


	def set_env(self, env_name, gym_params):
		### Modified
		self.env = envs.create(env_name, **gym_params)
		key, self.key = jax.random.split(self.key)
		self.jit_env_reset = jax.jit(self.env.reset)
		self.jit_step = jax.jit(self.env.step)
		self.jit_env_reset(rng=key)
		self.env_name = env_name
		print("Environment set to", self.env_name)

		if(self.controller is None): # Build controller
			if(self.controller_type is None):
				raise RuntimeError("Please either give a controller or specify controller type")
			self.controller = self.controller_type(self.env.observation_size,self.env.action_size, params=self.controller_params)
			self.jit_model = jax.jit(self.controller.predict)
			self.jit_array_to_fdict = jax.jit(self.controller.array_to_fdict)
		else:
			if(self.controller_type is not None or self.controller_params is not None):
				print("WARNING: EvaluationFunctor built with both controller and controller_type/controller_params. controller_type/controller_params arguments  will be ignored")

	@partial(jax.jit, static_argnums=(0,))
	def evaluate_indiv(self, state, params):
		def eval_step(carry, _):
			state, = carry
			state = self.jit_step(state, self.jit_model(params, state.obs))
			return (state,), (state.reward, state.metrics)

		(state,), (rewards, traj) = jax.lax.scan(eval_step, (state,), (), length=self.max_step)
		return state, rewards, traj

	def __call__(self, genotype):
		#print("Eval functor CALL")
		# Load genotype
		#print("Load gen")
		if(type(genotype)==tuple):
			gen, ngeneration, idx = jnp.array(genotype)
			# print("Start main eval loop -- #%d evals for this functor so far" % self.evals)
			# print("Evaluating indiv %d of gen %d" % (idx, ngeneration))
			# print('Eval thread: memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		else:
			gen = jnp.array(genotype)
		key, self.key = jax.random.split(self.key)
		state = self.jit_env_reset(key)
		params = self.jit_array_to_fdict(gen)
		start = time.time()
		state, rewards, traj = self.evaluate_indiv(state, params)
		print("Eval time: %f" % (time.time()-start))
		# Select fitness
		outdata = str(self.out)
		# Detect minus sign
		if(outdata[0] == '-'):
			outdata = outdata[1:]
			sign = -1
		else:
			sign = 1
		
		if(outdata=='total_reward'):
			fitness = [jnp.sum(rewards)]
		elif(outdata=='final_reward'):
			fitness = [rewards[-1]]
		elif(outdata==None or self.out=='none'):
			fitness = [None]
		elif(outdata in state.metrics):
			fitness = [state.metrics[outdata]]
		else:
			print("ERROR: No known output %s" % outdata)
			return None
		
		# Change sign if needed
		fitness = list(map(lambda x:sign*x, fitness))
		
		if self.get_behavior_descriptor is None:
			return fitness
		else:
			bd = self.get_behavior_descriptor(traj)
			return [fitness,[bd]]


