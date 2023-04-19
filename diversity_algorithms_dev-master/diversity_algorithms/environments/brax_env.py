
# coding: utf-8

# pyMaze expriments

from brax.envs import wrapper
from brax import envs
from brax.v1 import jumpy as jp
import jax
from functools import partial


# Fitness/evaluation function

default_max_step = 2000 # same as C++ sferes experiments

# change vmapwrapper reset to use jax.vmap on already generated random keys
class custom_vmap(wrapper.VmapWrapper):
    def reset(self, rng):
        return jp.vmap(self.env.reset)(rng)
wrapper.VmapWrapper = custom_vmap

class EvaluationFunctor:
	def __init__(self, gym_env_name=None, gym_params={}, controller=None, controller_type=None, controller_params=None, output='total_reward',max_step=default_max_step, bd_function=None):
		global current_serial
		#print("Eval functor created")
		#Env
		#Controller
		self.key = jax.random.PRNGKey(0) # key used for random state reset
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
		self.env = envs.create(env_name, batch_size=1, auto_reset=False, **gym_params)
		self.env_name = env_name
		print("Environment set to", self.env_name)

		if(self.controller is None): # Build controller
			if(self.controller_type is None):
				raise RuntimeError("Please either give a controller or specify controller type")
			self.controller = self.controller_type(self.env.observation_size,self.env.action_size, params=self.controller_params)

		else:
			if(self.controller_type is not None or self.controller_params is not None):
				print("WARNING: EvaluationFunctor built with both controller and controller_type/controller_params. controller_type/controller_params arguments  will be ignored")

		# jit functions
		self.jit_env_reset = jax.jit(self.env.reset)
		self.jit_step = jax.jit(self.env.step)
		self.jit_model = jax.jit(self.controller.predict)
		self.jit_array_to_fdict = jax.jit(self.controller.array_to_fdict)


	@partial(jax.jit, static_argnums=(0,))
	def eval(self, init_states, params):
		def eval_step(carry, _):
			states = carry
			next_states = self.jit_step(states, jp.vmap(self.jit_model)(params, states.obs))
			return (next_states), (next_states.reward, next_states.metrics)
		return jp.scan(eval_step, init_states, None, length=self.max_step)

	def __call__(self, genotypes):
		#print("Eval functor CALL")
		# Load genotype
		#print("Load gen")
		if(type(genotypes)==tuple):
			gen, ngeneration, idx = jp.array(genotypes)
			# print("Start main eval loop -- #%d evals for this functor so far" % self.evals)
			# print("Evaluating indiv %d of gen %d" % (idx, ngeneration))
			# print('Eval thread: memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		else:
			gen = jp.array(genotypes)

		params = jp.vmap(self.jit_array_to_fdict)(gen) 		# convert array type gen to fdict

		keys = jp.random_split(self.key, gen.shape[0]) 		# generate random keys for each individual
		self.key = jp.random_split(keys[-1], 1)[0] 			# update key for next call
		init_states = self.jit_env_reset(keys) 				# get different initial states for each individual

		states, (rewards, metrics) = self.eval(init_states, params) # evaluate
		for key in metrics:
			metrics[key] = metrics[key].T				# transpose metrics to have shape (n_indiv, n_metrics)	

		# Select fitness
		outdata = str(self.out)
		# Detect minus sign
		if(outdata[0] == '-'):
			outdata = outdata[1:]
			sign = -1
		else:
			sign = 1
		
		if(outdata=='total_reward'):
			fitness = jp.sum(rewards, axis=0)
		elif(outdata=='final_reward'):
			fitness = states.reward
		elif(outdata==None or self.out=='none'):
			fitness = [[None] for _ in range(gen.shape[0])]
		elif(outdata in states.metrics):
			fitness = [states.metrics[outdata]]
		else:
			print("ERROR: No known output %s" % outdata)
			return None
		
		# Change sign if needed
		fitness = jax.lax.map(lambda x:sign*x, fitness).tolist()
		if type(fitness[0]) is not list:
			fitness = [[x] for x in fitness]

		if self.get_behavior_descriptor is None:
			return fitness
		else:
			res = jax.lax.map(self.get_behavior_descriptor, metrics)
			bd = jp.stack((res[0], res[1]), axis=1).tolist()
			return [(f, r) for f, r in zip(fitness, bd)]


