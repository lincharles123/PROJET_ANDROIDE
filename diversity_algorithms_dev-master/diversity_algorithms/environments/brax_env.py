
# coding: utf-8

# pyMaze expriments

from brax import envs
from brax import jumpy as jp
import numpy as np
import jax

#import resource

from diversity_algorithms.controllers import SimpleNeuralController #, DNNController
from diversity_algorithms.analysis.data_utils import listify


# Fitness/evaluation function

default_max_step = 2000 # same as C++ sferes experiments
    

class EvaluationFunctor:
	def __init__(self, gym_env_name=None, gym_params={}, controller=None, controller_type=None, controller_params=None, output='-dist_obj',max_step=default_max_step, bd_function=None):
		global current_serial
		#print("Eval functor created")
		#Env
		#Controller
		self.out = output
		self.max_step = max_step
		self.evals = 0
		self.traj=None
		self.controller=controller
		self.controller_type=controller_type
		self.controller_params=controller_params
		self.key = jp.random_prngkey(0)
		if (gym_env_name is not None):
			self.set_env(gym_env_name, gym_params)
		else:
			self.env = None
		self.get_behavior_descriptor = bd_function
		




	def set_env(self, env_name, gym_params):
		### Modified
		self.env = envs.create(env_name, **gym_params)
		self.env.reset(rng=self.key)
		self.env_name = env_name
		print("Environment set to", self.env_name)

		if(self.controller is None): # Build controller
			if(self.controller_type is None):
				raise RuntimeError("Please either give a controller or specify controller type")
			self.controller = self.controller_type(self.env.observation_size,self.env.action_size, params=self.controller_params)
		else:
			if(self.controller_type is not None or self.controller_params is not None):
				print("WARNING: EvaluationFunctor built with both controller and controller_type/controller_params. controller_type/controller_params arguments  will be ignored")



	def load_indiv(self, genotype):
		if(self.controller is None):
			print("ERROR: controller is None")
		self.controller.set_parameters(genotype)


	def evaluate_indiv(self):
		"""
		Evaluate individual genotype (list of controller.n_weights floats) in environment env using
		given controller and max step number, and returns the required output:
		- dist_to_goal: final distance to goal (list of 1 scalar)
		- bd_finalpos: final robot position and orientation (list [x,y,theta])
		- total_reward: cumulated reward on episode (list of 1 scalar)
		"""
		# Inits
		print("env reset")
		self.evals += 1
		self.traj=[]
		state = self.env.reset(self.key)
		cumulative_reward = 0.

		def run_step(carry, unused_target_t):
			(env_state, cumulative_reward) = carry
			obs = env_state.obs
			actions = self.controller(obs)
			nstate = self.env.step(env_state, actions)
			cumulative_reward = cumulative_reward + nstate.reward
			return (nstate, cumulative_reward), (env_state.obs)

		state, _ = jax.lax.scan(run_step, (state, cumulative_reward), (),length=self.max_step)
		return state.reward, state.done, self.total_reward, state.info

        
	def __call__(self, genotype):
		#print("Eval functor CALL")
		# Load genotype
		#print("Load gen")
		if(type(genotype)==tuple):
			gen, ngeneration, idx = genotype
#			print("Start main eval loop -- #%d evals for this functor so far" % self.evals)
#			print("Evaluating indiv %d of gen %d" % (idx, ngeneration))
#			print('Eval thread: memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		else:
			gen = genotype
		self.load_indiv(gen)
		# Run eval genotype
		#print("Start eval")
		final_reward, end, total_reward, info = self.evaluate_indiv()
		#print("Eval done !")
		# Select fitness
		
		outdata = str(self.out)
		# Detect minus sign
		if(outdata[0] == '-'):
			outdata = outdata[1:]
			sign = -1
		else:
			sign = 1
		
		if(outdata=='total_reward'):
			fitness = [total_reward]
		elif(outdata=='final_reward'):
			fitness = [final_reward]
		elif(outdata==None or self.out=='none'):
			fitness = [None]
		elif(outdata in info):
			fitness = listify(info[outdata])
		else:
			print("ERROR: No known output %s" % outdata)
			return None
		
		# Change sign if needed
		fitness = list(map(lambda x:sign*x, fitness))
		
		if self.get_behavior_descriptor is None:
			self.traj=None # to avoid taking too much memory
			return fitness
		else:
			bd = self.get_behavior_descriptor(self.traj)
			self.traj=None # to avoid taking too much memory
			return [fitness,bd,info]


