#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    19/12/2019
""" 
import os

from diversity_algorithms.environments.behavior_descriptors import *

from diversity_algorithms.environments import gym_env, dummy_env, brax_env
from diversity_algorithms.environments import wrappers
from brax.v1 import envs

registered_environments = dict()

registered_environments["Fastsim-LS2011"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{}}, # Default
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-LS2011-EnergyFitness"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"reward_func":"minimize_energy"},
		"output":"total_reward"},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-Pugh2015"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/pugh_maze.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-16x16"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/realhard_maze.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-12x12"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/maze_12x12.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-8x8"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/maze_8x8.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["BipedalWalker"] = {
	"bd_func": bipedal_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"BipedalWalker-v3",
		"gym_params":{}}, # Default
	"grid_features": {
		"min_x": [-600,-600],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}


registered_environments["DummyMapping3D"] = {
	"eval": dummy_env.SimpleMappingEvaluator,
	"eval_params": {
		"geno_size":3,
		"mapping":"fitness_last"}, # Default
	"grid_features": {
		"min_x": [-5,-5],
		"max_x": [5, 5],
		"nb_bin": 50
	}
}

registered_environments["Billiard"] = {
	"bd_func": billiard_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"Billiard-v0",
		"gym_params":{},
		"output":"final_reward"}, # Default
	"grid_features": {
		"min_x": [-1.35,-1.35],
		"max_x": [1.35, 1.35],
		"nb_bin": 50
	}
}


registered_environments["ant-uni"] = {
	"bd_func": ant_behavior_descriptor,
	"eval": brax_env.EvaluationFunctor,
	"wrapper": wrappers.FeetContactWrapper,
	"eval_params": {
		"env_name":"ant",
		"output":"final_reward"}, # Default
	"grid_features": {
		"min_x": [0, 0, 0, 0],
		"max_x": [1, 1, 1, 1],
		"nb_bin": 50
	},
	"kwargs": [{}, {}]
}


def create(env_name,
           episode_length = 100,
           action_repeat = 1,
           auto_reset = True,
           **kwargs):
	"""Creates an Env with a specified brax system."""
	name = registered_environments[env_name]["eval_params"]["env_name"]
	env = envs._envs[name](**kwargs)
	if "wrapper" in registered_environments[env_name]:
		
		env = registered_environments[env_name]["wrapper"](env, name)
	if episode_length is not None:
		env = envs.wrappers.EpisodeWrapper(env, episode_length, action_repeat)
	if auto_reset:
		env = envs.wrappers.AutoResetWrapper(env)

	return env  # type: ignore