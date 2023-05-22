#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    19/12/2019
""" 
import os
from diversity_algorithms.environments.behavior_descriptors import *
from diversity_algorithms.environments import brax_env

registered_environments = dict()


registered_environments["ant-uni"] = {
	"bd_func": feet_contact_descriptor,
	"eval": brax_env.EvaluationFunctor,
	"eval_params": {
		"env_name":"ant-uni",
		"output":"reward_forward+reward_survive+torque"}, # Default
	"grid_features": {
		"min_x": [0, 0, 0, 0],
		"max_x": [1, 1, 1, 1],
		"nb_bin": 10
	},
	"kwargs": [{}, {}]
}

registered_environments["ant-omni"] = {
	"bd_func": final_pos_descriptor,
	"eval": brax_env.EvaluationFunctor,
	"eval_params": {
		"env_name":"ant-omni",
		"output":"reward_survive+torque"}, # Default
	"grid_features": {
		"min_x": [-15, -15],
		"max_x": [15, 15],
		"nb_bin": 100,
	},
	"kwargs": [{}, {}]
}
