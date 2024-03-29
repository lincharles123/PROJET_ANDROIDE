#!/usr/bin python -w

import random
from scipy.spatial import KDTree
import numpy as np
import datetime
import os
import time

creator = None
def set_creator(cr):
	global creator
	creator = cr

import pickle

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *

from diversity_algorithms.algorithms.novelty_management import *

import alphashape
from shapely.geometry import Point, Polygon, LineString

import jax
from jax import numpy as jnp
from diversity_algorithms.algorithms.jax_utils import *

__all__=["novelty_ea"]

import sys

def dist_to_shape(pp, s):
	p=Point(pp)
	d=p.distance(s)
	if (d==0.0):
		d=-p.distance(s.exterior)
	return d

def dist_to_shapes(pp, ls):
	if (not hasattr(ls, '__iter__')):
		ls=[ls] 
	p=Point(pp)
	imin=-1
	dmin=sys.float_info.max
	for i in range(len(ls)):
		d=p.distance(ls[i])
		if (d<dmin):
			imin=i
			dmin=d
	if (dmin==0.0):
		d=-p.distance(ls[i].exterior)
	else:
		d=dmin
	return d



def build_toolbox_ns(evaluate, params):
	toolbox = base.Toolbox()

	if(params["geno_type"] == "realarray"):
		print("** Using fixed structure networks (MLP) parameterized by a real array **")
		# With fixed NN
		# -------------
		# toolbox.register("population", init_pop_numpy, params=params)
		toolbox.register("population", init_pop_controller, controller=evaluate.get_controller())
	
		# Polynomial mutation with eta=15, and p=0.1 as for Leni
		toolbox.register("mutate", mutate, eta=params["eta_m"], min_val=params["min"], max_val=params["max"], indpb=params["indpb"])
		toolbox.register("mate", cxBLend, alpha=0.2)
	else:
		raise RuntimeError("Unknown genotype type %s" % geno_type)

	#Common elements - selection and evaluation
	
	v=str(params["variant"])
	variant=v.replace(",","")
	if (variant == "NS"): 
		toolbox.register("select", selBest, fit_attr='novelty')
	elif (variant == "Fit"):
		toolbox.register("select", selBest, fit_attr='fitness')
	elif (variant == "Random"):
		toolbox.register("select", random.sample)
	# elif (variant == "DistExplArea"):
	# 	toolbox.register("select", selBest, fit_attr='dist_to_explored_area')
	else:
		print("Variant not among the authorized variants (NS, Fit, Random, DistExplArea), assuming multi-objective variant")
		toolbox.register("select", tools.selNSGA2)
		
	toolbox.register("map_eval", evaluate)
	return toolbox

## DEAP compatible algorithm
def novelty_ea(evaluate, params, random_key):
	"""Novelty Search algorithm

	Novelty Search algorithm. Parameters:
	:param evaluate: the evaluation function
	:param params: the dict of run parameters
	   * params["pop_size"]: the number of parent individuals to keep from one generation to another
	   * params["lambda"]: the number of offspring to generate, given as a multiplying coefficent on params["pop_size"] (the number of generated individuals is: params["lambda"]*params["pop_size"]
	   * params["nb_gen"]: the number of generations to compute
	   * params["stats"]: the statistics to use (from DEAP framework))
	   * params["stats_offspring"]: the statistics to use (from DEAP framework))
	   * params["variant"]: the different supported variants ("NS", "Fit", "NS+Fit", "NS+BDDistP", "NS+Fit+BDDistP"), add "," at the end of the variant to select in offspring only (as the ES "," variant). By default, selects within the parents and offspring. "NS" uses the novelty criterion, "Fit" the fitness criterion and "BDDistP' the distance to the parent in the behavior space. If a single criterion is used, an elitist selection scheme is used. If more than one criterion is used, NSGA-II is used (see build_toolbox_ns function) 
	   * params["cxpb"]: the crossover rate
	   * params["mutpb"]: the mutation rate
	:param dump_period_bd: the period for dumping behavior descriptors
	:param dump_period_pop: the period for dumping the current population
	:param evolvability_period: period of the evolvability computation
	:param evolvability_nb_samples: the number of samples to generate from each individual in the population to estimate their evolvability (WARNING: it will significantly slow down a run and it is used only for statistical reasons
	"""
	print("Novelty search algorithm")

	# alphas=params["alphas"] # parameter to compute the alpha shape, to estimate the distance to explored area
	params["lambda_nov"] = int(0.06*params["pop_size"]) # number of individuals to add to the archive at each generation
	variant=params["variant"]
	if ("+" in variant):
		emo=True
	else:
		emo=False

	nb_eval=0

	lambda_ = int(params["lambda"]*params["pop_size"])

	toolbox = build_toolbox_ns(evaluate,params)

	population, random_key = toolbox.population(random_key, params["pop_size"])
	
	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals']

	if (params["stats"] is not None):
		logbook.header += params["stats"].fields
	if (params["stats_offspring"] is not None):
		logbook.header += params["stats_offspring"].fields

	# Evaluate the individuals
	fit, bd, random_key = toolbox.map_eval(jnp.asarray(population), random_key)
	nb_eval += bd.shape[0]
	
	for ind, f, b in zip(population, fit, bd):
		ind.fit = f
		ind.parent_bd = None
		ind.bd = b
		ind.id = generate_uuid()
		ind.parent_id = None

	for ind in population:
		ind.am_parent=0
 
	# Update the archive
	archive=updateNovelty(population,population,None,params)
 
	bd_dimension = bd.shape[-1]
 
	# alpha_shape = None
	# # Only 2D and 3D BD are supported
	# if bd_dimension == 2 or bd_dimension == 3:
	# 	alpha_shape = alphashape.alphashape(archive.all_bd, alphas)
  
	# Compute the novelty rank
	nov = [ind.novelty for ind in population]
	isortednov = np.argsort(nov)[::-1]
	rank = np.empty_like(isortednov)
	rank[isortednov] = np.arange(len(isortednov))
	
	varian=params["variant"].replace(",","")

	for i,ind in enumerate(population):
		# if alpha_shape:
		# 	ind.dist_to_explored_area=dist_to_shapes(ind.bd,alpha_shape)
		ind.rank_novelty = rank[i]
		ind.dist_to_parent=0
		if (emo): 
			if (varian == "NS+Fit"):
				ind.fitness.values = (ind.novelty, ind.fit)
			elif (varian == "NS+BDDistP"):
				ind.fitness.values = (ind.novelty, 0)
			elif (varian == "NS+Fit+BDDistP"):
				ind.fitness.values = (ind.novelty, ind.fit, 0)
			else:
				print("WARNING: unknown variant: "+variant)
				ind.fitness.values=ind.fit
		else:
			ind.fitness.values=ind.fit
		# if it is not a multi-objective experiment, the select tool from DEAP 
		# has been configured above to take the right attribute into account
		# and the fitness.values is thus ignored
	gen=0

	# record = params["stats"].compile(population) if params["stats"] is not None else {}
	# record_offspring = params["stats_offspring"].compile(population) if params["stats_offspring"] is not None else {}
	# logbook.record(gen=0, nevals=len(invalid_ind), **record, **record_offspring)
	# if (verbosity(params)):
	#	 print(logbook.stream)
	
	#generate_dumps(params, population, None, gen, pop1label="population", archive=None, logbook=None)

	# for ind in population:
	# 	ind.evolvability_samples=None # To avoid memory to inflate too much..
		
	# Begin the generational process
	for gen in range(1, params["nb_gen"] + 1):
		print("Generation %d" % gen)
		if (gen==params["restart"]):
			print("Restart: we reinitialize the population")
			offspring, random_key = toolbox.population(random_key, params["pop_size"])
			
			for ind in offspring:
				ind.bd=None
				ind.id = generate_uuid()
				ind.parent_id = None
    
			population=[]
		else:
			# Vary the population
			t = time.time()
			offspring, random_key = varOr(random_key, population, toolbox, lambda_, params["cxpb"], params["mutpb"])
			print("Mutation time: ", time.time() - t)
   
		# Evaluate the individuals with an invalid fitness
		t = time.time()
		fit, bd, random_key = toolbox.map_eval(jnp.asarray(offspring), random_key)
		print("Evaluation time: ", time.time() - t)
		nb_eval += bd.shape[0]

		for ind, f, b in zip(offspring, fit, bd):
			ind.fit = f
			ind.fitness.values = f
			ind.parent_bd = ind.bd
			ind.parent_id = ind.id
			ind.id = generate_uuid()
			ind.bd = b
   
		for ind in population:
			ind.am_parent=1
   
		for ind in offspring:
			ind.am_parent=0
			
		pq = population+offspring

		if(params["stop_archive_update"]==gen):
			params["add_strategy"]="none"

		if(params["pop_for_novelty_estimation"]==0):
			pop_for_novelty_estimation = None
		elif (params["freeze_pop"]==-1) or (gen<=params["freeze_pop"]):
			pop_for_novelty_estimation = pq

		t = time.time()
		archive = updateNovelty(pq, offspring, archive, params, pop_for_novelty_estimation)
		print("Novelty update time: ", time.time() - t)		

		# Only 2D and 3D BD are supported
		# if bd_dimension == 2 or bd_dimension == 3:
		# 	alpha_shape = alphashape.alphashape(archive.all_bd, alphas)
		
		t = time.time()
		# Compute the novelty rank
		nov = [ind.novelty for ind in pq]
		isortednov = np.argsort(nov)[::-1]
		rank = np.empty_like(isortednov)
		rank[isortednov] = np.arange(len(isortednov))
		
		for i,ind in enumerate(pq):
			# if alpha_shape:
			# 	ind.dist_to_explored_area = dist_to_shapes(ind.bd,alpha_shape)
			ind.rank_novelty = rank[i]
			#print("Indiv #%d: novelty=%f rank=%d"%(i, ind.novelty, ind.rank_novelty))
			if (ind.parent_bd is None):
				ind.dist_to_parent=0
			else:
				ind.dist_to_parent=np.linalg.norm(np.asarray(ind.bd)-np.asarray(ind.parent_bd))
			if (emo):
				if (varian == "NS+Fit"):
					ind.fitness.values = (ind.novelty, ind.fit)
				elif (varian == "NS+BDDistP"):
					if (ind.parent_bd is None):
						bddistp=0
					else:
						bddistp=np.linalg.norm(np.asarray(ind.bd) - np.asarray(ind.parent_bd))
					ind.fitness.values = (ind.novelty, bddistp)
				elif (varian == "NS+Fit+BDDistP"):
					if (ind.parent_bd is None):
						bddistp=0
					else:
						bddistp=np.linalg.norm(np.asarray(ind.bd) - np.asarray(ind.parent_bd))
					ind.fitness.values = (ind.novelty, ind.fit, bddistp)
				else:
					print("WARNING: unknown variant: "+variant)
					ind.fitness.values=ind.fit
			else:
				ind.fitness.values=ind.fit
		print("Data updating time: ", time.time() - t)

		if (verbosity(params)):
			print("Gen %d"%(gen))
		else:
			if(gen%100==0):
				print(" %d "%(gen), end='', flush=True)
			elif(gen%10==0):
				print("+", end='', flush=True)
			else:
				print(".", end='', flush=True)

		
		# Select the next generation population
		t =  time.time()
		if ("," in variant):
			population[:] = toolbox.select(offspring, params["pop_size"])		
		else:
			population[:] = toolbox.select(pq, params["pop_size"])		
		print("Selection time: ", time.time() - t)

		if (("eval_budget" in params.keys()) and (params["eval_budget"]!=-1) and (nb_eval>=params["eval_budget"])): 
			params["nb_gen"]=gen
			terminates=True
		else:
			terminates=False

		dump_data(population, gen, params, prefix="population", attrs=["all", "dist_to_explored_area", "dist_to_parent", "rank_novelty"], force=terminates)
		dump_data(population, gen, params, prefix="bd", complementary_name="population", attrs=["bd"], force=terminates)
		dump_data(offspring, gen, params, prefix="bd", complementary_name="offspring", attrs=["bd"], force=terminates)
		dump_data(archive.get_content_as_list(), gen, params, prefix="archive", attrs=["all"], force=terminates)
		
		# Update the statistics with the new population
		# record = params["stats"].compile(population) if params["stats"] is not None else {}
		# record_offspring = params["stats_offspring"].compile(offspring) if params["stats_offspring"] is not None else {}
		# logbook.record(gen=gen, nevals=len(invalid_ind), **record, **record_offspring)
		# if (verbosity(params)):
		#	 print(logbook.stream)

		for ind in population:
			ind.evolvability_samples=None

		if (terminates):
			break
			
			
	return population, archive, logbook, nb_eval




  
if (__name__=='__main__'):
	print("Test of the Novelty-based ES")

	printf("TODO...")
