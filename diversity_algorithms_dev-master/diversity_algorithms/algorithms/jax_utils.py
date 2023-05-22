from deap import creator
from functools import partial
import jax
from jax import numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames={"eta", "min_val", "max_val", "indpb"})
def mutate(random_key, gen, eta, min_val, max_val, indpb):
	"""
 	Polynomial mutation as implemented in deap (mutPolynomialBounded) using jax to parrallelize the computation.
	https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py   
 	"""
	random_key, subkey = jax.random.split(random_key)
	mut_id = jnp.arange(gen.shape[0])
	mut_id = jax.random.choice(subkey, mut_id, (int(indpb*gen.shape[0]),), replace=False)

	mut_var = gen[mut_id]
	delta_1 = (gen[mut_id] - min_val) / (max_val - min_val)
	delta_2 = (max_val - gen[mut_id]) / (max_val - min_val)
	mut_pow = 1.0 / (eta + 1.)

	random_key, subkey = jax.random.split(random_key)
	rands = jax.random.uniform(subkey, mut_var.shape, jnp.float32, 0, 1)

	val1 = 2.0 * rands + ((1.0 - 2.0*rands) * jnp.power((1.0 - delta_1), (mut_pow + 1.0)))
	val1 = jnp.power(val1, mut_pow) - 1.0
	val2 = 2.0 * (1.0 - rands) + (2.0 * (rands - 0.5) * jnp.power(1.0 - delta_2, eta + 1))
	val2 = 1.0 - jnp.power(val2, mut_pow)

	zero_arr = jnp.zeros_like(mut_var)
	delta_q = jnp.where(rands < 0.5, val1, zero_arr)
	delta_q = jnp.where(rands >= 0.5, val2, delta_q)

	new_val = mut_var + delta_q * (max_val - min_val)
	new_val = jnp.where(new_val < min_val, min_val, new_val)
	new_val = jnp.where(new_val > max_val, max_val, new_val)
	
	new_gen = gen.at[mut_id].set(new_val)
	return new_gen

@partial(jax.jit, static_argnames={"alpha"})
def cxBLend(random_key, gen1, gen2, alpha):
	"""
	Equivalent to deap algorithms.cxBlend but using jax to parrallelize the computation.
	https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
 	"""
	blend_id = jnp.arange(gen1.shape[0])
	blend_id = jax.random.choice(random_key, blend_id, (int(alpha*gen1.shape[0]),), replace=False)

	# Compute the new values
	new_val1 = gen1[blend_id] + (gen2[blend_id] - gen1[blend_id]) * 0.5 * (1.0 + alpha)
	new_val2 = gen2[blend_id] + (gen1[blend_id] - gen2[blend_id]) * 0.5 * (1.0 + alpha)

	# Put the new values in the new individuals
	new_gen1 = gen1.at[blend_id].set(new_val1)
	new_gen2 = gen2.at[blend_id].set(new_val2)

	return new_gen1, new_gen2

def varOr(random_key, population, toolbox, lambda_, cxpb, mutpb):
	"""
	Equivalent to deap algorithms.varOr but using jax to parrallelize the computation.
	https://github.com/DEAP/deap/blob/master/deap/algorithms.py
 	"""
	assert cxpb + mutpb == 1.0, ("The sum of the crossover and mutation probabilities must be 1.0.")
  
	# Crossover
	random_key, subkey = jax.random.split(random_key)
	cx_ind = jnp.arange(len(population))
	cx_ind = jax.random.choice(subkey, cx_ind, (int(cxpb*lambda_), 2))	# indices of the individuals to crossover
	cx_ind1, cx_ind2 = cx_ind[:,0], cx_ind[:,1]	
 
	# Crossover the geneotypes
	random_key, subkey = jax.random.split(random_key)
	keys = jax.random.split(subkey, cx_ind.shape[0])
	cx_gen, _ = jax.vmap(toolbox.mate)(keys, jnp.asarray(population)[cx_ind1], jnp.asarray(population)[cx_ind2])
 
	# Mutation
	random_key, mut_key = jax.random.split(random_key)
	mut_ind = jnp.arange(len(population))
	mut_ind = jax.random.choice(mut_key, mut_ind, (lambda_-cx_ind.shape[0],))	# indices of the individuals to mutate

	# Mutate the geneotypes
	random_key, subkey = jax.random.split(random_key)
	keys = jax.random.split(subkey, mut_ind.shape[0])
	mutate_gen = jax.vmap(toolbox.mutate)(keys, jnp.asarray(population)[mut_ind])
	
 
	# Create the offsprings
	off_gen = jnp.concatenate((cx_gen, mutate_gen), axis=0)
	off_ind = jnp.concatenate((cx_ind1, mut_ind), axis=0)
	offspring = [creator.Individual([x]) for x in np.asarray(off_gen)]
	for i in range(len(offspring)):
		offspring[i] =  offspring[i][0]
		offspring[i].fitness = creator.FitnessMax()

	# Copy bd and id from the mutated individuals
	bd_id = [(population[i].bd, population[i].id) for i in off_ind]
	for ind, val in zip(offspring, bd_id):
		ind.bd = val[0]
		ind.id = val[1]

	return offspring, random_key


def selBest(population, size, fit_attr="fitness"):
	values = np.asarray(([getattr(ind, fit_attr) for ind in population]))
	index = np.argsort(values)[-size:]
	return [population[i] for i in index]


def init_pop_controller(random_key, size, controller):
	""" 
	Weird behavior of creator.Individual, it is very slow when giving it an array but if given 
	the array wrapped in a list it is working fine.
	=> We create the population with jax and then wrap it in a list to create the individuals.
	   Then we get the array out of the list for each individual and initialize a new fitness.
	"""
	all_pop, random_key = controller.generate_random_parameters(random_key, size)
	population = [creator.Individual([x]) for x in np.asarray(all_pop)]
	for i in range(len(population)):
		population[i] =  population[i][0]
		population[i].fitness = creator.FitnessMax()
	return population, random_key

def init_pop_numpy(random_key, size, params):
	""" 
	Weird behavior of creator.Individual, it is very slow when giving it an array but if given 
	the array wrapped in a list it is working fine.
	=> We create the population with jax and then wrap it in a list to create the individuals.
	   Then we get the array out of the list for each individual and initialize a new fitness.
	"""
	random_key, subkey = jax.random.split(random_key)
	all_pop = jax.random.uniform(subkey, (size,params["ind_size"]), jnp.float32, params["min"], params["max"],)
	population = [creator.Individual([x]) for x in np.asarray(all_pop)]
	for i in range(len(population)):
		population[i] =  population[i][0]
		population[i].fitness = creator.FitnessMax()
	return population, random_key
