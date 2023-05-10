from deap import creator
from functools import partial
import jax
from jax import numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames={"eta", "min_val", "max_val", "indpb"})
def mutate(random_key, gen, eta, min_val, max_val, indpb):
	"""Polynomial mutation as implemented in deap (mutPolynomialBounded).
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


def varOr(random_key, population, toolbox, lambda_, cxpb, mutpb):
	"""
	The mutation calcution on itself is very fast but the creation of the offspring is slow.
	"""
	random_key, mut_key = jax.random.split(random_key)
 
	mut_ind = jnp.arange(len(population))
	mut_ind = jax.random.choice(mut_key, mut_ind, (int(mutpb*lambda_),))	# indices of the individuals to mutate

	# Mutate the geneotypes
	random_key, subkey = jax.random.split(random_key)
	keys = jax.random.split(subkey, mut_ind.shape[0])
	mutate_gen = jax.vmap(toolbox.mutate)(keys, jnp.asarray(population)[mut_ind])
	
	# Create the offsprings
	offspring = [creator.Individual([x]) for x in np.asarray(mutate_gen)]
	for i in range(len(offspring)):
		offspring[i] =  offspring[i][0]
		offspring[i].fitness = creator.FitnessMax()

	# Copy bd and id from the mutated individuals
	bd_id = [(population[i].bd, population[i].id) for i in mut_ind]
	for ind, val in zip(offspring, bd_id):
		ind.bd = val[0]
		ind.id = val[1]

	return offspring, random_key


def selBest(population, size, fit_attr="fitness"):
	values = np.asarray(([getattr(ind, fit_attr) for ind in population]))
	index = np.argsort(values)[-size:]
	return [population[i] for i in index]


def init_pop(random_key, size, params):
	""" 
	Weird behavior of creator.Individual, it is very slow when giving it an array but if given 
	the array wrapped in a list it is working fine.
	=> We create the population with jax and then wrap it in a list to create the individuals.
	   Then we get the array out of the list for each individual and initialize a new fitness.
	"""
	random_key, subkey = jax.random.split(random_key)
	all = jax.random.uniform(subkey, (size,params["ind_size"]), jnp.float32, params["min"], params["max"],)
	population = [creator.Individual([x]) for x in np.asarray(all)]
	for i in range(len(population)):
		population[i] =  population[i][0]
		population[i].fitness = creator.FitnessMax()
	return population, random_key

