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
	random_key, mut_key = jax.random.split(random_key)
 
	mut_ind = jnp.arange(len(population))
	mut_ind = jax.random.choice(mut_key, mut_ind, (int(mutpb*lambda_),))	# indices of the individuals to mutate

	# Mutate the geneotypes
	random_key, subkey = jax.random.split(random_key)
	keys = jax.random.split(subkey, mut_ind.shape[0])
	mutate_gen = jax.vmap(toolbox.mutate)(keys, jnp.array(population)[mut_ind])
	
	offspring = []
	for i in range(len(population)):
		off = toolbox.clone(population[i])
		off[:] = mutate_gen[i]
		offspring.append(off)
	return offspring, random_key