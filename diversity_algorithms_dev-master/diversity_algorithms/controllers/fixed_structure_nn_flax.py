from flax import linen as nn
from jax import random
from jax import numpy as jnp
from jax.nn.initializers import lecun_uniform
from jax.lax import dynamic_slice
import numpy as np
from flax.core.frozen_dict import freeze
from typing import Any, Callable, Tuple, Optional
import jax

class MLP(nn.Module):
    n_out: int
    n_hidden_layers: int
    n_per_hidden: int
    kernel_init: Callable[..., Any] = lecun_uniform()
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.sigmoid
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = nn.tanh

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i in range(self.n_hidden_layers):
            x = nn.Dense(
                self.n_per_hidden,
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation(x)
        x = nn.Dense(
            self.n_out,
            kernel_init=self.kernel_init,
        )(x)
        return self.activation(x)


class SimpleNeuralControllerFlax:
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5, params=None):
        self.dim_in = n_in
        self.dim_out = n_out
        if (not params==None):
            if ("n_hidden_layers" in params.keys()):
                n_hidden_layers=params["n_hidden_layers"]
            if ("n_neurons_per_hidden" in params.keys()):
                n_neurons_per_hidden=params["n_neurons_per_hidden"]
        #print("Creating a simple mlp with %d inputs, %d outputs, %d hidden layers and %d neurons per layer"%(n_in, n_out,n_hidden_layers, n_neurons_per_hidden))
        self.n_per_hidden = n_neurons_per_hidden
        self.n_hidden_layers = n_hidden_layers
        self.model = MLP(self.dim_out, n_hidden_layers, n_neurons_per_hidden)
        self.weights = None
        self.init_random_params()
    
    
    def init_random_params(self, ):
        key1, key2 = random.split(random.PRNGKey(0))
        x = random.uniform(key1, (self.dim_in,))
        self.weights = self.model.init(key2, x)
        weights = self.weights['params'].unfreeze()
        self.keys = weights.keys()
        self.shapes = [(weights[k]['bias'].shape, weights[k]['kernel'].shape) for k in self.keys]
        self.n_weights = sum([np.prod(shape[0]) + np.prod(shape[1]) for shape in self.shapes])
    

    def gen_indiv(self, size, random_key):
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=size)
        fake_batch = jnp.zeros(shape=(size, self.dim_in))
        new_gen = jax.vmap(jax.jit(self.model.init))(keys, fake_batch)
        return new_gen, random_key

    def array_to_dict(self, flat_parameters):
        """
        Set all network parameters from a single array
        """
        i = 0 # index
        dict = {}
        for key, shape in zip(self.keys, self.shapes):
            dict[key] = {}
            dict[key]['bias'] = dynamic_slice(flat_parameters, (i,), shape[0]).reshape(shape[0])
            i = i + shape[0][0]
            dict[key]['kernel'] = dynamic_slice(flat_parameters, (i,), (shape[1][0]*shape[1][1],)).reshape(shape[1])
            i = i + shape[1][0]*shape[1][1]
        return {'params': dict}
    
    def predict(self, params, obs):
        return self.model.apply(params, obs)

    def get_n_weights(self):
        return self.n_weights

