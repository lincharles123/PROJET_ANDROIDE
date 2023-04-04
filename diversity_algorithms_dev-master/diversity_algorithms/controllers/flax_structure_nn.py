from flax import linen as nn
from jax import random
from jax import numpy as jnp
from flax.core.frozen_dict import freeze

class MLP(nn.Module):                    # create a Flax Module dataclass
    n_out: int
    n_hidden_layers: int
    n_per_hidden: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i in range(self.n_hidden_layers):
            x = nn.Dense(self.n_per_hidden, name=f'layers_{i}')(x)
            x = nn.sigmoid(x)
        x = nn.Dense(self.n_out, name='layers_out')(x)
        x = nn.tanh(x)
        return x

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
    
    def init_random_params(self):
        key1, key2 = random.split(random.PRNGKey(0), 2)
        x = random.uniform(key1, (4,))
        self.params = self.model.init(key2, x)
        params = self.params['params'].unfreeze()
        self.keys = params.keys()
        self.shapes = [(params[k]['bias'].shape, params[k]['kernel'].shape) for k in self.keys]
    
    
    def set_parameters(self, flat_parameters):
        """
        Set all network parameters from a single array
        """
        if (jnp.nan in flat_parameters):
            print("WARNING: NaN in the parameters of the NN: "+str(list(flat_parameters)))
        if (max(flat_parameters)>1000):
            print("WARNING: max value of the parameters of the NN >1000: "+str(list(flat_parameters)))
        
        i = 0 # index
        dict = {}
        for key, shape in zip(self.keys, self.shapes):
            dict[key] = {}
            dict[key]['bias'] = flat_parameters[i:i+np.prod(shape[0])].reshape(shape[0])
            i += np.prod(shape[0])
            dict[key]['kernel'] = flat_parameters[i:i+np.prod(shape[1])].reshape(shape[1])
            i += np.prod(shape[1])
        self.params = freeze({'params': dict})