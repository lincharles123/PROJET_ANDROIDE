
# coding: utf-8

from jax import numpy as jnp, random
import jax

## Suppress TF info messages

import os

def sigmoid(x):
    return 1./(1 + jnp.exp(-x))

def tanh(x):
    return jnp.tanh(x)


def gen_simplemlp(n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5):
    n_neurons = [n_neurons_per_hidden]*n_hidden_layers if jnp.isscalar(n_neurons_per_hidden) else n_neurons_per_hidden
    i = Input(shape=(n_in,))
    x = i
    for n in n_neurons:
        x = Dense(n, activation='sigmoid')(x)
    o = Dense(n_out, activation='tanh')(x)
    m = Model(inputs=i, outputs=o)
    return m
    

class SimpleNeuralControllerNumpy():
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5, params=None):
        self.dim_in = n_in
        self.dim_out = n_out
        self.key = random.PRNGKey(0)
        # if params is provided, we look for the number of hidden layers and neuron per layer into that parameter (a dicttionary)
        if (not params==None):
            if ("n_hidden_layers" in params.keys()):
                n_hidden_layers=params["n_hidden_layers"]
            if ("n_neurons_per_hidden" in params.keys()):
                n_neurons_per_hidden=params["n_neurons_per_hidden"]
        self.n_per_hidden = n_neurons_per_hidden
        self.n_hidden_layers = n_hidden_layers
        self.weights = None 
        self.n_weights = None
        self.init_random_params()
        self.out = jnp.zeros(n_out)
        #print("Creating a simple mlp with %d inputs, %d outputs, %d hidden layers and %d neurons per layer"%(n_in, n_out,n_hidden_layers, n_neurons_per_hidden))


    def init_random_params(self):
        keys = random.split(self.key, self.n_hidden_layers+1)
        w_key, b_key = random.split(keys[0])
        if(self.n_hidden_layers > 0):
            self.weights = [random.uniform(w_key, (self.dim_in,self.n_per_hidden),float, 0, 1)] # In -> first hidden
            self.bias = [random.uniform(b_key, (self.n_hidden_layers,),float, 0, 1)] # In -> first hidden
            for i in range(self.n_hidden_layers-1): # Hidden -> hidden
                w_key, b_key = random.split(keys[i+1])
                self.weights.append(random.uniform(w_key, (self.n_per_hidden,self.n_per_hidden),float, 0, 1))
                self.bias.append(random.uniform(b_key, (self.n_per_hidden,),float, 0, 1))
            w_key, b_key = random.split(keys[-1])
            self.weights.append(random.uniform(w_key, (self.n_hidden_layers,self.dim_out),float, 0, 1)) # -> last hidden -> out
            self.bias.append(random.uniform(b_key, (self.dim_out,),float, 0, 1))
        else:
            self.weights = [random.uniform(w_key, (self.dim_in,self.dim_out),float, 0, 1)] # Single-layer perceptron
            self.bias = [random.uniform(b_key, (self.dim_out),float, 0, 1)]
        n_w = self.dim_in*self.n_per_hidden + self.n_per_hidden*self.n_per_hidden*(self.n_hidden_layers-1) + self.n_per_hidden*self.dim_out
        n_b = self.n_per_hidden*self.n_hidden_layers + self.dim_out
        self.n_weights = n_w + n_b
        self.key = random.split(keys[-1], 1)

    def get_parameters(self):
        """
        Returns all network parameters as a single array
        """
        flat_weights = jnp.hstack([arr.flatten() for arr in (self.weights+self.bias)])
        return flat_weights

    def set_parameters(self, flat_parameters):
        """
        Set all network parameters from a single array
        """
        if (jnp.nan in flat_parameters):
            print("WARNING: NaN in the parameters of the NN: "+str(list(flat_parameters)))
        if (max(flat_parameters)>1000):
            print("WARNING: max value of the parameters of the NN >1000: "+str(list(flat_parameters)))
            
                
        i = 0 # index
        to_set = []
        self.weights = list()
        self.bias = list()
        if(self.n_hidden_layers > 0):
            # In -> first hidden
            w0 = jnp.array(flat_parameters[i:(i+self.dim_in*self.n_per_hidden)])
            self.weights.append(w0.reshape(self.dim_in,self.n_per_hidden))
            i += self.dim_in*self.n_per_hidden
            for l in range(self.n_hidden_layers-1): # Hidden -> hidden
                w = jnp.array(flat_parameters[i:(i+self.n_per_hidden*self.n_per_hidden)])
                self.weights.append(w.reshape((self.n_per_hidden,self.n_per_hidden)))
                i += self.n_per_hidden*self.n_per_hidden
            # -> last hidden -> out
            wN = jnp.array(flat_parameters[i:(i+self.n_per_hidden*self.dim_out)])
            self.weights.append(wN.reshape((self.n_per_hidden,self.dim_out)))
            i += self.n_per_hidden*self.dim_out
            # Samefor bias now
            # In -> first hidden
            b0 = jnp.array(flat_parameters[i:(i+self.n_per_hidden)])
            self.bias.append(b0)
            i += self.n_per_hidden
            for l in range(self.n_hidden_layers-1): # Hidden -> hidden
                b = jnp.array(flat_parameters[i:(i+self.n_per_hidden)])
                self.bias.append(b)
                i += self.n_per_hidden
            # -> last hidden -> out
            bN = jnp.array(flat_parameters[i:(i+self.dim_out)])
            self.bias.append(bN)
            i += self.dim_out
        else:
            n_w = self.dim_in*self.dim_out
            w = jnp.array(flat_parameters[:n_w])
            self.weights = [w.reshape((self.dim_in,self.dim_out))]
            self.bias = [jnp.array(flat_parameters[n_w:])]


    def predict(self,x):
        """
        Propagage
        """
        if(self.n_hidden_layers > 0):
            #Input
            a = jnp.matmul(x,self.weights[0]) + self.bias[0]
            y = sigmoid(a)
            # hidden -> hidden
            for i in range(1,self.n_hidden_layers):
                a = jnp.matmul(y, self.weights[i]) + self.bias[i]
                y = sigmoid(a)
            # Out
            a = jnp.matmul(y, self.weights[-1]) + self.bias[-1]
            out = tanh(a)
            return out
        else: # Simple monolayer perceptron
            return tanh(jnp.matmul(x,self.weights[0]) + self.bias[0])

    def __call__(self,x):
        """Calling the controller calls predict"""
        return self.predict(x)


