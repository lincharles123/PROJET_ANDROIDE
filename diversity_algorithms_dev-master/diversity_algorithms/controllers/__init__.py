# coding: utf-8

#Keras controller
#from diversity_algorithms.controllers.fixed_structure_nn import SimpleNeuralControllerKeras as SimpleNeuralController

#Flax controller
from diversity_algorithms.controllers.fixed_structure_nn_flax import SimpleNeuralControllerFlax as SimpleNeuralController

#Numpy controller
#from diversity_algorithms.controllers.fixed_structure_nn_numpy import SimpleNeuralControllerNumpy as SimpleNeuralController

__all__ = ["fixed_structure_nn", "fixed_structure_nn_numpy", "fixed_structure_nn_flax"]
