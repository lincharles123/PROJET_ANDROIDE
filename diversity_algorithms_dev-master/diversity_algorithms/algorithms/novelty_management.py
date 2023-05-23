
import random
from scipy.spatial import cKDTree as KDTree
import numpy as np
from diversity_algorithms.algorithms.utils import verbosity
import jax.numpy as jnp

__all__ = ["NovArchive", "updateNovelty"]

# ### Novelty-based Evolution Strategies

class NovArchive:
    """Archive used to compute novelty scores."""
    def __init__(self, lbd, k=15):
        self.all_bd = lbd
        if (self.all_bd.shape[0]>0):
            self.kdtree=KDTree(self.all_bd)
        else:
            self.kdtree=None # for archive-less experiments
        self.k=k
        #print("Archive constructor. size = %d"%(len(self.all_bd)))

    def ready(self):
        return self.size()>self.k

    def get_content_as_list(self):
        return self.all_bd.tolist()

    def update(self,new_bd):
        new_bd = new_bd
        if (new_bd.shape[0]==0):
            return
        oldsize = self.all_bd.shape[0]
        if (oldsize>0):
            for bd in new_bd:
                assert bd.shape == self.all_bd[0].shape, "update archive, bd of different sizes: bd.shape[1]=%d, all_bd.shape[1]=%d"%(bd.shape[1],self.all_bd.shape[1])

        self.all_bd = jnp.concatenate((self.all_bd, new_bd))
        self.kdtree = KDTree(self.all_bd)
        #print("Archive updated, old size = %d, new size = %d"%(oldsize, self.all_bd.shape[0]))
    
    def get_nov(self, popbd, population=jnp.asarray([])):
        dpop = jnp.linalg.norm(popbd[:, None] - population, axis=2)

        if (self.kdtree is None):
            darch=[] # archive-less NS (i.e. behavior diversity)
        else:
            darch = self.kdtree.query(popbd,self.k, workers=-1)[0]
            
        d = jnp.concatenate((dpop, darch), axis=1)
        d = jnp.sort(d, axis=1)
        d = d[:, :self.k+1]
        #    print("WARNING in novelty search: the smallest distance should be 0 (distance to itself). If you see it, you probably try to get the novelty with respect to a population your indiv is not in. The novelty value is then the sum of the distance to the k+1 nearest divided by k. d[0]=%f"%(d[0]))
        return jnp.mean(d, axis=1) # as the indiv is in the population, the first value is necessarily a 0.

    def size(self):
        return self.all_bd.shape[0]


def updateNovelty(population, offspring, archive, params, population_saved=None):
    """Update the novelty criterion (including archive update) 
 
    Implementation of novelty search following (Gomes, J., Mariano, P., & Christensen, A. L. (2015, July). Devising effective novelty search algorithms: A comprehensive empirical study. In Proceedings of GECCO 2015 (pp. 943-950). ACM.).
    :param population: is the set of indiv for which novelty needs to be computed
    :param offspring: is the set of new individuals that need to be taken into account to update the archive (may be the same as population, but it may also be different as population may contain the set of parents)
    :param params: dictionary containing run parameters. The relevant parameters are:
       * params["k"] is the number of nearest neighbors taken into account
       * params["add_strategy"] is either "random" (a random set of indiv is added to the archive) or "novel" (only the most novel individuals are added to the archive).
       * params ["lambda_nov"] is the number of individuals added to the archive for each generation
     :returns: The function returns the new archive
    """
    k=params["k"]
    add_strategy=params["add_strategy"]
    _lambda=params["lambda_nov"]

    if (population_saved is not None):
        ref_pop=population_saved
    else: 
        ref_pop=population
 
    # Novelty scores updates
    if (archive) and (archive.size()+len(ref_pop)>=k):
        if (verbosity(params,["all", "novelty"])):
            print("Update Novelty. Archive size=%d"%(archive.size()))
        pop_bd = jnp.asarray(np.asarray([ind.bd for ind in population]))
        ref_bd = jnp.asarray(np.asarray([ind.bd for ind in ref_pop]))
        nov = archive.get_nov(pop_bd, population=ref_bd)    # compute the novelty of the population with respect to the archive
        for ind, n in zip(population, nov):
            if (jnp.isnan(n)):
                ind.novelty=-1
            else:
                ind.novelty = n
    else:
        if (verbosity(params,["all", "novelty"])):
            print("Update Novelty. Initial step...") 
        for ind in population:
            ind.novelty=0.
 
    if (verbosity(params,["all", "novelty"])):
        print("Fitness (novelty): ",end="") 
        for ind in population:
            print("%.2f, "%(ind.novelty),end="")
        print("")
    if (len(offspring)<_lambda):
        print("ERROR: updateNovelty, lambda(%d)<offspring size (%d)"%(_lambda, len(offspring)))
        return None
 
    lbd = []
    # Update of the archive
    # we remove indivs with NAN values
    offspring2 = list(filter(lambda x: not (True in np.isnan(x.bd)), offspring))

    if (len(offspring)!=len(offspring2)):
        print("WARNING: in updateNovelty, some individuals have a behavior descriptor with NaN values ! Initial offspring size: %d, filtered offspring size: %d"%(len(offspring), len(offspring2)))
    if (len(offspring2)<_lambda):
        print("WARNING: too few individuals have a non NaN value. We limit the number of added individuals to %d (number of offspring with non NaN bd)..."%(len(offspring2)))
        _lambda=len(offspring2)
    if(add_strategy=="random"):
        l = list(range(len(offspring2)))
        random.shuffle(l)
        if (verbosity(params,["all", "novelty"])):
            print("Random archive update. Adding offspring: "+str(l[:_lambda])) 
        lbd=[offspring2[l[i]].bd for i in range(_lambda)]
    elif(add_strategy=="novel"):
        nov = jnp.asarray(np.asarray([ind.novelty for ind in offspring2]))
        index = jnp.argsort(nov)
        ilast = index.shape[0] - _lambda
        lbd = [offspring2[i].bd for i in index[ilast:]]
        if (verbosity(params,["all", "novelty"])):
            print("Novel archive update. Adding offspring: ")
            for ind in index[ilast:]:
                offs = offspring2[ind]
                print("    nov="+str(offs.novelty)+" fit="+str(offs.fitness.values)+" bd="+str(offs.bd))
    elif(add_strategy=="none"):
        # nothing to do...
        pass
    else:
        print("ERROR: updateNovelty: unknown add strategy(%s), valid alternatives are \"random\" and \"novel\""%(add_strategy))
        return None
 
    lbd = jnp.asarray(np.asarray(lbd))
    if(archive==None):
        archive=NovArchive(lbd,k)
    else:
        archive.update(lbd)
 
    return archive