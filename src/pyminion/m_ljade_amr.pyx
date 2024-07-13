import numpy as np
cimport numpy as np

from .de_base cimport DE_Base 
from .minimizer_base cimport MinionResult
from .utility cimport *

cdef class M_LJADE_AMR (DE_Base):
    """
    Implementation of the jDE optimization algorithm using Cython for improved performance.
    
    Attributes:
    ----------
    func : callable
        The objective function to be minimized.
    bounds : np.ndarray
        Array defining the lower and upper bounds for each dimension.
    x0 : list or np.ndarray 
        initial guess. 
    args : tuple
        Additional arguments to pass to the objective function.
    population_size : int
        Number of candidates in the population.
    maxevals : int
        Maximum number of function evaluations allowed.
    strategy : str
        Strategy used for mutation and crossover.
    relTol : float
        Relatove tolerance for convergence.
    minPopSize : int
        Minimum population size for the algorithm.
    c : float
        Coefficient for adapting parameters.
    callback : callable
        Function to be called at the end of each iteration.
    """
    def __init__(self,  func, bounds, args=(),  x0=None, int population_size=20, int maxevals=100000, 
                str strategy='pbest1bin', double relTol=0.0001, int minPopSize=10, double c=0.1, object callback=None, seed=None):
        super(M_LJADE_AMR, self).__init__(func, bounds,args, x0, population_size, maxevals, strategy, relTol, minPopSize, callback, seed )
        self.F= np.random.uniform(0.4, 0.6, self.popsize) 
        self.CR= np.random.uniform(0.8, 1.0, self.popsize) 
        self.meanCR = 0.5 
        self.meanF = 0.8
        self.c= c

    cpdef void _adapt_parameters(self):
        self.CR = np.random.normal(self.meanCR, 0.1, self.popsize)
        self.F = np.random.normal(self.meanF, 0.1, self.popsize)

        spread = np.std(self.fitness)/self.best_fitness
        if spread < 0.1 : 
            etaF = 0.5 - 5*spread  
            etaCR = 0.1-spread
            for i in range(self.popsize) :
                if np.random.rand() < etaF : self.F[i] = np.random.uniform(0.5, 1.5)
                if np.random.rand() < etaCR : self.CR[i] = np.random.uniform(0.01, 1.0)

        self.CR = np.clip(self.CR, 0.01, 1.0)
        self.F = np.clip(self.F, 0.01, 2.0)
        self.muCR.append(np.mean(self.CR))
        self.muF.append(np.mean(self.F))
        self.stdCR.append(np.std(self.CR))
        self.stdF.append(np.std(self.F))
                
    cpdef MinionResult optimize(self):
        self._initialize_population()
        cdef int iter, new_population_size, memindex=0
        cdef np.ndarray[np.float64_t, ndim=2] new_population, all_trials
        cdef np.ndarray[np.float64_t, ndim=1] new_fitness, all_trial_fitness

        for iter in range(self.maxiter + 1):
            S_CR = []
            S_F =  []
            weights = []
            weights_Lehmer = []

            self._adapt_parameters()            
            new_population = np.empty_like(self.population)
            new_fitness = np.empty_like(self.fitness)
            all_trials = np.empty_like(self.population)

            for i in range(self.popsize):  all_trials[i] = self._crossover(self.population[i], self._mutate(i), self.CR[i])
            all_trials = np.array(all_trials)
            enforce_bounds(all_trials, self.bounds, "random-leftover")

            # Call self.func once with all trials
            all_trial_fitness = self.func(all_trials, *self.funcArgs)
            all_trial_fitness = np.nan_to_num(all_trial_fitness, nan=1e+100)
            self.Nevals += self.popsize

            for i in range(self.popsize):
                trial_fitness = all_trial_fitness[i]

                if trial_fitness < self.fitness[i]:
                    new_population[i] = all_trials[i]
                    new_fitness[i] = trial_fitness
                    S_CR.append(self.CR[i])
                    S_F.append(self.F[i])
                    w = (self.fitness[i]- trial_fitness)/(1e-100+self.fitness[i])
                    if np.isnan(w) : w=0
                    weights.append( w)
                else:
                    new_population[i] = self.population[i]
                    new_fitness[i] = self.fitness[i]

            self.population = new_population
            self.fitness = new_fitness
            self.best_idx = np.argmin(self.fitness)
            self.best = self.population[self.best_idx]
            self.best_fitness = self.fitness[self.best_idx]
            self.history.append([self.best, self.best_fitness])

            if len(S_CR) !=0 : 
                weights = np.array(weights)
                S_CR = np.array(S_CR)
                S_F = np.array(S_F)
                weights_Lehmer = (S_F**2*weights)/np.sum(S_F**2*weights)
                weights = np.array(weights)/np.sum(weights)
                muCR, stdCR = getMeanStd(S_CR, weights) 
                muF, stdF = getMeanStd(S_F, weights_Lehmer) 
                
                self.meanCR = (1-self.c)*self.meanCR + self.c*muCR 
                self.meanF = (1-self.c)*self.meanF + self.c*muF

            self.minionResult = MinionResult(x=self.best, fun=self.best_fitness, success=True, nit=iter+1, nfev=self.popsize * (iter + 1))
            if self.callback is not None: self.callback(self.minionResult)
            
            if self.popDecrease :
                new_population_size = round((self.minPopSize - self.original_popsize) / self.maxevals * self.Nevals + self.original_popsize)
                if self.popsize > new_population_size:
                    self.popsize = new_population_size
                    best_indexes = np.argsort(self.fitness)[:self.popsize]
                    self.population = self.population[best_indexes]
                    self.fitness = self.fitness[best_indexes]
                    self.best_idx = np.argmin(self.fitness) 
                    self.best = self.population[self.best_idx]
                    self.best_fitness = self.fitness[self.best_idx]
            
            if self.relTol !=0.0 :
                if (np.max(self.fitness)-np.min(self.fitness)) / np.average(self.fitness) <= self.relTol:
                    break

        return self.minionResult
