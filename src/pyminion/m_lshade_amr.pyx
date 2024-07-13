import numpy as np
cimport numpy as np

from .de_base cimport DE_Base
from .utility cimport *
from .minimizer_base cimport *

cdef class M_LSHADE_AMR (DE_Base):
    """
    An improved Memory-based DE  optimization algorithm implementation.

    This class implements the LSHADE algorithm, which is an extension of Differential Evolution
    (DE) that uses a memory-based adaptation mechanism for control parameters (CR and F).

    Parameters:
        ----------
        func : callable
            The objective function to be minimized.
        bounds : list
            List of (min, max) pairs for each dimension defining the bounds.
        args : tuple
            Additional arguments to pass to the objective function.
        x0 : array-like, optional
            Initial solution for the optimization.
        population_size : int
            Number of individuals in the population.
        maxevals : int
            Maximum number of function evaluations.
        strategy : str
            DE strategy to use for mutation and crossover.
        tol : float
            Tolerance for convergence.
        memorySize : int
            Size of the memory for CR and F values.
        minPopSize : int
            Minimum population size.
        callback : callable, optional
            A function to be called after each iteration.
        seed : int, optional
            Seed for random number generation.
    """
    def __init__(self,  func, bounds, args=(),  x0=None, int population_size=30, int maxevals=100000, str strategy='pbest1bin', double relTol=0.0001, 
                                        int memorySize=30, int minPopSize=10, object callback=None, seed=None):
        super(M_LSHADE_AMR, self).__init__(func, bounds,args, x0, population_size, maxevals, strategy, relTol, minPopSize, callback, seed)
        self.H = memorySize  # Memory size for CR and F
        self.memProb = [1.0/self.H for _ in range(self.H)]

    cpdef void _initialize_population(self):
        super(M_LSHADE_AMR, self)._initialize_population()
        self.M_CR = np.random.uniform(0.4, 0.6, self.H)
        self.M_F = np.random.uniform(0.6, 1.0, self.H)
        self.F= np.full(self.H, 0.8)
        self.CR= np.random.uniform(0.8, 1.0, self.popsize) 

    cpdef void _adapt_parameters(self):
        cdef np.ndarray[np.int_t, ndim=1] idx
        idx = np.random.randint(0, self.H, self.popsize)

        self.CR = np.random.normal(self.M_CR[idx], 0.1)
        self.F = np.random.normal(self.M_F[idx], 0.1)

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
        cdef int iter
        cdef np.ndarray[np.float64_t, ndim=2] new_population, all_trials
        cdef np.ndarray[np.float64_t, ndim=1] new_fitness, all_trial_fitness, mutant, trial

        self._initialize_population()
        for iter in range(self.maxiter + 1):
            self._adapt_parameters()
            new_population = np.empty_like(self.population)
            new_fitness = np.empty_like(self.fitness)
            S_CR = []
            S_F = []
            weights = []
            weights_Lehmer = []

            all_trials = np.empty_like(self.population)
            for i in range(self.popsize):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant, self.CR[i])
                all_trials[i] = trial

            all_trials = np.array(all_trials)
            enforce_bounds(all_trials, self.bounds, "random-leftover")

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

                self.M_CR = np.random.choice(np.concatenate((self.M_CR, np.random.normal(muCR, stdCR, len(S_CR)))), self.H, replace=False)
                self.M_F = np.random.choice(np.concatenate((self.M_F, np.random.normal(muF, stdF, len(S_F)) )), self.H, replace=False)

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

            if self.relTol != 0.0 : 
                if (np.max(self.fitness)-np.min(self.fitness))/ np.average(self.fitness) <= self.relTol:
                    break  

        return self.minionResult

