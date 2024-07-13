import numpy as np 
cimport numpy as np 

from .utility cimport latin_hypercube_sampling
from .minimizer_base cimport MinimizerBase

cdef class DE_Base (MinimizerBase) : 
    """
    Base class for all variants of DE.
    """
    def __init__(self,  func, bounds, args=(),  x0=None, int population_size=20, int maxevals=1000000,
                         str strategy='pbest1bin', double relTol=0.0001, int minPopSize=5, 
                         object callback=None, seed=None) :

        super(DE_Base, self).__init__(func, bounds, x0, args, callback, relTol, maxevals)

        if population_size < 10 : population_size =10
        self.original_popsize = population_size
        self.popsize = population_size
        self.minPopSize = minPopSize
        if self.minPopSize > self.original_popsize : raise ValueError("minPopSize must be smaller or equal to population_size.")
        self.popDecrease = True if self.minPopSize != self.original_popsize else False
        self.maxiter = self.getMaxIter()
        self.Nevals = 0
        self.rangeScale = 1.0
        self.strategy = strategy 
        if seed is not None: np.random.seed(seed)
        self.muCR = []
        self.muF = [] 
        self.stdCR = []
        self.stdF= []
        self.use_clip = False

    cpdef int getMaxIter(self) : 
        if not self.popDecrease  : return int(self.maxevals/self.original_popsize)
        cdef int i=0, n=self.popsize, max_iters = 0
        while i < self.maxevals:
            max_iters += 1
            n = round(self.original_popsize + (self.minPopSize - self.original_popsize) / self.maxevals * i )
            i += n
        return max_iters

    cpdef void _initialize_population(self):
        cdef np.ndarray[np.float64_t, ndim=1] lower_bounds, upper_bounds
        self.history = []
        lower_bounds, upper_bounds = self.bounds.T
        #self.population = np.random.rand(self.popsize, self.dim) * (upper_bounds - lower_bounds) + lower_bounds
        
        bounds_ = self.bounds.copy()
        midpoints = (bounds_[:, 0] + bounds_[:, 1]) / 2.0  # Calculate midpoints
        ranges = bounds_[:, 1] - bounds_[:, 0]  # Calculate original ranges
        new_ranges = ranges * self.rangeScale  # Calculate new ranges

        # Calculate new lower and upper bounds
        new_lower = midpoints - new_ranges / 2.0
        new_upper = midpoints + new_ranges / 2.0

        # Update self.bounds with the new bounds
        bounds_[:, 0] = new_lower
        bounds_[:, 1] = new_upper

        self.population = latin_hypercube_sampling(bounds_, self.popsize)
        if self.x0 is not None : self.population[0] = self.x0
        self.fitness = self.func(self.population, *self.funcArgs)
        self.fitness = np.nan_to_num(self.fitness, nan=1e+100)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        self.Nevals += self.popsize
        self.history.append([self.best, self.best_fitness])

    cpdef np.ndarray[np.float64_t, ndim=1] _mutate(self, int idx):
        cdef int r1, r2, r3, pbestind, frac, p
        cdef np.ndarray[np.float64_t, ndim=1] mutant
        cdef np.ndarray[np.float64_t, ndim=1] x1, x2, x3, target

        lower_bounds, upper_bounds = self.bounds.T
        available_indices = np.delete(np.arange(self.popsize), idx)

        if self.strategy in ['best1bin', 'best1exp']:
            r1, r2 = np.random.choice(available_indices, 2, replace=False)
            mutant = self.best + self.F[idx] * (self.population[r1] - self.population[r2])
        elif self.strategy in ['rand1bin', 'rand1exp']:
            r1, r2, r3 = np.random.choice(available_indices, 3, replace=False)
            mutant = self.population[r1] + self.F[idx] * (self.population[r2] - self.population[r3])
        elif self.strategy in ['current_to_best1bin', 'current_to_best1exp']:
            r1, r2 = np.random.choice(available_indices, 2, replace=False)
            mutant = self.population[idx] + self.F[idx] * (self.best - self.population[idx]) + self.F[idx] * (self.population[r1] - self.population[r2])
        elif self.strategy in ['current_to_pbest1bin', 'current_to_pbest1exp']:
            frac = int(0.2*self.popsize)
            if frac <= 1 : p=1 
            else : p = np.random.choice([1+i for i in range(frac)])
            pbestind= np.random.choice(np.argsort(self.fitness)[:p])
            r1, r2 = np.random.choice(available_indices, 2, replace=False)
            mutant = self.population[idx] + self.F[idx] * (self.population[pbestind] - self.population[idx]) + self.F[idx] * (self.population[r1] - self.population[r2])
        else:
            raise ValueError(f"Unknown mutation strategy: {self.strategy}")
        return mutant

    cpdef np.ndarray[np.float64_t, ndim=1] _crossover_bin(self, np.ndarray[np.float64_t, ndim=1] target, np.ndarray[np.float64_t, ndim=1] mutant, double CR):
        cdef np.ndarray crossover_mask
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    cpdef np.ndarray[np.float64_t, ndim=1] _crossover_exp(self, np.ndarray[np.float64_t, ndim=1] target, np.ndarray[np.float64_t, ndim=1] mutant, double CR):
        cdef int n, L
        cdef np.ndarray[np.float64_t, ndim=1] trial
        cdef np.ndarray[np.int_t, ndim=1] J
        trial = target.copy()
        n = np.random.randint(0, self.dim)
        L = 0
        while (np.random.rand() < CR) and (L < self.dim):
            trial[n] = mutant[n]
            n = (n + 1) % self.dim
            L += 1
        return trial

    cpdef np.ndarray[np.float64_t, ndim=1] _crossover(self, np.ndarray[np.float64_t, ndim=1] target, np.ndarray[np.float64_t, ndim=1] mutant, double CR):
        if 'bin' in self.strategy:
            return self._crossover_bin(target, mutant, CR)
        elif 'exp' in self.strategy:
            return self._crossover_exp(target, mutant, CR)
        else:
            raise ValueError(f"Unknown crossover strategy in mutation strategy: {self.strategy}")        
