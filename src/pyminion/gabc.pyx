import numpy as np 
cimport numpy as np 

from .utility cimport *
from .minimizer_base cimport *

cdef class GABC (MinimizerBase):
    """
    implementation of Global Best Artificial Bee Colony (GABC) optimization algorithm.
    This assume that the function is to be minimize, and always positive.
    
    Parameters:
    -----------
    objective_function : callable
        The objective function to be minimized. It should take an array (population) 
        of shape (n, N) where `n` is the population size and `N` is the number of dimensions.
    bounds : array_like
        Bounds for variables. It should be a list of tuples specifying the (min, max) 
        for each dimension.
    x0 : array_like, optional
        Initial guess for the population (default is None).
    args : tuple, optional
        Additional arguments to be passed to the objective function.
    population_size : int, optional
        The number of individuals in the population (default is 50).
    max_iter : int, optional
        The maximum number of iterations to perform (default is 1000).
    scoutBeeProb : float, optional 
        Define the fraction of population which defines the scout bee.
    influence_global_best : float, optional
        The influence of factor of the population global best.
    mutation_prob : float, optional
        The probability of mutation (default is 0.1).
    mutation_rate : float, optional
        The rate of mutation (default is 0.1).
    relTol : float, optional 
        Define the relative tolerance for stopping. The stopping condition is np.ptp(values)/np.average(values) < relTol.
    verbose : bool, optional
        If True, prints progress messages (default is False).
    patience : int, optional
        Number of iterations with no improvement before stopping (default is 50).
    """
    def __init__(self, func, bounds, x0=None, args=(), 
                 population_size=20, maxevals=1000000, influence_global_best=2.0, 
                 mutation_prob=0.1, mutation_rate=0.05, patience=50, relTol= 0.00001, callback= None, verbose=False,):
        
        super(GABC, self).__init__(func, bounds, x0, args, callback, relTol, maxevals)
        self.population_size = int(population_size)
        self.max_iter = round(self.maxevals/(2*self.population_size))
        self.verbose = verbose
        self.c = influence_global_best
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.patience = patience
        self.population = latin_hypercube_sampling(self.bounds, self.population_size)
        if self.x0 is not None:
            assert len(x0) == self.dim
            self.population[0] = np.array(self.x0, dtype=np.float64)

        self.values = self.func(self.population, *self.funcArgs)
        self.best_solution = self.population[np.argmin(self.values)]
        self.best_value = np.min(self.values)
        self.function_evaluations = population_size  # Initial evaluations during population setup
        self.history.append([self.best_solution, self.best_value])
        self.no_improvement_counter = 0

    cpdef np.ndarray[np.float64_t, ndim=1] mutate(self, np.ndarray[np.float64_t, ndim=1] candidate, np.ndarray[np.float64_t, ndim=2] new_candidates):
        """
        Apply mutation to a candidate solution. The mutation is inspired by differential eviolution.
        """
        cdef int d
        for d in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                agents = np.random.choice(self.population_size, size=3, replace=False)
                agents = new_candidates[agents]
                #candidate[d] += self.mutation_rate * (np.random.rand() - 0.5) * (self.bounds[d, 1] - self.bounds[d, 0])
                candidate[d] =  agents[0][d]+self.mutation_rate * np.random.rand()*(agents[1][d]-agents[2][d])
        return candidate

    cpdef MinionResult optimize(self):
        """
        Perform the optimization process using the Global Best ABC algorithm.
        """
        cdef int i, j
        cdef np.ndarray[np.float64_t, ndim=2] phi, new_candidates, nc
        cdef np.ndarray[np.float64_t, ndim=1] probabilities
        cdef int min_idx
        cdef np.ndarray[np.float64_t, ndim=1] new_values, fitness
        cdef np.ndarray[long, ndim=1] partner_indices, onlooker_indices

        for i in range(self.max_iter):
            # Phase 1: Employed bees phase
            phi = np.random.uniform(low=-1, high=1, size=(self.population_size, self.dim))
            r = np.random.rand(self.population_size, 1)
            partner_indices = np.random.choice(self.population_size, size=self.population_size, replace=False)
            new_candidates = self.population + phi * (self.population - self.population[partner_indices])+ self.c*r * (self.best_solution - self.population)

            nc = new_candidates.copy()
            # Apply mutation
            for j in range(self.population_size):
                new_candidates[j] = self.mutate(new_candidates[j], nc)
                
            # Enforce bounds on the new candidates
            enforce_bounds(new_candidates, self.bounds, "random-leftover")
            
            new_values = self.func(new_candidates, *self.funcArgs)
            new_values = np.nan_to_num(new_values, nan=1e+100)
            self.function_evaluations += self.population_size
            
            # Greedy selection
            improved = new_values < self.values
            self.population[improved] = new_candidates[improved]
            self.values[improved] = new_values[improved]

            self.values = np.nan_to_num(self.values, nan=1e+100)
            
            # Update best solution
            min_idx = np.argmin(self.values)
            if self.values[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx]
                self.best_value = self.values[min_idx]
                self.no_improvement_counter = 0
            else:
                self.no_improvement_counter += 1
            
            # Phase 2: Onlooker bees phase
            min_value = np.min(self.values)
            shifted_values = self.values - min_value + 1e-10 
            fitness = 1 / shifted_values  # Fitness calculation for minimization

            probabilities = fitness / fitness.sum()
            onlooker_indices = np.random.choice(self.population_size, size=self.population_size, p=probabilities)
            phi = np.random.uniform(low=-1, high=1, size=(self.population_size, self.dim))
            partner_indices = np.random.choice(self.population_size, size=self.population_size, replace=False)
            r = np.random.rand(self.population_size, 1)

            new_candidates = self.population[onlooker_indices] + phi * (self.population[onlooker_indices] - self.population[partner_indices])+self.c* r * (self.best_solution - self.population[onlooker_indices])

            # Apply mutation
            nc = new_candidates.copy()
            for j in range(self.population_size):
                new_candidates[j] = self.mutate(new_candidates[j], nc)
                
            # Enforce bounds on the new candidates
            enforce_bounds(new_candidates, self.bounds, "random-leftover")
            
            new_values = self.func(new_candidates, *self.funcArgs)
            new_values = np.nan_to_num(new_values, nan=1e+100)
            self.function_evaluations += self.population_size
            
            # Greedy selection
            improved = new_values < self.values[onlooker_indices]
            self.population[onlooker_indices[improved]] = new_candidates[improved]
            self.values[onlooker_indices[improved]] = new_values[improved]
            
            # Update best solution
            min_idx = np.argmin(self.values)
            if self.values[min_idx] < self.best_value:
                self.best_solution = self.population[min_idx]
                self.best_value = self.values[min_idx]
                self.no_improvement_counter = 0
            else:
                self.no_improvement_counter += 1
            
            self.history.append([self.best_solution, self.best_value])
            self.minionResult = MinionResult(x=self.best_solution, fun=self.best_value, success=True, nit=i+1, nfev=self.function_evaluations)
            if self.callback is not None : self.callback(self.minionResult)
            # Check for convergence
            if self.no_improvement_counter >= self.patience:
                if self.verbose: print(f"Convergence reached at iteration {i}")
                break

            if self.verbose and (i % 20 == 0):
                print(f"Iteration {i}: Best Value = {self.best_value}")

        return self.minionResult