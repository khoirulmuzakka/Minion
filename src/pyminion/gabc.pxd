import numpy as np 
cimport numpy as np 
from .minimizer_base cimport *

cdef class GABC(MinimizerBase):
    cdef public:
        np.ndarray population
        np.ndarray values
        int population_size
        int max_iter
        float c
        float mutation_prob, mutation_rate
        int patience, no_improvement_counter
        np.ndarray best_solution
        float best_value
        int function_evaluations
        bint verbose

    cpdef np.ndarray[np.float64_t, ndim=1] mutate(self, np.ndarray[np.float64_t, ndim=1] candidate, np.ndarray[np.float64_t, ndim=2] new_candidates)
    cpdef MinionResult  optimize(self)
