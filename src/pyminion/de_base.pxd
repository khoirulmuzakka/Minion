import numpy as np
cimport numpy as np
from .minimizer_base cimport MinimizerBase

cdef class DE_Base(MinimizerBase):
    """
    SHADE optimization algorithm implementation using Cython for performance.
    """
    cdef public int maxiter, Nevals, original_popsize
    cdef public int popsize
    cdef public double rangeScale
    cdef public int minPopSize
    cdef public np.ndarray population
    cdef public np.ndarray fitness
    cdef public int best_idx
    cdef public np.ndarray best
    cdef public double best_fitness
    cdef public object seed
    cdef public str strategy
    cdef public list muCR, muF, stdCR, stdF
    cdef public bint popDecrease, use_clip

    cpdef int getMaxIter(self)
    cpdef np.ndarray[np.float64_t, ndim=1] _crossover_bin(self, np.ndarray[np.float64_t, ndim=1] target, np.ndarray[np.float64_t, ndim=1] mutant, double CR)
    cpdef np.ndarray[np.float64_t, ndim=1] _crossover_exp(self, np.ndarray[np.float64_t, ndim=1] target, np.ndarray[np.float64_t, ndim=1] mutant, double CR)
    cpdef void _initialize_population(self)
    cpdef np.ndarray[np.float64_t, ndim=1] _mutate(self, int idx)
    cpdef np.ndarray[np.float64_t, ndim=1] _crossover(self, np.ndarray[np.float64_t, ndim=1] target, np.ndarray[np.float64_t, ndim=1] mutant, double CR)
