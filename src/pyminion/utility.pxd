import numpy as np 
cimport numpy as np

cdef np.ndarray latin_hypercube_sampling(object bounds, int population_size)
cdef tuple getMeanStd(object arr, object weight)
cdef void enforce_bounds(np.ndarray[np.float64_t, ndim=2] new_candidates, np.ndarray[np.float64_t, ndim=2] bounds, object strategy)