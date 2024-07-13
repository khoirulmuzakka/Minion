import numpy as np 
cimport numpy as np 
from .de_base cimport DE_Base 
from .minimizer_base cimport MinionResult

cdef class M_LJADE_AMR (DE_Base):
    """
    jDE optimization algorithm implementation using Cython for performance.
    """
    cdef public np.ndarray F
    cdef public np.ndarray  CR
    cdef public double meanCR, meanF, c
    cpdef void _adapt_parameters(self)
    cpdef MinionResult optimize(self)