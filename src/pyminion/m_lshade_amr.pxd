import numpy as np
cimport numpy as np
from .de_base cimport DE_Base
from .minimizer_base cimport MinionResult

cdef class M_LSHADE_AMR (DE_Base):
    """
    SHADE optimization algorithm implementation using Cython for performance.
    """
    cdef public int H
    cdef public np.ndarray  M_CR
    cdef public np.ndarray  M_F
    cdef public np.ndarray CR
    cdef public np.ndarray F
    cdef public object memProb

    cpdef void _initialize_population(self)
    cpdef void _adapt_parameters(self)
    cpdef MinionResult optimize(self)


