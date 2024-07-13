import numpy as np
cimport numpy as np


cdef class MinionResult : 
    cdef public :
        np.ndarray  x
        double fun
        int nit
        int nfev
        bint success
        str message


cdef class MinimizerBase:
    cdef public  : 
        np.ndarray  bounds, x0
        int maxevals, dim
        double relTol
        object func, callback
        tuple funcArgs
        MinionResult minionResult
        list history

    cpdef MinionResult optimize(self)