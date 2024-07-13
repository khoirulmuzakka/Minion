
import numpy as np 
cimport numpy as np


cdef class MinionResult:
    """
    A class to encapsulate the results of an optimization process.

    This class holds the output from optimization algorithms, providing
    a structured way to access the results, including the best solution,
    objective function value, iteration count, and other relevant information.

    Attributes:
    ----------
    x : np.ndarray
        The best parameter vector found during optimization.
    fun : float
        The value of the objective function at the best solution.
    nit : int
        The number of iterations performed by the optimization algorithm.
    nfev : int
        The number of function evaluations made during the optimization.
    success : bool
        A flag indicating whether the optimization was successful.
    message : str
        A message providing additional information about the optimization outcome.

    Methods:
    -------
    __getitem__(key)
        Retrieves the value associated with the given key from the result.
    __setitem__(key, value)
        Sets the value for the given key in the result.
    __contains__(key)
        Checks if a key is present in the result.
    keys()
        Returns the keys of the result as a view.
    items()
        Returns a view of the result's items (key-value pairs).
    values()
        Returns a view of the result's values.
    get(key, default=None)
        Retrieves the value for the given key, returning a default if not found.
    """
    def __cinit__(self):
        self.x = np.array([], dtype=np.float64)
        self.fun = 0.0
        self.nit = 0
        self.nfev = 0
        self.success = False
        self.message = ""

    def __init__(self, x=None, fun=None, nit=None, nfev=None, success=None, message=''):
        if x is not None:
            self.x = np.asarray(x, dtype=np.float64)
        if fun is not None:
            self.fun = fun
        if nit is not None:
            self.nit = nit
        if nfev is not None:
            self.nfev = nfev
        if success is not None:
            self.success = success
        self.message = message

    def __repr__(self):
        return (f"MinionResult(x={self.x}, fun={self.fun}, nit={self.nit}, "
                f"nfev={self.nfev}, success={self.success}, message={self.message})")

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


cdef class MinimizerBase:
    """
    Base class for minimization algorithms.

    This class provides a common interface for various optimization 
    methods, such as Genetic Algorithms (GA), Differential Evolution (DE),
    and Particle Swarm Optimization (PSO). It requires a vectorized target 
    fitness function, bounds for the optimization parameters, an 
    initial guess, and optional arguments for customization.

    Parameters:
    ----------
    func : callable
        The objective function to minimize. It should accept a parameter 
        vector and return a scalar value.
    bounds : list of tuples
        The bounds for each parameter as a list of (lower_bound, upper_bound) pairs.
    x0 : array_like, optional
        Initial guess for the parameters.
    args : tuple, optional
        Additional arguments to pass to the objective function.
    callback : callable, optional
        A function to call after each iteration, useful for monitoring 
        progress or implementing custom stopping criteria.
    relTol : float, optional
        Relative tolerance for termination criteria (default is 0.0001).
    """

    def __init__(self,  object func, object bounds, x0=None, args=(),  object callback= None,
             double relTol= 0.0001, int maxevals=100000) : 
        self.func = func 
        try : self.bounds = np.array(bounds).astype(np.float64)
        except : raise ValueError("Invalid bounds.")
        if self.bounds.shape[1]!=2 : raise ValueError("Invalid bounds. Bounds must be a list of (lower_bound, upper_bound).")
        self.dim = self.bounds.shape[0]
        self.funcArgs = args 
        self.x0 = x0 
        if self.x0 is not None : 
            if len(self.x0) != self.dim : raise ValueError("x0 must have the same dimension as the length of the bounds.")   
        self.callback = callback  
        self.relTol = relTol
        self.maxevals = maxevals
        self.minionResult = MinionResult(x=x0, fun=None, nit=0, 
                                nfev=0, success=False, message='Optimization is not yet started.')
        self.history = []
        
    cpdef MinionResult optimize(self):
        raise NotImplementedError("Subclasses should implement this method to perform optimization")
