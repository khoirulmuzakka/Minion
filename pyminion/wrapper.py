import sys
import os 

current_file_directory = os.path.dirname(os.path.abspath(__file__))
custom_path = os.path.join(current_file_directory, '../lib/Release/')
sys.path.append(custom_path)

import numpy as np
from pyminioncpp import M_LJADE_AMR as cppM_LJADE_AMR
from pyminioncpp import M_LSHADE_AMR as cppM_LSHADE_AMR
from pyminioncpp import MinionResult as cppMinionResult

  
class MinionResult:
    """
    @class MinionResult
    @brief A class to encapsulate the results of an optimization process.

    Stores the optimization result including solution vector, function value,
    number of iterations, number of function evaluations, success status, and a message.
    """

    def __init__(self, minRes):
        """
        @brief Constructor for MinionResult class.

        @param minRes The C++ MinionResult object to initialize from.
        """
        self.x = minRes.x
        self.fun = minRes.fun
        self.nit = minRes.nit
        self.nfev = minRes.nfev
        self.success = minRes.success
        self.message = minRes.message
        self.result = minRes

    def __repr__(self):
        """
        @brief Get a string representation of the MinionResult object.

        @return String representation containing key attributes.
        """
        return (f"MinionResult(x={self.x}, fun={self.fun}, nit={self.nit}, "
                f"nfev={self.nfev}, success={self.success}, message={self.message})")
    

class CalllbackWrapper: 
    """
    @class CalllbackWrapper
    @brief Wrap a Python function that takes cppMinionResult as an argument to work with MinionResult.

    Convert a callback function from working with cppMinionResult to MinionResult.
    """

    def __init__(self, callback):
        """
        @brief Constructor for CalllbackWrapper.

        @param callback Callback function that takes cppMinionResult as argument.
        """
        self.callback = callback

    def __call__(self, minRes):
        """
        @brief Call operator to invoke the callback function.

        @param minRes MinionResult object to pass to callback function.
        @return Result of the callback function.
        """
        minionResult = MinionResult(minRes)
        return self.callback(minionResult)
    

class MinimizerBase:
    """
    @class MinimizerBase
    @brief Base class for minimization algorithms.

    Provides common functionality for optimization algorithms.
    """

    def __init__(self,  func, bounds, data=None,  x0=None, relTol= 0.0001, maxevals=100000, callback= None, boundStrategy="reflect-random", seed=None) : 
        """
        @brief Constructor for MinimizerBase class.

        @param func Objective function to minimize.
        @param bounds Bounds for the decision variables.
        @param data Additional data to pass to the objective function.
        @param x0 Initial guess for the solution.
        @param relTol Relative tolerance for convergence.
        @param maxevals Maximum number of function evaluations.
        @param callback Callback function called after each iteration.
        @param boundStrategy Strategy when bounds are violated. Available strategy : "random", "reflect", "reflect-random", "clip".
        @param seed Seed for the random number generator.
        """

        self.func = func 
        self.bounds = self._validate_bounds(bounds)
        self.x0 = x0 
        if self.x0 is not None : 
            if len(self.x0) != len(self.bounds) : raise ValueError("x0 must have the same dimension as the length of the bounds.")   
        self.x0cpp = self.x0 if self.x0 is not None else []
        self.data = data

        self.callback = callback  
        self.cppCallback = CalllbackWrapper(self.callback) if callback is not None else None

        self.relTol = relTol
        self.maxevals = maxevals
        self.seed = seed if seed is not None else -1
        self.history = []
        self.minionResult = None
        self.boundStrategy = boundStrategy

    def _validate_bounds(self, bounds):
        """
        @brief Validate the bounds format.

        @param bounds Bounds for the decision variables.
        @return Validated bounds in the required format.
        @throws ValueError if bounds are invalid.
        """

        try:
            bounds = np.array(bounds)
        except:
            raise ValueError("Invalid bounds.")
        if np.any(bounds[:, 0]>= bounds[:,1]): raise ValueError ("upper bound must be larger than lower bound.")
        if bounds.shape[1] != 2:
            raise ValueError("Invalid bounds. Bounds must be a list of (lower_bound, upper_bound).")
        return [(b[0], b[1]) for b in bounds]


class M_LJADE_AMR(MinimizerBase):
    """
    @class M_LJADE_AMR
    @brief Implementation of the modified JADE with linear population size reduction with adaptive mutation rate (M-LJADE-AMR) algorithm.

    Inherits from MinimizerBase and implements the optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, population_size=30, maxevals=100000, 
                 strategy="current_to_pbest1bin", relTol=0.0, minPopSize=10, c=0.5, callback=None, boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for M_LJADE_AMR.

        @param func Objective function to minimize.
        @param bounds Bounds for the decision variables.
        @param data Additional data to pass to the objective function.
        @param x0 Initial guess for the solution.
        @param population_size Population size.
        @param maxevals Maximum number of function evaluations.
        @param strategy DE strategy to use.
        @param relTol Relative tolerance for convergence.
        @param minPopSize Minimum population size.
        @param c Control parameter for M-LJADE-AMR.
        @param callback Callback function called after each iteration.
        @param boundStrategy Strategy when bounds are violated. Available strategy : "random", "reflect", "reflect-random", "clip".
        @param seed Seed for the random number generator.
        """

        super().__init__(func, bounds, data, x0, relTol, maxevals, callback, boundStrategy, seed )
        self.population_size = population_size
        self.strategy = strategy
        self.minPopSize = minPopSize
        self.c = c
        self.optimizer = cppM_LJADE_AMR(self.func, self.bounds, self.data, self.x0cpp, population_size, maxevals, 
                                        strategy, relTol, minPopSize, c, self.cppCallback, boundStrategy, self.seed)
    
    def optimize(self):
        """
        @brief Optimize the objective function using M-LJADE-AMR.

        @return MinionResult object containing the optimization results.
        """
        self.minionResult = MinionResult(self.optimizer.optimize())
        self.history = [MinionResult(res) for res in self.optimizer.history]
        self.muCR = self.optimizer.muCR
        self.muF = self.optimizer.muF
        self.stdCR = self.optimizer.muCR
        self.stdF = self.optimizer.muF
        return self.minionResult
    
class M_LSHADE_AMR(MinimizerBase):
    """
    @class M_LSHADE_AMR
    @brief Implementation of the modified SHADE with linear population size reduction with adaptive mutation rate (M-LSHADE-AMR) algorithm.

    Inherits from MinimizerBase and implements the optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, population_size=30, maxevals=100000, 
                 strategy="current_to_pbest1bin", relTol=0.0, minPopSize=10, memeorySize=30, callback=None, boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for M_LSHADE_AMR.

        @param func Objective function to minimize.
        @param bounds Bounds for the decision variables.
        @param data Additional data to pass to the objective function.
        @param x0 Initial guess for the solution.
        @param population_size Population size.
        @param maxevals Maximum number of function evaluations.
        @param strategy DE strategy to use.
        @param relTol Relative tolerance for convergence.
        @param minPopSize Minimum population size.
        @param memorySize memory size for CR and F.
        @param callback Callback function called after each iteration.
        @param boundStrategy Strategy when bounds are violated. Available strategy : "random", "reflect", "reflect-random", "clip".
        @param seed Seed for the random number generator.
        """

        super().__init__(func, bounds, data, x0, relTol, maxevals, callback, boundStrategy, seed )
        self.population_size = population_size
        self.strategy = strategy
        self.minPopSize = minPopSize
        self.memorySize=memeorySize
        self.optimizer = cppM_LSHADE_AMR(self.func, self.bounds, self.data, self.x0cpp, population_size, maxevals, 
                                        strategy, relTol, minPopSize, self.memorySize, self.cppCallback, self.boundStrategy, self.seed)
    
    def optimize(self):
        """
        @brief Optimize the objective function using M-LSHADE-AMR.

        @return MinionResult object containing the optimization results.
        """
        self.minionResult = MinionResult(self.optimizer.optimize())
        self.history = [MinionResult(res) for res in self.optimizer.history]
        self.muCR = self.optimizer.muCR
        self.muF = self.optimizer.muF
        self.stdCR = self.optimizer.muCR
        self.stdF = self.optimizer.muF
        return self.minionResult