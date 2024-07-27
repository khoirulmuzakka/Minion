import sys
import os 

current_file_directory = os.path.dirname(os.path.abspath(__file__))
custom_path = os.path.join(current_file_directory, '../lib/Release/')
sys.path.append(custom_path)

import numpy as np
from pyminioncpp import MFADE as cppMFADE
from pyminioncpp import LSHADE as cppLSHADE
from pyminioncpp import EBR_LSHADE as cppEBR_LSHADE
from pyminioncpp import MinionResult as cppMinionResult
from pyminioncpp import GWO_DE as cppGWO_DE
from pyminioncpp import Powell as cppPowell  # Import Powell
from pyminioncpp import NelderMead as cppNelderMead 
from pyminioncpp import CEC2020Functions as cppCEC2020Functions
from pyminioncpp import CEC2022Functions as cppCEC2022Functions


  
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
    
class CEC2020Functions:
    """
    @class CEC2020Functions
    @brief A class to encapsulate CEC2020 test functions.

    Allows the loading of shift and rotation matrices and the evaluation of test functions.
    """

    def __init__(self, function_number, dimension):
        """
        @brief Constructor for CEC2020Functions class.

        @param function_number Function number (1-10).
        @param dimension Dimension of the problem.
        """
        if function_number not in range(1, 11) : raise Exception("Function number must be between 1-12.")
        if int(dimension) not in [2, 10, 20] : raise Exception("Dimension must be 2, 10, or 20.")
        self.cpp_func = cppCEC2020Functions(function_number, int(dimension))

    def __call__(self, X, data=None):
        """
        @brief Evaluate the CEC2020 test function.

        @param X Input vectors to evaluate.
        @return Vector of function values corresponding to each input vector.
        """
        return self.cpp_func(X)

class CEC2022Functions:
    """
    @class CEC2022Functions
    @brief A class to encapsulate CEC2022 test functions.

    Allows the loading of shift and rotation matrices and the evaluation of test functions.
    """

    def __init__(self, function_number, dimension):
        """
        @brief Constructor for CEC2020Functions class.

        @param function_number Function number (1-10).
        @param dimension Dimension of the problem.
        """
        if function_number not in range(1, 11) : raise Exception("Function number must be between 1-10.")
        if int(dimension) not in [2, 10, 20] : raise Exception("Dimension must be 2, 10, or 20.")
        self.cpp_func = cppCEC2022Functions(function_number, int(dimension))

    def __call__(self, X, data=None):
        """
        @brief Evaluate the CEC2022 test function.

        @param X Input vectors to evaluate.
        @return Vector of function values corresponding to each input vector.
        """
        return self.cpp_func(X)
    
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

    
class MFADE(MinimizerBase):
    """
    @class MFADE : Fully Adaptive Differential Evolution with Memory
    @brief Implementation of the MFADE algorithm.

    Inherits from MinimizerBase and implements the optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, population_size=30, maxevals=100000, 
                 strategy="current_to_pbest1bin", relTol=0.0, minPopSize=10, memeorySize=30, callback=None, boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for MFADE.

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
        self.optimizer = cppMFADE(self.func, self.bounds, self.data, self.x0cpp, population_size, maxevals, 
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


class EBR_LSHADE(MinimizerBase):
    """
    @class EBR_LSHADE : EBR_LSHADE : Exclusion-Based Restart LSHADE algorithm
    @brief Implementation of the EBR-LSHADE algorithm.

    Inherits from MinimizerBase and implements the optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, population_size=0, maxevals=100000, 
                 relTol_firstRun=0.01, minPopSize=5, memorySize=50, callback=None, max_restarts=10, 
                 startRefine=0.75, boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for EBR_LSHADE.

        @param func Objective function to minimize.
        @param bounds Bounds for the decision variables.
        @param data Additional data to pass to the objective function.
        @param x0 Initial guess for the solution.
        @param population_size Population size. 
        @param maxevals Maximum number of function evaluations.
        @param relTol_firstRun Relative tolerance for convergence in the first run.
        @param minPopSize Minimum population size.
        @param memorySize Memory size for CR and F.
        @param callback Callback function called after each iteration.
        @param max_restarts Maximum number of restarts.
        @param startRefine Start refinement threshold.
        @param boundStrategy Strategy when bounds are violated. Available strategies: "random", "reflect", "reflect-random", "clip".
        @param seed Seed for the random number generator.
        """

        super().__init__(func, bounds, data, x0, 0.0, maxevals, callback, boundStrategy, seed)
        self.population_size = population_size
        self.relTol_firstRun = relTol_firstRun
        self.minPopSize = minPopSize
        self.memorySize = memorySize
        self.max_restarts = max_restarts
        self.startRefine = startRefine
        self.optimizer = cppEBR_LSHADE(self.func, self.bounds, self.data, self.x0cpp, population_size, maxevals, 
                                      relTol_firstRun, minPopSize, memorySize, self.cppCallback, max_restarts, 
                                      startRefine, self.boundStrategy, self.seed)
    
    def optimize(self):
        """
        @brief Optimize the objective function using EBR-LSHADE.

        @return MinionResult object containing the optimization results.
        """
        self.minionResult = MinionResult(self.optimizer.optimize())
        self.history = [MinionResult(res) for res in self.optimizer.history]
        self.muCR = self.optimizer.muCR
        self.muF = self.optimizer.muF
        self.stdCR = self.optimizer.stdCR
        self.stdF = self.optimizer.stdF
        return self.minionResult
    
class LSHADE(MinimizerBase):
    """
    @class LSHADE 
    @brief Implementation of the LSHADE algorithm.

    Inherits from MinimizerBase and implements the optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, population_size=30, maxevals=100000, 
                 strategy="current_to_pbest1bin", relTol=0.0, minPopSize=10, memeorySize=30, callback=None, boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for LSHADE.

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
        self.optimizer = cppLSHADE(self.func, self.bounds, self.data, self.x0cpp, population_size, maxevals, 
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

class GWO_DE(MinimizerBase):
    """
    @class GWO_DE
    @brief Implementation of the Grey Wolf Optimizer with Differential Evolution (GWO-DE) algorithm.

    Inherits from MinimizerBase and implements the optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, population_size=20, maxevals=1000, F=0.5, CR=0.7, elimination_prob=0.1, relTol=0.0001, callback=None, boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for GWO_DE.

        @param func Objective function to minimize.
        @param bounds Bounds for the decision variables.
        @param data Additional data to pass to the objective function.
        @param x0 Initial guess for the solution.
        @param population_size Population size.
        @param maxevals Maximum number of function evaluations.
        @param F Differential evolution scaling factor.
        @param CR Crossover probability.
        @param elimination_prob Probability of elimination.
        @param relTol Relative tolerance for convergence.
        @param callback Callback function called after each iteration.
        @param boundStrategy Strategy when bounds are violated. Available strategies: "random", "reflect", "reflect-random", "clip".
        @param seed Seed for the random number generator.
        """

        super().__init__(func, bounds, data, x0, relTol, maxevals, callback, boundStrategy, seed)
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.elimination_prob = elimination_prob
        self.optimizer = cppGWO_DE(self.func, self.bounds, self.x0cpp, population_size, maxevals, F, CR, elimination_prob, relTol, boundStrategy, self.seed, self.data, self.cppCallback)
    
    def optimize(self):
        """
        @brief Optimize the objective function using GWO-DE.

        @return MinionResult object containing the optimization results.
        """
        self.minionResult = MinionResult(self.optimizer.optimize())
        self.history = [MinionResult(res) for res in self.optimizer.history]
        return self.minionResult
    
class Powell(MinimizerBase):
    """
    @class Powell
    @brief Implementation of Powell's method for multidimensional optimization.

    Inherits from MinimizerBase and implements the Powell optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, relTol=0.0001, maxevals=100000, callback=None,
                 boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for Powell.

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

        super().__init__(func, bounds, data, x0, relTol, maxevals, callback, boundStrategy, seed)
        self.optimizer = cppPowell(self.func, self.bounds, self.x0cpp, self.data, self.cppCallback, relTol, maxevals, boundStrategy, self.seed)

    def optimize(self):
        """
        @brief Optimize the objective function using Powell's method.

        @return MinionResult object containing the optimization results.
        """
        self.minionResult = MinionResult(self.optimizer.optimize())
        self.history = [MinionResult(res) for res in self.optimizer.history]
        return self.minionResult


class NelderMead(MinimizerBase):
    """
    @class AdaptiveNelderMead
    @brief Implementation of the Adaptive Nelder-Mead algorithm.

    Inherits from MinimizerBase and implements the Adaptive Nelder-Mead optimization algorithm.
    """

    def __init__(self, func, bounds, data=None, x0=None, relTol=0.0001, maxevals=100000, callback=None,
                 boundStrategy="reflect-random", seed=None):
        """
        @brief Constructor for AdaptiveNelderMead.

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

        super().__init__(func, bounds, data, x0, relTol, maxevals, callback, boundStrategy, seed)
        self.optimizer = cppNelderMead(self.func, self.bounds,  self.x0cpp, self.data, self.cppCallback, relTol, maxevals, boundStrategy, self.seed)

    def optimize(self):
        """
        @brief Optimize the objective function using Adaptive Nelder-Mead.

        @return MinionResult object containing the optimization results.
        """
        self.minionResult = MinionResult(self.optimizer.optimize())
        self.history = [MinionResult(res) for res in self.optimizer.history]
        return self.minionResult
