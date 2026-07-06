"""BBOB 2009 benchmark wrappers."""

from __future__ import annotations

from .minionpycpp import BBOB2009Problem as cppBBOB2009Problem

__all__ = ["BBOB2009Functions", "BBOB2009Problem"]


class BBOB2009Functions:
    """Provides access to the BBOB 2009 benchmark test functions.

    This class exposes the COCO BBOB 2009 suite through the compiled C++
    extension. The interface mirrors the CEC wrappers in this package: the
    wrapped C++ object is stored as ``cpp_func`` and the instance is callable.

    Available dimensions: **2, 5, 10, 20, 40**
    Available functions: **1-24**
    """

    def __init__(self, function_number, dimension, year=2009):
        """
        Initialize a `BBOB2009Functions` instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1-24).
        dimension : int
            The problem dimensionality (must be one of {2, 5, 10, 20, 40}).
        year : int, optional
            Must be 2009. Kept for API symmetry with the C++ constructor.
        """
        self.cpp_func = cppBBOB2009Problem(int(function_number), int(dimension), int(year))
        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.year = int(year)
        self.bounds = [tuple(b) for b in self.cpp_func.bounds]
        self.initial_solution = list(self.cpp_func.initial_solution)
        self.best_value = float(self.cpp_func.best_value)
        self.id = str(self.cpp_func.id)
        self.name = str(self.cpp_func.name)

    def __call__(self, X):
        """
        Evaluate the selected BBOB 2009 test function.

        Parameters
        ----------
        X : list[list[float]]
            Input vectors to evaluate.

        Returns
        -------
        list
            A vector of function values corresponding to each input vector.
        """
        return self.cpp_func(X)

    def get_bounds(self):
        """Return the box constraints as ``[(lb, ub), ...]``."""
        return list(self.bounds)


BBOB2009Problem = BBOB2009Functions
