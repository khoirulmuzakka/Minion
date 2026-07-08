import numpy as np

from .minionpycpp import CEC2017Functions as cppCEC2017Functions
from .minionpycpp import CEC2014Functions as cppCEC2014Functions
from .minionpycpp import CEC20142017Functions as cppCEC20142017Functions
from .minionpycpp import CEC2019Functions as cppCEC2019Functions
from .minionpycpp import CEC2020Functions as cppCEC2020Functions
from .minionpycpp import CEC2022Functions as cppCEC2022Functions
from .minionpycpp import CEC2011Functions as cppCEC2011Functions


def _cec2011_problem02_bounds(ndim: int):
    lb = np.zeros(ndim)
    for idx in range(3, ndim):
        lb[idx] = -4 - 0.25 * int((idx - 4) / 3)

    ub = np.zeros(ndim)
    ub[0] = ub[1] = 4.0
    ub[2] = np.pi
    for idx in range(3, ndim):
        ub[idx] = 4 + 0.25 * int((idx - 4) / 3)
    return lb, ub


def _cec2011_problem05_bounds(ndim: int):
    lb = -1.0 * np.ones(ndim)
    lb[:3] = 0.0
    ub = np.zeros(ndim)
    ub[0] = ub[1] = 4.0
    ub[2] = np.pi
    for idx in range(3, ndim):
        ub[idx] = 4 + 0.25 * int((idx - 4) / 3)
    return lb, ub


def _cec2011_problem06_bounds(ndim: int):
    lb = -1.0 * np.ones(ndim)
    lb[:3] = 0.0
    ub = np.zeros(ndim)
    ub[0] = ub[1] = 4.0
    ub[2] = np.pi
    for idx in range(3, ndim, 3):
        ub[idx] = 4 + 0.25 * int((1 - 4) / 3)
        if idx + 1 < ndim:
            ub[idx + 1] = 4 + 0.25 * int((2 - 4) / 3)
        if idx + 2 < ndim:
            ub[idx + 2] = 4 + 0.25 * int((3 - 4) / 3)
    return lb, ub


def _np_from_str(data: str):
    return np.fromstring(data.strip().replace("\n", " "), sep=" ")


CEC2011_METADATA = {
    1: (6, np.full(6, -6.4), np.full(6, 6.35)),
    2: (30, *_cec2011_problem02_bounds(30)),
    3: (1, np.array([-0.6]), np.array([0.9])),
    4: (1, np.array([0.0]), np.array([5.0])),
    5: (30, *_cec2011_problem05_bounds(30)),
    6: (30, *_cec2011_problem06_bounds(30)),
    7: (20, np.zeros(20), (2 * np.pi) * np.ones(20)),
    8: (7, np.zeros(7), 15.0 * np.ones(7)),
    9: (
        126,
        np.zeros(126),
        _np_from_str(
            """
            0.217 0.024 0.076 0.892 0.128 0.25 0.058 0.112 0.062 0.082 0.035 0.09 0.032 0.095 0.022 0.175 0.032
            0.087 0.035 0.024 0.106 0.217 0.024 0.026 0.491 0.228 0.3 0.058 0.112 0.062 0.082 0.035 0.09 0.032
            0.095 0.022 0.175 0.032 0.087 0.035 0.024 0.106 0.216 0.024 0.076 0.216 0.216 0.216 0.058 0.112
            0.062 0.082 0.035 0.09 0.032 0.095 0.022 0.175 0.032 0.087 0.035 0.024 0.081 0.217 0.024 0.076
            0.228 0.228 0.228 0.058 0.112 0.062 0.082 0.035 0.09 0.032 0.095 0.022 0.025 0.032 0.087 0.035
            0.024 0.081 0.124 0.024 0.076 0.124 0.124 0.124 0.058 0.112 0.062 0.082 0.035 0.065 0.032 0.095
            0.022 0.124 0.032 0.087 0.035 0.024 0.106 0.116 0.024 0.076 0.116 0.116 0.116 0.058 0.087 0.062
            0.082 0.035 0.09 0.032 0.095 0.022 0.116 0.032 0.087 0.035 0.024 0.106
            """
        ),
    ),
        10: (12, np.array([0.2] * 6 + [-180.0] * 6), np.array([1.0] * 6 + [180.0] * 6)),
}


def _repeat_bounds(dimension: int, lower: float, upper: float):
    return [(float(lower), float(upper)) for _ in range(int(dimension))]


def _cec2019_bounds(function_number: int, dimension: int):
    if function_number == 1:
        return _repeat_bounds(dimension, -8192.0, 8192.0)
    if function_number == 2:
        return _repeat_bounds(dimension, -16384.0, 16384.0)
    if function_number == 3:
        return _repeat_bounds(dimension, -4.0, 4.0)
    return _repeat_bounds(dimension, -100.0, 100.0)


def _cec_global_optimum(year: int, function_number: int):
    if year == 2022:
        values = [300.0, 400.0, 600.0, 800.0, 900.0, 1800.0, 2000.0, 2200.0, 2300.0, 2400.0, 2600.0, 2700.0]
        if 1 <= function_number <= len(values):
            return values[function_number - 1]
    elif year == 2020:
        values = [100.0, 1100.0, 700.0, 1900.0, 1700.0, 1600.0, 2100.0, 2200.0, 2400.0, 2500.0]
        if 1 <= function_number <= len(values):
            return values[function_number - 1]
    elif year in (2017, 2014):
        if 1 <= function_number <= 30:
            return 100.0 * float(function_number)
    elif year == 2019:
        return 1.0
    return None


class _BenchmarkMetadataMixin:
    def get_bounds(self):
        return list(self.bounds)

    def get_f_opt(self):
        return self.f_opt

CEC2011_METADATA.update(
    {
        11: (120, np.array([10, 20, 30, 40, 50] * 24), np.array([75, 125, 175, 250, 300] * 24)),
        12: (
            240,
            np.array([150, 135, 73, 60, 73, 57, 20, 47, 20, 55] * 24),
            np.array([470, 460, 340, 300, 243, 160, 130, 120, 80, 55.1] * 24),
        ),
        13: (6, np.array([100, 50, 80, 50, 50, 50]), np.array([500, 200, 300, 150, 200, 120])),
        14: (
            13,
            np.array([0, 0, 0, 60, 60, 60, 60, 60, 60, 40, 40, 55, 55]),
            np.array([680, 360, 360, 180, 180, 180, 180, 180, 180, 120, 120, 120, 120]),
        ),
        15: (
            15,
            np.array([150, 150, 20, 20, 150, 135, 135, 60, 25, 25, 20, 20, 25, 15, 15]),
            np.array([455, 455, 130, 130, 470, 460, 465, 300, 162, 160, 80, 80, 85, 55, 55]),
        ),
        16: (
            40,
            _np_from_str(
                """
                36 36 60 80 47 68 110 135 135 130 94 94 125 125 125 125 220 220 242 242
                254 254 254 254 254 254 10 10 10 47 60 60 60 90 90 90 25 25 25 242
                """
            ),
            _np_from_str(
                """
                114 114 120 190 97 140 300 300 300 300 375 375 500 500 500 500 500 500 550 550
                550 550 550 550 550 550 150 150 150 97 190 190 190 200 200 200 110 110 110 550
                """
            ),
        ),
        17: (
            140,
            _np_from_str(
                """
                71 120 125 125 90 90 280 280 260 260 260 260 260 260 260 260 260 260 260 260
                260 260 260 260 280 280 280 280 260 260 260 260 260 260 260 260 120 120 423 423
                3 3 160 160 160 160 160 160 160 160 165 165 165 165 180 180 103 198 100 153 163
                95 160 160 196 196 196 196 130 130 137 137 195 175 175 175 175 330 160 160 200 56
                115 115 115 207 207 175 175 175 175 360 415 795 795 578 615 612 612 758 755 750
                750 713 718 791 786 795 795 795 795 94 94 94 244 244 244 95 95 116 175 2 4 15 9
                12 10 112 4 5 5 50 5 42 42 41 17 7 7 26
                """
            ),
            _np_from_str(
                """
                119 189 190 190 190 190 490 490 496 496 496 496 506 509 506 505 506 506 505 505
                505 505 505 505 537 537 549 549 501 501 506 506 506 506 500 500 241 241 774 769
                19 28 250 250 250 250 250 250 250 250 504 504 504 504 471 561 341 617 312 471 500
                302 511 511 490 490 490 490 432 432 455 455 541 536 540 538 540 574 531 531 542 132
                245 245 245 307 307 345 345 345 345 580 645 984 978 682 720 718 720 964 958 1007
                1006 1013 1020 954 952 1006 1013 1021 1015 203 203 203 379 379 379 190 189 194 321
                19 59 83 53 37 34 373 20 38 19 98 10 74 74 105 51 19 19 40
                """
            ),
        ),
        18: (96, np.array([5, 6, 10, 13] * 24), np.array([15, 15, 30, 25] * 24)),
        19: (96, np.array([5, 6, 10, 13] * 24), np.array([15, 15, 30, 25] * 24)),
        20: (96, np.array([5, 6, 10, 13] * 24), np.array([15, 15, 30, 25] * 24)),
        21: (
            26,
            _np_from_str(
                "1900 2.5 0 0 100 100 100 100 100 100 0.01 0.01 0.01 0.01 0.01 0.01 1.1 1.1 1.05 1.05 1.05 "
                "-{pi} -{pi} -{pi} -{pi} -{pi}".format(pi=np.pi)
            ),
            _np_from_str(
                "2200 5 1 1 500 500 500 500 500 600 0.99 0.99 0.99 0.99 0.99 0.99 6 6 6 6 6 "
                "{pi} {pi} {pi} {pi} {pi}".format(pi=np.pi)
            ),
        ),
        22: (
            22,
            _np_from_str(
                "-1000 3 0 0 100 100 30 400 800 0.01 0.01 0.01 0.01 0.01 1.05 1.05 1.15 1.7 "
                "-{pi} -{pi} -{pi} -{pi}".format(pi=np.pi)
            ),
            _np_from_str(
                "0 5 1 1 400 500 300 1600 2200 0.9 0.9 0.9 0.9 0.9 6 6 6.5 291 "
                "{pi} {pi} {pi} {pi}".format(pi=np.pi)
            ),
        ),
    }
)
class CEC2014Functions(_BenchmarkMetadataMixin):
    """
    Provides access to the CEC2014 benchmark test functions.

    This class implements 30 benchmark optimization problems from CEC 2014
    at various dimensions.

    Available dimensions: **2, 10, 20, 30, 50, 100**  
    Available functions: **1–30**
    """

    def __init__(self, function_number, dimension):
        """
        Initialize a CEC2014Functions instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1–30).
        dimension : int
            The problem dimensionality (must be one of {2, 10, 20, 30, 50, 100}).

        """
        if function_number not in range(1, 31) : raise Exception("Function number must be between 1-30.")
        if int(dimension) not in [2, 10, 20, 30, 50, 100] : raise Exception("Dimension must be 2, 10, 20, 30, 50, 100")
        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.bounds = _repeat_bounds(self.dimension, -100.0, 100.0)
        self.f_opt = _cec_global_optimum(2014, self.function_number)
        self.cpp_func = cppCEC2014Functions(self.function_number, self.dimension)

    def __call__(self, X):
        """
        Evaluate the selected CEC2014 test function.

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

class CEC20142017Functions(_BenchmarkMetadataMixin):
    """
    Provides access to the combined CEC2014 + CEC2017 benchmark suite.

    This class implements 60 benchmark optimization problems:
    F1-F30 map to CEC2014 and F31-F60 map to CEC2017.

    Available dimensions: **2, 10, 20, 30, 50, 100**
    Available functions: **1–60**
    """

    def __init__(self, function_number, dimension):
        """
        Initialize a CEC20142017Functions instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1–60).
            Functions 41–49 are not available for dimensions 2 and 20.
        dimension : int
            The problem dimensionality (must be one of {2, 10, 20, 30, 50, 100}).
        """
        if function_number not in range(1, 61):
            raise Exception("Function number must be between 1-60.")
        if int(dimension) not in [2, 10, 20, 30, 50, 100]:
            raise Exception("Dimension must be 2, 10, 20, 30, 50, 100")

        local_function = int(function_number) if int(function_number) <= 30 else int(function_number) - 30
        if int(dimension) in [2, 20] and local_function in range(11, 20):
            raise Exception("At dimension 2 and 20, function number 41-49 are not available")

        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.bounds = _repeat_bounds(self.dimension, -100.0, 100.0)
        self.f_opt = 100.0
        self.cpp_func = cppCEC20142017Functions(self.function_number, self.dimension)

    def __call__(self, X):
        """
        Evaluate the selected combined CEC2014/CEC2017 test function.

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

class CEC2017Functions(_BenchmarkMetadataMixin):
    """
    Provides access to the CEC2014 benchmark test functions.

    This class implements 30 benchmark optimization problems from CEC 2017
    at various dimensions.

    Available dimensions: **2, 10, 20, 30, 50, 100**  
    Available functions: **1–30**
    """

    def __init__(self, function_number, dimension):
        """
        Initialize a `CEC2017Functions` instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1–30).  
            **Note:** Functions 11–19 are not available for dimensions 2 and 20.
        dimension : int
            The problem dimensionality (must be one of {2, 10, 20, 30, 50, 100}).
        """
        if function_number not in range(1, 31) : raise Exception("Function number must be between 1-30.")
        if int(dimension) not in [2, 10, 20, 30, 50, 100] : raise Exception("Dimension must be 2, 10, 20, 30, 50, 100")
        if int(dimension)==20 and function_number in range (11, 20) : raise Exception ("At dimension 20, function number 11-19 are not available")
        if int(dimension)==2 and function_number in range (11, 20) : raise Exception ("At dimension 2, function number 11-19 are not available")
        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.bounds = _repeat_bounds(self.dimension, -100.0, 100.0)
        self.f_opt = _cec_global_optimum(2017, self.function_number)
        self.cpp_func = cppCEC2017Functions(self.function_number, self.dimension)

    def __call__(self, X):
        """
        Evaluate the selected CEC2014 test function.

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
    
class CEC2019Functions(_BenchmarkMetadataMixin):
    """
    Provides access to the CEC2019 benchmark test functions.

    This class implements 10 benchmark optimization problems from CEC 2019
    at various dimensions.

    Available functions: **1–10**
    """

    def __init__(self, function_number, dimension=None):
        """
        Initialize a CEC2019Functions instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1–10).
        dimension : int, optional
            Accepted for API consistency with the C++ constructor. The CEC2019
            suite uses a fixed problem dimension for each function, so this
            argument is ignored.
        """
        if function_number not in range(1, 11) : raise Exception("Function number must be between 1-10.")
        if function_number==1 : dimension=9
        elif function_number==2:  dimension = 16
        elif function_number==3 : dimension=18
        else: dimension =10
        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.bounds = _cec2019_bounds(self.function_number, self.dimension)
        self.f_opt = _cec_global_optimum(2019, self.function_number)
        self.cpp_func = cppCEC2019Functions(self.function_number, self.dimension)

    def __call__(self, X):
        """
        Evaluate the selected CEC2014 test function.

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
       
class CEC2020Functions(_BenchmarkMetadataMixin):
    """
    Provides access to the CEC2020 benchmark test functions.

    This class implements 30 benchmark optimization problems from CEC 2020
    at various dimensions.

    Available dimensions: **5, 10, 15, 20**  
    Available functions: **1–10**
    """

    def __init__(self, function_number, dimension):
        """
        Initialize a CEC2020Functions instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1–10). 
        dimension : int
            The problem dimensionality (must be one of {5, 10, 15, 20}).
        """
        if function_number not in range(1, 11) : raise Exception("Function number must be between 1-10.")
        if int(dimension) not in [2, 5, 10, 15, 20] : raise Exception("Dimension must be 2, 10, or 20.")
        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.bounds = _repeat_bounds(self.dimension, -100.0, 100.0)
        self.f_opt = _cec_global_optimum(2020, self.function_number)
        self.cpp_func = cppCEC2020Functions(self.function_number, self.dimension)

    def __call__(self, X):
        """
        Evaluate the selected CEC2014 test function.

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

class CEC2022Functions(_BenchmarkMetadataMixin):
    """
    Provides access to the CEC2022 benchmark test functions.

    This class implements 12 benchmark optimization problems from CEC 2022
    at various dimensions.

    Available dimensions: **10, 20**  
    Available functions: **1–12**
    """

    def __init__(self, function_number, dimension):
        """
        Initialize a CEC2022Functions instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1–12). 
        dimension : int
            The problem dimensionality (must be one of {10, 20).
        """
        if function_number not in range(1, 13) : raise Exception("Function number must be between 1-12.")
        if int(dimension) not in [2, 10, 20] : raise Exception("Dimension must be 2, 10, or 20.")
        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.bounds = _repeat_bounds(self.dimension, -100.0, 100.0)
        self.f_opt = _cec_global_optimum(2022, self.function_number)
        self.cpp_func = cppCEC2022Functions(self.function_number, self.dimension)

    def __call__(self, X):
        """
        Evaluate the selected CEC2014 test function.

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

class CEC2011Functions(_BenchmarkMetadataMixin):
    """
    Provides access to the 22 real-world benchmark problems from CEC2011.
    Each problem has a fixed dimension and bound box defined by the original suite.
    """

    def __init__(self, function_number, dimension=None):
        """
        Initialize a CEC2011Functions instance.

        Parameters
        ----------
        function_number : int
            The problem index (must be in the range 1–22).
        dimension : int, optional
            Accepted for API consistency with the C++ constructor. CEC2011
            problems have fixed suite-defined dimensions, so this argument is
            ignored.
        """
        if function_number not in CEC2011_METADATA:
            raise Exception("Function number must be between 1 and 22.")

        dim, lb, ub = CEC2011_METADATA[function_number]
        self.function_number = int(function_number)
        self.dimension = int(dim)
        self.lb = lb.tolist() if hasattr(lb, "tolist") else list(lb)
        self.ub = ub.tolist() if hasattr(ub, "tolist") else list(ub)
        if len(self.lb) != self.dimension or len(self.ub) != self.dimension:
            raise ValueError("Bounds length does not match problem dimension.")
        self.bounds = list(zip(self.lb, self.ub))
        self.f_opt = None
        self.cpp_func = cppCEC2011Functions(self.function_number, self.dimension)

    def __call__(self, X):
        return self.cpp_func(X)

    def evaluate(self, xs):
        arr = np.asarray(xs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {arr.shape[1]}")
        return np.array(self.cpp_func(arr.tolist()))

    def get_bounds(self):
        return list(self.bounds)
