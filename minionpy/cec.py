import sys
import os 

current_file_directory = os.path.dirname(os.path.abspath(__file__))
custom_path = os.path.join(current_file_directory, 'lib')
sys.path.append(custom_path)

from minionpycpp import CEC2017Functions as cppCEC2017Functions
from minionpycpp import CEC2014Functions as cppCEC2014Functions
from minionpycpp import CEC2019Functions as cppCEC2019Functions
from minionpycpp import CEC2020Functions as cppCEC2020Functions
from minionpycpp import CEC2022Functions as cppCEC2022Functions


class CEC2014Functions:
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
        self.cpp_func = cppCEC2014Functions(function_number, int(dimension))

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
    
class CEC2017Functions:
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
        self.cpp_func = cppCEC2017Functions(function_number, int(dimension))

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
    
class CEC2019Functions:
    """
    Provides access to the CEC2019 benchmark test functions.

    This class implements 10 benchmark optimization problems from CEC 2019
    at various dimensions.

    Available functions: **1–10**
    """

    def __init__(self, function_number):
        """
        Initialize a CEC2019Functions instance.

        Parameters
        ----------
        function_number : int
            The function index (must be in the range 1–10). 
        """
        if function_number not in range(1, 11) : raise Exception("Function number must be between 1-10.")
        if function_number==1 : dimension=9
        elif function_number==2:  dimension = 16
        elif function_number==3 : dimension=18
        else: dimension =10
        self.cpp_func = cppCEC2019Functions(function_number, int(dimension))

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
       
class CEC2020Functions:
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
        self.cpp_func = cppCEC2020Functions(function_number, int(dimension))

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

class CEC2022Functions:
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
        self.cpp_func = cppCEC2022Functions(function_number, int(dimension))

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