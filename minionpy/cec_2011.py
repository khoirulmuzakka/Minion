#!/usr/bin/env python
                     
import numpy as np
import matlab.engine
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class Problem01:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 6
        self.lb = [-6.4,]*6
        self.ub = [6.35,]*6

    def evaluate(self, x):
        return self.eng.bench_func(x, 1)


class Problem02:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 30
        self.lb = self.get_lb(self.ndim)
        self.ub = self.get_ub(self.ndim)

    def get_lb(self, ndim):
        lb = np.zeros(ndim)
        for idx in range(3, ndim):
            lb[idx] = -4 - 1.0/4 * int((idx - 4.0) / 3)
        return lb

    def get_ub(self, ndim):
        ub = np.zeros(ndim)
        ub[0] = ub[1] = 4
        ub[2] = np.pi
        for idx in range(3, ndim):
            ub[idx] = 4 + 1.0/4 * int((idx - 4.0) / 3)
        return ub

    def evaluate(self, x):
        return self.eng.bench_func(x, 2)


class Problem03:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 1
        self.lb = [-0.6,]
        self.ub = [0.9,]

    def evaluate(self, x):
        return self.eng.bench_func(x, 3)


class Problem04:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 1
        self.lb = [0.,]
        self.ub = [5.,]

    def evaluate(self, x):
        return self.eng.bench_func(x, 4)


class Problem05:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 30
        self.lb = self.get_lb(self.ndim)
        self.ub = self.get_ub(self.ndim)

    def get_lb(self, ndim):
        lb = -1.0 * np.ones(ndim)
        lb[0] = lb[1] = lb[2] = 0.0
        return lb

    def get_ub(self, ndim):
        ub = np.zeros(ndim)
        ub[0] = ub[1] = 4
        ub[2] = np.pi
        for idx in range(3, ndim):
            ub[idx] = 4 + 1.0/4 * int((idx - 4.0) / 3)
        return ub

    def evaluate(self, x):
        return self.eng.bench_func(x.reshape((1, self.ndim)), 5)


class Problem06:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 30
        self.lb = self.get_lb(self.ndim)
        self.ub = self.get_ub(self.ndim)

    def get_lb(self, ndim):
        lb = -1.0 * np.ones(ndim)
        lb[0] = lb[1] = lb[2] = 0.0
        return lb

    def get_ub(self, ndim):
        ub = np.zeros(ndim)
        ub[0] = ub[1] = 4
        ub[2] = np.pi
        for idx in range(3, ndim, 3):
            ub[idx] = 4 + 1.0/4 * int((1 - 4.0) / 3)
            ub[idx + 1] = 4 + 1.0 / 4 * int((2 - 4.0) / 3)
            ub[idx + 2] = 4 + 1.0 / 4 * int((3 - 4.0) / 3)
        return ub

    def evaluate(self, x):
        return self.eng.bench_func(x, 6)


class Problem07:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 20
        self.lb = np.zeros(self.ndim)
        self.ub = 2*np.pi*np.ones(self.ndim)

    def evaluate(self, x):
        return self.eng.bench_func(x, 7)


class Problem08:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_1_to_8"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 7
        self.lb = np.zeros(self.ndim)
        self.ub = 15.0*np.ones(self.ndim)

    def evaluate(self, x):
        return self.eng.bench_func(x, 8)


class Problem09:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_9"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file, nargout=0)
        self.ndim = 126
        self.lb = np.zeros(self.ndim)
        self.ub = np.array(
           [0.217, 0.024, 0.076, 0.892, 0.128, 0.25, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.175, 0.032, 0.087, 0.035, 0.024, 0.106,
             0.217, 0.024, 0.026, 0.491, 0.228, 0.3, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.175, 0.032, 0.087, 0.035, 0.024, 0.106,
             0.216, 0.024, 0.076, 0.216, 0.216, 0.216, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.175, 0.032, 0.087, 0.035, 0.024, 0.081,
             0.217, 0.024, 0.076, 0.228, 0.228, 0.228, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.025, 0.032, 0.087, 0.035, 0.024, 0.081,
             0.124, 0.024, 0.076, 0.124, 0.124, 0.124, 0.058, 0.112, 0.062, 0.082, 0.035, 0.065, 0.032, 0.095, 0.022, 0.124, 0.032, 0.087, 0.035, 0.024, 0.106,
             0.116, 0.024, 0.076, 0.116, 0.116, 0.116, 0.058, 0.087, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.116, 0.032, 0.087, 0.035, 0.024, 0.106])

    def evaluate(self, x):
        return self.eng.cost_fn( x.reshape((6, 21)))    

class Problem10:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_10"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)

        self.ndim = 12
        self.lb = np.array(6 * [0.2, ] + 6 * [-180., ])
        self.ub = np.array(6 * [1., ] + 6 * [180., ])

    def evaluate(self, x):
        return self.eng.antennafunccircular(x, matlab.double([50.0, 120.0]), 18.00, 0.5)

class Problem11_1:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 120     # 5x24
        self.lb = np.array([10, 20, 30, 40, 50]*24)
        self.ub = np.array([75, 125, 175, 250, 300]*24)

    def evaluate(self, x):
        return self.eng.fn_DED_5(x.reshape(5, 24))

class Problem11_2:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 240     # 10x24
        self.lb = np.array([150,135,73,60,73,57,20,47,20, 55]*24)
        self.ub = np.array([470,460,340,300,243,160,130,120,80, 55.1]*24)

    def evaluate(self, x):
        return self.eng.fn_DED_10(x)

class Problem11_3:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 6
        self.lb = np.array([100, 50, 80, 50, 50, 50])
        self.ub = np.array([500, 200, 300, 150, 200, 120])

    def evaluate(self, x):
        return self.eng.fn_ELD_6(x)

class Problem11_4:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 13
        self.lb = np.array([0, 0, 0, 60, 60, 60, 60, 60, 60, 40, 40, 55, 55])
        self.ub = np.array([680, 360, 360, 180, 180, 180, 180, 180, 180, 120, 120, 120, 120])

    def evaluate(self, x):
        return self.eng.fn_ELD_13(x)

class Problem11_5:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 15
        self.lb = np.array([150, 150, 20, 20, 150, 135, 135, 60, 25, 25, 20, 20, 25, 15, 15])
        self.ub = np.array([455, 455, 130, 130, 470, 460, 465, 300, 162, 160, 80, 80, 85, 55, 55])

    def evaluate(self, x):
        return self.eng.fn_ELD_15(x)

class Problem11_6:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 40
        self.lb = np.array([36,36,60,80,47,68,110,135,135,130,94,94,125,125,125,125,220,220,242,
                            242,254,254,254,254,254,254,10,10,10,47,60,60,60,90,90,90,25,25,25,242])
        self.ub = np.array([114,114,120,190,97,140,300,300,300,300,375,375,500,500,500,500,500,500,550,550,
                            550,550,550,550,550,550,150,150,150,97,190,190,190,200,200,200,110,110,110,550])

    def evaluate(self, x):
        return self.eng.fn_ELD_40(x)

class Problem11_7:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 140
        self.lb = np.array([71,120,125,125,90,90,280,280,260,260,260,260,260,260,260,260,260,260,260,260,260,260,260,260,
                            280,280,280,280,260,260,260,260,260,260,260,260,120,120,423,423,3,3,160,160,160,160,160,160,
                            160,160,165,165,165,165,180,180,103,198,100,153,163,95,160,160,196,196,196,196,130,130,137,137,
                            195,175,175,175,175,330,160,160,200,56,115,115,115,207,207,175,175,175,175,360,415,795,795,578,
                            615,612,612,758,755,750,750,713,718,791,786,795,795,795,795,94,94,94,244,244,244,95,95,116,175,
                            2,4,15,9,12,10,112,4,5,5,50,5,42,42,41,17,7,7,26])
        self.ub = np.array([119,189,190,190,190,190,490,490,496,496,496,496,506,509,506,505,506,506,505,505,505,505,505,505,537,
                            537,549,549,501,501,506,506,506,506,500,500,241,241,774,769,19,28,250,250,250,250,250,250,250,250,504,
                            504,504,504,471,561,341,617,312,471,500,302,511,511,490,490,490,490,432,432,455,455,541,536,540,538,
                            540,574,531,531,542,132,245,245,245,307,307,345,345,345,345,580,645,984,978,682,720,718,720,964,958,
                            1007,1006,1013,1020,954,952,1006,1013,1021,1015,203,203,203,379,379,379,190,189,194,321,19,59,83,53,
                            37,34,373,20,38,19,98,10,74,74,105,51,19,19,40])

    def evaluate(self, x):
        return self.eng.fn_ELD_140(x)

class Problem11_8:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 96
        self.lb = np.array([5, 6, 10, 13]*24)
        self.ub = np.array([15, 15, 30, 25]*24)

    def evaluate(self, x):
        return self.eng.fn_HT_ELD_Case_1(x)

class Problem11_9:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 96
        self.lb = np.array([5, 6, 10, 13]*24)
        self.ub = np.array([15, 15, 30, 25]*24)

    def evaluate(self, x):
        return self.eng.fn_HT_ELD_Case_2(x)

class Problem11_10:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_11"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 96
        self.lb = np.array([5, 6, 10, 13]*24)
        self.ub = np.array([15, 15, 30, 25]*24)

    def evaluate(self, x):
        return self.eng.fn_HT_ELD_Case_3(x)

class Problem12:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_12_13"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 26
        self.lb = np.array([1900, 2.5, 0, 0, 100, 100, 100, 100, 100, 100, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                            1.1, 1.1, 1.05, 1.05, 1.05, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        self.ub = np.array([2300, 4.05, 1, 1, 500, 500, 500, 500, 500, 600, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                            6, 6, 6, 6, 6, np.pi, np.pi, np.pi, np.pi, np.pi])
        self.problem = self.eng.load("messengerfull.mat")

    def evaluate(self, x):
        return self.eng.feval('mga_dsm', x, self.problem['MGADSMproblem'])


class Problem13:
    def __init__(self, path="/cec_input_data/matlab_cec2011/pro_12_13"):
        self.path_file = os.path.dirname(os.path.abspath(__file__)) + path
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.path_file)
        self.ndim = 22
        self.lb = np.array([-1000, 3, 0, 0, 100, 100, 30, 400, 800, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.15, 1.7, -np.pi, -np.pi, -np.pi, -np.pi])
        self.ub = np.array([0, 5, 1, 1, 400, 500, 300, 1600, 2200, 0.9, 0.9, 0.9, 0.9, 0.9, 6, 6, 6.5, 291, np.pi, np.pi, np.pi, np.pi])
        self.problem = self.eng.load("cassini2.mat")

    def evaluate(self, x):
        return self.eng.feval('mga_dsm', x, self.problem['MGADSMproblem'])


class CEC2011:

    """
    Initializes the problem instance based on the provided problem number.
    Sets up a thread pool for parallel evaluation and thread-local storage
    for handling problem instances per thread.
    
    Attributes:
        num (int): The problem number, which determines the specific problem instance to create.
        problems (list): A list to hold problem instances.
        problemInstance: The problem instance corresponding to the given problem number.
        dimension (int): The dimensionality of the problem.
        executor (ThreadPoolExecutor): A thread pool for parallel processing.
        thread_local (threading.local): Thread-local storage for problem instances.
    """
    def __init__(self, function_number, max_workers=1):
        self.num = function_number
        self.problems = []
        self.problemInstance = self.createProblemInstance()
        self.dimension = self.problemInstance.ndim
        
        # Define the ThreadPoolExecutor as a class member
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize thread-local storage
        self.thread_local = threading.local()

    def createProblemInstance(self) : 
        """
        Creates a problem instance based on the specified problem number.
        
        Returns:
            An instance of the problem class corresponding to the problem number.
        
        Raises:
            Exception: If the problem number is invalid.
        """
        if self.num==1 : 
            p = Problem01 ()
        elif self.num==2 : 
            p= Problem02()
        elif self.num==3 : 
            p= Problem03()
        elif self.num==4 : 
            p= Problem04()
        elif self.num==5 : 
            p= Problem05()
        elif self.num==6 : 
            p= Problem06()
        elif self.num==7 : 
            p= Problem07()
        elif self.num==8 : 
            p= Problem08()
        elif self.num==9 : 
            p= Problem09()
        elif self.num==10 : 
            p= Problem10()
        elif self.num==11 : 
            p= Problem11_1()
        elif self.num==12 : 
            p= Problem11_2()
        elif self.num==13 : 
            p= Problem11_3()
        elif self.num==14 : 
            p= Problem11_4()
        elif self.num==15 : 
            p= Problem11_5()
        elif self.num==16 : 
            p= Problem11_6()
        elif self.num==17 : 
            p= Problem11_7()
        elif self.num==18 : 
            p= Problem11_8()
        elif self.num==19 : 
            p= Problem11_9()
        elif self.num==20 : 
            p= Problem11_10()
        elif self.num==21 : 
            p= Problem12()
        elif self.num==22 : 
            p= Problem13()
        else : raise Exception("Invalid function number")
        self.problems.append(p)
        return p
    
    def __del__(self):
        try :
            for p in self.problems : p.eng.quit()
            self.problemInstance.eng.quit()
        except Exception as e : print(e)
        self.executor.shutdown(wait=True)
    
    def _get_problem_instance(self):
        """
        Retrieves a thread-local problem instance.
        
        Creates a new problem instance for the current thread if one does not already exist.
        This ensures that each thread has its own instance of the problem class.
        
        Returns:
            The thread-local problem instance.
        """
        # Create a new instance of ProblemClass for the current thread if not already created
        if not hasattr(self.thread_local, 'problem_instance'):
            self.thread_local.problem_instance = self.createProblemInstance()
        return self.thread_local.problem_instance

    def _evaluate_single(self, x, i):
        """
        Evaluates a single solution using a thread-local problem instance.
        
        This method is intended to be run in a separate thread and evaluates the
        provided solution vector `x`. The result is stored in the `self.results` list
        at the index `i`.
        
        Args:
            x (list or np.array): The solution vector to be evaluated.
            i (int): The index at which to store the result in `self.results`.
        """
        # Get a thread-local instance of ProblemClass
        problem_instance = self._get_problem_instance()
        result = problem_instance.evaluate(np.array(x))
        self.results[i] = result

    def evaluate(self, xs):
        """
        Evaluates multiple solutions in parallel.
        
        This method submits each solution in `xs` to the ThreadPoolExecutor for
        parallel evaluation. The results are collected and returned as a numpy array.
        
        Args:
            xs (list of lists or np.array): A list of solution vectors to be evaluated.
        Returns:
            np.array: An array containing the evaluation results for each solution vector.
        """
        self.results = [1e+200 for x in xs]
        # Submit tasks to the executor
        futures = [self.executor.submit(self._evaluate_single, xs[i], i) for i in range(len(xs))]
        # Collect the results as they complete
        res = [future.result() for future in as_completed(futures)]
    
        return np.array(self.results)

    def getBounds(self):
        return list(zip(self.problemInstance.lb, self.problemInstance.ub))
