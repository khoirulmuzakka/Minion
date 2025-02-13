from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
import sys

_instance = None  # Global instance per process

def _init_process(class_name, args, kwargs):
    """
    Initializer function to create a per-process instance.

    This function is called when a new process is initialized. It creates an 
    instance of the specified class (`class_name`) with the provided arguments 
    (`args` and `kwargs`). The instance is stored in the global `_instance` 
    variable so that it can be accessed by the worker function.

    Parameters
    ----------
    class_name : class
        The class to instantiate.
    args : tuple
        Positional arguments to pass to the class constructor.
    kwargs : dict
        Keyword arguments to pass to the class constructor.
    """
    global _instance
    _instance = class_name(*args, **kwargs)

def _worker_function(x, index):
    """
    Worker function to evaluate the objective function.

    This function is executed by each worker process. It evaluates the objective 
    function for a given input (`x`) using the instance of the class created 
    during initialization. The result is returned along with the input index.

    Parameters
    ----------
    x : list or numpy.ndarray
        The input data for the objective function evaluation.
    index : int
        The index of the current input `x` for result tracking.

    Returns
    -------
    tuple
        A tuple containing the index and the result of the objective function 
        evaluation.
    """
    global _instance
    return index, _instance.objective_function(x)


class Process_Parallel:
    """
    Parallelizes the evaluation of an objective function using multiple processes.

    This class uses a process pool executor to parallelize the evaluation of the 
    objective function across multiple processes. Each process is provided its own 
    instance of the `class_name` class, ensuring independence between processes 
    and enabling the use of multiple CPU cores for computation.

    Working Principle:

    - When the `objective` method is called, a list of inputs (`X`) is passed, and the 
      function evaluations are distributed across the available processes.
    - Each process calls the `__get_instance` method, which ensures that each process 
      gets its own instance of the class `class_name`. This avoids issues that may arise 
      when sharing objects between processes.
    - The `__calc_func` method evaluates the objective function for each input `x` by 
      calling the `objective_function` method of the process-local instance.
    - The results are collected from the processes as they complete, and the final output 
      is returned after all evaluations have finished.

    Parameters
    ----------
    Nprocesses : int
        The number of processes to use for parallel processing.
    class_name : class
        The class containing the 'objective_function' method.
    args : tuple
        Positional arguments passed to the class constructor.
    kwargs : dict
        Keyword arguments passed to the class constructor.
    
    Raises
    ------
    ValueError
        If the provided class does not have an 'objective_function' method, 
        which is required for the parallel execution.

    Attributes
    ----------
    Nprocesses : int
        The number of processes to use.
    class_name : class
        The class that provides the 'objective_function' method.
    args : tuple
        Positional arguments passed to `class_name`'s constructor.
    kwargs : dict
        Keyword arguments passed to `class_name`'s constructor.
    executor : ProcessPoolExecutor
        Executor used to manage the parallel processes.
    """

    def __init__(self, Nprocesses, class_name, *args, **kwargs):
        """
        Initializes the Parallel object with a specified number of processes.

        Parameters
        ----------
        Nprocesses : int
            The number of processes to use for parallel execution.
        class_name : class
            The class that contains the 'objective_function' method.
        args : tuple
            Positional arguments passed to the class constructor.
        kwargs : dict
            Keyword arguments passed to the class constructor.
        """
        if not hasattr(class_name, "objective_function"):
            raise ValueError(f"Class {class_name.__name__} does not have the required 'objective_function' method.")

        self.Nprocesses = Nprocesses
        self.class_name = class_name
        self.args = args
        self.kwargs = kwargs

        # Initialize the executor with the provided class and arguments
        self.executor = ProcessPoolExecutor(
            max_workers=self.Nprocesses,
            initializer=_init_process,
            initargs=(self.class_name, self.args, self.kwargs)
        )

    def __del__(self):
        """Shuts down the executor and cleans up resources."""
        self.cleanUp()

    def objective(self, X):
        """
        Evaluates the objective function in parallel using multiple processes.

        Parameters
        ----------
        X : list of list of float
            A list of input arrays to evaluate the objective function.

        Returns
        -------
        list of float
            A list containing the results of the objective function evaluations.
        """
        results = [sys.float_info.max] * len(X)
        futures = {self.executor.submit(_worker_function, x, i): i for i, x in enumerate(X)}

        for future in as_completed(futures):
            try:
                idx, value = future.result()
                results[idx] = value
            except Exception as e:
                print(f"Error evaluating function: {e}", file=sys.stderr)

        return results

    def cleanUp(self):
        """Shuts down the executor and waits for all processes to finish."""
        self.executor.shutdown(wait=True)

    def __call__(self, X):
        """
        Allows the object to be called directly.

        Parameters
        ----------
        X : list[list[float]]
            A list of input arrays to evaluate the objective function.

        Returns
        -------
        list of float
            A list containing the results of the objective function evaluations.
        """
        return self.objective(X)


class Thread_Parallel:
    """
    Parallelizes the evaluation of an objective function using multiple threads.

    This class uses a thread pool executor to parallelize the evaluation of the 
    objective function across multiple threads. Each thread is provided its own 
    instance of the `class_name` class, ensuring thread safety and allowing 
    the objective function to be evaluated independently on each input. 

    Working Principle:

    - When the `objective` method is called, a list of inputs (`X`) is passed, and the 
      function evaluations are distributed across the available threads.
    - Each thread calls the `__get_instance` method, which ensures that each thread 
      gets its own instance of the class `class_name`. This avoids race conditions by 
      ensuring that each thread works with a separate object.
    - The `__calc_func` method evaluates the objective function for each input `x` by 
      calling the `objective_function` method of the thread-local instance.
    - The results are collected from the threads as they complete, and the final output 
      is returned after all evaluations have finished.

    Parameters
    ----------
    Nthreads : int
        The number of threads to use for parallel processing.
    class_name : class
        The class containing the 'objective_function' method.
    args : tuple
        Positional arguments passed to the class constructor.
    kwargs : dict
        Keyword arguments passed to the class constructor.
        
    Raises
    ------
    ValueError : If the provided class does not have an 'objective_function' method.
    """

    def __init__(self, Nthreads, class_name, *args, **kwargs):
        """
        Initializes the Thread_Parallel object with the specified number of threads.

        Parameters
        ----------
        Nthreads : int
            The number of threads to use for parallel execution.
        class_name : class
            The class that contains the 'objective_function' method.
        args : tuple
            Positional arguments passed to the class constructor.
        kwargs : dict
            Keyword arguments passed to the class constructor.
        """
        if not hasattr(class_name, "objective_function"):
            raise ValueError(f"Class {class_name.__name__} does not have the required 'objective_function' method.")

        self.Nthreads = Nthreads
        self.class_name = class_name
        self.args = args
        self.kwargs = kwargs

        # Initialize the thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=self.Nthreads)
        
        # Thread-local storage to ensure thread-safe instances
        self.thread_local = threading.local()

    def __del__(self):
        """Shuts down the executor and cleans up resources."""
        self.cleanUp()

    def __get_instance(self):
        """
        Returns a thread-local instance of the class.

        Each thread is provided its own instance of the class to avoid race conditions.

        Returns
        -------
        object
            An instance of the class.
        """
        if not hasattr(self.thread_local, "obj"):
            # Create and store the instance in thread-local storage
            self.thread_local.obj = self.class_name(*self.args, **self.kwargs)
        return self.thread_local.obj

    def __calc_func(self, x):
        """
        Evaluates the objective function in a thread-safe manner.

        Parameters
        ----------
        x : list
            Input to the objective function.

        Returns
        -------
        float
            The value of the objective function at the input `x`.
        """
        obj = self.__get_instance()
        return obj.objective_function(x)

    def objective(self, X):
        """
        Evaluates the objective function in parallel for multiple inputs.

        Parameters
        ----------
        X : list[list[float]]
            A list of input arrays to evaluate the objective function.

        Returns
        -------
        list of float
            A list containing the results of the objective function evaluations.
        """
        # Initialize the results list with a maximum float value
        results = [sys.float_info.max] * len(X)

        # Create futures dictionary to track tasks and their indices
        futures = {self.executor.submit(self.__calc_func, x): i for i, x in enumerate(X)}

        # Process the completed futures and store results
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Print error if evaluation fails
                print(f"Error evaluating function at index {idx}: {e}", file=sys.stderr)

        return results

    def cleanUp(self):
        """Shuts down the executor and waits for all threads to finish."""
        self.executor.shutdown(wait=True)

    def __call__(self, X):
        """
        Allows the object to be called directly.

        Parameters
        ----------
        X : list of list of float
            A list of input arrays to evaluate the objective function.

        Returns
        -------
        list of float
            A list containing the results of the objective function evaluations.
        """
        return self.objective(X)
