L-BFGS-B and L-BFGS (Unconstrained) Algorithms
===============================================

The implementation of the limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm (L-BFGS) and its bound-constrained counterpart (L-BFGS-B) in Minion is designed to comply with the library's vectorization requirements. Consequently, function evaluations and their derivatives are computed in batches, enabling the algorithm to achieve parallelization speeds comparable to other population-based algorithms. The L-BFGS-B and L-BFGS algorithms use the LBFGSpp library as a backend, with enhancements for improved vectorization and noise handling.

Robustness in Noisy Optimization Problems
-----------------------------------------

In real-world problems, objective functions are rarely smooth. Generally, quasi-Newton methods such as L-BFGS-B and L-BFGS are not recommended for noisy functions, as gradient calculations can become inaccurate. However, the Minuit optimization library, which implements the quasi-Newton algorithm Migrad, has been reliably used in high-energy physics for decades despite handling noisy objective functions. Inspired by this, Minion aims to achieve a similar level of robustness in its L-BFGS-B and L-BFGS implementations.

To enhance robustness under noisy conditions, the derivative must be computed as accurately as possible. For a noisy function using forward differences, the optimal step size for derivative estimation is given by:

.. math::
   h = 2\sqrt{\frac{\epsilon_f}{|f^{(2)}|}}

where :math:`f^{(2)}` is the second derivative. Here, the noise is assumed to be multiplicative: :math:`\epsilon_f(x) = \epsilon_r f(x)`, where :math:`\epsilon_r \ll 1`. The second derivative is estimated from the L-BFGS updates and computed recursively using results from previous iterations, eliminating the need for additional function evaluations. Similarly, the function values used to calculate :math:`\epsilon_f` are obtained from previous iterations.

Noise-Robust Derivative Estimation
-----------------------------------

To further improve robustness, Minion employs the Lanczos noise-robust derivative estimation:

.. math::
   f'(x) \approx \frac{3}{h} \sum_{k=1}^m \frac{k}{m(m+1)(2m+1)} (f(x+kh)-f(x-kh)), \quad m=\frac{N-1}{2}

This formula is derived by fitting a quadratic function to :math:`N` sample points. Notably, when :math:`N=3`, it reduces to the standard central difference formula. In Minion, setting :math:`N=1` corresponds to forward differences.

Performance Comparison
----------------------

The following notebook presents a comparison of Minion's L-BFGS-B and L-BFGS algorithms for solving CEC2017 benchmark problems. The results are compared against Minuit's Migrad and SciPy's L-BFGS-B implementation.

.. toctree::
   :maxdepth: 2
   :caption: Comparison Notebook:

   l_bfgs_b








Notes Regarding Convergence Criteria
========================================

Minion/MinionPy is designed to solve black-box, potentially expensive objective functions. 
As a result, the computational budget is primarily limited by the maximum number of function calls (maxevals). 
This differs from other optimization libraries, where an algorithm stops either when the function no longer improves 
or when a predefined maximum number of iterations is reached.

In Minion, we do use tolerance-based convergence criteria, but some algorithms, especially population-based ones, 
do not support this. The primary reason for this is that the population is not designed to converge completely; 
instead, it is maintained to ensure continued exploration of the search space. 
Additionally, we do not use iteration-based stopping criteria, as the number of function calls per iteration can vary, 
making it less intuitive to map to maxevals.

Below is a list of algorithms that **ignore** the ``relTol`` parameter. Even if it is set, it will have no effect:

1) ARRDE  
2) LSRTDE  
3) NL-SHADE-RSP  
4) j2020  
5) Dual Annealing  
6) GWO-DE  


The following DE-based algorithms **do** support tolerance-based stopping:

- DE
- JADE
- LSHADE
- jSO
- Nelder-Mead

For these, the ``relTol`` (in Python) or ``tol`` (in C++) parameters specify the maximum allowed value for the standard deviation of the 
function values divided by the average of the function values before the algorithm stops.

Note that L-BFGS-B has its own stopping criteria, which is specified in the algorithm options (``g_epsilon``, ``g_epsilon_rel``, ``f_reltol``).



Notes Regarding Vectorization Support
=========================================

As mentioned, Minion requires the objective function to be vectorized. This ensures that algorithms capable of batch function calls can 
fully utilize the parallelization implemented by the user in the vectorized function. However, some algorithms do not support batch function 
calls natively. Population-based algorithms are generally known for their support of batch function calls, while sequential ones, 
such as Nelder-Mead, do not.

Here is a list of algorithms implemented in Minion/MinionPy that **support** batch function calls, and therefore can fully take advantage 
of parallelization:

1) DE  
2) LSHADE  
3) JADE  
4) jSO  
5) ARRDE  
6) NL-SHADE-RSP  
7) LSRTDE  
8) GWO-DE  
9) ABC  
10) L-BFGS-B  

Algorithms that **do not** support batch function calls:

1) j2020  
2) Nelder-Mead  

Algorithms that **partially** support batch function calls:

1) Dual Annealing  

The L-BFGS-B algorithm in Minion supports batch function calls because, in every function evaluation, 
a derivative is also computed. Evaluations of the objective function at the evaluation point and the shifted point (for the derivative) 
can be performed in parallel. Note that this parallelization feature is not available in other optimization library that implement L-BFGS-B such as SciPy. 
Since Dual Annealing uses L-BFGS-B for local search, it can also benefit from this feature.

The j2020 algorithm, although population-based, is not written in a way that supports batch function calls. 
This is because an update is performed after every function call.


Algorithm Details
==================

This section provides detailed explanations of the optimization algorithms implemented in the Minion library, along with their key parameters and how to configure them.

Each optimization algorithm has a set of parameters that can be adjusted to influence the optimization process. These parameters can be obtained and modified as follows:

**C++ Example:**

In C++, you can retrieve the default settings for an algorithm and adjust the parameters as needed:

.. code-block:: cpp

    minion::DefaultSettings settings;
    // Retrieve the default settings for the ARRDE algorithm
    std::map<std::string, minion::ConfigValue> options = settings.getDefaultSettings("ARRDE");

    // Override default parameters
    options["population_size"] = 50; 
    options["restart-refine-duration"] = 0.9;  

    // Initialize the minimizer with custom settings
    auto min = minion::Minimizer(rosenbrock_vect, bounds, {}, nullptr, nullptr, "ARRDE", 0.0, max_evals, -1, options);

    // Perform optimization
    auto min_result = min.optimize();
    

**Python Example:**

In Python, you can pass the parameters via a dictionary when creating the `Minimizer` object. Here’s an example for configuring and using the ARRDE algorithm:

.. code-block:: Python 

    import minionpy as mpy 

    options = {
        "population_size"             : 0,           # Default population size (0 means auto-determined)
        "archive_size_ratio"          : 2.0,         # Archive size relative to population size
        "converge_reltol"             : 0.005,       # Convergence tolerance for relative error
        "refine_decrease_factor"      : 0.9,         # Factor by which the mutation step size is reduced
        "restart-refine-duration"     : 0.8,         # Duration for restart and refinement phase
        "maximum_consecutive_restarts": 2,           # Maximum number of consecutive restarts
        "bound_strategy"              : "reflect-random"  # Bound strategy for out-of-bounds values
    }

    # Use ARRDE algorithm for optimization
    min = mpy.Minimizer(func=objective_function, 
                        x0=x0, 
                        bounds=[(-10, 10)] * dimension, 
                        algo="ARRDE", 
                        relTol=0.0, 
                        maxevals=10000, 
                        callback=None, 
                        seed=None, 
                        options=options)

    # Run optimization
    result = min.optimize()



Each algorithm comes with a set of configurable parameters that affect its behavior during the optimization process. For example, the ``population_size`` parameter controls the number of candidate solutions in the population, while the ``mutation_rate`` defines the probability of modifying a candidate solution during the mutation process. 

Please refer to the respective algorithm section for detailed descriptions of each parameter.



Differential Evolution (DE)
---------------------------
Differential Evolution (DE) is a population-based stochastic optimization algorithm that applies mutation, crossover, and selection to evolve a population of candidate solutions.

*Reference : Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.*

Algorithm name : ``"DE"``

Parameters : 

- ``population_size``: 0  

  .. note:: Initial population size (N). If set to `0`, it will be automatically determined. 

                .. math::

                        N = 5 \cdot D

                where *D* is the dimensionality of the problem.

- ``mutation_rate``: 0.5 

  .. note:: The value of the mutation rate (F).

- ``crossover_rate``: 0.8  

  .. note:: The probability of recombining parent and mutant vectors (CR value).

- ``mutation_strategy``: ``best1bin``  

  .. note:: Mutation strategy used in the optimization process. Available strategies:  
                    ``"best1bin"``, ``"best1exp"``, ``"rand1bin"``, ``"rand1exp"``,  
                    ``"current_to_pbest1bin"``, ``"current_to_pbest1exp"``.

- ``bound_strategy``: ``reflect-random`` 

  .. note:: Method for handling boundary violations. Available strategies:  
                    ``"random"``, ``"reflect-random"``, ``"clip"``.


Adaptive Restart-Refine Differential Evolution (ARRDE)
------------------------------------------------------
ARRDE is an extension of the Differential Evolution algorithm with adaptive restart and refinement strategies. 

Algorithm name : ``"ARRDE"``

Parameters : 

- ``population_size``: 0  

  .. note:: Initial population size (N). If set to ``0``, it will be automatically determined as follows:

                .. math::

                    N = 2 \cdot D + \log(N_{maxevals})^2

                where *D* is the dimensionality of the problem and :math:`N_{maxevals}` is the maximum number of function evaluations.

- ``minimum_population_size``: 4

  .. note:: final (minimum) population size during linear population size reduction.

- ``archive_size_ratio``: 2.0  

  .. note:: The ratio of archive size to the current population size .

- ``converge_reltol``: 0.005  

  .. note:: The value of std(f)/mean(f) below which a population is said to be converged.

- ``refine_decrease_factor``: 0.9 

  .. note:: The decrease factor of *converge_relTol* in the refinement phase. 

- ``restart-refine-duration``: 0.8  

  .. note:: A fraction of evaluation budget during which restart-refine phase occurs. The remaining fraction is dedicated to final refinement. 

- ``maximum_consecutive_restarts``: 2  

  .. note::  maximum number of consecutive restarts during restart-refine phase.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  
                    ``"random"``, ``"reflect-random"``, ``"clip"``.


Grey Wolf Optimizer Differential Evolution (GWO-DE)
----------------------------------------------------
GWO-DE combines Differential Evolution with the Grey Wolf Optimizer, leveraging the social structure of wolves for optimization.

Algorithm name : ``"GWO_DE"``

Parameters : 

- ``population_size``: 0  

  .. note:: The number of individuals in the population. If set to 0, a default size based on problem dimension is used.

- ``mutation_rate``: 0.5  

  .. note:: The probability of mutating a parameter value during evolution.

- ``crossover_rate``: 0.7  

  .. note:: The probability of recombining parent and mutant vectors.

- ``elimination_prob``: 0.1 

  .. note:: The probability of eliminating a wolf from the population.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  
                    ``"random"``, ``"reflect-random"``, ``"clip"``.

j2020 Algorithm
----------------
The j2020 algorithm is a variation of Differential Evolution that incorporates parameter-specific strategies for mutation and recombination.

Reference : J. Brest, M. S. Maučec and B. Bošković, "Differential Evolution Algorithm for Single Objective Bound-Constrained Optimization: Algorithm j2020," 2020 IEEE Congress on Evolutionary Computation (CEC), Glasgow, UK, 2020, pp. 1-8, doi: 10.1109/CEC48606.2020.9185551.

Algorithm name : ``"j2020"``

Parameters : 

- ``population_size``: 0 

  .. note:: Initial population size (N). If set to ``0``, it will be automatically determined as follows:

                .. math::

                        N = 8 \cdot D

- ``tau1``: 0.1  

  .. note:: The value of *tau1* variable. The value must be between 0 and 1. 

- ``tau2``: 0.1  

  .. note:: The value of *tau1* variable. The value must be between 0 and 1. 

- ``myEqs``: 0.25  

  .. note:: The value of *myEqs* variable. The value must be between 0 and 1. 

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  
                    ``"random"``, ``"reflect-random"``, ``"clip"``.

LSRTDE Algorithm
----------------
The LSRTDE algorithm. Designed to solve CEC2024.

*Reference : V. Stanovov and E. Semenkin, "Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition," 2024 IEEE Congress on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: 10.1109/CEC60901.2024.10611907.*

Algorithm name : ``"LSRTDE"``

Parameters : 

- ``population_size``: 0  

  .. note:: nitial population size (N). If set to ``0``, it will be automatically determined as follows:

                .. math::

                        N = 20 \cdot D

                where *D* is the dimensionality of the problem.

- ``memory_size``: 5  

  .. note:: Memory size for storing the values of ``CR`` and ``F``.

- ``success_rate``: 0.5  

  .. note:: The success rate required for selecting an individual for the next generation.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  
                    ``"random"``, ``"reflect-random"``, ``"clip"``.

NLSHADE-RSP Algorithm
----------------------
NLSHADE-RSP is an extension of the SHADE algorithm designed to solve CEC2021 problems.

*Reference : V. Stanovov, S. Akhmedova and E. Semenkin, "NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for CEC 2021 Numerical Optimization," 2021 IEEE Congress on Evolutionary Computation (CEC), Kraków, Poland, 2021, pp. 809-816, doi: 10.1109/CEC45853.2021.9504959.*

Algorithm name : ``"NLSHADE_RSP"``

Parameters : 

- ``population_size``: 0  

  .. note:: Initial population size (N). If set to `0`, it will be automatically determined. 

                .. math::

                        N = 30 \cdot D

                where *D* is the dimensionality of the problem.

- ``memory_size``: 100  

  .. note:: Memory size for storing the values of ``CR`` and ``F`` 

- ``archive_size_ratio``: 2.6  

  .. note:: The ratio of the archive size relative to the population size.

- ``bound_strategy``: ``reflect-random`` 

  .. note:: Method for handling boundary violations. Available strategies:  ``"random"``, ``"reflect-random"``, ``"clip"``, ``"periodic"``.

JADE Algorithm
--------------
JADE is a variant of Differential Evolution that introduces adaptive strategies for mutation and selection.

*Reference : J. Zhang and A. C. Sanderson, "JADE: Adaptive Differential Evolution With Optional External Archive," in IEEE Transactions on Evolutionary Computation, vol. 13, no. 5, pp. 945-958, Oct. 2009, doi: 10.1109/TEVC.2009.2014613.*

Algorithm name : ``"JADE"``

Parameters : 

- ``population_size``: 0  

  .. note:: Initial population size (N). If set to ``0``, it will be automatically determined as follows:

                - If the dimensionality :math:`D` of the problem is :math:`D < 10`, then :math:`N = 30`.
                - If :math:`10 \leq D \leq 30`, then :math:`N = 100`.
                - If :math:`30 < D \leq 50`, then :math:`N = 200`.
                - If :math:`50 < D \leq 70`, then :math:`N = 300`.
                - Else, :math:`N = 400`.

- ``c``: 0.1  

  .. note::  The value of *c* variable. The value must be between 0 and 1. 

- ``mutation_strategy``: ``current_to_pbest_A_1bin``  

  .. note::  Mutation strategy used in the optimization process. Available strategies:  
                    ``"best1bin"``, ``"best1exp"``, ``"rand1bin"``, ``"rand1exp"``,  
                    ``"current_to_pbest1bin"``, ``"current_to_pbest1exp"``,  
                    ``"current_to_pbest_A_1bin"``, ``"current_to_pbest_A_1exp"``.

- ``archive_size_ratio``: 1.0  

  .. note:: The ratio of the archive size relative to the population size.

- ``minimum_population_size``: 4  

  .. note:: The minimum population size.

- ``reduction_strategy``: ``linear``  

  .. note:: Strategy used to reduce the population size. Available strategies:  ``"linear"``, ``"exponential"``, ``"agsk"``.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  ``"random"``, ``"reflect-random"``, ``"clip"``, ``"periodic"``.

jSO Algorithm
-------------
The jSO algorithm is a variant of LSHADE, designed to solve CEC2017 problems.

*Reference : J. Brest, M. S. Maučec and B. Bošković, "Single objective real-parameter optimization: Algorithm jSO," 2017 IEEE Congress on Evolutionary Computation (CEC), Donostia, Spain, 2017, pp. 1311-1318, doi: 10.1109/CEC.2017.7969456.*

Algorithm name : ``"jSO"``

Parameters : 

- ``population_size``: 0  

  .. note:: Initial population size (N). If set to `0`, it will be automatically determined as:

                .. math::

                    N = 25 \cdot \log(D) \cdot \sqrt{D}

                where *D* is the dimensionality of the problem.

- ``memory_size``: 5  

  .. note:: Memory size for storing the values of ``CR`` and ``F``.

- ``archive_size_ratio``: 1.0  

  .. note:: The ratio of the archive size relative to the population size.

- ``minimum_population_size``: 4  

  .. note:: The minimum population size.

- ``reduction_strategy``: ``linear``  

  .. note:: Strategy used to reduce the population size. Available strategies:  ``"linear"``, ``"exponential"``, ``"agsk"``.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  ``"random"``, ``"reflect-random"``, ``"clip"``, ``"periodic"``.

LSHADE Algorithm
----------------
Linear Population Reduction - Success History Adaptive Differential Evolution (LSHADE) algorithm. Originally designed to solve CEC2014.

*Reference : R. Tanabe and A. S. Fukunaga, "Improving the search performance of SHADE using linear population size reduction," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, China, 2014, pp. 1658-1665, doi: 10.1109/CEC.2014.6900380.*

Algorithm name : ``"LSHADE"``

Parameters : 

- ``population_size``: 0  

  .. note:: Initial population size (N). If set to `0`, it will be automatically determined. 

                .. math::

                        N = 5 \cdot D

                where *D* is the dimensionality of the problem.

- ``memory_size``: 6  

  .. note:: Memory size for storing the values of ``CR`` and ``F``.

- ``mutation_strategy``: ``current_to_pbest_A_1bin``  

  .. note:: Mutation strategy used in the optimization process. Available strategies:  
                    ``"best1bin"``, ``"best1exp"``, ``"rand1bin"``, ``"rand1exp"``,  
                    ``"current_to_pbest1bin"``, ``"current_to_pbest1exp"``,  
                    ``"current_to_pbest_A_1bin"``, ``"current_to_pbest_A_1exp"``.

- ``archive_size_ratio``: 2.6  

  .. note:: The ratio of the archive size relative to the population size.

- ``minimum_population_size``: 4  

  .. note:: The minimum population size.

- ``reduction_strategy``: ``linear``  

  .. note:: The strategy for reducing the population size over time. Options include linear, exponential, or agsk.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  ``"random"``, ``"reflect-random"``, ``"clip"``, ``"periodic"``.

Artificial Bee Colony (ABC)
---------------------------
The Artificial Bee Colony (ABC) algorithm is a swarm intelligence-based optimization algorithm inspired by the foraging behavior of honey bees.

Algorithm name : ``"ABC"``

Parameters : 

- ``population_size``: 0  

  .. note:: Initial population size (N). If set to `0`, it will be automatically determined. 

                .. math::

                        N = 5 \cdot D

                where *D* is the dimensionality of the problem.

- ``mutation_strategy``: ``rand1``  

  .. note::  Mutation strategy, default is "rand1", available : ``"rand1"``, ``"best1"``.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:  ``"random"``, ``"reflect-random"``, ``"clip"``, ``"periodic"``.

Dual Annealing (DA)
----------------------------
Dual Annealing combines simulated annealing with local search to provide a flexible and robust global optimization algorithm.

*Reference : Tsallis C, Stariolo DA. Generalized Simulated Annealing. Physica A, 233, 395-406 (1996).*

Algorithm name : ``"DA"``

Parameters : 

- ``acceptance_par``: -5.0  

  .. note:: The acceptance parameter controlling the probability of accepting worse solutions. The value must be between -1.0e+4 and -5.

- ``visit_par``: 2.67  

  .. note:: The parameter controlling the annealing rate during the search. The value must be between 1.0 and 3.0.

- ``initial_temp``: 5230.0  

  .. note:: The initial temperature for the annealing process. The value must be between 0.01 and 5.0e+4.

- ``restart_temp_ratio``: 2e-5  

  .. note:: The temperature ratio for restart condition. The value must be between 0 and 1.

- ``use_local_search``: true  

  .. note:: Whether to use local search (e.g., L-BFGS-B) to refine the solutions.

- ``local_search_algo``: ``L_BFGS_B`` 

  .. note:: The local search algorithm to be used. Available : ``"NelderMead"`` and ``"L_BFGS_B"``

- ``finite_diff_rel_step``: 1e-10 

  .. note:: The relative step size for finite difference computations. The default value 0.0 means that the relative step is given by the square root of machine epsilon.

- ``bound_strategy``: ``periodic``  

  .. note:: Method for handling boundary violations. Available strategies:  ``"random"``, ``"reflect-random"``, ``"clip"``, ``"periodic"``.

Nelder-Mead Algorithm
---------------------
The Nelder-Mead algorithm is a derivative-free optimization method that relies on reflection, expansion, contraction, and shrinkage to search for an optimum.

*Reference : Nelder, John A.; R. Mead (1965). "A simplex method for function minimization". Computer Journal. 7 (4): 308–313. doi:10.1093/comjnl/7.4.308.*

Algorithm name : ``"NelderMead"``

Parameters : 

- ``locality_factor``: 1.0  

  .. note:: The factor controlling the step size for reflection and expansion during optimization.

- ``bound_strategy``: ``reflect-random``  

  .. note:: Method for handling boundary violations. Available strategies:   ``"random"``, ``"reflect-random"``, ``"clip"``, ``"periodic"``.

L-BFGS-B Algorithm
------------------

L-BFGS-B is a quasi-Newton method that approximates the Hessian matrix while handling bound constraints. The implementation here utilizes the back-end code from the `LBFGSpp` library (`https://github.com/yixuan/LBFGSpp`), which provides L-BFGS-B updates and Hessian approximation functionality.

Minion implements a customized **derivative calculation** method that ensures both **vectorization** and **noise robustness**. To improve stability under noise, the derivative is computed using an **adaptive step size**, and a **noise-robust Lanczos derivative** is employed. 

Additionally, **function calls are vectorized**, meaning the objective function and its derivative can be evaluated in a **single batch**. This batch execution can be further parallelized using **multithreading** or **multiprocessing**, leading to significant computational efficiency improvements.

*Reference : Byrd, R. H.; Lu, P.; Nocedal, J.; Zhu, C. (1995).  "A Limited Memory Algorithm for Bound Constrained Optimization", SIAM J. Sci. Comput. 16 (5): 1190–1208.*

Algorithm Name : ``"L_BFGS_B"``

Parameters

- **``max_iterations``**: *15000*  

  .. note:: The maximum number of iterations allowed for the algorithm.

- **``m``**: *15*  

  .. note:: The number of previous iterations used to approximate the Hessian matrix.

- **``g_epsilon``**: *1e-8*  

  .. note:: The absolute gradient convergence tolerance.

- **``g_epsilon_rel``**: *0.0*  

  .. note:: The relative gradient convergence tolerance.

- **``f_reltol``**: *1e-8*  

  .. note:: The function value convergence tolerance.

- **``max_linesearch``**: *20*  

  .. note:: The maximum number of iterations allowed during line search.

- **``c_1``**: *1e-3*  

  .. note:: The first Wolfe condition parameter for line search.

- **``c_2``**: *0.9*  

  .. note:: The second Wolfe condition parameter for line search.

- **``func_noise_ratio``**: *1e-16*  

  .. note:: Noise level (ratio), defined as the deviation of the function value from its ideal smooth counterpart, relative to the function value. If the function is smooth, set this to zero.

- **``N_points_derivative``**: *3*  

  .. note:: The number of sample points used for derivative calculations.  
            If set to an even number, it is automatically increased by 1 to make it odd.  
            Given ``N``, the total function call batch size for one function evaluation and derivative calculation is computed as:  
            **1 + D * (N - 1)**, where ``D`` is the dimensionality of the problem.


L-BFGS Algorithm
------------------

L-BFGS is a quasi-Newton method that approximates the Hessian matrix for *unconstrained* optimization rpoblem. The implementation here utilizes the back-end code from the `LBFGSpp` library (`https://github.com/yixuan/LBFGSpp`), which provides L-BFGS updates and Hessian approximation functionality.

Minion implements a customized **derivative calculation** method that ensures both **vectorization** and **noise robustness**. To improve stability under noise, the derivative is computed using an **adaptive step size**, and a **noise-robust Lanczos derivative** is employed. 

Additionally, **function calls are vectorized**, meaning the objective function and its derivative can be evaluated in a **single batch**. This batch execution can be further parallelized using **multithreading** or **multiprocessing**, leading to significant computational efficiency improvements.

*Reference: Liu, D. C.; Nocedal, J. (1989). "On the Limited Memory Method for Large Scale Optimization". *Mathematical Programming B*, 45(3): 503-528.*

Algorithm Name : ``"L_BFGS"``

Parameters

- **``max_iterations``**: *15000*  

  .. note:: The maximum number of iterations allowed for the algorithm.

- **``m``**: *15*  

  .. note:: The number of previous iterations used to approximate the Hessian matrix.

- **``g_epsilon``**: *1e-8*  

  .. note:: The absolute gradient convergence tolerance.

- **``g_epsilon_rel``**: *0.0* 

  .. note:: The relative gradient convergence tolerance.

- **``f_reltol``**: *1e-8*  

  .. note:: The function value convergence tolerance.

- **``max_linesearch``**: *20* 

  .. note:: The maximum number of iterations allowed during line search.

- **``c_1``**: *1e-3*  

  .. note:: The first Wolfe condition parameter for line search.

- **``c_2``**: *0.9*  

  .. note:: The second Wolfe condition parameter for line search.

- **``func_noise_ratio``**: *1e-16*  

  .. note:: Noise level (ratio), defined as the deviation of the function value from its ideal smooth counterpart, relative to the function value. If the function is smooth, set this to zero.

- **``N_points_derivative``**: *3*  

  .. note:: The number of sample points used for derivative calculations.  
            If set to an even number, it is automatically increased by 1 to make it odd.  
            Given ``N``, the total function call batch size for one function evaluation and derivative calculation is computed as:  
            **1 + D * (N - 1)**, where ``D`` is the dimensionality of the problem.
