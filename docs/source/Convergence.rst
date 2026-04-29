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

- ARRDE  
- AGSK  
- j2020  
- Dual Annealing  
- RCMAES
- BIPOP-aCMAES

For these, the ``relTol`` (in Python) or ``tol`` (in C++) parameters specify the maximum allowed value for the standard deviation of the 
function values divided by the average of the function values before the algorithm stops.

Note that L-BFGS-B has its own stopping criteria, which is specified in the algorithm options (``g_epsilon``, ``g_epsilon_rel``, ``f_reltol``).
