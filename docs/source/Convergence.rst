Notes Regarding Convergence Criteria
========================================

Minion/MinionPy is designed to solve black-box, potentially expensive objective functions. 
As a result, the computational budget is primarily limited by the maximum number of function calls (maxevals). 
This differs from other optimization libraries, where an algorithm stops either when the function no longer improves 
or when a predefined maximum number of iterations is reached.

In Minion, tolerance-based convergence is configured through the algorithm options map or dictionary, using the
``"convergence_tol"`` key for algorithms that support it. We do not use a global constructor-level tolerance anymore.
Additionally, we do not use iteration-based stopping criteria, as the number of function calls per iteration can vary,
making it less intuitive to map to ``maxevals``.

For supported population-based algorithms, ``"convergence_tol"`` typically controls a diversity-based stopping rule
derived from the spread of population fitness values. For ``NelderMead`` and ``DA``, the stopping rule is algorithm-specific
rather than population-diversity-based.

Note that L-BFGS and L-BFGS-B has its own stopping criteria, which is specified in the algorithm options (``g_epsilon``, ``g_epsilon_rel``, ``f_reltol``).
