Notes Regarding Convergence Criteria
========================================

Minion/MinionPy is designed to solve black-box, potentially expensive objective functions.
As a result, the computational budget is primarily limited by the maximum number of function calls (maxevals). Nevertheless,
Minion also implements tolerance-based convergence criteria for algorithms that support them. The tolerances are
configured through the algorithm options map or dictionary, using
``"x_tol"`` and ``"f_tol"`` for algorithms that support them. The default values are ``x_tol = 1e-8`` and
``f_tol = -1.0``. We do not use a global constructor-level tolerance anymore.
Set either tolerance to a negative value to disable that side of the convergence check. For example,
``f_tol = -1.0`` disables objective-value spread convergence while leaving ``x_tol`` active.
Algorithms without built-in restart strategies also accept ``maxiters`` as an iteration cap. The default ``maxiters = -1`` disables the iteration cap.
``ARRDE``, ``j2020``, ``RCMAES``, and ``BIPOP_aCMAES`` do not expose ``maxiters``.
Because the number of function calls per iteration can vary by algorithm, ``maxevals`` remains the recommended primary
budget for comparable runs.

For supported algorithms, ``"x_tol"`` controls the spread of candidate coordinates and ``"f_tol"`` controls relative
objective-value spread. The optimizer reports convergence when either condition is satisfied:

.. math::

   \max_j \left(\max_i x_{i,j} - \min_i x_{i,j}\right) \le x_{\mathrm{tol}}

or

.. math::

   \frac{f_{\max} - f_{\min}}{\left|0.5(f_{\max} + f_{\min})\right|} \le f_{\mathrm{tol}}.

If all objective values are identical, the relative objective-value spread is treated as zero, including the stable
case ``f_max == f_min == 0.0``. If the denominator is zero but the objective-value range is nonzero, the spread is
treated as infinite and does not trigger ``f_tol`` convergence.

The same ``x_tol`` / ``f_tol`` names are used for ``NelderMead`` and ``DA``. For ``CMAES`` and ``ACMAES``, coordinate
spread is measured in the original bounded coordinates, even though the internal sampling uses normalized coordinates.
For ``BIPOP_aCMAES`` and ``RCMAES``, the tolerance options are used as restart triggers; ``max_restarts = -1`` means
unlimited restarts.

Note that L-BFGS and L-BFGS-B have their own stopping criteria, which are specified in the algorithm options (``g_epsilon``, ``g_epsilon_rel``, ``f_reltol``).
