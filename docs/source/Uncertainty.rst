Statistical uncertainty
=======================

Minion estimates local statistical uncertainty from a numerical Hessian at a
supplied optimum. All differentiation, matrix calculations, normal quantiles,
and diagnostics run in C++; minionpy exposes the same implementation.

C++ uncertainty types live in ``minion::uncertainty`` and their implementation
is in ``minion/uncertainty``. Algorithms retain their existing ``minion`` namespace.
``minion.h`` includes both APIs; ``uncertainty.h`` can also be included directly.
Aliases such as ``minion::Uncertainty`` preserve source compatibility with the
initial uncertainty API. The Python API remains ``minionpy.Uncertainty``.

The objective remains a vectorized black-box function. Statistical interpretation
is supplied by the caller through the required positive ``scale`` argument:
covariance equals ``scale`` times the inverse Hessian. Use 1 for negative log
likelihood, or 2 for chi-squared with known observation variances. An arbitrary
objective requires its own justified scaling. Unknown observation variance is
not estimated automatically.

Python
------

.. code-block:: python

   import numpy as np
   import minionpy

   # t, observed and errors contain the data and known standard deviations.
   def chi_squared(parameters):
       p = np.asarray(parameters)
       predicted = p[:, 0, None] + p[:, 1, None] * t
       return np.sum(((predicted - observed) / errors)**2, axis=1)

   bounds = [(-10, 10), (-10, 10)]
   fit = minionpy.Minimizer(chi_squared, bounds).optimize()
   result = minionpy.Uncertainty(chi_squared, fit.x, bounds).hessian(scale=2)
   if result.covariance_available:
       print(result.std, result.covariance, result.correlation)
       print(result.ci_lower, result.ci_upper)
   print(result.status, result.warnings)

C++
---

.. code-block:: cpp

   #include "minion.h"

   // objective is a MinionFunction; fit is a MinionResult.
   minion::uncertainty::Uncertainty analysis(objective, fit.x, bounds, data);
   minion::uncertainty::HessianOptions options;
   options.confidence_level = 0.95;
   auto result = analysis.hessian(2.0, options);

The caller owns the optional C++ ``data`` pointer and keeps it alive for the
analysis lifetime. Bounds can be omitted for an unconstrained objective.

Numerical controls and interpretation
-------------------------------------

``step_sizes`` and ``parameter_scales`` are optional positive vectors in parameter
units. Automatic steps equal ``relative_step`` (default 0.001) times the supplied
parameter scale, or ``max(1, abs(x[i]))`` when no scale is supplied. ``batch_size``
defaults to -1, evaluating all points of each stencil in one objective call.
Positive values limit the number of points in each call; zero and values below
-1 are rejected. A central stencil uses
``1 + 2*D*D`` points. ``check_stability=True`` repeats differentiation with half
the actual steps and reports the relative Frobenius difference of scaled Hessians
as ``differentiation_error``. This measures step sensitivity, not a rigorous error
bound. ``nfev`` counts all points and ``objective_calls`` counts batches.

The implementation uses one-sided stencils where central stencils do not fit;
it never evaluates beyond bounds. Fixed parameters (zero-width bounds) are
rejected. ``stencil_types``, ``step_sizes``, ``gradient`` and ``warnings`` expose
the differentiation and stationarity diagnostics.

``eigenvalues`` and ``eigenvectors`` describe the Hessian in original parameter
units. ``condition_number`` and the relative ``eigenvalue_tolerance`` (default
1e-10) use the Hessian transformed by parameter scales. By default,
``discard_nonpositive=True`` excludes eigenvalues at or below the tolerance and
calculates covariance using only resolved positive modes. When modes are
discarded, ``status`` is ``partial``, ``covariance_available`` is true,
``uncertainty_complete`` and ``covariance_valid`` are false. The resulting standard
uncertainties and intervals describe only retained contributions and are
incomplete. A zero contribution from a discarded direction does not mean zero
uncertainty. Correlations involving zero retained variance are NaN; the full
condition number remains infinite for partial results.

``retained_rank`` counts retained modes. ``discarded_modes`` contains indices into
``scaled_eigenvalues`` and columns of ``scaled_eigenvectors``. Covariance is computed
in scaled coordinates and transformed back to original parameter units. Negative
curvature can arise from differentiation error (including too-small steps),
nonstationarity, or actual non-positive curvature. Try larger steps and inspect
the optimum; discarding modes does not resolve these causes.

Set ``discard_nonpositive=False`` for strict behavior: non-positive or unresolved
curvature leaves uncertainty arrays empty. Arrays are also empty when no resolved
positive modes remain, regardless of the option. Inspect ``status`` and
``warnings`` before using results. Step-sensitive positive curvature can still
produce covariance, accompanied by warnings.

Intervals are marginal Gaussian intervals at ``confidence_level`` (default 0.95).
They assume a smooth, locally quadratic objective and the supplied statistical
interpretation. They are not clipped to bounds; active bounds can invalidate
the Gaussian approximation even when numerical covariance is available.
Profile likelihood is not implemented yet.

Bootstrap uncertainty
---------------------

``Uncertainty.bootstrap()`` refits user-generated objectives. The sampling
mechanism is supplied by the user: simulating from a fitted statistical model
is a parametric bootstrap; resampling observations with replacement is a
nonparametric bootstrap. The generic name supports both, as in the
`R bootstrap documentation <https://stat.ethz.ch/R-manual/R-devel/library/boot/html/boot.html>`_.

.. code-block:: python

   fitted_theory = model(t, fit.x)

   def objective_sampler(index, sample_seed):
       rng = np.random.default_rng(sample_seed)
       replica = fitted_theory + rng.normal(0, errors)

       def objective(parameters):
           return chi_squared_for_data(parameters, replica)

       return objective

   bootstrap = analysis.bootstrap(
       objective_sampler, n_samples=100, optimizer="NelderMead",
       maxevals=6000, seed=2026,
       options={"x_tol": 1e-9, "f_tol": 1e-12})
   print(bootstrap.std, bootstrap.covariance)
   print(bootstrap.ci_lower, bootstrap.ci_upper)
   print(bootstrap.n_converged, bootstrap.n_used, bootstrap.warnings)

The sampler is called once per requested realization with its zero-based index
and a seed. Its returned vectorized objective must capture one fixed dataset
and return one finite objective value per parameter vector. No data are
resampled during evaluations. C++ performs all repeated optimizations and
statistical calculations; Python only supplies callbacks. Optimizer options
are passed through the existing configuration map. Execution is sequential.

.. code-block:: cpp

   minion::uncertainty::BootstrapOptions options;
   options.n_samples = 100;
   options.optimizer = "NelderMead";
   options.maxevals = 6000;
   options.seed = 2026;
   options.optimizer_options = {{"x_tol", 1e-9}, {"f_tol", 1e-12}};
   auto bootstrap = analysis.bootstrap(sampler, options);

The C++ ``minion::uncertainty::BootstrapSampler`` signature is
``MinionFunction(size_t index, uint64_t sample_seed)``. Returned objectives own
their captured realization and receive a null ``data`` pointer. Unlike the
original objective, bootstrap objectives should capture any additional data
directly. Sampling seeds are independent of optimizer draws and budgets.
The recorded master ``seed``, ``sample_seeds`` and ``optimizer_seeds`` support
reproducibility; use each supplied seed in the sampling callback. Minion's
thread-local RNG state is restored after the analysis, including on failure.

``warm_start=True`` supplies the original optimum to each fit; disabling it
lets the bounded optimizer initialize independently. Bounded optimizers require
finite bounds. Unconstrained bootstrap is supported with ``optimizer="L_BFGS"``
and warm starts; use L_BFGS_B when finite bounds are needed.

By default ``require_convergence=False`` accepts valid finite fits reporting
convergence or stopped by a budget, iteration limit or stagnation. Fits without
confirmed convergence are included with a warning about optimization error;
global optimizers such as ARRDE commonly stop at the evaluation budget.
Set ``require_convergence=True`` to accept only results with termination status
``Converged``. Failed fits
and exceptions are recorded in ``realization_results`` and excluded. Inspect
these records: discarding failed or unconverged fits may bias the bootstrap,
while accepting unconverged fits may introduce optimization error. There are
no automatic retries or replacement samples.

``samples`` has one row per requested realization; invalid fits have NaN rows.
``used_indices`` selects rows used for statistics. Counts ``n_requested``,
``n_valid``, ``n_converged`` and ``n_used`` distinguish the requested fits from
the accepted ones. ``estimate`` is the original optimum and ``mean`` is the mean
of used bootstrap fits. Covariance uses denominator ``n_used-1``; ``std`` and
``correlation`` follow from that covariance. Correlations involving zero sample
variance are NaN. Marginal percentile intervals use linear interpolation between
sorted samples, equivalent to NumPy's default percentile calculation.

At least two accepted fits are required. Otherwise ``status`` is
``insufficient_samples`` and uncertainty arrays are empty. ``covariance_available``
indicates that numerical statistics were returned; ``covariance_valid`` additionally
requires every requested realization to converge, without certifying the user's
statistical assumptions. Small sample counts and bound hits produce diagnostics.
Percentile intervals can have imperfect coverage, particularly with few replicas.

See ``examples/curve_fit.ipynb`` for an executed Gaussian parametric bootstrap
comparison with Hessian uncertainties. Run the integration tests with
``python tests/test_minionpy.py`` after building the extension. Hessian and
bootstrap checks are included in this existing Python implementation suite.
