"""Statistical uncertainty calculated by the Minion C++ backend."""
from .minionpycpp import Uncertainty as _Uncertainty, HessianOptions, HessianResult
from .minionpycpp import BootstrapOptions, BootstrapResult, CurvaturePolicy

__all__ = ["Uncertainty", "HessianOptions", "HessianResult", "BootstrapOptions", "BootstrapResult", "CurvaturePolicy"]


class Uncertainty:
    """Analyze uncertainty at a fixed optimum using a vectorized objective.

    Parameters
    ----------
    objective : callable
        Objective accepting parameter vectors with shape ``(n_points, n_parameters)``
        and returning one objective value per point, with shape ``(n_points,)``.
        It follows the same vectorized contract as ``Minimizer``.
    optimum : array_like of shape (n_parameters,)
        Finite parameter values at which uncertainty is evaluated. Must be nonempty.
    bounds : sequence of tuple of float, optional
        Lower and upper bounds for each parameter. Each lower bound must be
        strictly less than its upper bound, and the optimum must lie within
        the bounds. If None, the analysis is unconstrained.

    Raises
    ------
    TypeError
        If `objective` is not callable.
    RuntimeError
        If the backend rejects the optimum or bounds.

    See Also
    --------
    Uncertainty.hessian : Estimate local Gaussian uncertainty.
    Uncertainty.bootstrap : Refit statistically resampled objectives.
    """

    def __init__(self, objective, optimum, bounds=None):
        if not callable(objective):
            raise TypeError("objective must be callable")
        self._objective = objective
        self._backend = _Uncertainty(
            lambda points, data: objective(points), optimum,
            [] if bounds is None else bounds, None)

    def hessian(self, scale, *, parameter_scales=None, step_sizes=None,
                relative_step=1e-3, step_candidates=4, step_tolerance=0.05,
                eigenvalue_tolerance=1e-10, curvature_policy="regularize",
                confidence_level=0.95, batch_size=-1):
        """Estimate local Gaussian uncertainty from a numerical Hessian.

        Parameters
        ----------
        scale : float
            Positive covariance multiplier. Use 1 for negative log likelihood
            and 2 for chi-squared with known observation variances. Other
            objectives require a user-supplied statistical interpretation.
        parameter_scales : array_like of shape (n_parameters,), optional
            Positive parameter scales used for initial steps, eigenvalue
            thresholds, and conditioning. If None, use ``max(1, abs(optimum[i]))``
            for each parameter. These are distinct from the statistical `scale`.
        step_sizes : array_like of shape (n_parameters,), optional
            Positive initial steps in parameter units. If None, use
            ``relative_step * parameter_scales``. Probing can enlarge these steps.
        relative_step : float, default: 1e-3
            Positive initial step relative to each parameter scale. Ignored for
            initial step construction when `step_sizes` is supplied.
        step_candidates : int, default: 4
            Number of coordinate-only step candidates in a single preliminary
            batch. The default probes ``h, 2h, 4h, 8h``. Use 1 to calculate one
            full Hessian directly at the initial steps, without preliminary
            probing. Other accepted values are 3 through 30; 2 is invalid
            because automatic selection requires three distinct scales.
            Probe values are reused in the single final Hessian calculation.
        step_tolerance : float, default: 0.05
            Positive relative agreement required between coordinate curvature
            estimates during step selection. Unused when `step_candidates` is 1.
        eigenvalue_tolerance : float, default: 1e-10
            Relative cutoff in (0, 1). Scaled Hessian eigenvalues at or below
            this fraction of the largest absolute scaled eigenvalue are unresolved.
        curvature_policy : {'strict', 'regularize', 'positive_modes'}, default: 'regularize'
            Action when the final scaled Hessian has non-positive or unresolved
            modes. 'strict' returns diagnostics without covariance. 'regularize'
            applies MINUIT-style diagonal inflation and marks the covariance as
            a regularized approximation. 'positive_modes' returns only resolved
            positive-mode contributions and marks the uncertainty as incomplete.
        confidence_level : float, default: 0.95
            Confidence level in (0, 1) for marginal Gaussian intervals.
        batch_size : int, default: -1
            Maximum number of final stencil points per objective call. Use -1
            for one final batch. Preliminary probes always use one separate
            batch, regardless of this setting. Zero and values below -1 are invalid.

        Returns
        -------
        result : HessianResult
            Parameter estimates, covariance, standard uncertainties (``std``),
            correlation, and marginal intervals (``ci_lower``, ``ci_upper``).
            Includes the original Hessian, gradient, raw and scaled eigenpairs,
            selected ``step_sizes``, ``stencil_types``, ``step_probe_errors``,
            ``condition_number``, ``retained_rank``,
            ``discarded_modes``, ``status``, and ``warnings``. The flags
            ``covariance_available``, ``covariance_valid``, and
            ``uncertainty_complete`` describe availability and completeness.
            ``nfev`` and ``objective_calls`` count actual evaluations and
            callbacks across probing and final differentiation.

        Raises
        ------
        ValueError
            If `curvature_policy` is not one of the accepted policy names.
        RuntimeError
            If the backend rejects options, cannot construct a representable
            stencil within bounds, or receives an incorrect objective batch
            length or non-finite objective values or derivatives.

        See Also
        --------
        Uncertainty.bootstrap : Estimate uncertainty by refitting sampled data.

        Notes
        -----
        Step selection is independent for each parameter. Choose the largest
        candidate whose last three distinct steps have positive curvature and
        adjacent relative differences within `step_tolerance`. If none
        passes, warn and use the most consistent positive pair's larger step,
        or the largest candidate when no positive pair exists. Repeated capped
        steps do not establish stability. ``step_probe_errors`` contains the
        selected consistency measures, or infinity when unavailable. Coordinate
        probes cannot validate weak combinations of parameters. Four probes
        may not reach a large enough step for a noisy objective.

        Only one full Hessian is calculated after step selection. Set
        ``step_candidates=1`` to use fixed initial steps without probing.

        With ``curvature_policy='positive_modes'``, discarded modes produce
        ``status='partial'``,
        ``uncertainty_complete=False``, and ``covariance_valid=False``. Their
        zero covariance contributions do not establish zero uncertainty.
        ``discarded_modes`` indexes ``scaled_eigenvalues`` and columns of
        ``scaled_eigenvectors``. Correlations involving zero retained variance
        are NaN. With no resolved positive modes and no successful repair,
        uncertainty arrays are empty.

        Repaired results have ``status='forced_positive_definite'``,
        ``uncertainty_complete=True`` (all parameter directions are included),
        and ``covariance_valid=False``. The original Hessian and eigenpairs
        remain unchanged. ``regularized_hessian`` is the matrix used for the
        covariance; ``regularization_relative_change`` measures the relative
        Frobenius change in parameter-scaled coordinates. A completely zero
        Hessian remains unavailable because no curvature scale exists.

        Intervals are marginal normal intervals and are not clipped to bounds.
        Warnings identify bounds, step sensitivity, and nonstationary points.
        A positive-definite matrix or ``covariance_valid=True`` alone does not
        establish statistical coverage or reliable curvature under noise.

        Examples
        --------
        >>> import numpy as np
        >>> import minionpy
        >>> def nll(points):
        ...     return 0.5 * np.sum(np.asarray(points)**2, axis=1)
        >>> analysis = minionpy.Uncertainty(nll, [0.0, 0.0])
        >>> result = analysis.hessian(scale=1)
        >>> np.allclose(result.std, [1.0, 1.0])
        True
        """
        options = HessianOptions()
        options.step_sizes = [] if step_sizes is None else step_sizes
        options.parameter_scales = [] if parameter_scales is None else parameter_scales
        options.relative_step = relative_step
        options.confidence_level = confidence_level
        options.eigenvalue_tolerance = eigenvalue_tolerance
        options.step_tolerance = step_tolerance
        options.batch_size = batch_size
        options.step_candidates = step_candidates
        policies = {"strict": CurvaturePolicy.Strict,
                    "regularize": CurvaturePolicy.Regularize,
                    "positive_modes": CurvaturePolicy.PositiveModes}
        if not isinstance(curvature_policy, str) or curvature_policy not in policies:
            raise ValueError("curvature_policy must be 'strict', 'regularize', or 'positive_modes'")
        options.curvature_policy = policies[curvature_policy]
        return self._backend.hessian(scale, options)

    def bootstrap(self, objective_sampler, n_samples=200, *, optimizer="ARRDE",
                  maxevals=10000, seed=None, confidence_level=0.95,
                  warm_start=True, require_convergence=False, options=None):
        """Refit statistically resampled objectives and summarize parameter samples.

        Parameters
        ----------
        objective_sampler : callable
            Called as ``objective_sampler(index, sample_seed)`` for each requested
            realization. Must return a vectorized objective accepting parameter
            vectors and returning one value per vector. Generate and capture a
            fixed realization in this sampler; do not resample on each objective
            evaluation. The sampling mechanism defines whether the bootstrap is
            parametric or nonparametric; Minion does not infer a data model.
        n_samples : int, default: 200
            Number of requested bootstrap realizations. Must be at least 2.
        optimizer : str, default: 'ARRDE'
            Optimizer used for each refit. Bounded optimizers require finite
            bounds. Use ``'L_BFGS'`` with ``warm_start=True`` for unconstrained
            refits, or ``'L_BFGS_B'`` for bounded refits.
        maxevals : int, default: 10000
            Positive objective evaluation budget for each refit.
        seed : int, optional
            Non-negative master seed for reproducible sampling and optimization.
            None or -1 generates a master seed, returned in ``result.seed``.
            Sampling seeds and optimizer seeds are derived separately; use the
            supplied ``sample_seed`` in `objective_sampler`.
        confidence_level : float, default: 0.95
            Confidence level in (0, 1) for marginal percentile intervals.
        warm_start : bool, default: True
            Use the original optimum as an initial guess for every refit.
        require_convergence : bool, default: False
            If True, use only fits reporting ``Converged``. Otherwise, accept
            valid finite converged, budget-limited, or stagnated fits, warning
            about optimization error for fits without confirmed convergence.
        options : dict, optional
            Existing optimizer configuration map passed to each refit. If None,
            use an empty map and the optimizer's defaults.

        Returns
        -------
        result : BootstrapResult
            Original estimate, bootstrap mean, standard uncertainties,
            covariance, correlation, and percentile intervals. ``samples`` has
            one row per requested realization, with NaN rows for invalid fits;
            ``used_indices`` selects rows used in the summary.
            ``realization_results`` preserves each fit or failure. Includes
            sampling and optimizer seeds, requested/valid/converged/used counts,
            total ``nfev``, ``status``, ``warnings``, ``covariance_available``,
            and ``covariance_valid``.

        Raises
        ------
        TypeError
            If `objective_sampler` is not callable.
        RuntimeError
            If the backend rejects the configuration, optimizer, or bounds.

        See Also
        --------
        Uncertainty.hessian : Estimate uncertainty from local curvature.

        Notes
        -----
        Repeated optimizations and statistical summaries execute sequentially
        in C++. Covariance uses the sample denominator ``n_used - 1``; intervals
        use linearly interpolated percentiles. ``covariance_available`` requires
        at least two used fits. ``covariance_valid`` additionally requires every
        requested realization to converge. Excluding fits may bias results.
        Sampler failures and failed refits are retained in the result diagnostics.
        """
        if not callable(objective_sampler):
            raise TypeError("objective_sampler must be callable")

        def sampler_for_cpp(index, sample_seed):
            objective = objective_sampler(index, sample_seed)
            if not callable(objective):
                raise TypeError("objective_sampler must return a callable objective")
            return lambda points, data: objective(points)

        config = BootstrapOptions()
        config.n_samples = n_samples
        config.optimizer = optimizer
        config.maxevals = maxevals
        config.seed = -1 if seed is None else seed
        config.confidence_level = confidence_level
        config.warm_start = warm_start
        config.require_convergence = require_convergence
        config.optimizer_options = {} if options is None else options
        return self._backend.bootstrap(sampler_for_cpp, config)
