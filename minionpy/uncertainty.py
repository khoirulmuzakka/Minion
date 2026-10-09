"""Statistical uncertainty calculated by the Minion C++ backend."""
from .minionpycpp import Uncertainty as _Uncertainty, HessianOptions, HessianResult
from .minionpycpp import BootstrapOptions, BootstrapResult

__all__ = ["Uncertainty", "HessianOptions", "HessianResult", "BootstrapOptions", "BootstrapResult"]


class Uncertainty:
    """Analyze a fixed optimum using a vectorized objective, as in Minimizer.

    The objective accepts a batch of parameter vectors and returns one value
    per vector. Bounds may be omitted for an unconstrained objective.
    """

    def __init__(self, objective, optimum, bounds=None):
        if not callable(objective):
            raise TypeError("objective must be callable")
        self._objective = objective
        self._backend = _Uncertainty(
            lambda points, data: objective(points), optimum,
            [] if bounds is None else bounds, None)

    def hessian(self, scale, *, step_sizes=None, parameter_scales=None,
                relative_step=1e-3, confidence_level=0.95,
                eigenvalue_tolerance=1e-10, stability_tolerance=0.05,
                batch_size=-1, check_stability=True, discard_nonpositive=True):
        """Return local Gaussian uncertainty and numerical diagnostics.

        Explicit scale is required: 1 for negative log likelihood, 2 for
        chi-squared with known variances. Other objectives require a user
        supplied statistical interpretation. Intervals are marginal normal
        intervals and are not clipped to parameter bounds.

        Steps and parameter scales are positive vectors in parameter units.
        The condition number uses the scaled Hessian. By default, non-positive
        or unresolved modes are discarded and positive-mode uncertainty is
        returned with status='partial', uncertainty_complete=False and
        covariance_valid=False. covariance_available indicates whether numbers
        were returned. These are incomplete contributions, not full parameter
        uncertainties; zero contributions do not establish zero uncertainty.
        discarded_modes index scaled_eigenvalues and scaled_eigenvectors.
        Correlations involving zero retained variance are NaN. With no resolved
        positive modes, uncertainty arrays are empty. Set discard_nonpositive=False
        to require positive definite curvature and retain strict behavior.
        Warnings identify bounds, step sensitivity and nonstationary points.
        batch_size=-1 evaluates each complete stencil in one objective call;
        positive values limit the number of parameter vectors per call.
        """
        options = HessianOptions()
        options.step_sizes = [] if step_sizes is None else step_sizes
        options.parameter_scales = [] if parameter_scales is None else parameter_scales
        options.relative_step = relative_step
        options.confidence_level = confidence_level
        options.eigenvalue_tolerance = eigenvalue_tolerance
        options.stability_tolerance = stability_tolerance
        options.batch_size = batch_size
        options.check_stability = check_stability
        options.discard_nonpositive = discard_nonpositive
        return self._backend.hessian(scale, options)

    def bootstrap(self, objective_sampler, n_samples=200, *, optimizer="ARRDE",
                  maxevals=10000, seed=None, confidence_level=0.95,
                  warm_start=True, require_convergence=False, options=None):
        """Fit statistically resampled objectives and summarize parameter samples.

        objective_sampler(index, sample_seed) returns one vectorized objective.
        Generate/capture the replica once in the sampler; do not resample on
        each objective evaluation. The sampling mechanism determines whether
        this is a parametric or nonparametric bootstrap. Minion does not infer
        a statistical model. Returned objectives take batches as in Minimizer.

        Minion performs all repeated optimizations, covariance calculations
        (sample denominator n_used-1), correlations and interpolated percentile
        intervals in C++. Sampling and optimizer seeds are separate. Pass seed
        for reproducibility and use the provided sample_seed in your sampler.
        warm_start uses the original optimum as an initial guess for each fit.

        By default, valid finite converged, budget-limited or stagnated fits
        are used, with warnings about possible optimization error for fits
        without confirmed convergence. Set require_convergence=True to accept
        only fits reporting Converged.
        samples contains one row per requested realization; invalid fits have
        NaN rows. used_indices selects rows used in the statistical summary.
        realization_results preserves every fit or failure. covariance_available
        requires at least two used fits; covariance_valid additionally requires
        every requested realization to converge. Exclusions may bias results.

        options passes the existing optimizer configuration map. Bounded
        optimizers require finite bounds; use L_BFGS with warm_start=True for
        an unconstrained fit. Execution is sequential.
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
