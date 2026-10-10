Statistical uncertainty
=======================

Minion estimates statistical uncertainty from local Hessian curvature at a
supplied optimum or from repeated bootstrap fits. All differentiation,
optimizations, matrix calculations, quantiles, and diagnostics run in C++;
minionpy exposes the same implementation.

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

What statistical uncertainty means
----------------------------------

Let :math:`Y` denote the measured data and :math:`\theta\in\mathbb{R}^D`
the model parameters. A fit produces an estimator
:math:`\hat\theta(Y)=\operatorname*{arg\,min}_{\theta} f(\theta;Y)`.
Repeated experiments produce different datasets and therefore different fitted
parameters. Statistical uncertainty describes this variation, conditional on
the assumed data-generating model and fitting procedure:

.. math::

   C = \operatorname{Cov}_{Y}(\hat\theta)
     = \mathbb{E}_{Y}\left[
       (\hat\theta-\mathbb{E}_{Y}\hat\theta)
       (\hat\theta-\mathbb{E}_{Y}\hat\theta)^T\right].

The standard uncertainty of parameter :math:`i` is :math:`u_i=\sqrt{C_{ii}}`.
It has the same units as that parameter. Off-diagonal entries describe how fitted
parameters vary together; their dimensionless correlations are

.. math::

   \rho_{ij}=\frac{C_{ij}}{\sqrt{C_{ii}C_{jj}}}.

Minion estimates this covariance from either local likelihood curvature or
repeated bootstrap fits. Statistical uncertainty is distinct from optimizer
error and numerical differentiation error. It does not automatically include
model misspecification, systematic measurement errors, or estimator bias.
Such effects require a statistical model or resampling procedure that includes
them. Gaussian errors in a fixed dataset do not make objective evaluations
stochastic: the resulting chi-squared function can still be deterministic.

Likelihood curvature and the Hessian derivation
-----------------------------------------------

For likelihood :math:`L(\theta;Y)`, define the negative log likelihood
:math:`\ell(\theta)=-\log L(\theta;Y)`. At a smooth interior optimum,
:math:`\nabla\ell(\hat\theta)=0`. With
:math:`\delta=\theta-\hat\theta`, the quadratic Taylor expansion is

.. math::

   \ell(\hat\theta+\delta)
   \simeq \ell(\hat\theta)+\frac{1}{2}\delta^T H_\ell\delta,
   \qquad
   (H_\ell)_{ij}=
   \left.\frac{\partial^2\ell}{\partial\theta_i\partial\theta_j}
   \right|_{\hat\theta}.

Consequently, the local likelihood ratio has Gaussian form:

.. math::

   \frac{L(\hat\theta+\delta;Y)}{L(\hat\theta;Y)}
   \simeq \exp\left(-\frac{1}{2}\delta^T H_\ell\delta\right).

Comparing this quadratic form with a multivariate Gaussian identifies the local
covariance estimate :math:`\hat C=H_\ell^{-1}` when curvature is positive
definite. Under regularity conditions and sufficiently informative data, this
inverse observed information approximates the sampling covariance of the
maximum likelihood estimator. The likelihood ratio itself is not a sampling
distribution for the parameter; the frequentist interpretation relies on those
conditions.

Minion differentiates the supplied objective :math:`f`, returning its **raw
second-derivative matrix** in ``result.hessian``:

.. math::

   H_f=\nabla^2 f(\hat\theta),
   \qquad \hat C=s H_f^{-1},
   \qquad s=\texttt{scale}.

If :math:`f=s\ell+\text{constant}`, then :math:`H_f=sH_\ell`, which explains
the required scale. Multiplying the objective by a positive constant requires
multiplying ``scale`` by the same constant to preserve uncertainty.

For :math:`f=\chi^2=-2\log L+\text{constant}`, the expansion becomes

.. math::

   \Delta\chi^2\simeq\frac{1}{2}\delta^T H_{\chi^2}\delta
   =\delta^T\hat C^{-1}\delta,
   \qquad \hat C=2H_{\chi^2}^{-1}.

Thus ``scale=2`` absorbs the Taylor expansion factor. If the statistical Hessian
is instead defined as :math:`\mathcal H=\tfrac12\nabla^2\chi^2`, then
:math:`\hat C=\mathcal H^{-1}`. Minion returns :math:`H_{\chi^2}`, rather than
:math:`\mathcal H`, so these conventions agree without changing the reported
raw Hessian. For :math:`f=-\log L`, use ``scale=1``.

Gaussian curve fitting
~~~~~~~~~~~~~~~~~~~~~~

For independent observations with known standard deviations,

.. math::

   Y_k=m(t_k;\theta_0)+\varepsilon_k,\qquad
   \varepsilon_k\sim\mathcal N(0,\sigma_k^2),\qquad
   \chi^2(\theta)=\sum_k\frac{[m(t_k;\theta)-Y_k]^2}{\sigma_k^2}.

Writing :math:`r_k=m(t_k;\theta)-Y_k`,
:math:`J_{ki}=\partial m(t_k;\theta)/\partial\theta_i`, and
:math:`W=\operatorname{diag}(\sigma_k^{-2})`, the exact objective Hessian is

.. math::

   (H_{\chi^2})_{ij}
   =2(J^TWJ)_{ij}
    +2\sum_k\frac{r_k}{\sigma_k^2}
       \frac{\partial^2m(t_k;\theta)}{\partial\theta_i\partial\theta_j}.

For a polynomial :math:`m(t;\theta)=\sum_{j=0}^{D-1}\theta_j t^j`, the model
is linear in its coefficients. The second term vanishes, :math:`J=X` with
:math:`X_{kj}=t_k^j`, and

.. math::

   H_{\chi^2}=2X^TWX,\qquad \hat C=(X^TWX)^{-1}.

For a full-rank, unconstrained linear model with the stated Gaussian errors,
this is the exact sampling covariance of the fitted coefficients. For nonlinear
models the Hessian covariance is generally a local approximation. Unknown
noise variances need additional estimation; Minion does not automatically
multiply covariance by reduced chi-squared.

Standard uncertainties and confidence intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For confidence level :math:`1-\alpha`, Minion returns marginal normal intervals:

.. math::

   [\hat\theta_i-z_{1-\alpha/2}u_i,
    \hat\theta_i+z_{1-\alpha/2}u_i],\qquad
   z_p=\Phi^{-1}(p).

At 95 percent confidence, :math:`z_{0.975}\simeq1.96`. Repeated intervals
constructed this way have approximately the stated coverage under the Gaussian
approximation. These are individual parameter intervals; they are not a joint
95 percent region for all parameters. With a complete covariance, the Gaussian
joint region instead uses
:math:`\delta^T\hat C^{-1}\delta\leq\chi^2_{D,1-\alpha}`.
Minion does not return that joint region automatically.

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
defaults to -1, evaluating all candidate stencil points in one objective call.
Positive values limit the number of points in each call; zero and values below
-1 are rejected. ``n_scales`` defaults to 1 and accepts integers from 1 to 30.
Candidate steps are h, 2h, 4h, etc. Central stencils use
``1 + 2*n_scales*D*D`` unique points, sharing the center across scales.
With multiple scales, the larger member of the first consistent adjacent pair
is selected; if none meets ``stability_tolerance``, the most consistent pair is
used with a warning. Consistency is the relative Frobenius difference of scaled
Hessians, normalized by the larger norm. It is reported as
``differentiation_error`` (zero when only one scale is requested), and is not a
rigorous error bound or proof that evaluation noise is absent.
``check_stability=True`` requests at least two scales for compatibility.
Near bounds, steps may be capped and candidate stencils may coincide.
``nfev`` counts unique points and ``objective_calls`` counts batches.

For a central stencil, diagonal and mixed entries are estimated by

.. math::

   (H_f)_{ii}\simeq
   \frac{f(\hat\theta+h_i e_i)-2f(\hat\theta)
         +f(\hat\theta-h_i e_i)}{h_i^2},

.. math::

   (H_f)_{ij}\simeq
   \frac{f(\hat\theta+h_i e_i+h_j e_j)
        -f(\hat\theta+h_i e_i-h_j e_j)
        -f(\hat\theta-h_i e_i+h_j e_j)
        +f(\hat\theta-h_i e_i-h_j e_j)}{4h_i h_j},\qquad i\ne j.

Here :math:`e_i` is the unit vector for coordinate :math:`i`. One triangle is
computed and mirrored. A single central scale needs
:math:`1+2D+4\binom{D}{2}=1+2D^2` evaluations. Evaluation noise is amplified by
:math:`h_i^{-2}` on the diagonal and :math:`(h_i h_j)^{-1}` off the diagonal;
larger candidate steps can reduce that amplification but increase truncation
bias. One scale does not perform a step-sensitivity comparison.

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

Specifically, let :math:`S=\operatorname{diag}(s_1,\ldots,s_D)` contain
``parameter_scales`` (distinct from the statistical ``scale`` :math:`s`). In
scaled coordinates :math:`\theta=\hat\theta+Sz`, write

.. math::

   SH_fS=V\Lambda V^T,\qquad
   \tau=\texttt{eigenvalue\_tolerance}\max_k|\lambda_k|,\qquad
   P=\{k:\lambda_k>\tau\}.

The returned positive-mode covariance is

.. math::

   \hat C_+=sS\left(\sum_{k\in P}
                    \frac{v_kv_k^T}{\lambda_k}\right)S.

If every mode is retained, this equals :math:`sH_f^{-1}`. Otherwise it is only
the retained contribution. A zero-curvature direction has no local restoring
curvature and cannot be assigned a finite uncertainty by this approximation;
negative curvature cannot define a Gaussian variance. Setting those inverse
eigenvalues to zero is a computational fallback, not a claim that the omitted
directions are known exactly. The partial intervals consequently do not carry
the complete Gaussian coverage interpretation described above.

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

For the Gaussian parametric bootstrap, each replica is simulated around the
fitted theory, using the known observation errors, and fitted independently:

.. math::

   Y_k^{*(b)}=m(t_k;\hat\theta)+\sigma_k Z_k^{(b)},\qquad
   Z_k^{(b)}\sim\mathcal N(0,1),\qquad
   \hat\theta^{*(b)}=\operatorname*{arg\,min}_{\theta}
                     \chi^2(\theta;Y^{*(b)}).

The bootstrap replaces the unknown true parameters with the original fit and
approximates repeated experiments under that fitted model. For :math:`B`
accepted fits (``n_used``), Minion computes

.. math::

   \bar\theta^*=\frac{1}{B}\sum_{b=1}^{B}\hat\theta^{*(b)},\qquad
   \hat C_{\mathrm{boot}}=
     \frac{1}{B-1}\sum_{b=1}^{B}
       (\hat\theta^{*(b)}-\bar\theta^*)
       (\hat\theta^{*(b)}-\bar\theta^*)^T,\qquad
   u_{i,\mathrm{boot}}=\sqrt{(\hat C_{\mathrm{boot}})_{ii}}.

This standard deviation estimates the uncertainty of a single fitted parameter;
it is not divided by :math:`\sqrt B`. Increasing the replica count reduces the
Monte Carlo error of the uncertainty estimate, not the experimental uncertainty
itself. ``mean`` is :math:`\bar\theta^*`; ``estimate`` remains
:math:`\hat\theta`. Their difference can indicate bootstrap bias, but Minion
does not automatically correct it.

Marginal percentile intervals are

.. math::

   [Q_i^*(\alpha/2),Q_i^*(1-\alpha/2)],

where :math:`Q_i^*` is the empirical quantile of the accepted parameter samples.
For sorted samples :math:`a_0\leq\cdots\leq a_{B-1}`, the implemented quantile
uses :math:`r=(B-1)p`, :math:`j=\lfloor r\rfloor`, and
:math:`Q(p)=(1-r+j)a_j+(r-j)a_{j+1}` when :math:`j<B-1`; at :math:`p=1`
it returns :math:`a_{B-1}`. These intervals allow asymmetric sampling
distributions but still depend on the sampling model and successful fitting.
For Gaussian linear models, bootstrap and Hessian covariance should agree up
to bootstrap Monte Carlo error and numerical fitting error.

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
