# Noisy Hessian validation

Reproduce with `python examples/validate_hessian_noise.py`.
Seed: 20261118. Uses the actual C++ backend through the Python API.

## Problem and independent reference

A two-parameter Gaussian linear fit to 40 observations with known unit errors.
The design has nearly collinear columns; the exact negative-log-likelihood Hessian
has eigenvalues 200 and 0.0002 (condition number 1,000,000).
The true optimum is zero. The objective at the optimum is 19.
The exact covariance follows analytically from the known singular decomposition,
independently of numerical differentiation.
Both parameter standard deviations are **50**.
50,000 independent datasets are refitted using their closed-form least-squares
solution; empirical standard deviations are 50.0385 and
50.0384.

A deterministic +1e-8 evaluation error is added at the stencil center. This is
computational noise, separate from Gaussian observation errors. It is only
5.3e-10 of the objective baseline but shifts both estimated diagonal curvatures
by -2e-8/h^2. It overwhelms the weak curvature at small steps. The underlying
data model and reference covariance remain unchanged.

## Results

Coverage is the fraction of simulated refits whose marginal 95% interval includes
the true parameter, using each fixed reported covariance. Interval widths are
constant in this linear model. Parameter 0 is shown; parameter 1 behaves the same
by symmetry. This measures the calibration of the reported widths against the
underlying model; computational noise is not re-simulated for each refit.

| Case | Status | Selected h | Raw minimum eigenvalue | Reported SD | SD / exact SD | 95% coverage | Relative covariance error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean, h=0.01 | warning | 0.01 | 0.0002 | 50 | 1 | 94.982% | 2.28132e-07 |
| Noisy, strict, h=0.001 | indefinite | 0.001 | -0.0198 | unavailable | unavailable | unavailable | unavailable |
| Noisy, discard, h=0.001 | partial | 0.001 | -0.0198 | 0.0500025 | 0.00100005 | 0.140% | 1 |
| Noisy, repair, h=0.001 | forced_positive_definite | 0.001 | -0.0198 | 1.58201 | 0.0316401 | 4.970% | 0.999 |
| Noisy, repair, n_scales=9 | forced_positive_definite | 0.002 | -0.0048 | 1.58195 | 0.0316389 | 4.970% | 0.999 |
| Noisy, default probes=4 | partial | 0.008 | -0.0001125 | 0.05 | 0.001 | 0.140% | 1 |
| Noisy, default probes=4, repair | forced_positive_definite | 0.008 | -0.0001125 | 1.58193 | 0.0316386 | 4.970% | 0.999 |
| Noisy, default probes=4, h0=0.032 | warning | 0.256 | 0.000199695 | 50.0382 | 1.00076 | 95.000% | 0.00152821 |
| Noisy, one-shot probes=9 | warning | 0.256 | 0.000199695 | 50.0382 | 1.00076 | 95.000% | 0.00152821 |
| Noisy, repair, h=0.256 | warning | 0.256 | 0.000199695 | 50.0382 | 1.00076 | 95.000% | 0.00152821 |

## Independent random evaluation noise

Each configuration is repeated 300 times with independent N(0, 1e-8^2) errors
at the evaluated stencil points. All configurations enable repair. The coverage
below is calculated analytically from the exact Gaussian estimator distribution
conditional on each reported width, then averaged over the 300 numerical-noise
realizations. It is not an additional Monte Carlo estimate of the fit distribution.

| Configuration | Repaired | covariance_valid=True | SD/exact SD: 5th / median / 95th percentile | Mean expected 95% coverage |
| --- | --- | --- | --- | --- |
| h=0.001 | 163/300 | 137/300 | 0.03164 / 0.03164 / 0.2332 | 12.500% |
| n_scales=9 | 127/300 | 173/300 | 0.03164 / 0.1502 / 0.6053 | 25.757% |
| default probes=4 | 78/300 | 222/300 | 0.03164 / 0.6539 / 1.676 | 64.435% |
| four probes, h0=0.032 | 0/300 | 300/300 | 0.9985 / 0.9999 / 1.001 | 94.999% |
| one-shot probes=9 | 0/300 | 300/300 | 0.9987 / 1 / 1.001 | 95.001% |
| h=0.256 | 0/300 | 300/300 | 0.9987 / 1 / 1.001 | 95.000% |

## Interpretation

The clean calculation and adequately enlarged steps agree with the independent
covariance and achieve approximately nominal coverage. Strict mode correctly
declines uncertainty for the indefinite numerical Hessian. Discarding the weak
mode severely understates uncertainty.

MINUIT-style repair produces a finite positive-definite covariance, but the weak
curvature is set primarily by diagonal inflation. Its intervals are much too
narrow, despite a relative matrix correction of only
0.00155423. A small whole-matrix correction does not
imply a small error in uncertainty along a nearly flat direction.

Nine candidate scales do not fix this example: the whole-matrix stability metric
accepts an early adjacent pair because the strong mode masks changes in the weak
mode. Explicit h=0.256 resolves the curvature without repair. The new one-shot
coordinate probe option selects that same step and also recovers the uncertainty,
using 37 axis probe points in one call followed by 4 mixed points in a second
call. Nine full candidate Hessians cost 73 points in this two-dimensional case;
the savings grow with dimension. This succeeds here because the largest stable
coordinate step is preferred, not because coordinate probes resolve degeneracy.

The random-noise experiment checks whether the result extends beyond a special
center perturbation. A noisy but positive-definite estimate can also return
`covariance_valid=True` while giving inaccurate uncertainty: that flag currently
indicates numerical admissibility of the covariance, not demonstrated calibration.

This is a deliberately adverse deterministic numerical-error example, not a claim
that all repaired covariances fail. It demonstrates that positive definiteness
and finite intervals are insufficient validation. The existing
`covariance_valid=False` flag on repaired results is essential. More reliable
automatic handling needs mode-sensitive step checks or independent profile or
resampling checks. The default is now four probe scales (h through 8h). This range does not reach
the steps needed for this adverse example; increasing probe_scales to 9 or
supplying a larger initial step remains necessary. Starting four probes at
relative_step=0.032 reaches h=0.256 and recovers the reference uncertainty. The default change reduces
cost relative to nine scales but does not guarantee trustworthy uncertainty.
