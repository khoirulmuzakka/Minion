# Impact of bound-aware steps and curvature diagnostics

Baseline: git revision `546a835`. Current: working-tree C++ source snapshots.
Both versions were compiled as isolated Python extensions with the same compiler,
Eigen headers and bindings. The installed extension was not modified.

Reproduce: `python examples/compare_hessian_changes.py`.

## Accuracy of local curvature

The nonlinear cases compare with the exact inverse local Hessian; these are
derivative-accuracy checks, not claims of Gaussian statistical coverage.

| Case | Version | Selected h | Stencil | SD | Exact local SD | SD error | Evaluations/calls |
|---|---|---:|---|---:|---:|---:|---:|
| Interior nonlinear curvature | baseline | 0.08 | central | 0.662266 | 1 | 33.773% | 8/1 |
| Interior nonlinear curvature | current | 0.01 | central | 0.990148 | 1 | 0.985% | 3/1 |
| Active-bound nonlinear curvature | baseline | 0.333333 | forward | 3 | 1 | 200.000% | 4/1 |
| Active-bound nonlinear curvature | current | 0.1 | forward | 1.06 | 1 | 6.000% | 4/1 |
| Clean bounded quadratic | baseline | 0.16 | central | 0.1 | 0.1 | 0.000% | 9/1 |
| Clean bounded quadratic | current | 0.05 | central | 0.1 | 0.1 | 0.000% | 7/1 |

## Gaussian mean fits: paired coverage experiments

Each case uses 1,000 independent datasets summarized by their exact Gaussian
sample means, with both implementations evaluated at the same exact constrained
optimum. Nominal intervals are the returned symmetric marginal 95% intervals.
The bounded quadratic likelihood has a known unconstrained estimator SD.
At an active bound, the constrained estimator is non-Gaussian; Gaussian
intervals need not attain 95% coverage even with exact curvature.

For quantization, the objective is rounded in increments of 0.6 with a random
phase per dataset, fixed for all evaluations and shared between versions.
This is computational noise, separate from the Gaussian measurement noise.
Optimization error is deliberately excluded. The tight-bound case is adverse:
shrinking steps can make curvature indistinguishable from rounding noise.

Coverage is conditional on covariance being available. The accompanying
available count matters; unavailable results are not silently excluded from
the JSON's `covered_fraction_all`. Monte Carlo sampling errors for coverage
near 95% are about 0.7 percentage points for 1,000 trials.

| Case | Version | Available / 1000 | Median SD / reference SD | 95% coverage, available | Repairs | Mean evaluations/calls |
|---|---|---:|---:|---:|---:|---:|
| Clean interior Gaussian fits | baseline | 1000 | 1 | 95.60% | 0 | 9.002/1.000 |
| Clean interior Gaussian fits | current | 1000 | 1 | 95.60% | 0 | 6.350/1.000 |
| Tight bounds and quantized objective | baseline | 1000 | 1.0127 | 93.80% | 0 | 9.000/1.000 |
| Tight bounds and quantized objective | current | 701 | 0.86002 | 88.02% | 0 | 5.000/1.000 |
| Gaussian fits at an active bound | baseline | 1000 | 1 | 97.50% | 0 | 6.495/1.000 |
| Gaussian fits at an active bound | current | 1000 | 1 | 97.50% | 0 | 4.000/1.000 |

## New diagnostics

| Case | Unverified step selections / 1000 |
|---|---:|
| Clean interior Gaussian fits | 321 |
| Tight bounds and quantized objective | 1000 |
| Gaussian fits at an active bound | 1000 |

The unbounded nearly-degenerate control returns the same SD before and after: **1.58193**, against an exact SD of **50**.
Both versions use **21 evaluations / 2 calls**.
The new implementation flags unresolved raw curvature.
Regularization still does not recover the true uncertainty.

## Interpretation

A bound-distance cap is a locality constraint, not a general accuracy guarantee.
It can reduce nonlinear finite-difference bias, but can increase noise error.
A verification flag means coordinate probes passed the consistency heuristic;
it does not certify weak coupled directions or statistical coverage.
If the admissible local region is too small
relative to evaluation noise, improve objective accuracy, reconsider bounds,
or use an independent inference method rather than treating repaired intervals
as validated uncertainties.
