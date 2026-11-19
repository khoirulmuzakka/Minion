"""Run from the repository root: python examples/validate_hessian_noise.py."""
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minionpy import Uncertainty


def main():
    rng = np.random.default_rng(20261118)
    # Nearly collinear columns, with an exactly known singular decomposition.
    u, _ = np.linalg.qr(rng.normal(size=(40, 2)))
    q = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2.)
    eigenvalues = np.array([200., 0.0002])
    design = (u * np.sqrt(eigenvalues)) @ q.T
    residual = rng.normal(size=40)
    residual -= u @ (u.T @ residual)
    residual *= np.sqrt(38.) / np.linalg.norm(residual)
    exact_cov = (q / eigenvalues) @ q.T
    exact_std = np.sqrt(np.diag(exact_cov))

    def objective(points):
        r = np.asarray(points) @ design.T - residual
        return .5 * np.sum(r**2, axis=1)

    # Numerical evaluation error, separate from the observation noise.
    def noisy(points):
        points = np.asarray(points)
        return objective(points) + 1e-8*np.all(points == 0, axis=1)

    # Closed-form refits of independent datasets, each with unit Gaussian errors.
    noise = rng.normal(size=(50000, 40))
    estimates = ((noise @ u) / np.sqrt(eigenvalues)) @ q.T
    empirical_std = estimates.std(axis=0, ddof=1)
    assert np.allclose(empirical_std, exact_std, rtol=.015)
    z = NormalDist().inv_cdf(.975)
    cases = [
        ('Clean, h=0.01', objective, dict(step_sizes=[.01]*2)),
        ('Noisy, strict, h=0.001', noisy, dict(curvature_policy="strict")),
        ('Noisy, positive modes, h=0.001', noisy, dict(curvature_policy='positive_modes')),
        ('Noisy, repair, h=0.001', noisy, dict(curvature_policy="regularize")),
        ('Noisy, default candidates=4', noisy, {}),
        ('Noisy, four candidates, strict', noisy, dict(step_candidates=4, curvature_policy="strict")),
        ('Noisy, default candidates=4, h0=0.032', noisy, dict(relative_step=.032)),
        ('Noisy, one-shot candidates=9', noisy, dict(step_candidates=9)),
        ('Noisy, repair, h=0.256', noisy, dict(curvature_policy="regularize", step_sizes=[.256]*2)),
    ]
    rows, results = [], {}
    diagnostic_rows = []
    for name, f, options in cases:
        # Preserve fixed-step reference cases; default cases exercise the API default.
        if 'default candidates=4' not in name:
            options = {'step_candidates': 1, **options}
        r = Uncertainty(f, [0., 0.]).hessian(1, **options)
        results[name] = r
        diagnostic_rows.append('| ' + ' | '.join([
            name, str(all(r.step_selection_verified)), str(r.curvature_resolved),
            str(r.unresolved_modes)]) + ' |')
        if r.covariance_available:
            sd = np.asarray(r.std)
            coverage = np.mean(np.abs(estimates) <= z*sd, axis=0)
            error = np.linalg.norm(r.covariance-exact_cov)/np.linalg.norm(exact_cov)
            values = [f'{sd[0]:.6g}', f'{sd[0]/exact_std[0]:.6g}',
                      f'{100*coverage[0]:.3f}%', f'{error:.6g}']
        else:
            values = ['unavailable']*4
        rows.append('| ' + ' | '.join([name, r.status, f'{r.step_sizes[0]:.6g}',
                    f'{r.scaled_eigenvalues[0]:.6g}']+values) + ' |')

    clean = results['Clean, h=0.01']
    recovered = results['Noisy, repair, h=0.256']
    bad = results[cases[3][0]]
    assert np.allclose(clean.covariance, exact_cov, rtol=.005)
    assert np.allclose(recovered.covariance, exact_cov, rtol=.005)
    assert bad.forced_positive_definite and not bad.covariance_valid
    assert bad.std[0] < .1*exact_std[0]
    assert not bad.curvature_resolved
    assert bad.unresolved_modes == [0]
    assert bad.objective_calls == 1 and bad.nfev == 9
    assert not recovered.forced_positive_definite and recovered.covariance_valid
    probed = results['Noisy, one-shot candidates=9']
    assert np.allclose(probed.covariance, exact_cov, rtol=.005)
    assert probed.objective_calls == 2 and probed.nfev == 41

    # Repeat with independent random errors at every evaluated stencil point.
    # The center is evaluated once and reused by the C++ differentiation plan.
    random_rows = []
    normal = NormalDist()
    for label, options in [
        ('h=0.001', {}),
        ('default candidates=4', {}),
        ('four probes, h0=0.032', dict(step_candidates=4, relative_step=.032)),
        ('one-shot candidates=9', dict(step_candidates=9)),
        ('h=0.256', dict(step_sizes=[.256]*2)),
    ]:
        if label != 'default candidates=4':
            options = {'step_candidates': 1, **options}
        ratios, coverage, forced, valid = [], [], 0, 0
        for _ in range(300):
            def random_noisy(points):
                return objective(points) + rng.normal(0., 1e-8, len(points))
            r = Uncertainty(random_noisy, [0., 0.]).hessian(
                1, curvature_policy="regularize", **options)
            assert r.covariance_available
            ratio = r.std[0]/exact_std[0]
            ratios.append(ratio)
            # Exact normal coverage conditional on this reported interval width.
            coverage.append(2*normal.cdf(z*ratio)-1)
            forced += r.forced_positive_definite
            valid += r.covariance_valid
        lo, med, hi = np.quantile(ratios, [.05, .5, .95])
        random_rows.append(
            f'| {label} | {forced}/300 | {valid}/300 | '
            f'{lo:.4g} / {med:.4g} / {hi:.4g} | {100*np.mean(coverage):.3f}% |')
        if label in ('h=0.256', 'one-shot candidates=9', 'four probes, h0=0.032'):
            assert abs(med-1) < .005
            assert .945 < np.mean(coverage) < .955

    report = f"""# Noisy Hessian validation

Reproduce with `python examples/validate_hessian_noise.py`.
Seed: 20261118. Uses the actual C++ backend through the Python API.

## Problem and independent reference

A two-parameter Gaussian linear fit to 40 observations with known unit errors.
The design has nearly collinear columns; the exact negative-log-likelihood Hessian
has eigenvalues 200 and 0.0002 (condition number 1,000,000).
The true optimum is zero. The objective at the optimum is 19.
The exact covariance follows analytically from the known singular decomposition,
independently of numerical differentiation.
Both parameter standard deviations are **{exact_std[0]:.6g}**.
50,000 independent datasets are refitted using their closed-form least-squares
solution; empirical standard deviations are {empirical_std[0]:.6g} and
{empirical_std[1]:.6g}.

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
""" + '\n'.join(rows) + f"""

## Step verification and unresolved curvature

Fixed steps have no probe verification. Unresolved mode indices refer to the
original scaled Hessian, even if regularization returns a full covariance.
These diagnostics do not change the returned covariance or add function calls.

| Case | All coordinate steps verified | Raw curvature resolved | Unresolved modes |
| --- | --- | --- | --- |
""" + '\n'.join(diagnostic_rows) + f"""

## Independent random evaluation noise

Each configuration is repeated 300 times with independent N(0, 1e-8^2) errors
at the evaluated stencil points. All configurations enable repair. The coverage
below is calculated analytically from the exact Gaussian estimator distribution
conditional on each reported width, then averaged over the 300 numerical-noise
realizations. It is not an additional Monte Carlo estimate of the fit distribution.

| Configuration | Repaired | covariance_valid=True | SD/exact SD: 5th / median / 95th percentile | Mean expected 95% coverage |
| --- | --- | --- | --- | --- |
""" + '\n'.join(random_rows) + f"""

## Interpretation

The clean calculation and adequately enlarged steps agree with the independent
covariance and achieve approximately nominal coverage. The explicit strict policy correctly
declines uncertainty for the indefinite numerical Hessian. Discarding the weak
mode severely understates uncertainty.

MINUIT-style repair produces a finite positive-definite covariance, but the weak
curvature is set primarily by diagonal inflation. Its intervals are much too
narrow, despite a relative matrix correction of only
{bad.regularization_relative_change:.6g}. A small whole-matrix correction does not
imply a small error in uncertainty along a nearly flat direction.

Explicit h=0.256 resolves the curvature without repair. Nine coordinate
candidates select that same step and also recover the uncertainty,
using 37 axis probe points in one call followed by 4 mixed points in a second
call. Only one full Hessian is calculated after the preliminary probe batch.
This succeeds here because the largest stable
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
resampling checks. The default is four step candidates (h through 8h), with regularized curvature handling. This range does not reach
the steps needed for this adverse example; increasing step_candidates to 9 or
supplying a larger initial step remains necessary. Starting four probes at
relative_step=0.032 reaches h=0.256 and recovers the reference uncertainty. The default change reduces
cost relative to nine scales but does not guarantee trustworthy uncertainty.
"""
    path = Path(__file__).resolve().parents[1] / 'docs/hessian_noise_validation.md'
    path.write_text(report)
    print(report)


if __name__ == '__main__':
    main()
