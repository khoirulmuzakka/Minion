"""Compare 1,000 serial/parallel bootstrap fits against exact Gaussian mean fits.

Run from the repository root: python examples/bootstrap_parallel.py
This tiny NumPy objective may be slower in parallel because callback/threading
overhead dominates; expensive GIL-releasing objectives are the intended use case.
"""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minionpy import Uncertainty


def main():
    sigma, n_observations = .2, 40
    observations = np.random.default_rng(2026).normal(.5, sigma, n_observations)
    fitted_mean = observations.mean()

    def objective_for(data):
        fixed = np.array(data, copy=True)
        def objective(points):
            means = np.asarray(points)[:, 0, None]
            return np.sum(((means-fixed)/sigma)**2, axis=1)
        return objective

    def sampler(index, seed):
        # Each fitting worker calls this sampler and owns its fixed dataset.
        data = np.random.default_rng(seed).normal(fitted_mean, sigma, n_observations)
        return objective_for(data)

    analysis = Uncertainty(objective_for(observations), [fitted_mean], [(0., 1.)])
    results = {}
    for jobs in (1, 4):
        started = perf_counter()
        result = analysis.bootstrap(
            sampler, n_samples=1000, n_jobs=jobs, optimizer='NelderMead',
            maxevals=1000, seed=42, require_convergence=True,
            options={'x_tol': 1e-9, 'f_tol': -1.})
        elapsed = perf_counter()-started
        assert result.n_used == 1000 and result.n_jobs_used == jobs
        references = [np.random.default_rng(seed).normal(
            fitted_mean, sigma, n_observations).mean() for seed in result.sample_seeds]
        np.testing.assert_allclose(result.samples[:, 0], references, atol=1e-7)
        np.testing.assert_allclose(result.std, [np.std(references, ddof=1)], rtol=1e-6)
        results[jobs] = result
        print(f'n_jobs={jobs}: {result.n_used} fits, SD={result.std[0]:.8f}, '
              f'evaluations={result.nfev}, elapsed={elapsed:.3f}s')
    np.testing.assert_array_equal(results[1].samples, results[4].samples)
    np.testing.assert_array_equal(results[1].covariance, results[4].covariance)
    assert results[1].sample_seeds == results[4].sample_seeds
    assert results[1].optimizer_seeds == results[4].optimizer_seeds
    assert results[1].nfev == results[4].nfev
    print('Serial and parallel results are identical and match exact replica refits.')
    print(f'Population reference SD: {sigma/np.sqrt(n_observations):.8f}')


if __name__ == '__main__':
    main()
