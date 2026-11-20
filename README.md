# Minion: C++ and Python Optimization Library

<div align="center">
  <img src="docs/logo.png" alt="Minion Logo" width="600" />
</div>

![CI](https://github.com/khoirulmuzakka/Minion/actions/workflows/ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/minionpy.svg)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/minionpy)
![PyPI pip downloads](https://img.shields.io/pypi/dm/minionpy.svg)
![PyPI License](https://img.shields.io/pypi/l/minionpy.svg)
[![Documentation Status](https://readthedocs.org/projects/minion-py/badge/?version=latest)](https://minion-py.readthedocs.io/en/latest/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10145/status.svg)](https://doi.org/10.21105/joss.10145)

**Minion/MinionPy** is an optimization library designed for solving single-objective optimization problems. It features state-of-the-art evolutionary algorithms, including top-performing methods from IEEE CEC competitions, which are not commonly found in standard optimization libraries such as SciPy, NLopt, OptimLib, pyGMO, and pagmo2.

Minion also serves as a research platform for developing and testing new optimization algorithms. It includes benchmark functions from IEEE CEC competitions (2011, 2014, 2017, 2019, 2020, and 2022), providing a robust framework for algorithm evaluation and comparison.

## Why Minion?
- **State-of-the-art optimization algorithms** :
  - **Differential Evolution-based algorithms:**
    - Basic Differential Evolution (DE)
    - JADE  
    - LSHADE  
    - ARRDE *(Adaptive Restart-Refine DE)*  
    - other advanced DE variants
  - **Other population-based algorithms:**
    - Artificial Bee Colony (ABC)
    - Canonical PSO, SPSO-2011, and Dynamic Multi-Swarm PSO (DMS-PSO)  
    - CMA-ES *(Covariance Matrix Adaptation Evolution Strategy)*  
    - BIPOP-aCMAES 
    - RCMAES (Restart aCMAES)
  - **Classical optimization algorithms:**
    - Gradient-descent
    - Nelder-Mead  
    - Generalized Simulated Annealing (Dual Annealing) 
    - L-BFGS-B (vectorized & noise-robust) 
    - L-BFGS (vectorized & noise-robust) 
- **Native Batch Evaluation**
  - Algorithms work with **vectorized / batched objective evaluations** by default, making it straightforward to use multithreading or multiprocessing inside the objective when needed.
- **C++ Library with Python Bindings**
  - The core implementation is in C++, with matching Python access through `minionpy`.
- **CEC Benchmark Suite**  
  - Includes `CEC2011`, `CEC2014`, `CEC2017`, `CEC2019`, `CEC2020`, and `CEC2022` benchmark problems.  
  - The benchmark problems are directly adapted from the original C++ implementations.  
  - The CEC2011 suite, containing 22 real-world optimization problems, has been completely rewritten in C++ from the original MATLAB code. The CEC2011 rewrite has been checked by running the original MATLAB code through Octave with `comparison_scripts/verify_cec2011_octave.py` using 1000 random samples per problem drawn from the allowed bounds. With a relative tolerance of `1e-9`, all rewritten CEC2011 problems agree with the Octave-evaluated MATLAB reference except `F3`, `F4`, `F21`, and `F22`. Observed mismatches are:
    - `F3` (failed `239/1000`, max relative error `5.25e-6`)
    - `F4` (failed `986/1000`, max relative error `0.39`)
    - `F21` (failed `75/1000`, max relative error `1.46`)
    - `F22` (failed `89/1000`, max relative error `1.42`)

## Installation and Usage

### Build dependencies
- CMake >= 3.18
- C++17 compiler (GCC/Clang/MSVC)
- Eigen3 (or allow automatic fetch via CMake)
- Optional for Python bindings: Python 3 + `pybind11`
- Optional for documentation: `doxygen`, `pandoc`, Python 3, `sphinx`, `sphinx-rtd-theme`, `nbsphinx`, `breathe`

Downloaded Eigen and COCO dependencies are cached in `.dependencies/`, separately
from compiled output in `build/`. Removing `build/` keeps these downloads for the
next configure. Generator/platform variants use separate cache subdirectories.
The cache is shared by builds of this checkout; configure them sequentially.
To check upstream dependencies again, configure with
`-DFETCHCONTENT_UPDATES_DISCONNECTED=OFF` (set it back to `ON` afterward).
You can override the cache location with `-DFETCHCONTENT_BASE_DIR=/path/to/cache`.
On Windows, MSVC compiles C++ files with up to eight processes per project.

Detailed installation, native package, C++ build, and usage instructions are available in the official documentation.

## Algorithm Usage
Across both C++ and Python, Minion uses the same core model:
- The objective must be vectorized: it receives a batch of candidate points and returns one objective value per point.
- `x0` contains multiple initial guesses, where each inner vector/list is one candidate start.
- `x0` is not a full population specification. In population-based algorithms, Minion first initializes the population using the algorithm's normal rule, then evaluates and ranks all supplied guesses and replaces initialized individuals with the best `min(len(x0), N0)` guesses, where `N0` is the initial population size. Ranking evaluations count toward the evaluation budget, and retained guesses are not evaluated twice. In non-population-based algorithms, Minion evaluates the guesses and starts from the best one.
- After constructing the `Minimizer`, call `optimize()` to run the algorithm and get a result object.

### C++ Code

Minion expects a vectorized objective function:

```cpp
#include <minion.h>
#include <iostream>
#include <vector>

double rosenbrock_scalar(const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i + 1 < x.size(); ++i) {
        const double a = x[i + 1] - x[i] * x[i];
        const double b = 1.0 - x[i];
        sum += 100.0 * a * a + b * b;
    }
    return sum;
}

std::vector<double> rosenbrock_batch(const std::vector<std::vector<double>>& X, void*) {
    std::vector<double> out;
    out.reserve(X.size());
    for (const auto& x : X) {
        out.push_back(rosenbrock_scalar(x));
    }
    return out;
}

int main() {
    const size_t dim = 10;
    std::vector<std::pair<double, double>> bounds(dim, {-5.0, 5.0});

    // x0 is a vector of guesses: each inner vector is one candidate start.
    std::vector<std::vector<double>> x0 = {
        std::vector<double>(dim, 0.0),
        std::vector<double>(dim, 1.0)
    };

    auto options = minion::DefaultSettings().getDefaultSettings("DE");
    options["maxiters"] = -1;
    options["population_size"] = 50;
    options["x_tol"] = 1e-6;
    options["f_tol"] = 1e-12;

    minion::Minimizer optimizer(
        rosenbrock_batch, bounds, x0, nullptr, nullptr, "DE", 20000, 42, options
    );

    minion::MinionResult result = optimizer.optimize();
    std::cout << "best f  = " << result.fun << "\n";
    std::cout << "status  = " << result.status << "\n";
    std::cout << "message = " << result.message << "\n";
}
```

In C++, `MinionFunction` is the vectorized entry point: it takes `std::vector<std::vector<double>>` and returns one value per candidate point.
Callbacks have signature `bool(minion::MinionResult*)`; return `true` to stop optimization early and `false` to continue.

### Python Code

MinionPy uses the same idea: the objective function should be vectorized and accept a batch of points.

```python
import minionpy as mpy

def rosenbrock_batch(X):
    values = []
    for x in X:
        total = 0.0
        for i in range(len(x) - 1):
            a = x[i + 1] - x[i] * x[i]
            b = 1.0 - x[i]
            total += 100.0 * a * a + b * b
        values.append(total)
    return values

dimension = 10
bounds = [(-5.0, 5.0)] * dimension

# x0 is a list of guesses: each inner list is one candidate start.
x0 = [
    [0.0] * dimension,
    [1.0] * dimension,
]

options = {
    "maxiters": -1,
    "population_size": 50,
    "x_tol": 1e-6,
    "f_tol": 1e-6,
}

optimizer = mpy.Minimizer(
    func=rosenbrock_batch,
    x0=x0,
    bounds=bounds,
    algo="DE",
    maxevals=20000,
    callback=None,
    seed=42,
    options=options,
)

result = optimizer.optimize()
print("best f =", result.fun)
print("status =", result.status)
print("message =", result.message)
```

In Python, `func(X)` receives `X` as `list[list[float]]` and returns one value per candidate point.
Python callbacks receive a `MinionResult`; return `True` to stop optimization early, or `False`/`None` to continue.

## Statistical Uncertainty

Minion provides statistical uncertainty estimates for fitted parameters in both
C++ and Python through the same C++ implementation:

- **Hessian analysis** estimates local parameter uncertainties, covariance,
  correlations, and marginal normal confidence intervals at a supplied optimum.
- **Bootstrap analysis** repeatedly fits user-generated datasets and reports
  parameter standard deviations, covariance, correlations, and marginal
  percentile intervals.

For example, estimate uncertainties for a straight-line fit with known Gaussian
measurement errors:

```python
import numpy as np
import minionpy as mpy

t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([1.1, 2.9, 5.2, 6.8, 9.1])
errors = np.full(t.shape, 0.2)

def chi_squared(parameters):
    p = np.asarray(parameters)
    predictions = p[:, 0, None] + p[:, 1, None] * t
    return np.sum(((predictions - y) / errors) ** 2, axis=1).tolist()

bounds = [(-10.0, 10.0), (-10.0, 10.0)]
fit = mpy.Minimizer(
    chi_squared, bounds=bounds, x0=[[1.0, 2.0]],
    algo="NelderMead", maxevals=4000, seed=42,
    options={"x_tol": 1e-9},
).optimize()

analysis = mpy.Uncertainty(chi_squared, fit.x, bounds)
uncertainty = analysis.hessian(
    scale=2,
    step_candidates=4,               # Default: h, 2h, 4h, 8h in one probe batch.
    curvature_policy="regularize",    # Default: MINUIT-style repair if needed.
)
if uncertainty.covariance_available:
    print("parameter estimates:", fit.x)
    print("standard uncertainties:", uncertainty.std)
    print("covariance:", uncertainty.covariance)
    print("correlation:", uncertainty.correlation)
print("status:", uncertainty.status)
print("covariance valid:", uncertainty.covariance_valid)
print("uncertainty complete:", uncertainty.uncertainty_complete)
print("selected steps:", uncertainty.step_sizes)
print("diagnostics:", uncertainty.warnings)
```

For resolved positive-definite curvature, Hessian covariance is
`scale * inverse(Hessian)`: use `scale=2` for chi-squared with known observation
variances, or `scale=1` for negative log likelihood. Other objectives need a
justified statistical scaling. Repaired results use the adjusted Hessian instead.

The equivalent C++ configuration uses `minion::uncertainty`:

```cpp
// chi_squared_batch is a MinionFunction for the same statistical objective;
// fit and bounds come from the corresponding C++ fit.
minion::uncertainty::Uncertainty analysis(chi_squared_batch, fit.x, bounds);
minion::uncertainty::HessianOptions options;
options.step_candidates = 4;
options.curvature_policy = minion::uncertainty::CurvaturePolicy::Regularize;
auto uncertainty = analysis.hessian(2.0, options);
```

### Hessian controls and diagnostics

The Python and C++ APIs share the following options and defaults:

| Option | Default | Meaning |
|---|---|---|
| `step_candidates` | `4` | Probe `h, 2h, 4h, 8h`; use `1` for fixed steps without probing, or `3�30` for automatic selection. |
| `relative_step` | `1e-3` | Initial step relative to the parameter scale. |
| `step_sizes` | Not supplied | Explicit initial steps, overriding `relative_step`; probing may enlarge them. |
| `parameter_scales` | `max(1, abs(x[i]))` | Parameter units used for initial steps and conditioning. |
| `step_tolerance` | `0.05` | Relative agreement required between coordinate curvature estimates. |
| `eigenvalue_tolerance` | `1e-10` | Relative cutoff for unresolved scaled Hessian modes. |
| `curvature_policy` | `"regularize"` | Handling of non-positive or unresolved curvature; choices below. |
| `confidence_level` | `0.95` | Confidence level for marginal Gaussian intervals. |
| `batch_size` | `-1` | All missing final stencil points in one callback; positive values limit final batch size. |

All coordinate probes are evaluated in **one preliminary batch**. For each
parameter, select the largest candidate with three consistent positive curvature
estimates, then calculate **one full Hessian**, reusing probe evaluations. If no
triple passes, select the best positive pair or the largest candidate, with an
explicit warning. Probing always uses one batch, even when `batch_size` limits
the final batches. With central stencils and four candidates, the probe batch
contains `8*D + 1` points and the final batch adds `2*D*(D - 1)` mixed points.
`nfev` counts evaluated points; `objective_calls` counts actual callbacks.

The curvature policies are:

- **`"regularize"` (default):** apply MINUIT-style diagonal inflation when needed.
  Repaired results have `status="forced_positive_definite"`,
  `covariance_available=True`, and `covariance_valid=False`.
- **`"strict"`:** return Hessian diagnostics without covariance when any mode is
  non-positive or unresolved.
- **`"positive_modes"`:** return only resolved positive-mode contributions. Partial
  results have `status="partial"` and `uncertainty_complete=False`; zero
  contributions from omitted modes do not establish zero uncertainty.

C++ uses `CurvaturePolicy::Regularize`, `Strict`, or `PositiveModes` rather than
policy strings. A completely zero Hessian cannot be repaired without an
arbitrary curvature scale and leaves covariance unavailable.

Inspect `status`, `warnings`, `step_sizes`, and `eigenvalues` before interpreting
results. `regularized_hessian` and `regularization_relative_change` describe any
repair; the original Hessian remains available. `step_selection_verified` and
`step_limited_by_bounds` describe coordinate step selection. Central steps
are capped at 10% of the nearest-bound distance; near a bound, inward one-sided
steps use at most 10% of the available inward distance. `unresolved_modes`
and `curvature_resolved` describe the raw scaled curvature, even after repair.

Regularization provides a finite
approximation, but can substantially underestimate uncertainty in weak directions.
Even `covariance_valid=True` does not establish Gaussian interval coverage for
nonlinear fits or parameters at active bounds. Four candidates may not reach a
large enough step for a noisy objective; enlarge the initial steps or candidate
range and compare the resulting uncertainties.

### Bootstrap and worked examples

`analysis.bootstrap(objective_sampler, ...)` refits statistically resampled data.
The sampler returns one vectorized objective for each fixed resampled dataset;
it must not regenerate observations on each objective evaluation. Minion runs
the refits and computes the covariance, correlations, and percentile intervals
in C++.

Use `analysis.bootstrap(objective_sampler, n_samples=1000, n_jobs=4)` for up to
four concurrent replica fits (`n_jobs=1` keeps sequential execution). Each C++
worker calls the sampler with its replica's index and seed, then fits its
returned batch objective. C++ uses `BootstrapOptions::n_jobs`. Results remain
in replica order, and `n_jobs_used` reports actual workers. Samplers must use
independent per-replica data/RNG state and synchronize shared mutable state.
Inner objective threads are configured by the user; `n_jobs` does not change
BLAS/OpenMP settings. Python objectives need GIL-releasing work for CPU-bound
thread parallelism. The [parallel bootstrap example](examples/bootstrap_parallel.py)
compares 1,000 serial and parallel fits against exact Gaussian mean refits;
for such a tiny objective, thread overhead can exceed any speed benefit.

The executed [curve-fitting notebook](examples/curve_fit.ipynb) includes:

- A Gaussian polynomial fit with an analytic covariance and 100 Minion bootstrap fits.
- Nearly collinear parameters with tiny numerical errors, showing when repair fails
  and larger differentiation steps recover the reference uncertainty.
- A curved nonlinear parameter distribution, compared with exact transformed
  refits and another 100 Minion bootstrap fits.
- A non-negative amplitude at its bound, where symmetric Gaussian intervals miss
  the constrained estimator's point mass and support.

See the [statistical uncertainty guide](https://minion-py.readthedocs.io/en/latest/Uncertainty.html)
for the full API, numerical details, and interval interpretation. 

## CEC and BBOB2009 Benchmark Usage

Minion exposes the CEC suites and the BBOB2009 suite through the same benchmark API.
The general pattern is:
- construct a benchmark evaluator,
- pass it to `Minimizer` as a vectorized objective,
- provide `bounds` separately to `Minimizer`.

The benchmark evaluators are already vectorized, so in Python no wrapper is needed.
In C++, a thin adapter is usually used to forward the batch to the benchmark object.

The CEC suites all use the constructor form `(function_number, dimension)` for API consistency, although their effective dimensions are suite-defined. For `CEC2011`, MinionPy additionally exposes `get_bounds()`.

Example using `CEC2017` in C++:

```cpp
#include <minion.h>
#include <minion_cec.h>

std::vector<double> cec2017_batch(const std::vector<std::vector<double>>& X, void* data) {
    auto* cec = static_cast<minion::CECBase*>(data);
    return (*cec)(X);
}

int main() {
    const int function_number = 1;
    const int dimension = 30;
    const size_t maxevals = 30000;
    const int seed = 20250306;

    minion::CEC2017Functions cec_f1(function_number, dimension);
    std::vector<std::pair<double, double>> bounds(dimension, {-100.0, 100.0});
    std::vector<std::vector<double>> x0 = {std::vector<double>(dimension, 0.0)};

    minion::Minimizer optimizer(
        cec2017_batch, bounds, x0, &cec_f1, nullptr, "ARRDE", maxevals, seed
    );

    minion::MinionResult result = optimizer.optimize();
    std::cout << "status = " << result.status << "\n";
}
```

Example using `CEC2017` in Python:

```python
import minionpy as mpy

function_number = 1
dimension = 30
maxevals = 30000
seed = 20250306

cec_f1 = mpy.CEC2017Functions(function_number=function_number, dimension=dimension)
bounds = [(-100.0, 100.0)] * dimension
x0 = [[0.0] * dimension]

optimizer = mpy.Minimizer(
    func=cec_f1,
    x0=x0,
    bounds=bounds,
    algo="ARRDE",
    maxevals=maxevals,
    callback=None,
    seed=seed,
    options=None,
)

result = optimizer.optimize()
print("best f =", result.fun)
print("status =", result.status)
print("message =", result.message)
```
For `CEC2011`, the pattern is the same, but the bounds are problem-specific:

```python
cec2011 = mpy.CEC2011Functions(function_number=1, dimension=6)
bounds = cec2011.get_bounds()
```

Example using `BBOB2009` in Python:

```python
import minionpy as mpy

function_number = 1
dimension = 10
maxevals = 30000
seed = 20250306

bbob = mpy.BBOB2009Problem(function_number=function_number, dimension=dimension)
bounds = bbob.bounds
x0 = [bbob.initial_solution]

optimizer = mpy.Minimizer(
    func=bbob,
    x0=x0,
    bounds=bounds,
    algo="ARRDE",
    maxevals=maxevals,
    callback=None,
    seed=seed,
    options=None,
)

result = optimizer.optimize()
print("best f =", result.fun)
print("status =", result.status)
print("message =", result.message)
```

### C++ Benchmark Driver

For repeated benchmark runs, use `examples/main_run_benchmark.cpp`.
It is built as the `run_benchmark` example target when `MINION_BUILD_EXAMPLES=ON` and `MINION_BUILD_BENCHMARK=ON`.

Run it:

```bash
./build/bin/run_benchmark cec 1 10 ARRDE 0 2017 30000 1 8
./build/bin/run_benchmark bbob 1 10 ARRDE 0 2009 30000 1 8
```

The command-line layout is:

```text
cec|bbob Nruns dim algo popsize year maxevals nthreads accuracy
```

## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## 📜 Citing Minion
If you use **Minion** or **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2026). *Minion: A C++ and Python Library for Single-Objective Optimization.* Journal of Open Source Software, 11(127), 10145. [https://doi.org/10.21105/joss.10145](https://doi.org/10.21105/joss.10145)
