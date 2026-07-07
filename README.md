# Minion: C++ and Python Optimization Library

<div align="center">
  <img src="docs/minion_logo.png" alt="Minion Logo" width="200" />
</div>

![CI](https://github.com/khoirulmuzakka/Minion/actions/workflows/ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/minionpy.svg)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/minionpy)
![PyPI pip downloads](https://img.shields.io/pypi/dm/minionpy.svg)
![PyPI License](https://img.shields.io/pypi/l/minionpy.svg)
[![Documentation Status](https://readthedocs.org/projects/minion-py/badge/?version=latest)](https://minion-py.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14794240.svg)](https://doi.org/10.5281/zenodo.14794240)

**Minion/MinionPy** is an optimization library designed for solving single-objective optimization problems. It features state-of-the-art evolutionary algorithms, including top-performing methods from IEEE CEC competitions, which are not commonly found in standard optimization libraries such as SciPy, NLopt, OptimLib, pyGMO, and pagmo2.

Minion also serves as a research platform for developing and testing new optimization algorithms. It includes benchmark functions from IEEE CEC competitions (2011, 2014, 2017, 2019, 2020, and 2022), providing a robust framework for algorithm evaluation and comparison.

## 🔥 Why Minion?
- **State-of-the-art optimization algorithms** :
  - **Differential Evolution-based algorithms:**
    - Basic Differential Evolution (DE)
    - JADE  
    - LSHADE  
    - ARRDE *(Adaptive Restart-Refine DE)*  
    - other advanced DE variants
  - **Other population-based algorithms:**
    - Artificial Bee Colony (ABC)
    - Grey Wolf DE Optimization  
    - Canonical PSO, SPSO-2011, and Dynamic Multi-Swarm PSO (DMS-PSO)  
    - CMA-ES *(Covariance Matrix Adaptation Evolution Strategy)*  
    - BIPOP-aCMAES 
    - RCMAES (Restart aCMAES)
  - **Classical optimization algorithms:**
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

## 🚀 Installation and Usage

### Build dependencies
- CMake >= 3.18
- C++17 compiler (GCC/Clang/MSVC)
- Eigen3 (or allow automatic fetch via CMake)
- Optional for Python bindings: Python 3 + `pybind11`
- Optional for documentation: `doxygen`, `pandoc`, Python 3, `sphinx`, `sphinx-rtd-theme`, `nbsphinx`, `breathe`

Detailed installation, native package, C++ build, and usage instructions are available in the official documentation.

## Algorithm Usage

Across both C++ and Python, Minion uses the same core model:
- The objective must be vectorized: it receives a batch of candidate points and returns one objective value per point.
- `x0` contains multiple initial guesses, where each inner vector/list is one candidate start.
- `x0` is not a full population specification. In population-based algorithms, Minion first initializes the population using the algorithm's normal rule, then replaces some initialized individuals with the supplied guesses. In non-population-based algorithms, Minion evaluates the guesses and starts from the best one.
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
    options["population_size"] = 50;
    options["convergence_tol"] = 1e-6;

    minion::Minimizer optimizer(
        rosenbrock_batch, bounds, x0, nullptr, nullptr, "DE", 20000, 42, options
    );

    minion::MinionResult result = optimizer.optimize();
    std::cout << "best f = " << result.fun << "\n";
}
```

In C++, `MinionFunction` is the vectorized entry point: it takes `std::vector<std::vector<double>>` and returns one value per candidate point.

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
    "population_size": 50,
    "convergence_tol": 1e-6,
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
print("f_opt   =", cec_f1.f_opt)
```

In Python, `func(X)` receives `X` as `list[list[float]]` and returns one value per candidate point.

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
print("f_opt   =", cec_f1.f_opt)
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
print("f_opt   =", bbob.f_opt)
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

If you omit the leading `cec` or `bbob`, the driver defaults to `cec`.

### Python Benchmark API

The Python binding exposes the same benchmark machinery through:
- `minionpy.run_benchmark(mode="cec" | "bbob", ...)`
- `minionpy.Benchmark`
- `minionpy.BenchmarkConfig`
- `minionpy.BenchmarkMode` for lower-level use

Example:

```python
import minionpy as mpy

result = mpy.run_benchmark(
    mode="bbob",# "cec
    num_runs=51,
    dimension=10,
    algo="ARRDE",
    popsize=0,
    year=2009,
    max_evals=30000,
    nthreads=32,
    acc=8,
    dump_results=True,
    results_folder=".",
    log_min_ev=False,
)
print(result.results)
print(result.results_file)
```

If you prefer an object-oriented wrapper, `mpy.Benchmark(config).run()` is also available, and `BenchmarkConfig.mode` accepts the enum value `mpy.BenchmarkMode.Bbob`.

## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## 📜 Citing Minion
If you use **Minion** or **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2025).  
> *Minion: A high-performance derivative-free optimization library designed for solving complex optimization problems.*  
> Zenodo. [https://doi.org/10.5281/zenodo.14794239](https://doi.org/10.5281/zenodo.14794239)  
