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

**Minion/MinionPy** is an optimization library designed for solving single-objective optimization problems. It features **state-of-the-art evolutionary algorithms**, including top-performing methods from IEEE CEC competitions, which are not commonly found in standard optimization libraries such as SciPy, NLopt, OptimLib, pyGMO, and pagmo2.

Minion also serves as a **research platform** for developing and testing new optimization algorithms. It includes benchmark functions from **CEC competitions (2011, 2014, 2017, 2019, 2020, and 2022)**, providing a robust framework for algorithm evaluation and comparison.

## 🔥 Why Choose Minion?
- **State-of-the-art optimization algorithms** :
  - **Differential Evolution-based algorithms:**
    - Basic Differential Evolution (DE)
    - JADE  
    - L-SHADE  
    - LSHADE-cnEpSin 
    - IMODE  
    - jSO  
    - j2020  
    - NL-SHADE-RSP  
    - LSRTDE  
    - ARRDE *(Adaptive Restart-Refine DE)*  
    - AGSK 
    - IMODE
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
- **Highly parallelized**
  - Designed for **vectorized function evaluations**, supporting **multithreading and multiprocessing** to speed up optimization.
- **Optimized C++ backend with Python API**
  - Enjoy the performance of C++ with the simplicity of Python.
- **CEC Benchmark Suite**  
  - Includes `CEC2011`, `CEC2014`, `CEC2017`, `CEC2019`, `CEC2020`, and `CEC2022` benchmark problems for rigorous algorithm testing.  
  - The benchmark problems are directly adapted from the original C++ implementations.  
  - The CEC2011 suite, containing 22 real-world optimization problems, has been completely rewritten in C++ from the original MATLAB code. The CEC2011 rewrite has been checked by running the original MATLAB code through Octave with `comparison_scripts/verify_cec2011_octave.py` using 1000 random samples per problem drawn from the allowed bounds. With a relative tolerance of `1e-9`, all rewritten CEC2011 problems agree with the Octave-evaluated MATLAB reference except `F3`, `F4`, `F21`, and `F22`. Observed mismatches are:
    - `F3` (failed `239/1000`, max relative error `5.25e-6`)
    - `F4` (failed `986/1000`, max relative error `0.39`)
    - `F21` (failed `75/1000`, max relative error `1.46`)
    - `F22` (failed `89/100`, max relative error `1.42`)

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
- The objective must be **vectorized**: it receives a batch of candidate points and returns one objective value per point.
- `x0` contains **multiple initial guesses**, where each inner vector/list is one candidate start.
- `x0` is not a full population specification. In population-based algorithms, Minion first initializes the population using the algorithm's normal rule, then replaces some initialized individuals with the supplied guesses. In non-population-based algorithms, Minion evaluates the guesses and starts from the best one.
- After constructing the `Minimizer`, call `optimize()` to run the algorithm and get a result object.

### C++ Code

Minion expects a **vectorized** objective function:

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

MinionPy uses the same idea: the objective function should be **vectorized** and accept a batch of points.

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
```

In Python, `func(X)` receives `X` as `list[list[float]]` and returns one value per candidate point.

## CEC Benchmark Usage

The CEC wrappers follow the same optimizer workflow as ordinary objectives. The main difference is that the benchmark evaluators already implement batch evaluation logic.

### C++ Code

Minion provides CEC benchmark wrappers in C++. These already match Minion's vectorized interface, so you can pass them directly through a thin adapter:

```cpp
#include <minion.h>
#include <minion_cec.h>
#include <iostream>
#include <vector>

std::vector<double> cec2017_batch(const std::vector<std::vector<double>>& X, void* data) {
    auto* cec = static_cast<minion::CECBase*>(data);
    return (*cec)(X);
}

int main() {
    const int dim = 30;
    minion::CEC2017Functions cec_f1(1, dim);
    std::vector<std::pair<double, double>> bounds(dim, {-100.0, 100.0});
    std::vector<std::vector<double>> x0 = {
        std::vector<double>(dim, 0.0)
    };

    auto options = minion::DefaultSettings().getDefaultSettings("ARRDE");
    minion::Minimizer optimizer(
        cec2017_batch, bounds, x0, &cec_f1, nullptr, "ARRDE", 30000, 20250306, options
    );

    minion::MinionResult result = optimizer.optimize();
    std::cout << "best f = " << result.fun << "\n";
}
```

The CEC benchmark classes already support batch evaluation. In C++, you only need a thin adapter that forwards `X` to the benchmark object.

### Python Code

In Python, CEC benchmark wrappers are already vectorized, so they can be passed directly to `Minimizer`.

```python
import minionpy as mpy

dimension = 30
cec_f1 = mpy.CEC2017Functions(function_number=1, dimension=dimension)
bounds = [(-100.0, 100.0)] * dimension
x0 = [[0.0] * dimension]

optimizer = mpy.Minimizer(
    func=cec_f1,
    x0=x0,
    bounds=bounds,
    algo="ARRDE",
    maxevals=30000,
    callback=None,
    seed=20250306,
    options=None,
)

result = optimizer.optimize()
print("best f =", result.fun)
```

In Python, no extra wrapper is needed because `CEC2017Functions`, `CEC2014Functions`, `CEC2019Functions`, `CEC2020Functions`, `CEC2022Functions`, and `CEC2011Functions` already evaluate batches of candidate points.

## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## 📜 Citing Minion
If you use **Minion** or **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2025).  
> *Minion: A high-performance derivative-free optimization library designed for solving complex optimization problems.*  
> Zenodo. [https://doi.org/10.5281/zenodo.14794239](https://doi.org/10.5281/zenodo.14794239)  
