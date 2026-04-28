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
  - Includes CEC benchmark problems from 2011, 2014, 2017, 2019, 2020, and 2022 for rigorous algorithm testing.  
  - The benchmark problems are directly adapted from the original C++ implementations.  
  - The CEC 2011 suite, containing 22 real-world optimization problems, has been completely rewritten into C++ from the MATLAB version, making it much faster than the original.

## 🚀 Installation and Usage

### Python
Install from PyPI:

```sh
pip install --upgrade minionpy
```

### C++ / Native package
- Windows: compile from source with CMake (see "Compilation From Scratch" and "Using Minion in a C++ Project" below).
- Linux: install the `.deb` package from GitHub Release assets.
- macOS: use the release archive (`.tgz` / `tgz.zip` asset).

#### Linux `.deb` install
```sh
sudo dpkg -i minion_<version>_<arch>.deb
sudo apt-get install -f
```

#### macOS `.tgz` / `tgz.zip` install
If the asset is `tgz.zip`, unzip first:
```sh
unzip minion-<version>-macos.tgz.zip
```

Extract package content:
```sh
tar -xzf minion-<version>-macos.tgz -C /tmp/minion_pkg
```

Install manually (example to `/usr/local`):
```sh
sudo rsync -a /tmp/minion_pkg/ /usr/local/
```

### Compilation From Scratch
Dependencies:
- CMake >= 3.18
- C++17 compiler (GCC/Clang/MSVC)
- Eigen3 (or allow automatic fetch via CMake)
- Optional for Python bindings: Python 3 + `pybind11`
- Optional for documentation: `doxygen`, `pandoc`, Python 3, `sphinx`, `sphinx-rtd-theme`, `nbsphinx`, `breathe`

Build with helper scripts:
- Windows: `compile.bat`
- Linux/macOS: `compile.sh`

The helper scripts are intended for native C++ builds and configure:
- `MINION_BUILD_CEC=ON`
- `MINION_BUILD_EXAMPLES=ON`
- `MINION_BUILD_PYTHON=OFF`

If the optional documentation toolchain is already installed, the scripts also generate Doxygen and HTML docs. If any docs dependency is missing, the scripts print a warning and skip documentation generation.

Manual CMake build:
```sh
cmake -S . -B build \
  -DMINION_BUILD_CEC=ON \
  -DMINION_BUILD_PYTHON=OFF \
  -DMINION_BUILD_EXAMPLES=ON
cmake --build build --config Release
cmake --install build --prefix /usr/local
```

Manual CMake build with Python bindings:
```sh
cmake -S . -B build \
  -DMINION_BUILD_CEC=ON \
  -DMINION_BUILD_PYTHON=ON \
  -DMINION_BUILD_EXAMPLES=ON
cmake --build build --config Release
```

For Python usage, installing from PyPI is the recommended path:
```sh
pip install --upgrade minionpy
```

### Using Minion in a C++ Project
Header include:
```cpp
#include <minion/minion.h>
```

CMake with `find_package` first, then `FetchContent` fallback:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(Minion QUIET CONFIG)
if(NOT Minion_FOUND)
    FetchContent_Declare(
        minion
        GIT_REPOSITORY https://github.com/khoirulmuzakka/Minion.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    set(MINION_BUILD_CEC ON CACHE BOOL "Build CEC library" FORCE)
    set(MINION_BUILD_PYTHON OFF CACHE BOOL "Disable Python extension" FORCE)
    set(MINION_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
    FetchContent_MakeAvailable(minion)
endif()

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE minion)     # core algorithms
# target_link_libraries(my_app PRIVATE minion_cec)  # optional CEC benchmark suite
```

### Minimal Example (C++)
```cpp
#include <iostream>
#include <vector>
#include <minion/minion.h>

std::vector<double> sphere(const std::vector<std::vector<double>>& X, void*) {
    std::vector<double> out;
    out.reserve(X.size());
    for (const auto& x : X) {
        double s = 0.0;
        for (double xi : x) s += xi * xi;
        out.push_back(s);
    }
    return out;
}

int main() {
    const std::size_t dim = 10;
    std::vector<std::pair<double,double>> bounds(dim, {-5.0, 5.0});
    std::vector<std::vector<double>> x0 = {std::vector<double>(dim, 0.0)};
    auto settings = minion::DefaultSettings().getDefaultSettings("DE");
    auto res = minion::Minimizer(sphere, bounds, x0, nullptr, nullptr, "DE", 0.0, 20000, 123, settings).optimize();
    std::cout << "Best f(x) = " << res.fun << std::endl;
}
```

### Minimal Example (Python)
```python
import minionpy as mpy

def sphere(X):
    return [sum(xi * xi for xi in x) for x in X]

dim = 10
bounds = [(-5.0, 5.0)] * dim
x0 = [[0.0] * dim]  # optional for DE, required for some local-search algorithms

res = mpy.Minimizer(
    sphere,
    bounds,
    x0=x0,
    algo="DE",
    maxevals=20000,
    seed=123,
).optimize()

print("Best f(x) =", res.fun)
```

More examples are available in the official docs and in the repository `examples/` and `tests/` folders.

## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## 📜 Citing Minion
If you use **Minion** or **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2025).  
> *Minion: A high-performance derivative-free optimization library designed for solving complex optimization problems.*  
> Zenodo. [https://doi.org/10.5281/zenodo.14794239](https://doi.org/10.5281/zenodo.14794239)  
