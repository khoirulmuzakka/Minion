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

## 🚀 Installation
The Python wrapper (**minionpy**) is available on PyPI:

```sh
pip install --upgrade minionpy
```

For compiling the C++ version, please refer to the official Minion documentation.

### C++ Compilation (Quick)
Build with helper scripts:

- Windows: `compile.bat`
- Linux/macOS: `compile.sh`

Or with CMake manually:

```sh
cmake -S . -B build -DMINION_BUILD_CEC=OFF -DMINION_BUILD_PYTHON=OFF -DMINION_BUILD_EXAMPLES=OFF
cmake --build build --config Release
```

Enable all components (CEC, Python extension, examples):

```sh
cmake -S . -B build -DMINION_BUILD_CEC=ON -DMINION_BUILD_PYTHON=ON -DMINION_BUILD_EXAMPLES=ON
cmake --build build --config Release
```

### Use Minion in Another C++ Project
Install Minion first:

```sh
cmake -S . -B build
cmake --install build --prefix /path/to/minion/install
```

Then in your project CMake:

```cmake
find_package(Minion CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE minion)
# Optional CEC:
# target_link_libraries(my_app PRIVATE Minion_cec)
```

Find Minion with fallback download:

```cmake
include(FetchContent)

find_package(Minion QUIET CONFIG)
if(NOT Minion_FOUND)
    FetchContent_Declare(
        minion
        GIT_REPOSITORY https://github.com/khoirulmuzakka/Minion.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    # Typical dependency mode: algorithms only
    set(MINION_BUILD_CEC OFF CACHE BOOL "Disable CEC" FORCE)
    set(MINION_BUILD_PYTHON OFF CACHE BOOL "Disable Python extension" FORCE)
    set(MINION_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
    FetchContent_MakeAvailable(minion)
endif()

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE minion)
```

In exactly one translation unit:

```cpp
#define MINION_ALGORITHMS_IMPLEMENTATION
#include <minion/minion.h>
```

In all other translation units:

```cpp
#include <minion/minion.h>
```

## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## 📜 Citing Minion
If you use **Minion** or **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2025).  
> *Minion: A high-performance derivative-free optimization library designed for solving complex optimization problems.*  
> Zenodo. [https://doi.org/10.5281/zenodo.14794239](https://doi.org/10.5281/zenodo.14794239)  
