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
  - The CEC2011 suite, containing 22 real-world optimization problems, has been completely rewritten in C++ from the original MATLAB/Octave code.
  - The CEC2011 rewrite has been checked against the original Octave implementation with `comparison_scripts/verify_cec2011_octave.py` using 1000 random samples per problem drawn from the allowed bounds.
  - With a relative tolerance of `1e-9`, all rewritten CEC2011 problems agree with the Octave reference except `F3`, `F4`, `F21`, and `F22`.
  - Observed mismatches are:
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

## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## 📜 Citing Minion
If you use **Minion** or **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2025).  
> *Minion: A high-performance derivative-free optimization library designed for solving complex optimization problems.*  
> Zenodo. [https://doi.org/10.5281/zenodo.14794239](https://doi.org/10.5281/zenodo.14794239)  
