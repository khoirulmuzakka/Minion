# Minion: Derivative-Free Optimization Library

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

**Minion** is a high-performance **derivative-free optimization** library designed for solving complex optimization problems where gradients are unavailable or unreliable. It features **state-of-the-art evolutionary algorithms**, including top-performing methods from IEEE CEC competitions, which are not commonly found in standard optimization libraries such as SciPy, NLopt, OptimLib, pyGMO, and pagmo2.

Minion also serves as a **research platform** for developing and testing new optimization algorithms. It includes benchmark functions from **CEC competitions (2011, 2014, 2017, 2019, 2020, and 2022)**, providing a robust framework for algorithm evaluation and comparison.

## 🔥 Why Choose Minion?
- **State-of-the-art optimization algorithms** :
  - **Differential Evolution-based algorithms:**
    - Basic Differential Evolution (DE)
    - JADE  
    - L-SHADE 
    - jSO
    - j2020 
    - NL-SHADE-RSP 
    - LSRTDE 
    - ARRDE (our novel Adaptive Restart-Refine DE algorithm)  
  - **Other population-based algorithms:**
    - Artificial Bee Colony (ABC)
    - Grey Wolf DE Optimization  
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
  - Includes benchmark problems from **2011, 2014, 2017, 2019, 2020, and 2022** for rigorous algorithm testing.

## 🚀 Installation
The Python wrapper (**minionpy**) is available on PyPI:

```sh
pip install --upgrade minionpy
```

For compiling the C++ version, please refer to the official Minion documentation.

## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## 📜 Citing Minion
If you use **Minion** or **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2025).  
> *Minion: A high-performance derivative-free optimization library designed for solving complex optimization problems.*  
> Zenodo. [https://doi.org/10.5281/zenodo.14794240](https://doi.org/10.5281/zenodo.14794240)  



