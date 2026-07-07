# MinionPy

<div align="center">
  <img src="https://github.com/khoirulmuzakka/Minion/raw/main/docs/minion_logo.png" alt="Logo" width="200" />
</div>


![PyPI Python Version](https://img.shields.io/pypi/pyversions/minionpy)
![PyPI version](https://img.shields.io/pypi/v/minionpy.svg)
![PyPI downloads](https://img.shields.io/pypi/dm/minionpy.svg)
![PyPI License](https://img.shields.io/pypi/l/minionpy.svg)
[![Documentation Status](https://readthedocs.org/projects/minion-py/badge/?version=latest)](https://minion-py.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14794240.svg)](https://doi.org/10.5281/zenodo.14794240)


MinionPy is the Python interface to the Minion C++ optimization library. It focuses on single-objective, derivative-free optimization. The package includes several population-based and local optimization methods, along with CEC benchmark suites that can be used for testing and comparison.

## Features

- **Optimization Algorithms**  
    - **Differential Evolution-based algorithms:**
      - Basic Differential Evolution (DE)
      - JADE  
      - LSHADE  
      - ARRDE 
      - and other DE variants
    - **Other population-based algorithms:**
      - Artificial Bee Colony (ABC)
      - Grey Wolf DE Optimization  
      - Canonical PSO, SPSO-2011, Dynamic Multi-Swarm PSO (DMS-PSO)  
      - CMA-ES *(Covariance Matrix Adaptation Evolution Strategy)*  
      - BIPOP-aCMAES
      - RCMAES
    - **Classical optimization algorithms:**
      - Nelder-Mead  
      - Generalized Simulated Annealing (Dual Annealing)  
      - L-BFGS-B (vectorized & noise-robust) 
      - L-BFGS (vectorized & noise-robust) 
      
- **Benchmark Support**  
  The library includes benchmark functions from the CEC competitions (2011, 2014, 2017, 2019, 2020, 2022) and BBOB2009, providing a standardized environment for algorithm development, testing, and comparison.

- **Performance**  
  Most implemented algorithms are population-based, making them suitable for parallelization. MinionPy is optimized for vectorized functions, enabling efficient use of multithreading and multiprocessing capabilities.

- **Cross-Platform Compatibility**  
  MinionPy is implemented in C++ with a Python wrapper, supporting usage in both languages. It has been tested on the following platforms:
  - Windows 11
  - Linux Ubuntu 24.04
  - macOS Sequoia 15  

## Applications

MinionPy is applicable in scenarios where single objective, bound-constrained/unconstrauned optimization is required, including engineering, physics, and machine learning. Its standardized benchmarks and high-performance algorithms make it suitable for developing and evaluating new optimization techniques as well as solving real-world optimization problems.


## 📖 Documentation
For full usage instructions, API reference, and examples, visit the official documentation:

- **[Minion Documentation](https://minion-py.readthedocs.io/)**

## Citing Minion

If you use **MinionPy** in your research or projects, we would be grateful if you could cite the following publication:

> Muzakka, K. F., Möller, S., & Finsterbusch, M. (2025).  
> *Minion: A high-performance derivative-free optimization library designed for solving complex optimization problems.*  
> Zenodo. [https://doi.org/10.5281/zenodo.14794239](https://doi.org/10.5281/zenodo.14794239)  
