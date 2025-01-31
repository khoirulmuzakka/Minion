# MinionPy

<div align="center">
  <img src="https://github.com/khoirulmuzakka/Minion/raw/main/docs/minion_logo.png" alt="Logo" width="200" />
</div>


![PyPI Python Version](https://img.shields.io/pypi/pyversions/minionpy)
![PyPI version](https://img.shields.io/pypi/v/minionpy.svg)
![PyPI downloads](https://img.shields.io/pypi/dm/minionpy.svg)
![PyPI License](https://img.shields.io/pypi/l/minionpy.svg)

MinionPy is the Python implementation of the Minion C++ library, designed for derivative-free optimization. It provides tools for solving optimization problems where gradients are unavailable or unreliable, incorporating state-of-the-art algorithms recognized in IEEE Congress on Evolutionary Computation (CEC) competitions. The library offers researchers and practitioners access to advanced optimization techniques and benchmarks for testing and evaluation.

## Features

- **Optimization Algorithms**  
  MinionPy implements several high-performing algorithms, including:
  - JADE
  - L-SHADE (1st place, CEC2014)
  - jSO (1st place, CEC2017)
  - j2020 (3rd place, CEC2020)
  - NL-SHADE-RSP (1st place, CEC2021)
  - LSRTDE (1st place, CEC2024)
  - ARRDE (Adaptive Restart-Refine Differential Evolution)  

  These algorithms are robust and converge efficiently in complex optimization tasks. Additionally, classical methods like Nelder-Mead and the original Differential Evolution are included for reference.

- **Benchmark Support**  
  The library includes benchmark functions from the CEC competitions (2011, 2014, 2017, 2019, 2020, 2022), providing a standardized environment for algorithm development, testing, and comparison.

- **Performance**  
  Most implemented algorithms are population-based, making them suitable for parallelization. MinionPy is optimized for vectorized functions, enabling efficient use of multithreading and multiprocessing capabilities.

- **Cross-Platform Compatibility**  
  MinionPy is implemented in C++ with a Python wrapper, supporting usage in both languages. It has been tested on the following platforms:
  - Windows 11
  - Linux Ubuntu 24.04
  - macOS Sequoia 15  

## Applications

MinionPy is applicable in scenarios where derivative-free optimization is required, including engineering, physics, and machine learning. Its standardized benchmarks and high-performance algorithms make it suitable for developing and evaluating new optimization techniques as well as solving real-world optimization problems.


## Documentation

For comprehensive usage instructions, API references, and detailed examples, please refer to the official documentation:

- **[Minion Documentation](#)**  
  *Explore detailed guides, installation instructions, and examples at [ReadTheDocs](https://minion-py.readthedocs.io/).*
