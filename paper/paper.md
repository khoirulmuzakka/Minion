---
title: 'Minion: A C++ and Python Library for Single-Objective Optimization Algorithms'
tags:
  - C++ 
  - Python 
  - Optimization
  - Differential Evolution
  - CEC Benchmark Problems
authors:
  - name: Khoirul Faiq Muzakka
    orcid: 0000-0002-3888-1697
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Sören Möller
    orcid: 0000-0002-7948-4305
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Martin Finsterbusch
    orcid: 0000-0001-7027-7636
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Institute of Energy Materials and Devices (IMD-2), Forschungszentrum Jülich GmbH, Germany
   index: 1
date: 1 April 2025
bibliography: paper.bib

---

# Summary

Minion is a derivative-free optimization library designed for solving single-objective optimization problems where gradient-based methods are infeasible. It provides a C++ backend with a Python interface (MinionPy), supporting applications in engineering, machine learning, and scientific computing.

The library provides a centralized implementation of state-of-the-art Differential Evolution (DE) algorithms that have achieved strong performance in IEEE CEC competitions. Many existing optimization libraries include basic DE variants but lack these advanced methods and standardized benchmark problems. Minion addresses this by integrating multiple CEC benchmark functions (2011, 2014, 2017, 2019, 2020, and 2022) to facilitate algorithm evaluation and comparison.

Minion also supports batch (vectorized) function evaluations, enabling efficient parallel computation through multithreading and multiprocessing. This feature helps mitigate performance bottlenecks when dealing with computationally expensive objective functions. Additionally, the library includes an L-BFGS-B implementation designed to handle noisy function evaluations and batch optimization tasks.

By providing a structured framework for both practical optimization and research, Minion enables the development, testing, and benchmarking of optimization algorithms with a focus on modern DE techniques and parallel computing.

# Statement of need

Minion was created to address several limitations in existing optimization libraries:

1. Centralized Library for State-of-the-Art Differential Evolution Algorithms. Current optimization libraries often lack a unified framework that implements advanced Differential Evolution (DE) algorithms and the latest CEC benchmark problems with a simple, intuitive interface in both C++ and Python. While many libraries include basic DE algorithms, they typically overlook modern variants like L-SHADE, jSO, and NL-SHADE-RSP, which have demonstrated superior performance in CEC competitions. Minion fills this gap by offering a library that not only includes these algorithms but also provides a platform for researchers to create and test new optimization algorithms. It streamlines the benchmarking process and facilitates comparisons with both state-of-the-art methods and traditional optimization techniques.

2. Limited support for true multithreading and multiprocessing in optimization libraries. Many existing frameworks do not efficiently handle vectorized or batch function evaluations, leading to performance bottlenecks when optimizing expensive objective functions. Minion is designed to work with batch (vectorized) objective function calls, thus can efficiently utilize parallel computation. 

3. Lack of a robust L-BFGS-B implementation that performs well under noisy conditions and supports batch objective function calls. Traditional L-BFGS-B implementations struggle with noisy function evaluations, which are common in real-world applications such as machine learning hyperparameter tuning and experimental data fitting. Minion provides an improved L-BFGS-B implementation that mitigates these issues.

# Algorithms
Minion supports 13 optimization algorithms, including 8 based on Differential Evolution [@Storn1997]. The following algorithms are implemented:

  - Basic Differential Evolution [@Storn1997] 
  - LSHADE [@b6900380]
  - JADE [@5208221] 
  - jSO [@7969456]  
  - j2020 [@9185551]  
  - NL-SHADE-RSP [@9504959]    
  - LSRTDE [@10611907]        
  - Adaptive Restart-Refine Differential Evolution (ARRDE)    
  - Artificial Bee Colony (ABC) [@10.1007/978-3-540-72950-1_77] 
  - Nelder-Mead [@10.1093/comjnl/7.4.308] 
  - Generalized Simulated Annealing or Dual Annealing [@XIANG1997216] 
  - L-BFGS-B [@doi:10.1137/0916069] 
  - L-BFGS [@Liu1989] 

In addition, Minion implements benchmark problems from IEEE CEC competitions, including:

  - CEC 2011 [@cec2011]
  - CEC 2014 [@cec2014]
  - CEC 2017 [@cec2017]
  - CEC 2019 [@cec2019]
  - CEC 2020 [@cec2020]
  - CEC 2022 [@cec2022]

# Availability and Documentations
Minion is primarily implemented in C++. However, recognizing the popularity of Python and its ease of use, a Python wrapper (MinionPy) is also available. It can be installed via PIP, allowing for seamless integration into Python-based workflows. Documentation on how to use both Minion and MinionPy is available at: https://minion-py.readthedocs.io/ .


# References