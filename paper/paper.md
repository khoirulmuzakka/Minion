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
    affiliation: "1"
  - name: Sören Möller
    orcid: 0000-0002-7948-4305
    affiliation: "1"
  - name: Martin Finsterbusch
    orcid: 0000-0001-7027-7636
    affiliation: "1"
affiliations:
 - name: Institute of Energy Materials and Devices (IMD-2), Forschungszentrum Jülich GmbH, Germany
   index: 1
date: 5 October 2025
bibliography: paper.bib

---

# Summary

Minion is a derivative-free optimization library for single-objective problems in which gradient-based methods are impractical. It provides a C++ backend with a Python interface (MinionPy), supporting applications in engineering, machine learning, and scientific computing.

The library offers a centralized implementation of state-of-the-art Differential Evolution (DE) algorithms that have performed strongly in IEEE CEC competitions. Alongside these research-grade solvers, Minion ships widely used optimizers such as Nelder–Mead, Dual Annealing (generalized simulated annealing), Covariance Matrix Adaptation Evolution Strategy (CMA-ES), and several Particle Swarm Optimization (PSO) variants, allowing practitioners to combine established baselines with advanced heuristics within a single API. Many existing optimization libraries include only elementary DE variants and lack standardized benchmark problems. Minion addresses this by integrating multiple CEC benchmark suites (2011, 2014, 2017, 2019, 2020, and 2022) to facilitate algorithm evaluation and comparison.

Compared with widely adopted toolkits such as SciPy, NLopt, and pagmo2/pygmo, Minion emphasises a unified interface for batch-evaluated objective functions, provides native support for modern CEC-winning DE variants, and bundles curated benchmark suites for reproducible experimentation. These design choices serve researchers developing bespoke algorithms as well as practitioners seeking robust defaults for black-box optimisation.

# Review of existing optimization libraries

SciPy [@2020SciPy-NMeth] underpins a large fraction of scientific computing in Python and offers a stable interface to classical optimisation routines, including gradient-based methods and a handful of derivative-free heuristics such as Nelder–Mead and Powell’s method. Its design prioritises broad accessibility and numerical reliability; consequently, coverage of recent population-based metaheuristics or fully vectorised objective evaluations is therefore limited.

NLopt [@NLopt] collects an extensive set of local and global optimisers behind a C API with bindings to multiple languages. The library provides deterministic algorithms (e.g. COBYLA, BOBYQA) and stochastic search methods (e.g. CRS, ISRES, ESCH), yet relies on single-sample objective calls. Users who require Differential Evolution or particle-swarm heuristics typically integrate third-party implementations alongside NLopt’s core offerings.

pagmo2/pygmo [@Biscani2020] is geared towards island-based, massively parallel search. It excels at composing heterogeneous portfolios of solvers and supports sophisticated multi-objective workflows. For practitioners focused on single-objective, derivative-free problems, realising a streamlined setup—particularly when benchmarking CEC-style test suites or coupling to noisy quasi-Newton updates—can involve additional configuration effort.

DEAP [@DEAP_JMLR2012] provides a highly extensible Python framework for constructing evolutionary algorithms from modular operators. This flexibility is valuable for exploratory research, but achieving high-throughput optimisation requires users to supply their own performance-oriented backends, batched evaluation loops, and curated algorithm configurations.

Minion aims to complement these ecosystems by concentrating on single-objective optimisation with built-in support for batch evaluation, vectorised quasi-Newton updates, and implementations of recent Differential Evolution and swarm variants that have performed well on modern CEC benchmarks. The goal is not to replace these libraries, but to offer an option tailored to scenarios where such capabilities are central requirements.

# Statement of need

Minion was created to address several limitations in existing optimization libraries:

1. Centralized library for state-of-the-art Differential Evolution algorithms. Many optimization libraries lack a unified framework that combines advanced Differential Evolution (DE) variants with the latest CEC benchmark problems through a simple interface in both C++ and Python. While basic DE algorithms are common, modern variants such as L-SHADE and jSO—which have demonstrated superior performance in CEC competitions—are often absent. Minion fills this gap by providing these algorithms alongside a platform for researchers to create and test new optimizers, streamlining benchmarking and comparison with existing methods.

2. Limited support for straightforward batch evaluation. Several widely used libraries offer population-based optimisers, but their APIs typically accept one sample at a time, making it cumbersome to exploit highly parallel objective evaluations. Minion accepts batches natively so that vectorised or distributed functions can be used without additional wrappers.

3. Lack of a robust L-BFGS-B implementation that performs well under noise and supports batch objective evaluations. Traditional L-BFGS-B implementations struggle with noisy function calls, which are common in real-world applications such as experimental data fitting. Minion provides an improved L-BFGS-B implementation that mitigates these issues.

# Algorithms
Minion currently implements the following optimization algorithms:

- Basic Differential Evolution (DE) [@Storn1997]
- JADE [@5208221]
- LSHADE [@b6900380]
- LSHADE-cnEpSin [@7969336]
- jSO [@7969456]
- j2020 [@9185551]
- NL-SHADE-RSP [@9504959]
- LSRTDE [@10611907]
- Adaptive Restart-Refine Differential Evolution (ARRDE)
- Artificial Bee Colony (ABC) [@10.1007/978-3-540-72950-1_77]
- Canonical PSO [@Kennedy1995]
- SPSO-2011 [@ZambranoBigiarini2013]
- Dynamic Multi-Swarm PSO (DMS-PSO) [@1501611]
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [@Hansen1996]
- Generalized Simulated Annealing (Dual Annealing) [@XIANG1997216]
- Nelder–Mead [@10.1093/comjnl/7.4.308]
- L-BFGS-B [@doi:10.1137/0916069]
- L-BFGS [@Liu1989]

Additional algorithms are planned for future releases. Minion also ships benchmark suites from IEEE CEC competitions spanning 2011, 2014, 2017, 2019, 2020, and 2022. The library further bundles classic analytic test functions—such as sphere, Rosenbrock, and Rastrigin—for quick experimentation and unit testing.

Minion exposes a consistent `Minimizer` interface in both C++ and Python. Algorithms are selected via a simple identifier (e.g. `"ARRDE"`, `"L_BFGS_B"`), and option names remain stable across population-based and quasi-Newton methods. The result object, `MinionResult`, mirrors the structure of `scipy.optimize.OptimizeResult`, making it straightforward to exchange results with SciPy tooling or any code that expects that layout.

In Minion, the optional ``x0`` argument may contain multiple initial guesses—an uncommon capability in optimisation libraries. This is practical when prior knowledge suggests several promising starting points or when restart strategies are desired. Population methods treat the entries as explicit seeds for their initial populations, while single-trajectory solvers (e.g. CMA-ES, Nelder–Mead, L-BFGS variants) evaluate each candidate and proceed from the best-performing one. The behaviour streamlines the reuse of domain heuristics when launching new optimisation runs.

Minion’s L-BFGS and L-BFGS-B implementations build on LBFGSpp [@LBFGSpp] but introduce several features tailored to noisy, vectorised workloads. Gradient estimates are generated from batched finite differences, while noise-aware step sizes and a Lanczos-style smoothing filter reduce variance in the resulting updates. This design is motivated by the robustness that has kept Minuit/Migrad[@James:1994vla] a standard tool in high-energy physics for decades. The accompanying benchmark notebook shows that, under these modifications, Minion’s quasi-Newton solvers compete favourably with Minuit/Migrad on noisy CEC test suites while preserving a fully vectorised evaluation pipeline.

# Availability and documentation
Minion is primarily implemented in C++. However, recognizing the popularity of Python and its ease of use, a Python wrapper (MinionPy) is also available. It can be installed via PIP, allowing for seamless integration into Python-based workflows. Documentation on how to use both Minion and MinionPy is available at: https://minion-py.readthedocs.io/ .


# References
