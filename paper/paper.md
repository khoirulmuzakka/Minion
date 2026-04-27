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

Minion is a derivative-free optimization library for single-objective blackbox problems where gradient information is unavailable or unreliable. It provides a C++ backend with a Python interface (MinionPy), supporting applications in engineering, machine learning, and scientific computing.

The library offers a centralized implementation of advanced derivative-free optimization algorithms, including state-of-the-art Differential Evolution (DE) methods that have performed strongly in IEEE CEC competitions, CMA-ES variants, and several Particle Swarm Optimization (PSO) variants. Alongside these research-grade solvers, Minion ships widely used optimizers such as Nelder–Mead, Dual Annealing (generalized simulated annealing), L-BFGS, and L-BFGS-B, allowing practitioners to combine established baselines with advanced heuristics within a single API. Many existing optimization libraries include only elementary variants of these methods and lack standardized benchmark problems. Minion addresses this by integrating multiple CEC benchmark suites (2011, 2014, 2017, 2019, 2020, and 2022) to facilitate algorithm evaluation and comparison.

Compared with widely adopted toolkits such as SciPy, NLopt, and pagmo2/pygmo, Minion emphasizes a unified interface for batch-evaluated objective functions, provides native support for modern population-based methods and finite-difference quasi-Newton solvers, and bundles curated CEC benchmark suites for reproducible experimentation. These design choices serve researchers developing bespoke algorithms as well as practitioners seeking robust defaults for blackbox optimization.

# State of the field

SciPy [@2020SciPy-NMeth] underpins a large fraction of scientific computing in Python and offers a stable interface to classical optimization routines, including gradient-based methods and a handful of derivative-free heuristics such as Nelder–Mead and Powell’s method. Its design prioritizes broad accessibility and numerical reliability; consequently, coverage of recent population-based metaheuristics or fully vectorized objective evaluations is limited.

NLopt [@NLopt] collects an extensive set of local and global optimizers behind a C API with bindings to multiple languages. The library provides deterministic algorithms (e.g. COBYLA, BOBYQA) and stochastic search methods (e.g. CRS, ISRES, ESCH), yet relies on single-sample objective calls. Users who require Differential Evolution or particle-swarm heuristics typically integrate third-party implementations alongside NLopt’s core offerings.

pagmo2/pygmo [@Biscani2020] is geared towards island-based, massively parallel search. It excels at composing heterogeneous portfolios of solvers and supports sophisticated multi-objective workflows. For practitioners focused on single-objective, derivative-free problems, realizing a streamlined setup—particularly when benchmarking CEC-style test suites or coupling to noisy quasi-Newton updates—can involve additional configuration effort.

DEAP [@DEAP_JMLR2012] provides a highly extensible Python framework for constructing evolutionary algorithms from modular operators. This flexibility is valuable for exploratory research, but achieving high-throughput optimization requires users to supply their own performance-oriented backends, batched evaluation loops, and curated algorithm configurations.

Minion aims to complement these ecosystems by concentrating on single-objective optimization with built-in support for batch evaluation, finite-difference quasi-Newton solvers, modern DE, CMA-ES, and PSO variants, and integrated CEC benchmark suites. The goal is not to replace these libraries, but to offer an option tailored to scenarios where such capabilities are central requirements.

The decision to build Minion rather than contribute a single extension to an existing package follows from this scope. Adding one or two solvers to SciPy, NLopt, DEAP, or pagmo2 would not by itself provide a coherent benchmark platform, a shared batched-objective interface across both global and local solvers, and C++/Python access to a curated collection of recent CEC-oriented algorithms. Minion's scholarly contribution is therefore the integration of these components into a reproducible research tool: users can compare modern DE variants, CMA-ES variants, swarm methods, classical local search, and CEC benchmark suites without changing objective-function conventions or stitching together several libraries.

# Statement of need

Minion was created to address several limitations in existing optimization libraries:

1. Centralized library for advanced derivative-free optimization algorithms. Many optimization libraries lack a unified framework that combines modern DE variants, CMA-ES variants, PSO variants, and established local-search methods through a simple interface in both C++ and Python. While basic implementations of these algorithm families are common, recent research-grade variants such as L-SHADE, jSO, NL-SHADE-RSP, LSRTDE, BIPOP-ACMAES, and DMS-PSO are often absent or distributed across separate packages. Minion fills this gap by providing these algorithms alongside a platform for researchers to create and test new optimizers.

2. Centralized CEC benchmark support for reproducible comparison. Benchmarking derivative-free algorithms often requires separate implementations of CEC test suites, custom data files, and ad hoc scripts, making results harder to reproduce and compare. Minion integrates CEC benchmark suites from 2011, 2014, 2017, 2019, 2020, and 2022 under the same objective-function convention as user-defined problems, streamlining benchmarking and comparison with existing methods.

3. Limited support for straightforward batch evaluation. Several widely used libraries offer population-based optimizers, but their APIs typically accept one sample at a time, making it cumbersome to exploit highly parallel objective evaluations. Minion accepts batches natively so that vectorized or distributed functions can be used without additional wrappers.

4. Lack of robust L-BFGS-B and L-BFGS implementations that perform well under noise and support batch objective evaluations. Traditional finite-difference quasi-Newton implementations can struggle with noisy function calls, which are common in real-world applications such as experimental data fitting. Minion mitigates this by using batched derivative evaluations together with a curvature-aware adaptive finite-difference step size and an optional Lanczos-style derivative estimate.

# Software Design

Minion is organized around a small C++ core interface rather than around algorithm-specific entry points. The user-facing `Minimizer` class acts as a factory and dispatcher: it normalizes an algorithm identifier, constructs the corresponding solver, and exposes both `optimize()` and call-operator execution. All concrete optimizers inherit from `MinimizerBase`, which stores common state including the objective function, bounds, initial guesses, stopping tolerance, evaluation budget, random seed, callback, option map, current result, and optimization history. This structure keeps each algorithm implementation focused on its search logic while preserving uniform validation, configuration, and result reporting.

The central design choice is that objective functions are batched by default. In C++, a `MinionFunction` maps `std::vector<std::vector<double>>` candidate points to a vector of objective values, and the same convention is exposed in Python. This is less familiar than scalar objective calls, but it reflects the workload that motivated Minion: population-based algorithms, finite-difference gradients, noisy objectives, and benchmark campaigns often evaluate many candidates per iteration. A batched interface lets users vectorize evaluations in Python, run compiled objective kernels efficiently, or distribute objective calls without wrapping every solver separately.

The library balances this specialized interface with conventions that are familiar to scientific Python users. `MinionResult` follows the structure of `scipy.optimize.OptimizeResult`, with fields for the best point, objective value, iteration count, number of function evaluations, success flag, and diagnostic message. Solver options use a common map of typed values, and the `x0` argument accepts either one starting point or multiple candidate starts. This is useful when prior knowledge suggests more than one promising candidate solution. For population-based algorithms, the supplied points are inserted into the initial population and the remaining members are sampled from the bounds. For local methods, Minion evaluates the supplied candidates and starts from the best one. This design supports domain-specific initial guesses without making each algorithm expose a different API.

A second trade-off is that Minion implements the performance-critical algorithms in C++ while providing a Python interface for accessibility. This avoids the overhead of pure-Python inner loops in large benchmark runs, but keeps installation and use close to standard Python workflows through MinionPy. The project also exposes C++ headers for native applications, including a single umbrella header and separate headers for individual solvers and CEC test functions. This split supports both exploratory notebooks and compiled scientific applications.

Finally, the benchmark code is treated as part of the software design, not as an external afterthought. CEC suites are bundled behind the same batched evaluation convention as user objectives, and example scripts store reproducible benchmark outputs for multiple algorithms and budgets. This matters because Minion is intended not only as an optimizer library, but also as an environment for evaluating new single-objective derivative-free methods against established baselines.

# Algorithms
Minion currently implements the following optimization algorithms:

- Basic Differential Evolution (DE) [@Storn1997]
- JADE [@5208221]
- LSHADE [@b6900380]
- LSHADE-cnEpSin [@Awad2017]
- jSO [@7969456]
- j2020 [@9185551]
- NL-SHADE-RSP [@9504959]
- LSRTDE [@10611907]
- Adaptive Restart-Refine Differential Evolution (ARRDE) [@ARRDE]
- Artificial Bee Colony (ABC) [@Karaboga2005]
- Canonical PSO [@Kennedy1995]
- SPSO-2011 [@ZambranoBigiarini2013]
- Dynamic Multi-Swarm PSO (DMS-PSO) [@1501611]
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [@Hansen1996]
- BI-population CMA-ES (BIPOP-ACMAES) [@bipop2009]
- Generalized Simulated Annealing (Dual Annealing) [@XIANG1997216]
- Nelder–Mead [@10.1093/comjnl/7.4.308]
- L-BFGS-B [@doi:10.1137/0916069]
- L-BFGS [@Liu1989]

Additional algorithms are planned for future releases. Minion also ships benchmark suites from IEEE CEC competitions spanning 2011, 2014, 2017, 2019, 2020, and 2022. The library further bundles classic analytic test functions—such as sphere, Rosenbrock, and Rastrigin—for quick experimentation and unit testing.

Minion’s L-BFGS and L-BFGS-B implementations build on LBFGSpp [@LBFGSpp] but introduce derivative calculations tailored to noisy, vectorized workloads. Gradient estimates are generated from batched finite differences. The finite-difference step for each coordinate is adapted from an estimate of local curvature, obtained from L-BFGS curvature information accumulated in previous iterations, and from a multiplicative model of function noise. This follows the relation \(h = 2\sqrt{\epsilon_f / |f^{(2)}|}\), where \(\epsilon_f\) is the estimated function-value noise and \(f^{(2)}\) is the local second derivative. Minion also supports a Lanczos-style derivative formula derived from least-squares polynomial fitting, while preserving a single batched objective call for the derivative evaluation. The accompanying benchmark notebook demonstrates the robustness of these quasi-Newton solvers on noisy CEC test suites while preserving a fully vectorized evaluation pipeline.

# Availability and documentation
Minion is primarily implemented in C++. However, recognizing the popularity of Python and its ease of use, a Python wrapper (MinionPy) is also available. It can be installed via pip, allowing for seamless integration into Python-based workflows. Documentation on how to use both Minion and MinionPy is available at: https://minion-py.readthedocs.io/ .

# Research Impact Statement

Minion supports reproducible research in single-objective blackbox optimization by combining advanced optimizers, batched objective evaluation, and standardized CEC benchmark suites in one C++/Python package. It can also serve as a research platform where researchers can implement and experiment with new methods while comparing them directly against state-of-the-art baselines. This reduces the friction involved in comparing algorithms whose original implementations often differ in language, objective-function convention, stopping criteria, and benchmark setup.

The project already provides community-readiness signals and reusable research materials. MinionPy is packaged on PyPI, the source repository provides continuous-integration and test workflows, documentation is hosted on Read the Docs, and the software is archived with a Zenodo DOI [@muzakka_2025_14893994]. The repository includes C++ examples, Python examples, notebooks, benchmark comparison scripts, and stored CEC result files covering multiple algorithms, dimensions, and evaluation budgets. These materials make it possible for users to reproduce benchmark-style studies, inspect optimizer behavior, and extend the software with new methods.

Minion is particularly relevant for optimization studies where objective evaluations are expensive, noisy, or naturally parallel, including simulation calibration, experimental data fitting, machine-learning hyperparameter tuning, and engineering design studies. 

# AI usage disclosure

Generative AI tools were used to assist with drafting and language editing of this paper. The authors reviewed and edited the resulting text and remain responsible for the scientific claims, software description, citations, and final manuscript. Generative AI was not used as an autonomous contributor to the software's algorithmic implementation.


# References
