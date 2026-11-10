.. Minion documentation master file, created by
   sphinx-quickstart on Thu Jan 30 11:19:38 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Minion's documentation!
==================================


Welcome to the **Minion** library documentation. Minion is a high-performance, derivative-free optimization library designed for solving complex optimization problems where gradient information is unavailable or unreliable. The library implements state-of-the-art algorithms, many of which have won top positions in IEEE CEC competitions, making them highly competitive compared to traditional optimization libraries like SciPy, NLopt, OptimLib, pyGMO, and pagmo2.

Minion also provides a research-friendly environment for developing and evaluating new optimization algorithms. It includes benchmark functions from multiple CEC competitions (2011, 2014, 2017, 2019, 2020, and 2022) and the BBOB2009 suite to facilitate algorithm testing and comparison.

This documentation covers the installation, usage, and details of both the C++ and Python interfaces of Minion.

Citing Minion
-------------

If you use **Minion** or **MinionPy** in your research or projects, please cite:

Muzakka, K. F., Möller, S., & Finsterbusch, M. (2026). *Minion: A C++ and Python Library for Single-Objective Optimization.* Journal of Open Source Software, 11(127), 10145. https://doi.org/10.21105/joss.10145

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   minion/index
   minionpy/index
   algolist
   BFGS
   Convergence
   Uncertainty
   Vectorization
   Algorithms
   cec
   contribute
   changelog
   license


   







Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
