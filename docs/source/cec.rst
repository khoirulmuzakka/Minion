CEC Benchmarks
==============

This page describes how to minimize CEC benchmark functions with both the **C++** and **Python** Minion APIs.

General Pattern
===============

Using a CEC benchmark follows the same overall workflow as any other objective:

1. Construct a benchmark evaluator such as ``CEC2017Functions``.
2. Provide search ``bounds`` separately to ``Minimizer``.
3. Run ``optimize()``.

CEC benchmark evaluators are already batch evaluators:

- In **C++**, benchmark classes derive from ``minion::CECBase`` and evaluate batches through ``operator()(const std::vector<std::vector<double>>& X)``.
- In **Python**, benchmark wrappers are already vectorized and can be passed directly as ``func=...`` to ``minionpy.Minimizer``.


Constructor Pattern
===================

The benchmark suites use these constructor forms:

.. code-block:: cpp

    minion::CEC2014Functions(function_number, dimension)
    minion::CEC2017Functions(function_number, dimension)
    minion::CEC2019Functions(function_number, dimension)
    minion::CEC2020Functions(function_number, dimension)
    minion::CEC2022Functions(function_number, dimension)
    minion::CEC2011Functions(function_number, dimension)

.. code-block:: python

    mpy.CEC2014Functions(function_number, dimension)
    mpy.CEC2017Functions(function_number, dimension)
    mpy.CEC2019Functions(function_number, dimension)
    mpy.CEC2020Functions(function_number, dimension)
    mpy.CEC2022Functions(function_number, dimension)
    mpy.CEC2011Functions(function_number, dimension)


Valid Dimensions
================

Valid dimensions by suite:

- ``CEC2011``: fixed, problem-specific dimensions depending on the selected function
- ``CEC2014``: ``2, 10, 20, 30, 50, 100``
- ``CEC2017``: ``2, 10, 20, 30, 50, 100``
- ``CEC2019``: ``9`` for F1, ``16`` for F2, ``18`` for F3, and ``10`` for F4-F10
- ``CEC2020``:
  C++ implementation accepts ``2, 5, 10, 15, 20, 30, 50, 100``
  Python wrapper currently accepts ``2, 5, 10, 15, 20``
- ``CEC2022``:
  C++ implementation accepts ``2, 10, 20``
  Python wrapper currently accepts ``2, 10, 20``

Some suites also have function-specific restrictions at certain dimensions.


C++ Usage
=========

The usual C++ pattern is to adapt the CEC evaluator to ``MinionFunction``:

.. code-block:: cpp

    #include <minion.h>
    #include <minion_cec.h>

    std::vector<double> cec2017_batch(const std::vector<std::vector<double>>& X, void* data) {
        auto* cec = static_cast<minion::CECBase*>(data);
        return (*cec)(X);
    }

    int main() {
        const int function_number = 1;
        const int dimension = 30;
        const size_t maxevals = 30000;
        const int seed = 20250306;

        minion::CEC2017Functions cec_f1(function_number, dimension);
        std::vector<std::pair<double, double>> bounds(dimension, {-100.0, 100.0});
        std::vector<std::vector<double>> x0 = {};

        auto options = minion::DefaultSettings().getDefaultSettings("ARRDE");
        minion::Minimizer optimizer(
            cec2017_batch, bounds, x0, &cec_f1, nullptr, "ARRDE", maxevals, seed, options
        );

        minion::MinionResult result = optimizer.optimize();
    }


Python Usage
============

In Python, no adapter is needed because the benchmark wrapper is already vectorized:

.. code-block:: python

    import minionpy as mpy

    function_number = 1
    dimension = 30
    maxevals = 30000
    seed = 20250306

    cec_f1 = mpy.CEC2017Functions(function_number=function_number, dimension=dimension)
    bounds = [(-100.0, 100.0)] * dimension

    optimizer = mpy.Minimizer(
        func=cec_f1,
        x0=None,
        bounds=bounds,
        algo="ARRDE",
        maxevals=maxevals,
        callback=None,
        seed=seed,
        options=None,
    )

    result = optimizer.optimize()


About Bounds
============

The benchmark object evaluates the objective, but it does **not** supply bounds to ``Minimizer`` automatically. You should still pass ``bounds`` explicitly.

Typical examples:

- ``CEC2014`` / ``CEC2017`` / ``CEC2020`` / ``CEC2022``:
  the project benchmark drivers typically use ``[-100, 100]^D`` in C++, or ``[(-100, 100)] * dimension`` in Python
- ``CEC2019``: use the suite-specific ranges
  - F1: ``[-8192, 8192]^9``
  - F2: ``[-16384, 16384]^16``
  - F3: ``[-4, 4]^18``
  - F4-F10: ``[-100, 100]^10``
- ``CEC2011``: use the problem-specific bounds from the original suite

For ``CEC2011``, MinionPy exposes the suite-defined bounds directly:

.. code-block:: python

    cec2011 = mpy.CEC2011Functions(function_number=1, dimension=6)
    bounds = cec2011.get_bounds()

For the full C++ per-problem bound setup, see ``examples/main_cec.cpp`` or ``examples/main_cec_multi.cpp`` in the repository.


Further Examples
================

For broader benchmark examples and sweeps, see:

- ``examples/main_cec.cpp``
- ``examples/main_cec_multi.cpp``
- ``tests/test_minionpy.py``
