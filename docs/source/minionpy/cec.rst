CEC Benchmarks
==============

This page describes how to minimize CEC benchmark functions with **MinionPy**.

General Pattern
===============

Using a CEC benchmark in MinionPy follows the same workflow as any other objective:

1. Construct a benchmark evaluator such as ``minionpy.CEC2017Functions``.
2. Provide ``bounds`` to ``minionpy.Minimizer``.
3. Run ``optimize()``.

CEC benchmark wrappers in MinionPy are already **vectorized**. They accept a batch of candidate points and return one objective value per point, so they can be passed directly to ``Minimizer`` as ``func=...``.


Constructor Pattern
===================

The benchmark wrappers use these constructor forms:

.. code-block:: python

    mpy.CEC2014Functions(function_number, dimension)
    mpy.CEC2017Functions(function_number, dimension)
    mpy.CEC2019Functions(function_number, dimension)
    mpy.CEC2020Functions(function_number, dimension)
    mpy.CEC2022Functions(function_number, dimension)
    mpy.CEC2011Functions(function_number, dimension)

Valid dimensions by suite:

- ``CEC2011``: fixed, problem-specific dimensions depending on the selected function
- ``CEC2014``: ``2, 10, 20, 30, 50, 100``
- ``CEC2017``: ``2, 10, 20, 30, 50, 100``
- ``CEC2019``: ``9`` for F1, ``16`` for F2, ``18`` for F3, and ``10`` for F4-F10
- ``CEC2020``: currently accepts ``2, 5, 10, 15, 20``
- ``CEC2022``: currently accepts ``2, 10, 20``

Some suites also have function-specific restrictions at certain dimensions.


CEC2017 Example
===============

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

The benchmark wrapper evaluates the objective, but you still provide the search bounds to ``Minimizer``.

Typical examples:

- ``CEC2014`` / ``CEC2017`` / ``CEC2020`` / ``CEC2022``: the project benchmark drivers typically use ``[(-100, 100)] * dimension``
- ``CEC2019``: use the suite-specific ranges

  - F1: ``[(-8192, 8192)] * 9``
  - F2: ``[(-16384, 16384)] * 16``
  - F3: ``[(-4, 4)] * 18``
  - F4-F10: ``[(-100, 100)] * 10``

- ``CEC2011``: use the problem-specific bounds

For ``CEC2011``, MinionPy exposes the suite-defined bounds directly:

.. code-block:: python

    cec2011 = mpy.CEC2011Functions(function_number=1, dimension=6)
    bounds = cec2011.get_bounds()

Further Example
===============

For broader benchmark examples and sweeps, see ``tests/test_minionpy.py`` and the main documentation examples.
