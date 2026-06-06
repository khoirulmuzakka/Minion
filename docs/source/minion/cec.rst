CEC Benchmarks
==============

This page describes how to minimize CEC benchmark functions with the **C++** Minion API.

General Pattern
===============

Using a CEC benchmark in Minion follows the same overall workflow as any other objective:

1. Construct a benchmark evaluator such as ``minion::CEC2017Functions``.
2. Provide search ``bounds`` separately to ``minion::Minimizer``.
3. Run ``optimize()``.

CEC benchmark evaluators are already **batch evaluators**. In C++, they are exposed as callable objects derived from ``minion::CECBase`` and evaluate

.. code-block:: cpp

    std::vector<std::vector<double>>

input batches into one objective value per point.


Constructor Pattern
===================

Most suites use the constructor form:

.. code-block:: cpp

    minion::CEC2014Functions(function_number, dimension)
    minion::CEC2017Functions(function_number, dimension)
    minion::CEC2019Functions(function_number, dimension)
    minion::CEC2020Functions(function_number, dimension)
    minion::CEC2022Functions(function_number, dimension)
    minion::CEC2011Functions(function_number, dimension)

Valid dimensions by suite:

- ``CEC2011``: fixed, problem-specific dimensions depending on the selected function
- ``CEC2014``: ``2, 10, 20, 30, 50, 100``
- ``CEC2017``: ``2, 10, 20, 30, 50, 100``
- ``CEC2019``: ``9`` for F1, ``16`` for F2, ``18`` for F3, and ``10`` for F4-F10
- ``CEC2020``: ``2, 5, 10, 15, 20, 30, 50, 100``
- ``CEC2022``: ``2, 10, 20``

Some suites also have function-specific restrictions at certain dimensions.


CEC2017 Example
===============

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


About Bounds
============

The benchmark object evaluates the function, but it does **not** supply bounds to ``minion::Minimizer`` automatically. You should still pass ``bounds`` explicitly.

Typical examples:

- ``CEC2014`` / ``CEC2017`` / ``CEC2020`` / ``CEC2022``: the project benchmark drivers typically use ``[-100, 100]^D``
- ``CEC2019``: use the suite-specific ranges

  - F1: ``[-8192, 8192]^D`` with ``D = 9``
  - F2: ``[-16384, 16384]^D`` with ``D = 16``
  - F3: ``[-4, 4]^D`` with ``D = 18``
  - F4-F10: ``[-100, 100]^D`` with ``D = 10``

- ``CEC2011``: use the problem-specific bounds from the original suite

For ``CEC2011``, the bound box depends on the selected real-world problem. See ``examples/main_cec.cpp`` for the complete per-problem setup.

Further Example
===============

For a more complete benchmark driver covering multiple years, functions, and suite-specific bound handling, see ``examples/main_cec.cpp`` in the repository.
