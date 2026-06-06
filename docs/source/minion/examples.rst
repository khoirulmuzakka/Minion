Examples
========

This page shows practical C++ usage patterns for Minion.

Reference files in this repository:

- ``examples/main_minimizer.cpp``
- ``examples/main_cec.cpp``
- ``tests/test_minion.cpp``


Basic Pattern
=============

Minion expects a **vectorized objective**:

.. code-block:: cpp

    std::vector<double> objective(const std::vector<std::vector<double>>& X, void* data) {
        std::vector<double> out(X.size(), 0.0);
        for (size_t i = 0; i < X.size(); ++i) {
            const auto& x = X[i];
            // compute f(x)
            out[i] = /* ... */;
        }
        return out;
    }

Then call ``minion::Minimizer``:

.. code-block:: cpp

    std::vector<std::pair<double, double>> bounds(dim, {-5.0, 5.0});
    std::vector<std::vector<double>> x0 = {
        std::vector<double>(dim, 0.0)
    };
    std::string algo = "LSHADE";
    auto settings = minion::DefaultSettings().getDefaultSettings(algo);

    minion::MinionResult res = minion::Minimizer(
        objective, bounds, x0, nullptr, nullptr, algo, 100000, 42, settings
    ).optimize();

``x0`` is a ``std::vector<std::vector<double>>`` because Minion accepts **multiple initial guesses**.
Each inner vector is one candidate starting point.

.. note::

   Multiple initial guesses are **not** the same thing as directly specifying the full internal population.
   For population-based algorithms, Minion first initializes the population using the algorithm's usual rules, then replaces some of those individuals with the provided guesses.
   For single-trajectory algorithms, Minion evaluates the supplied guesses first and starts from the best one.


Understanding MinionResult
==========================

``optimize()`` returns ``minion::MinionResult`` with these fields:

- ``x``: best decision vector found.
- ``fun``: objective value at ``x``.
- ``nit``: number of iterations/generations completed.
- ``nfev``: number of objective evaluations.
- ``success``: solver status flag.
- ``message``: termination message (if provided by the algorithm).

Example:

.. code-block:: cpp

    minion::MinionResult res = minion::Minimizer(
        objective, bounds, x0, nullptr, nullptr, "ARRDE", 100000, 42
    ).optimize();

    std::cout << "success: " << std::boolalpha << res.success << "\n";
    std::cout << "fun: " << res.fun << "\n";
    std::cout << "nit: " << res.nit << ", nfev: " << res.nfev << "\n";
    std::cout << "x_best[0]: " << (res.x.empty() ? 0.0 : res.x[0]) << "\n";
    if (!res.message.empty()) {
        std::cout << "message: " << res.message << "\n";
    }


Using Callback
==============

You can pass a callback to monitor progress after each iteration:

.. code-block:: cpp

    void progress_callback(minion::MinionResult* state) {
        std::cout << "iter=" << state->nit
                  << " nfev=" << state->nfev
                  << " best=" << state->fun << "\n";
    }

    minion::MinionResult res = minion::Minimizer(
        objective, bounds, x0, nullptr, progress_callback, "LSHADE", 100000, 42
    ).optimize();

If you need custom early stopping, a practical pattern is to throw from the callback
when your condition is met, then catch outside:

.. code-block:: cpp

    struct StopNow : public std::exception {
        const char* what() const noexcept override { return "user stop"; }
    };

    void early_stop_callback(minion::MinionResult* state) {
        if (state->nfev >= 20000 || state->fun < 1e-8) {
            throw StopNow();
        }
    }

    try {
        auto res = minion::Minimizer(
            objective, bounds, x0, nullptr, early_stop_callback, "ARRDE", 100000, 42
        ).optimize();
        (void)res;
    } catch (const StopNow&) {
        std::cout << "Optimization stopped by user callback.\n";
    }


Override Default Options
========================

Start from defaults, then override only what you need.

.. code-block:: cpp

    std::string algo = "DE";
    auto settings = minion::DefaultSettings().getDefaultSettings(algo);

    // Common overrides
    settings["population_size"] = 80;
    settings["convergence_tol"] = 1e-6;
    settings["bound_strategy"] = std::string("reflect-random");

    // Algorithm-specific overrides
    if (algo == "DE") {
        settings["mutation_rate"] = 0.7;
        settings["crossover_rate"] = 0.9;
        settings["mutation_strategy"] = std::string("current_to_pbest1bin");
    }

    minion::MinionResult res = minion::Minimizer(
        objective, bounds, x0, nullptr, nullptr, algo, 100000, 42, settings
    ).optimize();

Notes:

- Option keys are algorithm-specific. Use ``DefaultSettings`` as the source of valid keys.
- Keep value types consistent with the expected type (``int``, ``double``, ``std::string``, ``bool``).


Multithreading Objective Evaluation
===================================

Minion calls your objective in batches (``X``).  
Parallelization for C++ workflows is typically implemented **inside your objective function**.


1) Standalone Function (Thread-Safe)
------------------------------------

If each evaluation is independent, parallelize the loop over ``X``.

.. code-block:: cpp

    #include <omp.h>

    double sphere(const std::vector<double>& x) {
        double s = 0.0;
        for (double v : x) s += v * v;
        return s;
    }

    std::vector<double> sphere_batch(const std::vector<std::vector<double>>& X, void*) {
        std::vector<double> out(X.size(), 0.0);
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            out[i] = sphere(X[i]);
        }
        return out;
    }


2) Non-Thread-Safe Class Method
-------------------------------

If the class has mutable shared state, protect access.

.. code-block:: cpp

    #include <mutex>

    class Model {
    public:
        double eval(const std::vector<double>& x) {
            // touches mutable shared state internally
            return /* ... */;
        }
    };

    struct ModelData {
        Model* model;
        std::mutex* mtx;
    };

    std::vector<double> model_batch(const std::vector<std::vector<double>>& X, void* data) {
        auto* md = static_cast<ModelData*>(data);
        std::vector<double> out(X.size(), 0.0);
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            std::lock_guard<std::mutex> lock(*md->mtx);
            out[i] = md->model->eval(X[i]);
        }
        return out;
    }

If possible, prefer a re-entrant/stateless evaluator per thread to avoid lock contention.


3) Lambda Function
------------------

Use a lambda and assign it to ``minion::MinionFunction``.

.. code-block:: cpp

    minion::MinionFunction objective = [](const std::vector<std::vector<double>>& X, void*) {
        std::vector<double> out(X.size(), 0.0);
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            const auto& x = X[i];
            double f = 0.0;
            for (size_t j = 0; j + 1 < x.size(); ++j) {
                const double a = x[j + 1] - x[j] * x[j];
                const double b = 1.0 - x[j];
                f += 100.0 * a * a + b * b;
            }
            out[i] = f;
        }
        return out;
    };

