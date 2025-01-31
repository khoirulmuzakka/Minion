This section provides examples of how to use the **Minion** library to solve bound-constrained optimization problems using various algorithms.

Basic Usage
===========

This example demonstrates how to use **Minion** to optimize the Rosenbrock and Rastrigin functions using different optimization algorithms.

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include "minion.h"

    // Define the Rosenbrock function
    double rosenbrock(const std::vector<double>& x) {
        double sum = 0.0;
        for (size_t i = 0; i < x.size() - 1; i++) {
            sum += 100 * pow(x[i + 1] - x[i] * x[i], 2) + pow(1 - x[i], 2);
        }
        return sum;
    };

    // Vectorized version for Minion
    std::vector<double> rosenbrock_vect(const std::vector<std::vector<double>>& X, void* data) {
        std::vector<double> ret;
        for (auto& x : X) ret.push_back(rosenbrock(x));
        return ret;
    }

    int main() {
        std::vector<std::string> algoList = {"ARRDE", "LSHADE", "LSRTDE"};
        size_t dimension = 10;
        std::vector<std::pair<double, double>> bounds(dimension, {-5.0, 5.0});
        size_t max_evals = 100000;

        for (auto& algo : algoList) {
            auto settings = minion::DefaultSettings().getDefaultSettings(algo);
            auto res = minion::Minimizer(rosenbrock_vect, bounds, {}, nullptr, nullptr, algo, 0.0, max_evals, -1, settings).optimize();
            std::cout << algo << " : " << res.fun << "\n";
        }
        return 0;
    }

Using Class-Based Objective Functions
======================================

You can also define objective functions as class methods:

.. code-block:: cpp

    class SomeObjective {
    public:
        double objective(std::vector<double> x) {
            return rosenbrock(x);
        }
    };

    std::vector<double> objective_function(const std::vector<std::vector<double>>& X, void* data) {
        SomeObjective* obj = static_cast<SomeObjective*>(data);
        std::vector<double> ret;
        for (auto& x : X) ret.push_back(obj->objective(x));
        return ret;
    }

    int main() {
        SomeObjective obj;
        auto res = minion::Minimizer(objective_function, bounds, {}, &obj, nullptr, "LSHADE", 0.0, max_evals, -1, settings).optimize();
        std::cout << "Result: " << res.fun << "\n";
        return 0;
    }

Using Callbacks
===============

You can define a callback function to track optimization progress:

.. code-block:: cpp

    void callBack(minion::MinionResult* res) {
        std::cout << "Current best fitness: " << res->fun << "\n";
    }

    int main() {
        auto res = minion::Minimizer(rosenbrock_vect, bounds, {}, nullptr, callBack, "ARRDE", 0.0, max_evals, -1, settings).optimize();
        return 0;
    }

These examples illustrate different ways to use Minion effectively. For more details, see the API reference and algorithm descriptions.


Default Options
===============

The `DefaultSettings` class in the `minion` namespace provides default configuration settings for various algorithms in the Minion library. Below is a summary of the default settings for each algorithm.

Algorithms and Their Default Settings
--------------------------------------

1. **DE (Differential Evolution)**

   - `population_size`: 0
   - `mutation_rate`: 0.5
   - `crossover_rate`: 0.8
   - `mutation_strategy`: `best1bin`
   - `bound_strategy`: `reflect-random`

2. **ARRDE (Adaptive Restart-Refine Differential Evolution)**

   - `population_size`: 0
   - `archive_size_ratio`: 2.0
   - `converge_reltol`: 0.005
   - `refine_decrease_factor`: 0.9
   - `restart-refine-duration`: 0.8
   - `maximum_consecutive_restarts`: 2
   - `bound_strategy`: `reflect-random`

3. **GWO_DE (Grey Wolf Optimization Differential Evolution)**

   - `population_size`: 0
   - `mutation_rate`: 0.5
   - `crossover_rate`: 0.7
   - `elimination_prob`: 0.1
   - `bound_strategy`: `reflect-random`

4. **j2020 (A variant of Differential Evolution)**

   - `population_size`: 0
   - `tau1`: 0.1
   - `tau2`: 0.1
   - `myEqs`: 0.25
   - `bound_strategy`: `reflect-random`

5. **LSRTDE (Local Search Restart Differential Evolution)**

   - `population_size`: 0
   - `memory_size`: 5
   - `success_rate`: 0.5
   - `bound_strategy`: `reflect-random`

6. **NLSHADE_RSP (Noisy Local Search Adaptive Differential Evolution with Randomized Restart)**

   - `population_size`: 0
   - `memory_size`: 100
   - `archive_size_ratio`: 2.6
   - `bound_strategy`: `reflect-random`

7. **JADE (Jiang's Adaptive Differential Evolution)**

   - `population_size`: 0
   - `c`: 0.1
   - `mutation_strategy`: `current_to_pbest_A_1bin`
   - `archive_size_ratio`: 1.0
   - `minimum_population_size`: 4
   - `reduction_strategy`: `linear` 
   - `bound_strategy`: `reflect-random`

8. **jSO (jSO - Another DE Variant)**

   - `population_size`: 0
   - `memory_size`: 5
   - `archive_size_ratio`: 1.0
   - `minimum_population_size`: 4
   - `reduction_strategy`: `linear` 
   - `bound_strategy`: `reflect-random`

9. **LSHADE (LSHADE - Local Search Variant)**

   - `population_size`: 0
   - `memory_size`: 6
   - `mutation_strategy`: `current_to_pbest_A_1bin`
   - `archive_size_ratio`: 2.6
   - `minimum_population_size`: 4
   - `reduction_strategy`: `linear` 
   - `bound_strategy`: `reflect-random`

10. **NelderMead (Nelder-Mead Simplex Algorithm)**

    - `bound_strategy`: `reflect-random`



Accessing Default Settings
---------------------------

To retrieve the default settings for a specific algorithm, use the `getDefaultSettings` method in the `DefaultSettings` class. This method takes the algorithm name as a string argument and returns a map of configuration values.

Example usage:

.. code-block:: cpp

    minion::DefaultSettings settings;
    std::map<std::string, ConfigValue> de_settings = settings.getDefaultSettings("DE");
