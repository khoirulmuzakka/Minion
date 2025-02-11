This section provides examples of how to use the **Minion** library to solve bound-constrained optimization problems using various algorithms.

Basic Usage
===========

This example demonstrates how to use **Minion** to optimize the Rosenbrock and Rastrigin functions using different optimization algorithms.

## Example: Optimizing Rosenbrock Function with Minion

The following C++ example showcases how to use Minion to optimize the Rosenbrock function using multiple evolutionary algorithms.

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
    }

    // Vectorized version for Minion
    std::vector<double> rosenbrock_vect(const std::vector<std::vector<double>>& X, void* data) {
        std::vector<double> ret;
        for (const auto& x : X) ret.push_back(rosenbrock(x));
        return ret;
    }

    int main() {
        // List of optimization algorithms to test
        std::vector<std::string> algoList = {"ARRDE", "LSHADE", "LSRTDE"};
        
        // Define the problem dimensions and bounds
        size_t dimension = 10;
        std::vector<std::pair<double, double>> bounds(dimension, {-5.0, 5.0});
        size_t max_evals = 100000;

        // Iterate over each algorithm and perform optimization
        for (const auto& algo : algoList) {
            auto settings = minion::DefaultSettings().getDefaultSettings(algo);
            minion::MinionResult res = minion::Minimizer(rosenbrock_vect, bounds, {}, nullptr, nullptr, algo, 0.0, max_evals, -1, settings).optimize();
            std::cout << algo << " : " << res.fun << "\n";
        }
        return 0;
    }


Minion supports the following optimization algorithms:

- **Differential Evolution (DE) Variants:**

  - ``"LSHADE"``  (Success History Adaptive DE with Linear Population Size Reduction)
  - ``"DE"``      (The original DE)
  - ``"JADE"``     (An adaptive DE)
  - ``"jSO"``      (A variant of LSHADE with some improvements)
  - ``"NLSHADE_RSP"`` (A variant of LSHADE with some improvements)
  - ``"j2020"``  (A variant of jDE algorithm)
  - ``"GWO_DE"``  (Grey Wolf-DE optimization)
  - ``"ARRDE"``   (Adaptive restart-refine DE)
  - ``"LSRTDE"``  (A variant of LSHADE with some improvements)

- **Swarm Intelligence Algorithms:**

  - ``"ABC"`` (Artificial Bee Colony)
  - ``"DA"`` (Generalized Simulated Annealing or Dual Annealing)

- **Classical Optimization Methods:**

  - ``"NelderMead"``
  - ``"L_BFGS_B"`` (Limited-memory BFGS with Bound Constraints)

Each of these algorithms can be selected using their corresponding names when calling Minion's `Minimizer`.


MinionResult Class
==================

The `MinionResult` class stores the result of an optimization process performed using the Minion optimization library.

**Attributes :**

- ``x`` (`std::vector<double>`): The solution vector containing the values for the decision variables at the optimal solution.
- ``fun`` (`double`): The objective function value at the optimal solution.
- ``nit`` (`size_t`): The number of iterations performed during the optimization process.
- ``nfev`` (`size_t`): The number of function evaluations performed during the optimization process.
- ``success`` (`bool`): A boolean flag indicating whether the optimization was successful.
- ``message`` (`std::string`): A message providing additional information about the result of the optimization process.


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
