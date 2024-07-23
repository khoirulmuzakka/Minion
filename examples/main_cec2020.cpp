#include <iostream>
#include <vector>
#include <utility>
#include "minion.h"

// Example function to minimize CEC 2020 Functions using MFADE
void minimize_cec2020_functions(int function_number, int dimension, int population_size, int max_evals) {
    // Check function number range
    if (function_number < 1 || function_number > 30) {
        std::cerr << "Function number must be between 1 and 10." << std::endl;
        return;
    }
    minion::CEC2020Functions cec2020(function_number, dimension);
    std::vector<std::pair<double, double>> bounds(dimension, std::make_pair(-100.0, 100.0));
    minion::LJADE ljade(
        [&](const std::vector<std::vector<double>> & x, void* data) {
            return cec2020(x); // Call the operator with a single vector
        },
        bounds,
        nullptr,  // No additional data needed
        {},
        population_size,
        max_evals,
        "current_to_pbest1bin",  // DE strategy
        0.0000,   // Relative tolerance
        10,        // Minimum population size
        0.5, 
        nullptr,   // No callback function
        "reflect-random", // Bound strategy
        -1         // No specific seed
    );

    // Optimize and get the result
    MinionResult result = ljade.optimize();
    // Output the results
    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "Best Value: " << result.fun << "\n";
    std::cout << std::endl;
}

int main() {
    // Example: Minimize function 1 with dimension 10
    for (int function_number = 1; function_number <= 30; ++function_number) {
        minimize_cec2020_functions(function_number, 10, 50, 100000);
    }
    return 0;
}
