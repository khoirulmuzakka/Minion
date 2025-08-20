#include <iostream>
#include <vector>
#include <chrono>
#include "minion.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define the Rastrigin function (scalar version)
double rastrigin(const std::vector<double>& x) {
    double A = 10.0;
    double sum = A * x.size();
    for (double xi : x) {
        sum += (xi * xi - A * cos(2 * M_PI * xi));
    }
    return sum;
};

// Define the Rosenbrock function (scalar version)
double rosenbrock(const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; i++) {
        sum += 100 * pow(x[i + 1] - x[i] * x[i], 2) + pow(1 - x[i], 2);
    }
    return sum;
};

// Vectorized version of the Rastrigin function
std::vector<double> rastrigin_vect(const std::vector<std::vector<double>>& X, void* data) {
    std::vector<double> ret;
    for (auto& x : X) ret.push_back(rastrigin(x));
    return ret;
};

// Vectorized version of the Rosenbrock function
std::vector<double> rosenbrock_vect(const std::vector<std::vector<double>>& X, void* data) {
    std::vector<double> ret;
    for (auto& x : X) ret.push_back(rosenbrock(x));
    return ret;
};

// Example of using a class-based objective function
class SomeObjective {
public:
    SomeObjective() {};
    ~SomeObjective() {};
    
    // Objective function implemented as a class method
    double objective(std::vector<double> x) {
        return rastrigin(x);
    };
};

// Wrapper function for the class-based objective function
std::vector<double> objective_function(const std::vector<std::vector<double>>& X, void* data) {
    SomeObjective* object = static_cast<SomeObjective*>(data);
    std::vector<double> ret;
    for (auto& x : X) ret.push_back(object->objective(x));
    return ret;
}

// Callback function for Minion (can be used for logging or monitoring)
void callBack(minion::MinionResult* res) {
    // std::cout << "Best fitness " << res->fun << "\n";
};

int main(int argc, char* argv[]) {
    // List of optimization algorithms to test
    std::vector<std::string> algoList = { "ARRDE", "LSHADE", "LSRTDE", "NLSHADE_RSP", "j2020", "jSO",
                                     "JADE", "L_BFGS_B", "L_BFGS", "DA", "ABC", "NelderMead"};

    // Define the dimensionality of the optimization problem
    size_t dimension = 20;
    
    // Define the search bounds for all dimensions
    std::vector<std::pair<double, double>> bounds = std::vector<std::pair<double, double>>(dimension, std::make_pair(-100.0, 100.0));
    
    // Initial guess (empty in this case, meaning random initialization is used)
    std::vector<std::vector<double>> x0 = {}; //note that some algorithms such as NelderMead, DA, L_BFGS, L_BFGS_B require non-empty initial guess

    //alternatively, you can add initial guesses into the x0 
    x0 = minion::latin_hypercube_sampling(bounds, 2); //here, two initial guesses.
    
    // Maximum number of function evaluations
    size_t max_evals = dimension*1000;

    // Minimizing Rosenbrock function
    std::cout << "Minimizing Rosenbrock function: \n";
    for (auto& algo : algoList) {
        auto settings = minion::DefaultSettings().getDefaultSettings(algo);
        settings["population_size"] = 0;  // Let Minion decide the best population size
        auto res = minion::Minimizer(rosenbrock_vect, bounds, x0, nullptr, callBack, algo, 0.0, max_evals, -1, settings).optimize();
        std::cout << "\t " << algo << " : " << res.fun << "\n";
    };

    // Minimizing Rastrigin function
    std::cout << "\nMinimizing Rastrigin function: \n";
    for (auto& algo : algoList) {
        auto settings = minion::DefaultSettings().getDefaultSettings(algo);
        settings["population_size"] = 0;
        auto res = minion::Minimizer(rastrigin_vect, bounds, x0, nullptr, callBack, algo, 0.0, max_evals, -1, settings).optimize();
        std::cout << "\t " << algo << " : " << res.fun << "\n";
    };

    // Minimizing an objective function that is a class member
    std::cout << "\nMinimizing function which is a class member: \n";
    SomeObjective* so = new SomeObjective();
    for (auto& algo : algoList) {
        auto settings = minion::DefaultSettings().getDefaultSettings(algo);
        settings["population_size"] = 0;
        auto res = minion::Minimizer(objective_function, bounds, x0, so, callBack, algo, 0.0, max_evals, -1, settings).optimize();
        std::cout << "\t " << algo << " : " << res.fun << "\n";
    };

    // Clean up dynamically allocated object
    delete so;

    return 0;
}
