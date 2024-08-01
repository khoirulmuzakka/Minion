#include <iostream>
#include <vector>
#include <utility>
#include "minion.h"

/**
 * @brief Example function to minimize CEC 2020 and 2022 Functions.
 * 
 * This function minimizes the specified CEC function using the LSHADE and LSHADE2 algorithms.
 * 
 * @param function_number The CEC function number (1-10).
 * @param dimension The dimension of the problem.
 * @param population_size The size of the population.
 * @param max_evals The maximum number of evaluations.
 * @param year The year of the CEC function (2020 or 2022). Default is 2022.
 * 
 * @throws std::runtime_error If the function number is not between 1 and 10, or if the year is invalid.
 */
void minimize_cec_functions(int function_number, int dimension, int population_size, int max_evals, int year=2022) {

    if (function_number < 1 || function_number > 12) throw std::runtime_error("Function number must be between 1 and 10.");

    minion::CECBase* cecfunc;
    if (year==2020) cecfunc = new minion::CEC2020Functions(function_number, dimension);
    else if (year==2022) cecfunc = new minion::CEC2022Functions(function_number, dimension);
    else throw std::runtime_error("Invalid year.");

    std::vector<std::pair<double, double>> bounds(dimension, std::make_pair(-100.0, 100.0));

    std::map<std::string, ConfigValue> options= std::map<std::string, ConfigValue> {
        {"mutation_strategy", std::string("current_to_pbest_A1_1bin")},
        {"memory_size", int(50)}, 
        {"archive_size_ratio", double(2.0)}, 
        {"population_reduction" , bool(true)}, 
        {"reduction_strategy", std::string("linear")}, //linear or exponential
        {"minimum_population_size", int(5)}, 
        { "refine_method" , std::string("jade")}
    };  

    minion::LSHADE lshade (
        [&](const std::vector<std::vector<double>> & x, void* data) {
            return cecfunc->operator()(x); // Call the operator with a single vector
        },
        bounds, options, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, population_size
    );

    minion::ARRDE arrde (
        [&](const std::vector<std::vector<double>> & x, void* data) {
            return cecfunc->operator()(x); // Call the operator with a single vector
        },
        bounds, options, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, population_size
    );
   
    // Optimize and get the result
    MinionResult result_lshade = lshade.optimize();
    MinionResult result_arrde = arrde.optimize();

    // Output the results
    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "\tAlgo : "<<" LSHADE, Best Value: " << result_lshade.fun << "\n";
    std::cout << "\tAlgo : "<<" LSHADE2, Best Value: " << result_arrde.fun << "\n";
    std::cout << std::endl;
    delete cecfunc;
}


/**
 * @brief Main function to run the example.
 * 
 * This function runs the `minimize_cec_functions` for CEC functions 1 to 10 with dimension 10.
 * 
 * @return int Returns 0 on successful completion.
 */
int main() {
    int Nmaxevals = int(1e+6), dimension = 20;
    int popsize = static_cast<int>(std::ceil(std::pow(std::log10(static_cast<double>(Nmaxevals)), 1.8) + static_cast<double>(dimension)));
    int year = 2022;

    std::vector<int> funcnums = {2, 11};
    for (auto& num : funcnums) {
        minimize_cec_functions(num, dimension, popsize, Nmaxevals, year);
        std::cout << "\n";
    }
    return 0;
}