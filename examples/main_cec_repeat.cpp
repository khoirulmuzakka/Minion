#include <iostream>
#include <vector>
#include <utility>
#include "minion.h"


double minimize_cec_functions(int function_number, int dimension, int population_size, int max_evals, int year=2022, std::string algo="ARRDE") {
    minion::CECBase* cecfunc;
    minion::MinimizerBase* optimizer;

    if (year==2020) cecfunc = new minion::CEC2020Functions(function_number, dimension);
    else if (year==2022) cecfunc = new minion::CEC2022Functions(function_number, dimension);
    else if (year==2017) cecfunc = new minion::CEC2017Functions(function_number, dimension);
    else throw std::runtime_error("Invalid year.");

    std::vector<std::pair<double, double>> bounds(dimension, std::make_pair(-100.0, 100.0));

    std::map<std::string, ConfigValue> options= std::map<std::string, ConfigValue> {
        {"mutation_strategy", std::string("current_to_pbest_AW_1bin")},
        {"memory_size", int(50)},
        {"archive_size_ratio", double(2.0)}, 
        {"population_reduction" , bool(true)}, 
        {"reduction_strategy", std::string("linear")}, //linear or exponential
        {"minimum_population_size", int(5)}, 
        {"c", 0.1},
    };  

    std::map<std::string, ConfigValue> options_jso = std::map<std::string, ConfigValue> {
        {"mutation_strategy", std::string("current_to_pbest_AW_1bin")},
        {"memory_size", int(5)},
        {"archive_size_ratio", double(1.0)}, 
        {"population_reduction" , bool(true)}, 
        {"reduction_strategy", std::string("linear")}, //linear or exponential
        {"minimum_population_size", int(4)}, 
    };  

    std::map<std::string, ConfigValue> options_arrde= std::map<std::string, ConfigValue> {
        {"mutation_strategy", std::string("current_to_pbest_AW_1bin")},
        {"archive_size_ratio", double(2.0)}, 
        {"population_reduction" , bool(true)}, 
        {"reduction_strategy", std::string("linear")}, //linear or exponential
        {"minimum_population_size", int(5)}, 
    };  

    int popsize=population_size;

    if (algo == "j2020") {
        optimizer = new minion::j2020 (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1
        );
    } else if (algo=="ARRDE"){
        if (population_size==0) popsize = static_cast<int>(std::ceil(std::pow(std::log10(static_cast<double>(max_evals)), 2.0) + static_cast<double>(dimension)/2.0));
        optimizer = new minion::ARRDE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options_arrde, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else if (algo == "NLSHADE_RSP") {
        if (population_size==0) popsize = std::min(int(30*dimension), 1000);
        optimizer = new minion::NLSHADE_RSP (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize , std::min(int(20*dimension), 1000), 2.1 
        );
    } else if (algo == "LSHADE"){ 
        if (population_size==0) popsize = static_cast<int>(std::ceil(std::pow(std::log10(static_cast<double>(max_evals)), 2.0) + static_cast<double>(dimension)/2.0));
        optimizer = new minion::LSHADE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else if (algo == "JADE"){ 
        if (population_size==0) popsize = static_cast<int>(std::ceil(std::pow(std::log10(static_cast<double>(max_evals)), 2.0) + static_cast<double>(dimension)/2.0));
        optimizer = new minion::JADE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else if (algo == "jSO"){ 
        if (population_size==0) popsize = int(25.0*log10(dimension)*sqrt(dimension));
        optimizer = new minion::jSO(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else throw std::runtime_error("unknown algorithm!");
    
   
    // Optimize and get the result
    MinionResult result = optimizer->optimize();
    double ret = result.fun;

    // Output the results
    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "\tAlgo : "<< algo << ". Best Value: " << result.fun << "\n";

    delete cecfunc;
    delete optimizer;
    return ret;
}

void dumpResultsToFile(const std::vector<std::vector<double>>& results, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Iterate through the 2D vector and write to the file
    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << "\t"; // Separate values by tabs
            }
        }
        file << "\n"; // New line for each row
    }

    file.close();
}


int main(int argc, char* argv[]) {
    int numRuns= 1;
    int dimension = 10;
    std::string algo="ARRDE";
    int popsize=0;
    int year = 2017;
    int Nmaxevals = int(1e+4*dimension);
    if (argc > 1) {
        numRuns = std::atoi(argv[1]); // Convert first argument to integer for numRuns
    }
    if (argc > 2) {
        dimension = std::atoi(argv[2]); // Convert second argument to integer for dimension
    }
    if (argc > 3) {
        algo = argv[3]; // Use third argument for algo, no conversion needed
    }
    if (argc > 4) {
        popsize = std::atoi(argv[4]); // Use third argument for algo, no conversion needed
    }

    if (argc > 5) {
        year = std::atoi(argv[5]); // Use third argument for algo, no conversion needed
    }

    if (argc > 6) {
        Nmaxevals = std::atoi(argv[6]); // Use third argument for algo, no conversion needed
    }

    std::cout << popsize << " "<< Nmaxevals<< "\n";
    
    

    std::vector<int> funcnums; 
    if (year==2017) funcnums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                11, 12,13, 14, 15, 16, 17, 18, 19, 20, 
                                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    else if (year==2020) funcnums =  {1,2,3,4,5,6,7,8,9, 10}; 
    else if (year==2022) funcnums = {1,2,3,4,5,6,7,8,9, 10, 11, 12}; 
    else throw std::runtime_error("Year invalid.");

    std::vector<std::vector<double>> results;
    for (int i=0; i<numRuns; i++){
        std::cout << "\nRun : "<< i+1 << "\n";
        std::vector<double> result_per_run;
        for (auto& num : funcnums) result_per_run.push_back(minimize_cec_functions(num, dimension, popsize, Nmaxevals, year, algo));
        results.push_back(result_per_run);
    };
    dumpResultsToFile(results, "results_"+std::to_string(year)+"_"+algo+"_" + std::to_string(dimension)+"_"+std::to_string(Nmaxevals)+".txt");

    return 0;
}
