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

    std::map<std::string, ConfigValue> options_lshade= std::map<std::string, ConfigValue> {
        {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
        {"memory_size", int(6)},
        {"archive_size_ratio", double(2.6)}, 
        {"population_reduction" , bool(true)}, 
        {"reduction_strategy", std::string("linear")}, //linear or exponential
        {"minimum_population_size", int(4)}, 
    };  

    std::map<std::string, ConfigValue> options_jade= std::map<std::string, ConfigValue> {
        {"mutation_strategy", std::string("current_to_pbest_A_1bin")},
        {"archive_size_ratio", double(1.0)}, 
        {"population_reduction" , bool(true)}, 
        {"reduction_strategy", std::string("linear")},
        {"c", 0.1},
    };

    int popsize=population_size;

    if (algo == "j2020") {
        optimizer = new minion::j2020 (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr,  max_evals, -1, popsize
        );
    } else if (algo=="SIMPLEX"){
        std::vector<double> x0; 
        for (auto& el : bounds) x0.push_back(0.5*(el.first+el.second));
        optimizer = new minion::NelderMead(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, x0, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1
        );
    }  else if (algo=="ARRDE"){
        optimizer = new minion::ARRDE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else if (algo == "NLSHADE_RSP") {
        optimizer = new minion::NLSHADE_RSP (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr,  max_evals, -1, popsize , std::min(int(20*dimension), 1000), 2.1 
        );
    } else if (algo == "LSRTDE") {
        optimizer = new minion::LSRTDE (
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, max_evals, -1, popsize 
        );
    } else if (algo == "LSHADE"){ 
        optimizer = new minion::LSHADE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options_lshade, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else if (algo == "LSHADE2"){ 
        if (population_size==0) popsize = 10*dimension;
        optimizer = new minion::LSHADE2(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options_lshade, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else if (algo == "JADE"){ 
        optimizer = new minion::JADE(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, options_jade, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
        );
    } else if (algo == "jSO"){ 
        optimizer = new minion::jSO(
            [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, 0.0, max_evals, "reflect-random", -1, popsize
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
    int dimension = 10;
    std::string algo="ARRDE";
    int popsize=0;
    int year = 2017;
    if (argc > 1) {
        dimension = std::atoi(argv[1]); 
    }
    if (argc > 2) {
        algo = argv[2]; 
    }
    if (argc > 3) {
        popsize = std::atoi(argv[3]);
    }

    if (argc > 4) {
        year = std::atoi(argv[4]); 
    }

    std::vector<int> Nevals = {5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000};
    std::vector<int> Nrepeats = {101, 101, 101, 51, 51, 31, 21, 21};

    for (int i=0; i<Nevals.size(); i++) {
        int Nmaxevals = Nevals[i]; 
        int numRuns = Nrepeats[i]; 
        std::vector<int> funcnums; 
        if (year==2017) funcnums =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20,  21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
        else if (year==2020) funcnums = {1,2,3,4,5,6,7,8,9, 10}; 
        else if (year==2022) funcnums = {1,2,3,4,5,6,7,8,9, 10, 11, 12}; 
        else throw std::runtime_error("Year invalid.");

        std::vector<std::vector<double>> results;
        for (int i=0; i<numRuns; i++){
            std::cout << "\nRun : "<< i+1 << ", with maxevals : " << Nmaxevals << "\n";
            std::vector<double> result_per_run;
            for (auto& num : funcnums) result_per_run.push_back(minimize_cec_functions(num, dimension, popsize, Nmaxevals, year, algo));
            results.push_back(result_per_run);
        };
        dumpResultsToFile(results, "results_"+std::to_string(year)+"_"+algo+"_" + std::to_string(dimension)+"_"+std::to_string(Nmaxevals)+".txt");
    };
    return 0;
}
