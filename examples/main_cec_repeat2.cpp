#include <iostream>
#include <vector>
#include <utility>
#include "minion.h"


double minimize_cec_functions(int function_number, int dimension, int population_size, int max_evals, int year=2022, std::string algo="ARRDE", int seed=-1) {
    minion::CECBase* cecfunc;
    if (year==2020) cecfunc = new minion::CEC2020Functions(function_number, dimension);
    else if (year==2022) cecfunc = new minion::CEC2022Functions(function_number, dimension);
    else if (year==2017) cecfunc = new minion::CEC2017Functions(function_number, dimension);
    else throw std::runtime_error("Invalid year.");

    std::vector<std::pair<double, double>> bounds(dimension, std::make_pair(-100.0, 100.0));


    int popsize=population_size;

    auto settings = minion::algoToSettingsMap[algo];
    settings["population_size"] = size_t (popsize);

    minion::Minimizer optimizer ( [&](const std::vector<std::vector<double>> & x, void* data) {
                return cecfunc->operator()(x); // Call the operator with a single vector
            },
            bounds, {}, nullptr, nullptr, algo, 0.0, max_evals,  seed, settings
    );
    // Optimize and get the result
    minion::MinionResult result = optimizer();

    double ret = result.fun;

    // Output the results
    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "\tAlgo : "<< algo << ". Best Value: " << result.fun << "\n";

    delete cecfunc;
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
            for (auto& num : funcnums) result_per_run.push_back(minimize_cec_functions(num, dimension, popsize, Nmaxevals, year, algo, i));
            results.push_back(result_per_run);
        };
        dumpResultsToFile(results, "results_"+std::to_string(year)+"_"+algo+"_" + std::to_string(dimension)+"_"+std::to_string(Nmaxevals)+".txt");
    };
    return 0;
}
