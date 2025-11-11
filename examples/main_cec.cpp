#include <iostream>
#include <vector>
#include <chrono>
#include "minion.h"
#include "utility.h"
#include "cec2011.h"
#include <fstream>
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<std::pair<double, double>> make_bounds(const std::vector<double>& lb,
                                                   const std::vector<double>& ub) {
    std::vector<std::pair<double, double>> bounds;
    bounds.reserve(lb.size());
    for (size_t i = 0; i < lb.size(); ++i) {
        bounds.emplace_back(lb[i], ub[i]);
    }
    return bounds;
}

std::vector<std::pair<double, double>> bounds_problem02(int dim) {
    std::vector<double> lb(dim, 0.0);
    std::vector<double> ub(dim, 0.0);
    ub[0] = ub[1] = 4.0;
    ub[2] = kPi;
    for (int idx = 3; idx < dim; ++idx) {
        int t = static_cast<int>((idx - 4.0) / 3.0);
        lb[idx] = -4.0 - 0.25 * static_cast<double>(t);
        ub[idx] = 4.0 + 0.25 * static_cast<double>(t);
    }
    return make_bounds(lb, ub);
}

std::vector<std::pair<double, double>> bounds_problem05(int dim) {
    std::vector<double> lb(dim, -1.0);
    lb[0] = lb[1] = lb[2] = 0.0;
    std::vector<double> ub(dim, 0.0);
    ub[0] = ub[1] = 4.0;
    ub[2] = kPi;
    for (int idx = 3; idx < dim; ++idx) {
        int t = static_cast<int>((idx - 4.0) / 3.0);
        ub[idx] = 4.0 + 0.25 * static_cast<double>(t);
    }
    return make_bounds(lb, ub);
}

std::vector<std::pair<double, double>> bounds_problem06(int dim) {
    std::vector<double> lb(dim, -1.0);
    lb[0] = lb[1] = lb[2] = 0.0;
    std::vector<double> ub(dim, 0.0);
    ub[0] = ub[1] = 4.0;
    ub[2] = kPi;
    for (int idx = 3; idx < dim; idx += 3) {
        ub[idx] = 4.0 + 0.25 * static_cast<double>(static_cast<int>((1.0 - 4.0) / 3.0));
        if (idx + 1 < dim) {
            ub[idx + 1] = 4.0 + 0.25 * static_cast<double>(static_cast<int>((2.0 - 4.0) / 3.0));
        }
        if (idx + 2 < dim) {
            ub[idx + 2] = 4.0 + 0.25 * static_cast<double>(static_cast<int>((3.0 - 4.0) / 3.0));
        }
    }
    return make_bounds(lb, ub);
}

struct CEC2011ProblemDef {
    int dimension;
    std::vector<std::pair<double, double>> bounds;
};

CEC2011ProblemDef get_cec2011_problem(int function_number) {
    switch (function_number) {
    case 1:
        return {6, std::vector<std::pair<double, double>>(6, {-6.4, 6.35})};
    case 2:
        return {30, bounds_problem02(30)};
    case 3:
        return {1, {{-0.6, 0.9}}};
    case 4:
        return {1, {{0.0, 5.0}}};
    case 5:
        return {30, bounds_problem05(30)};
    case 6:
        return {30, bounds_problem06(30)};
    case 7: {
        std::vector<std::pair<double, double>> bounds(20, {0.0, 2.0 * kPi});
        return {20, bounds};
    }
    case 8:
        return {7, std::vector<std::pair<double, double>>(7, {0.0, 15.0})};
    default:
        throw std::runtime_error("CEC2011 problem definition not available.");
    }
}

} // namespace

std::vector <double> objective_function (const std::vector<std::vector<double>> & x, void* data){
     minion::CECBase* func = static_cast<minion::CECBase* > (data);
    return func->operator()(x); // Call the operator with a single vector
}

void callBack(minion::MinionResult* res) {
    //std::cout << "Best fitness " << res->fun << "\n";
};

double minimize_cec_functions(int function_number, int dimension, int population_size, int max_evals, int year=2022, std::string algo="ARRDE", int seed=-1) {
    minion::CECBase* cecfunc;
    std::vector<std::pair<double, double>> bounds;
    int effective_dimension = dimension;
    if (year==2019) { 
        if (function_number ==1) effective_dimension =9; 
        else if (function_number==2) effective_dimension =16; 
        else if (function_number==3) effective_dimension =18;
        else effective_dimension=10;
        for (int i=0; i<effective_dimension; i++) {
            if (function_number ==1) bounds.push_back(std::make_pair(-8192, 8192)); 
            else if (function_number==2) bounds.push_back(std::make_pair(-16384, 16384)); 
            else if (function_number==3) bounds.push_back(std::make_pair(-4, 4)); 
            else bounds.push_back(std::make_pair(-100, 100));
        }
    } else if (year==2011) {
        auto problem = get_cec2011_problem(function_number);
        effective_dimension = problem.dimension;
        bounds = problem.bounds;
    } else bounds = std::vector<std::pair<double, double>>(effective_dimension, std::make_pair(-100.0, 100.0));

    if (year==2020) cecfunc = new minion::CEC2020Functions(function_number, effective_dimension);
    else if (year==2022) cecfunc = new minion::CEC2022Functions(function_number, effective_dimension);
    else if (year==2017) cecfunc = new minion::CEC2017Functions(function_number, effective_dimension);
    else if (year==2019) cecfunc = new minion::CEC2019Functions(function_number, effective_dimension);
    else if (year==2014) cecfunc = new minion::CEC2014Functions(function_number, effective_dimension);
    else if (year==2011) cecfunc = new minion::CEC2011Functions(function_number, effective_dimension);
    else throw std::runtime_error("Invalid year.");

    int popsize=population_size;

    auto settings = minion::DefaultSettings().getDefaultSettings(algo);
    settings["population_size"] = popsize;
    std::vector<std::vector<double>> x0={};
    if (algo == "NelderMead" || algo == "L_BFGS_B" || algo == "DA"){
        std::vector<double> x00;
        for (auto& el : bounds) x00.push_back(0.5*(el.first+el.second));
        x0 = {x00};
    };

    minion::Minimizer optimizer (objective_function,  bounds, x0, cecfunc, callBack, algo, 0.0, max_evals,  seed, settings);
    // Optimize and get the result
    minion::MinionResult result = optimizer();
    double ret = result.fun;

    // Output the results
    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "\tAlgo : "<< algo << ". Best Value: " << result.fun << "\n";
    std::cout << "\tReal Ncalls : " << cecfunc->Ncalls << "\n";

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


    std::vector<int> funcnums; 
    if (year==2017 || year == 2014) funcnums =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20,  21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    else if (year==2020 || year == 2019) funcnums = {1,2,3,4,5,6,7,8,9, 10}; 
    else if (year==2022) funcnums =  {1,2,3,4,5,6,7,8,9, 10, 11, 12}; 
    else if (year==2011) funcnums = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    else throw std::runtime_error("Year invalid.");

    std::vector<std::vector<double>> results;
    for (int i=0; i<numRuns; i++){
        std::cout << "========================\n";
        std::cout << "\nRun : "<< i+1 << "\n";
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> result_per_run;
        for (auto& num : funcnums) {
            try {
                result_per_run.push_back(minimize_cec_functions(num, dimension, popsize, Nmaxevals, year, algo, i));
            } catch (const std::exception& e) {
                continue;
            }
        };
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start);
        std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
        results.push_back(result_per_run);
        std::cout << "========================\n";
       
    };
    dumpResultsToFile(results, "results_"+std::to_string(year)+"_"+algo+"_" + std::to_string(dimension)+"_"+std::to_string(Nmaxevals)+".txt");

    return 0;
}
