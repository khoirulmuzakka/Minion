#include <iostream>
#include <vector>
#include <chrono>
#include <array>
#include <unordered_map>
#include "minion.h"
#include "utility.h"
#include "cec2011.h"
#include <fstream>
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::array<double, 126> kProblem9UpperBoundsExample = {
   0.217, 0.024, 0.076, 0.892, 0.128, 0.25, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.175, 0.032, 0.087, 0.035, 0.024, 0.106,
    0.217, 0.024, 0.026, 0.491, 0.228, 0.3, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.175, 0.032, 0.087, 0.035, 0.024, 0.106,
    0.216, 0.024, 0.076, 0.216, 0.216, 0.216, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.175, 0.032, 0.087, 0.035, 0.024, 0.081,
    0.217, 0.024, 0.076, 0.228, 0.228, 0.228, 0.058, 0.112, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.025, 0.032, 0.087, 0.035, 0.024, 0.081,
    0.124, 0.024, 0.076, 0.124, 0.124, 0.124, 0.058, 0.112, 0.062, 0.082, 0.035, 0.065, 0.032, 0.095, 0.022, 0.124, 0.032, 0.087, 0.035, 0.024, 0.106,
    0.116, 0.024, 0.076, 0.116, 0.116, 0.116, 0.058, 0.087, 0.062, 0.082, 0.035, 0.09, 0.032, 0.095, 0.022, 0.116, 0.032, 0.087, 0.035, 0.024, 0.106
};

static constexpr double kELD6Pmin[6] = {100.000000, 50.000000, 80.000000, 50.000000, 50.000000, 50.000000};
static constexpr double kELD6Pmax[6] = {500.000000, 200.000000, 300.000000, 150.000000, 200.000000, 120.000000};

static constexpr double kELD13Pmin[13] = {0.000000, 0.000000, 0.000000, 60.000000, 60.000000, 60.000000, 60.000000, 60.000000, 60.000000, 40.000000, 40.000000, 55.000000, 55.000000};
static constexpr double kELD13Pmax[13] = {680.000000, 360.000000, 360.000000, 180.000000, 180.000000, 180.000000, 180.000000, 180.000000, 180.000000, 120.000000, 120.000000, 120.000000, 120.000000};

static constexpr double kELD15Pmin[15] = {150.000000, 150.000000, 20.000000, 20.000000, 150.000000, 135.000000, 135.000000, 60.000000, 25.000000, 25.000000, 20.000000, 20.000000, 25.000000, 15.000000, 15.000000};
static constexpr double kELD15Pmax[15] = {455.000000, 455.000000, 130.000000, 130.000000, 470.000000, 460.000000, 465.000000, 300.000000, 162.000000, 160.000000, 80.000000, 80.000000, 85.000000, 55.000000, 55.000000};

static constexpr double kELD40Pmin[40] = {36.000000, 36.000000, 60.000000, 80.000000, 47.000000, 68.000000, 110.000000, 135.000000, 135.000000, 130.000000, 94.000000, 94.000000, 125.000000, 125.000000, 125.000000, 125.000000, 220.000000, 220.000000, 242.000000, 242.000000, 254.000000, 254.000000, 254.000000, 254.000000, 254.000000, 254.000000, 10.000000, 10.000000, 10.000000, 47.000000, 60.000000, 60.000000, 60.000000, 90.000000, 90.000000, 90.000000, 25.000000, 25.000000, 25.000000, 242.000000};
static constexpr double kELD40Pmax[40] = {114.000000, 114.000000, 120.000000, 190.000000, 97.000000, 140.000000, 300.000000, 300.000000, 300.000000, 300.000000, 375.000000, 375.000000, 500.000000, 500.000000, 500.000000, 500.000000, 500.000000, 500.000000, 550.000000, 550.000000, 550.000000, 550.000000, 550.000000, 550.000000, 550.000000, 550.000000, 150.000000, 150.000000, 150.000000, 97.000000, 190.000000, 190.000000, 190.000000, 200.000000, 200.000000, 200.000000, 110.000000, 110.000000, 110.000000, 550.000000};

static constexpr double kELD140Pmin[140] = {71.000000, 120.000000, 125.000000, 125.000000, 90.000000, 90.000000, 280.000000, 280.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 280.000000, 280.000000, 280.000000, 280.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 260.000000, 120.000000, 120.000000, 423.000000, 423.000000, 3.000000, 3.000000, 160.000000, 160.000000, 160.000000, 160.000000, 160.000000, 160.000000, 160.000000, 160.000000, 165.000000, 165.000000, 165.000000, 165.000000, 180.000000, 180.000000, 103.000000, 198.000000, 100.000000, 153.000000, 163.000000, 95.000000, 160.000000, 160.000000, 196.000000, 196.000000, 196.000000, 196.000000, 130.000000, 130.000000, 137.000000, 137.000000, 195.000000, 175.000000, 175.000000, 175.000000, 175.000000, 330.000000, 160.000000, 160.000000, 200.000000, 56.000000, 115.000000, 115.000000, 115.000000, 207.000000, 207.000000, 175.000000, 175.000000, 175.000000, 175.000000, 360.000000, 415.000000, 795.000000, 795.000000, 578.000000, 615.000000, 612.000000, 612.000000, 758.000000, 755.000000, 750.000000, 750.000000, 713.000000, 718.000000, 791.000000, 786.000000, 795.000000, 795.000000, 795.000000, 795.000000, 94.000000, 94.000000, 94.000000, 244.000000, 244.000000, 244.000000, 95.000000, 95.000000, 116.000000, 175.000000, 2.000000, 4.000000, 15.000000, 9.000000, 12.000000, 10.000000, 112.000000, 4.000000, 5.000000, 5.000000, 50.000000, 5.000000, 42.000000, 42.000000, 41.000000, 17.000000, 7.000000, 7.000000, 26.000000};
static constexpr double kELD140Pmax[140] = {119.000000, 189.000000, 190.000000, 190.000000, 190.000000, 190.000000, 490.000000, 490.000000, 496.000000, 496.000000, 496.000000, 496.000000, 506.000000, 509.000000, 506.000000, 505.000000, 506.000000, 506.000000, 505.000000, 505.000000, 505.000000, 505.000000, 505.000000, 505.000000, 537.000000, 537.000000, 549.000000, 549.000000, 501.000000, 501.000000, 506.000000, 506.000000, 506.000000, 506.000000, 500.000000, 500.000000, 241.000000, 241.000000, 774.000000, 769.000000, 19.000000, 28.000000, 250.000000, 250.000000, 250.000000, 250.000000, 250.000000, 250.000000, 250.000000, 250.000000, 504.000000, 504.000000, 504.000000, 504.000000, 471.000000, 561.000000, 341.000000, 617.000000, 312.000000, 471.000000, 500.000000, 302.000000, 511.000000, 511.000000, 490.000000, 490.000000, 490.000000, 490.000000, 432.000000, 432.000000, 455.000000, 455.000000, 541.000000, 536.000000, 540.000000, 538.000000, 540.000000, 574.000000, 531.000000, 531.000000, 542.000000, 132.000000, 245.000000, 245.000000, 245.000000, 307.000000, 307.000000, 345.000000, 345.000000, 345.000000, 345.000000, 580.000000, 645.000000, 984.000000, 978.000000, 682.000000, 720.000000, 718.000000, 720.000000, 964.000000, 958.000000, 1007.000000, 1006.000000, 1013.000000, 1020.000000, 954.000000, 952.000000, 1006.000000, 1013.000000, 1021.000000, 1015.000000, 203.000000, 203.000000, 203.000000, 379.000000, 379.000000, 379.000000, 190.000000, 189.000000, 194.000000, 321.000000, 19.000000, 59.000000, 83.000000, 53.000000, 37.000000, 34.000000, 373.000000, 20.000000, 38.000000, 19.000000, 98.000000, 10.000000, 74.000000, 74.000000, 105.000000, 51.000000, 19.000000, 19.000000, 40.000000};

constexpr std::array<double, 26> kMessengerLB = {1900.0,  2.5,  0.0,  0.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                                                 100.0,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.1,   1.1,
                                                 1.05,   1.05, 1.05, -kPi, -kPi, -kPi, -kPi, -kPi};
constexpr std::array<double, 26> kMessengerUB = {2300.0, 4.05, 1.0,  1.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                                                 600.0,  0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 6.0,   6.0,
                                                 6.0,    6.0,  6.0,  kPi,  kPi,  kPi,  kPi,  kPi};
constexpr std::array<double, 22> kCassiniLB = {-1000.0, 3.0,  0.0,  0.0, 100.0, 100.0, 30.0,  400.0, 800.0, 0.01, 0.01,
                                               0.01,    0.01, 0.01, 1.05, 1.05,  1.15,  1.7,  -kPi,  -kPi, -kPi, -kPi};
constexpr std::array<double, 22> kCassiniUB = {0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9,
                                               0.9, 0.9, 0.9, 6.0, 6.0, 6.5, 291.0, kPi, kPi, kPi, kPi};

std::vector<std::pair<double, double>> make_bounds(const std::vector<double>& lb,
                                                   const std::vector<double>& ub) {
    std::vector<std::pair<double, double>> bounds;
    bounds.reserve(lb.size());
    for (size_t i = 0; i < lb.size(); ++i) {
        bounds.emplace_back(lb[i], ub[i]);
    }
    return bounds;
}

std::vector<std::pair<double, double>> uniform_bounds(int dim, double lb, double ub) {
    return std::vector<std::pair<double, double>>(dim, std::make_pair(lb, ub));
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


const CEC2011ProblemDef &get_cec2011_problem(int function_number) {
    static const std::unordered_map<int, CEC2011ProblemDef> problems = [] {
        std::unordered_map<int, CEC2011ProblemDef> map;
        auto add = [&map](int id, int dim, std::vector<std::pair<double, double>> bounds) {
            map.emplace(id, CEC2011ProblemDef{dim, std::move(bounds)});
        };

        add(1, 6, uniform_bounds(6, -6.4, 6.35));
        add(2, 30, bounds_problem02(30));
        add(3, 1, {{-0.6, 0.9}});
        add(4, 1, {{0.0, 5.0}});
        add(5, 30, bounds_problem05(30));
        add(6, 30, bounds_problem06(30));
        add(7, 20, uniform_bounds(20, 0.0, 2.0 * kPi));
        add(8, 7, uniform_bounds(7, 0.0, 15.0));

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(kProblem9UpperBoundsExample.size());
            for (double ub : kProblem9UpperBoundsExample) {
                bounds.emplace_back(0.0, ub);
            }
            add(9, static_cast<int>(kProblem9UpperBoundsExample.size()), std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            for (int i = 0; i < 6; ++i) {
                bounds.emplace_back(0.2, 1.0);
            }
            for (int i = 0; i < 6; ++i) {
                bounds.emplace_back(-180.0, 180.0);
            }
            add(10, 12, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(24 * 5);
            const std::array<double, 5> lbs = {10, 20, 30, 40, 50};
            const std::array<double, 5> ubs = {75, 125, 175, 250, 300};
            for (int hour = 0; hour < 24; ++hour) {
                for (int u = 0; u < 5; ++u) {
                    bounds.emplace_back(lbs[u], ubs[u]);
                }
            }
            add(11, 120, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(24 * 10);
            const std::array<double, 10> lbs = {150, 135, 73, 60, 73, 57, 20, 47, 20, 55};
            const std::array<double, 10> ubs = {470, 460, 340, 300, 243, 160, 130, 120, 80, 55.1};
            for (int hour = 0; hour < 24; ++hour) {
                for (int u = 0; u < 10; ++u) {
                    bounds.emplace_back(lbs[u], ubs[u]);
                }
            }
            add(12, 240, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(6);
            for (size_t i = 0; i < 6; ++i) {
                bounds.emplace_back(kELD6Pmin[i], kELD6Pmax[i]);
            }
            add(13, 6, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(13);
            for (size_t i = 0; i < 13; ++i) {
                bounds.emplace_back(kELD13Pmin[i], kELD13Pmax[i]);
            }
            add(14, 13, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(15);
            for (size_t i = 0; i < 15; ++i) {
                bounds.emplace_back(kELD15Pmin[i], kELD15Pmax[i]);
            }
            add(15, 15, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(40);
            for (size_t i = 0; i < 40; ++i) {
                bounds.emplace_back(kELD40Pmin[i], kELD40Pmax[i]);
            }
            add(16, 40, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(140);
            for (size_t i = 0; i < 140; ++i) {
                bounds.emplace_back(kELD140Pmin[i], kELD140Pmax[i]);
            }
            add(17, 140, std::move(bounds));
        }

        auto hydro_bounds = [] {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(24 * 4);
            const std::array<double, 4> qmin = {5.0, 6.0, 10.0, 13.0};
            const std::array<double, 4> qmax = {15.0, 15.0, 30.0, 25.0};
            for (int hour = 0; hour < 24; ++hour) {
                for (int u = 0; u < 4; ++u) {
                    bounds.emplace_back(qmin[u], qmax[u]);
                }
            }
            return bounds;
        };

        add(18, 96, hydro_bounds());
        add(19, 96, hydro_bounds());
        add(20, 96, hydro_bounds());

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(26);
            for (size_t i = 0; i < kMessengerLB.size(); ++i) {
                bounds.emplace_back(kMessengerLB[i], kMessengerUB[i]);
            }
            add(21, 26, std::move(bounds));
        }

        {
            std::vector<std::pair<double, double>> bounds;
            bounds.reserve(22);
            for (size_t i = 0; i < kCassiniLB.size(); ++i) {
                bounds.emplace_back(kCassiniLB[i], kCassiniUB[i]);
            }
            add(22, 22, std::move(bounds));
        }

        return map;
    }();

    auto it = problems.find(function_number);
    if (it == problems.end()) {
        throw std::runtime_error("CEC2011 problem definition not available.");
    }
    return it->second;
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
        const auto &problem = get_cec2011_problem(function_number);
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
    else if (year==2011) funcnums = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};
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
