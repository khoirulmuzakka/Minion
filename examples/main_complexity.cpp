#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "minion.h"
#include "minion_cec.h"
#include "utility.h"

namespace {

constexpr size_t kNumProblems = 29;
constexpr size_t kMaxFunction = 30;
constexpr size_t kExcludedFunction = 2;
constexpr size_t kEvalCount = 10000;

std::vector<std::pair<double, double>> uniform_bounds(int dim, double lb, double ub) {
    return std::vector<std::pair<double, double>>(dim, std::make_pair(lb, ub));
}

std::vector<std::vector<double>> random_samples(const std::vector<std::pair<double, double>>& bounds,
                                                size_t count,
                                                std::mt19937& rng) {
    std::vector<std::vector<double>> samples(count, std::vector<double>(bounds.size(), 0.0));
    std::vector<std::uniform_real_distribution<double>> distributions;
    distributions.reserve(bounds.size());
    for (const auto& bound : bounds) {
        distributions.emplace_back(bound.first, bound.second);
    }

    for (size_t i = 0; i < count; ++i) {
        for (size_t d = 0; d < bounds.size(); ++d) {
            samples[i][d] = distributions[d](rng);
        }
    }
    return samples;
}

minion::CECBase* create_cec_function(int year, int function_number, int dimension) {
    if (year == 2017) {
        return new minion::CEC2017Functions(function_number, dimension);
    }
    if (year == 2014) {
        return new minion::CEC2014Functions(function_number, dimension);
    }
    throw std::runtime_error("main_complexity supports CEC2014 and CEC2017 only.");
}

std::vector<double> objective_function(const std::vector<std::vector<double>>& x, void* data) {
    minion::CECBase* func = static_cast<minion::CECBase*>(data);
    return func->operator()(x);
}

void callback(minion::MinionResult*) {}

} // namespace

int main(int argc, char* argv[]) {
    int dimension = 10;
    int year = 2017;
    std::string algo = "ARRDE";
    int popsize = 0;
    int seed = -1;

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
    if (argc > 5) {
        seed = std::atoi(argv[5]);
    }

    if (year != 2014 && year != 2017) {
        std::cerr << "main_complexity supports only CEC2014 or CEC2017.\n";
        return 1;
    }

    const std::vector<std::pair<double, double>> bounds = uniform_bounds(dimension, -100.0, 100.0);
    double t1_sum = 0.0;
    double t2_sum = 0.0;

    size_t counted = 0;
    for (size_t func = 1; func <= kMaxFunction; ++func) {
        if (func == kExcludedFunction) {
            continue;
        }
        {
            std::mt19937 rng(12345u + static_cast<unsigned>(func));
            std::vector<std::vector<double>> samples = random_samples(bounds, kEvalCount, rng);
            minion::CECBase* cecfunc = create_cec_function(year, static_cast<int>(func), dimension);
            auto start = std::chrono::high_resolution_clock::now();
            (void)cecfunc->operator()(samples);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            t1_sum += elapsed.count();
            delete cecfunc;
        }

        {
            minion::CECBase* cecfunc = create_cec_function(year, static_cast<int>(func), dimension);
            auto settings = minion::DefaultSettings().getDefaultSettings(algo);
            settings["population_size"] = popsize;
            std::vector<std::vector<double>> x0 = {};
            if (algo == "NelderMead" || algo == "L_BFGS_B" || algo == "DA") {
                std::vector<double> x00;
                x00.reserve(bounds.size());
                for (const auto& bound : bounds) {
                    x00.push_back(0.5 * (bound.first + bound.second));
                }
                x0 = {x00};
            }

            minion::Minimizer optimizer(
                objective_function,
                bounds,
                x0,
                cecfunc,
                callback,
                algo,
                0.0,
                static_cast<int>(kEvalCount),
                seed,
                settings);

            auto start = std::chrono::high_resolution_clock::now();
            (void)optimizer();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            t2_sum += elapsed.count();
            delete cecfunc;
        }

        ++counted;
    }

    if (counted != kNumProblems) {
        std::cerr << "Expected to evaluate " << kNumProblems
                  << " functions, but processed " << counted << ".\n";
        return 1;
    }

    const double t1 = t1_sum / static_cast<double>(kNumProblems);
    const double t2 = t2_sum / static_cast<double>(kNumProblems);
    const double complexity = (t2 - t1) / t1;

    std::cout << "T1: " << t1 << "\n";
    std::cout << "T2: " << t2 << "\n";
    std::cout << "Complexity (T2 - T1) / T1: " << complexity << "\n";
    return 0;
}
