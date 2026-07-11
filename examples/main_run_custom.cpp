#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "default_options.h"
#include "minimizer.h"
#include "minion_bench.h"
#include "utility.h"

namespace {

struct CustomRunConfig {
    std::string algo = "ARRDE";
    int dimension = 50;
    int max_evals = 500000;
    int num_functions = 20;
    int instance = 1;
    int num_hybrid = 10;
    int num_composition = 10;
    bool use_rotation = true;
};

std::vector<double> evaluate_benchmark_batch(
    const std::vector<std::vector<double>>& xs,
    void* data) {
    const auto* function = static_cast<const minion::FunctionBase*>(data);
    if (function == nullptr) {
        throw std::invalid_argument("benchmark function pointer must not be null");
    }
    return (*function)(xs);
}

CustomRunConfig parse_cli(int argc, char* argv[]) {
    CustomRunConfig config;
    if (argc > 1) config.algo = argv[1];
    if (argc > 2) config.dimension = std::max(1, std::atoi(argv[2]));
    if (argc > 3) config.max_evals = std::max(1, std::atoi(argv[3]));
    if (argc > 4) config.num_functions = std::max(1, std::atoi(argv[4]));
    if (argc > 5) config.instance = std::atoi(argv[5]);
    if (argc > 6) config.num_hybrid = std::max(0, std::atoi(argv[6]));
    if (argc > 7) config.num_composition = std::max(0, std::atoi(argv[7]));
    if (argc > 8) config.use_rotation = std::atoi(argv[8]) != 0;

    if (config.num_hybrid + config.num_composition != config.num_functions) {
        config.num_functions = config.num_hybrid + config.num_composition;
    }
    return config;
}

bool requires_initial_guesses(const std::string& algo) {
    const std::string canonical = minion::DefaultSettings::canonicalAlgoName(algo);
    return canonical == "NelderMead" || canonical == "DA" || canonical == "L_BFGS" || canonical == "L_BFGS_B";
}

} // namespace

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    try {
        std::cout << "Usage: run_custom [algo] [dimension] [maxevals] [num_functions] [instance] [num_hybrid] [num_composition] [use_rotation]" << std::endl;
        const CustomRunConfig config = parse_cli(argc, argv);

        std::vector<std::shared_ptr<minion::MinionBenchmark>> suite;
        suite.reserve(static_cast<std::size_t>(config.num_functions));
        const int basic_count = static_cast<int>(minion::list_basic_functions().size());
        for (int i = 0; i < config.num_hybrid; ++i) {
            suite.push_back(std::make_shared<minion::MinionBenchmark>(
                basic_count + 1 + i,
                config.dimension,
                config.num_hybrid,
                config.num_composition,
                config.instance,
                config.use_rotation));
        }
        for (int i = 0; i < config.num_composition; ++i) {
            suite.push_back(std::make_shared<minion::MinionBenchmark>(
                basic_count + config.num_hybrid + 1 + i,
                config.dimension,
                config.num_hybrid,
                config.num_composition,
                config.instance,
                config.use_rotation));
        }

        const auto bounds = suite.front()->bounds();
        std::vector<std::vector<double>> x0;
        if (requires_initial_guesses(config.algo)) {
            minion::set_global_seed(static_cast<unsigned int>(config.instance));
            x0 = minion::latin_hypercube_sampling(bounds, 4);
        }

        double total_fun = 0.0;
        double best_fun = std::numeric_limits<double>::infinity();
        double worst_fun = -std::numeric_limits<double>::infinity();

        std::cout << "Running " << suite.size()
                  << " generated MinionBenchmark functions with " << config.algo
                  << " in D=" << config.dimension
                  << ", maxevals=" << config.max_evals
                  << ", instance=" << config.instance
                  << ", hybrids=" << config.num_hybrid
                  << ", compositions=" << config.num_composition
                  << ", use_rotation=" << (config.use_rotation ? "1" : "0") << std::endl;

        for (std::size_t i = 0; i < suite.size(); ++i) {
            const auto& benchmark = suite[i];
            auto settings = minion::DefaultSettings().getDefaultSettings(config.algo);
            settings["population_size"] = 0;

            minion::Minimizer minimizer(
                evaluate_benchmark_batch,
                bounds,
                x0,
                benchmark.get(),
                nullptr,
                config.algo,
                static_cast<std::size_t>(config.max_evals),
                config.instance + static_cast<int>(i),
                settings);

            const minion::MinionResult result = minimizer.optimize();
            total_fun += result.fun;
            best_fun = std::min(best_fun, result.fun);
            worst_fun = std::max(worst_fun, result.fun);

            std::cout << std::setw(3) << benchmark->function_number() << "  "
                      << std::setw(18) << benchmark->function()->name << "  "
                      << "f=" << std::setw(14) << std::setprecision(8) << std::scientific << result.fun << "  "
                      << "nfev=" << std::setw(8) << result.nfev << "  "
                      << "ok=" << (result.success ? "1" : "0") << std::endl;
        }

        const double mean_fun = total_fun / static_cast<double>(suite.size());
        std::cout << "\nSummary" << std::endl;
        std::cout << "  mean f = " << std::setprecision(8) << std::scientific << mean_fun << std::endl;
        std::cout << "  best f = " << std::setprecision(8) << std::scientific << best_fun << std::endl;
        std::cout << "  worst f = " << std::setprecision(8) << std::scientific << worst_fun << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "run_custom failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "run_custom failed: unknown exception" << std::endl;
        return 1;
    }
}
