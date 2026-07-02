#include "minion.h"
#include "bbob2009.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void handle_sigint(int) {
    g_stop_requested = 1;
}

bool is_supported_bbob_dimension(int dimension) {
    return dimension == 2 || dimension == 3 || dimension == 5 || dimension == 10 || dimension == 20 || dimension == 40;
}

std::vector<std::string> parse_algorithms(const std::string& csv) {
    if (csv.empty()) {
        return {"CMAES", "ACMAES", "RCMAES", "BIPOP_aCMAES"};
    }

    std::vector<std::string> algos;
    size_t start = 0;
    while (start < csv.size()) {
        const size_t end = csv.find(',', start);
        const std::string item = csv.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) {
            algos.push_back(item);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return algos;
}

std::vector<int> parse_functions(int max_function) {
    std::vector<int> functions;
    for (int i = 1; i <= max_function; ++i) {
        functions.push_back(i);
    }
    return functions;
}

minion::MinionResult run_algorithm(
const minion::BBOB2009Problem& problem,
    const std::string& algo,
    int maxevals,
    int seed,
    double sigma0,
    int population_size) {
    const std::vector<std::pair<double, double>>& bounds = problem.bounds();
    const std::vector<std::vector<double>> x0 = {problem.initialSolution()};

    minion::MinionFunction objective = [&problem](const std::vector<std::vector<double>>& candidates, void*) {
        return problem.evaluateBatch(candidates);
    };

    std::map<std::string, minion::ConfigValue> options;
    options["population_size"] = population_size;
    options["rel_initial_step"] = sigma0;
    options["bound_strategy"] = std::string("reflect-random");

    minion::Minimizer optimizer(
        objective,
        bounds,
        x0,
        nullptr,
        nullptr,
        algo,
        static_cast<size_t>(maxevals),
        seed,
        options);
    return optimizer.optimize();
}

}  // namespace

int main(int argc, char* argv[]) {
    int numRuns = 1;
    int dimension = 10;
    std::string algo_csv = "CMAES,ACMAES,RCMAES,BIPOP_aCMAES";
    int popsize = 0;
    int year = 2018;
    int maxevals = static_cast<int>(1e4 * dimension);
    int Nthreads = 1;
    int log_min_ev = 0;

    if (argc > 1) {
        numRuns = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        dimension = std::max(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
        algo_csv = argv[3];
    }
    if (argc > 4) {
        popsize = std::atoi(argv[4]);
    }
    if (argc > 5) {
        year = std::atoi(argv[5]);
    }
    if (argc > 6) {
        maxevals = std::max(1, std::atoi(argv[6]));
    }
    if (argc > 7) {
        Nthreads = std::max(1, std::atoi(argv[7]));
    }
    if (argc > 8) {
        log_min_ev = std::atoi(argv[8]);
    }

    const std::vector<std::string> algorithms = parse_algorithms(algo_csv);
    const std::vector<int> functions = parse_functions(24);

    std::signal(SIGINT, handle_sigint);

    if (!is_supported_bbob_dimension(dimension)) {
        throw std::runtime_error("BBOB2009 only supports dimensions 2, 3, 5, 10, 20, and 40.");
    }

    Nthreads = std::max(1, Nthreads);
    if (numRuns > 0) {
        Nthreads = std::min(Nthreads, numRuns);
    }

    std::cout << "BBOB2009 single-objective comparison\n";
    std::cout << "Runs     : " << numRuns << "\n";
    std::cout << "Dimension: " << dimension << "\n";
    std::cout << "Algo     : " << (algo_csv.empty() ? "CMAES, ACMAES, RCMAES, BIPOP_aCMAES" : algo_csv) << "\n";
    std::cout << "Popsize  : " << popsize << "\n";
    std::cout << "Year     : " << year << "\n";
    std::cout << "Budget   : " << maxevals << "\n";
    std::cout << "Threads  : " << Nthreads << "\n";
    std::cout << "Log min  : " << log_min_ev << "\n";
    std::cout << "Algos    : ";
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << algorithms[i];
    }
    std::cout << "\n\n";

    std::vector<std::vector<double>> results(static_cast<size_t>(numRuns));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(Nthreads));
    std::atomic<int> nextRun{0};
    std::mutex coutMutex;

    for (int t = 0; t < Nthreads; ++t) {
        workers.emplace_back([&]() {
            while (!g_stop_requested) {
                const int run = nextRun.fetch_add(1, std::memory_order_relaxed);
                if (run >= numRuns) {
                    break;
                }

                {
                    std::lock_guard<std::mutex> lock(coutMutex);
                    std::cout << "========================\n";
                    std::cout << "Run : " << (run + 1) << "\n";
                }

                std::vector<double> result_per_run;
                result_per_run.reserve(functions.size());

                if (g_stop_requested) {
                    break;
                }
                const auto run_start = std::chrono::high_resolution_clock::now();

                for (int function_number : functions) {
                    if (g_stop_requested) {
                        break;
                    }
                    try {
                        minion::BBOB2009Problem problem(function_number, dimension, year);
                        const auto function_start = std::chrono::high_resolution_clock::now();
                        {
                            std::lock_guard<std::mutex> lock(coutMutex);
                            std::cout << "F" << function_number << "\n";
                            std::cout << "  name : " << problem.name() << "\n";
                        }

                        for (const std::string& algo : algorithms) {
                            const minion::MinionResult result = run_algorithm(problem, algo, maxevals, run, 0.3, popsize);
                            const double error = result.fun - problem.bestValue();

                            {
                                std::lock_guard<std::mutex> lock(coutMutex);
                                std::cout << "  nfev = " << result.nfev << "\n";
                                std::cout << "  err  = " << error << "\n";
                            }
                        }

                        const auto function_end = std::chrono::high_resolution_clock::now();
                        const std::chrono::duration<double> function_elapsed = function_end - function_start;
                        {
                            std::lock_guard<std::mutex> lock(coutMutex);
                            std::cout << "  time = " << std::fixed << std::setprecision(3)
                                      << function_elapsed.count() << "s\n\n";
                        }

                        result_per_run.push_back(function_elapsed.count());
                    } catch (const std::exception& ex) {
                        std::lock_guard<std::mutex> lock(coutMutex);
                        std::cout << "Function " << function_number << " failed: " << ex.what() << "\n\n";
                    }
                }

                const auto run_end = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double> run_elapsed = run_end - run_start;
                {
                    std::lock_guard<std::mutex> lock(coutMutex);
                    std::cout << "Run elapsed time: " << run_elapsed.count() << " seconds\n";
                    if (g_stop_requested) {
                        std::cout << "Interrupted by Ctrl+C, stopping cleanly.\n";
                    }
                }

                results[static_cast<size_t>(run)] = std::move(result_per_run);
                if (g_stop_requested) {
                    break;
                }
            }
        });
    }

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    return 0;
}
