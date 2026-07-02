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
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <filesystem>
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

std::vector<int> parse_functions(int max_function) {
    std::vector<int> functions;
    for (int i = 1; i <= max_function; ++i) {
        functions.push_back(i);
    }
    return functions;
}

struct MinEvLogger {
    size_t step = 0;
    size_t evals = 0;
    size_t next_index = 0;
    double best = std::numeric_limits<double>::infinity();
    double optimum = 0.0;
    std::vector<double> samples;

    MinEvLogger(size_t dimension, size_t max_evals, double global_optimum) {
        step = std::max<size_t>(1, 10 * dimension);
        optimum = global_optimum;
        size_t count = max_evals / step + 1;
        samples.assign(count, std::numeric_limits<double>::infinity());
    }

    void update(const std::vector<double>& values) {
        for (double v : values) {
            if (v < best) {
                best = v;
            }
            ++evals;
            while (next_index < samples.size() && evals >= next_index * step) {
                samples[next_index] = best - optimum;
                ++next_index;
            }
        }
    }

    void finalize() {
        while (next_index < samples.size()) {
            samples[next_index] = best - optimum;
            ++next_index;
        }
    }
};

struct LogContext {
    minion::BBOB2009Problem* problem = nullptr;
    MinEvLogger* logger = nullptr;
};

std::vector<double> objective_function_logged(const std::vector<std::vector<double>>& candidates, void* data) {
    auto* ctx = static_cast<LogContext*>(data);
    auto values = ctx->problem->evaluateBatch(candidates);
    if (ctx->logger) {
        ctx->logger->update(values);
    }
    return values;
}

void dumpResultsToFile(const std::vector<std::vector<double>>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i + 1 < row.size()) {
                file << "\t";
            }
        }
        file << "\n";
    }
}

void dumpMatrixToFile(const std::vector<std::vector<double>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    file << std::scientific << std::setprecision(7);
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i + 1 < row.size()) {
                file << "\t";
            }
        }
        file << "\n";
    }
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
    std::string algo = "ARRDE";
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
        algo = argv[3];
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
    std::cout << "Algo     : " << algo << "\n";
    std::cout << "Popsize  : " << popsize << "\n";
    std::cout << "Year     : " << year << "\n";
    std::cout << "Budget   : " << maxevals << "\n";
    std::cout << "Threads  : " << Nthreads << "\n";
    std::cout << "Log min  : " << log_min_ev << "\n";
    std::cout << "\n";

    std::vector<std::vector<double>> results(static_cast<size_t>(numRuns));
    std::map<int, std::vector<std::vector<double>>> min_ev_logs;
    if (log_min_ev) {
        for (int func : functions) {
            size_t step = std::max<size_t>(1, 10 * static_cast<size_t>(dimension));
            size_t count = static_cast<size_t>(maxevals) / step + 1;
            min_ev_logs.emplace(func, std::vector<std::vector<double>>(
                                          count, std::vector<double>(static_cast<size_t>(numRuns), std::numeric_limits<double>::infinity())));
        }
    }
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

                for (size_t func_idx = 0; func_idx < functions.size(); ++func_idx) {
                    const int function_number = functions[func_idx];
                    if (g_stop_requested) {
                        break;
                    }
                    try {
                        minion::BBOB2009Problem problem(function_number, dimension, year);
                        const auto function_start = std::chrono::high_resolution_clock::now();
                        std::unique_ptr<MinEvLogger> logger;
                        if (log_min_ev) {
                            logger = std::make_unique<MinEvLogger>(
                                static_cast<size_t>(dimension),
                                static_cast<size_t>(maxevals),
                                problem.bestValue());
                        }
                        {
                            std::lock_guard<std::mutex> lock(coutMutex);
                            std::cout << "F" << function_number << "\n";
                            std::cout << "  name : " << problem.name() << "\n";
                        }

                        LogContext ctx{&problem, logger ? logger.get() : nullptr};
                        minion::MinionFunction objective;
                        void* objective_data = nullptr;
                        if (logger) {
                            objective = objective_function_logged;
                            objective_data = &ctx;
                        } else {
                            objective = [&problem](const std::vector<std::vector<double>>& candidates, void*) {
                                return problem.evaluateBatch(candidates);
                            };
                            objective_data = nullptr;
                        }

                        const std::vector<std::pair<double, double>>& bounds = problem.bounds();
                        const std::vector<std::vector<double>> x0 = {problem.initialSolution()};
                        std::map<std::string, minion::ConfigValue> options;
                        options["population_size"] = popsize;
                        options["rel_initial_step"] = 0.3;
                        options["bound_strategy"] = std::string("reflect-random");

                        minion::Minimizer optimizer(
                            objective,
                            bounds,
                            x0,
                            objective_data,
                            nullptr,
                            algo,
                            static_cast<size_t>(maxevals),
                            run,
                            options);
                        const minion::MinionResult result = optimizer.optimize();
                        const double error = result.fun - problem.bestValue();

                        {
                            std::lock_guard<std::mutex> lock(coutMutex);
                            std::cout << "  nfev = " << result.nfev << "\n";
                            std::cout << "  err  = " << error << "\n";
                        }

                        result_per_run.push_back(result.fun);
                        if (log_min_ev && logger) {
                            auto it = min_ev_logs.find(function_number);
                            if (it != min_ev_logs.end()) {
                                auto& matrix = it->second;
                                for (size_t row = 0; row < logger->samples.size(); ++row) {
                                    matrix[row][static_cast<size_t>(run)] = logger->samples[row];
                                }
                            }
                            logger->finalize();
                        }

                        const auto function_end = std::chrono::high_resolution_clock::now();
                        const std::chrono::duration<double> function_elapsed = function_end - function_start;
                        {
                            std::lock_guard<std::mutex> lock(coutMutex);
                            std::cout << "  time = " << std::fixed << std::setprecision(3)
                                      << function_elapsed.count() << "s\n\n";
                        }

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

    dumpResultsToFile(results, "results_" + std::to_string(year) + "_" + algo + "_" +
                                 std::to_string(dimension) + "_" + std::to_string(maxevals) + ".txt");
    if (log_min_ev) {
        const std::string out_dir = algo;
        std::filesystem::create_directories(out_dir);
        for (const auto& entry : min_ev_logs) {
            const int func = entry.first;
            const auto& matrix = entry.second;
            dumpMatrixToFile(matrix, out_dir + "/" + algo + "_F" + std::to_string(func) + "_Min_EV.txt");
        }
    }

    return 0;
}
