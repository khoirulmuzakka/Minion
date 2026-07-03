#include <cstdint>
#include <memory>

#define main main_bbob_multi_minion
#include "main_bbob_multi.cpp"
#undef main

#ifdef CMAES_H
#undef CMAES_H
#endif
#include <libcmaes/cmaes.h>

namespace {

std::vector<std::pair<double, double>> normalized_bounds_for(size_t dimension) {
    return std::vector<std::pair<double, double>>(dimension, std::make_pair(-1.0, 1.0));
}

std::vector<double> normalize_point(
    const std::vector<double>& point,
    const std::vector<std::pair<double, double>>& bounds) {
    if (bounds.empty()) {
        return point;
    }

    std::vector<double> normalized(point.size(), 0.0);
    for (size_t i = 0; i < point.size(); ++i) {
        const double low = bounds[i].first;
        const double high = bounds[i].second;
        const double range = high - low;
        normalized[i] = range > 0.0 ? (2.0 * (point[i] - low) / range - 1.0) : 0.0;
    }
    return normalized;
}

std::vector<double> denormalize_point(
    const std::vector<double>& point,
    const std::vector<std::pair<double, double>>& bounds) {
    if (bounds.empty()) {
        return point;
    }

    std::vector<double> actual(point.size(), 0.0);
    for (size_t i = 0; i < point.size(); ++i) {
        const double low = bounds[i].first;
        const double high = bounds[i].second;
        const double range = high - low;
        actual[i] = range > 0.0 ? (low + 0.5 * (point[i] + 1.0) * range) : low;
    }
    return actual;
}

libcmaes::CMASolutions run_libcmaes_optimizer(
    const minion::BBOB2009Problem& problem,
    int population_size,
    int max_evals,
    const std::string& algo,
    int seed,
    MinEvLogger* logger) {
    const std::vector<std::pair<double, double>>& bounds = problem.bounds();
    const std::vector<std::pair<double, double>> normalized_bounds = normalized_bounds_for(bounds.size());
    const std::vector<double> x0 = normalize_point(problem.initialSolution(), bounds);

    using GenoPhenoT = libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;
    std::vector<double> lbounds;
    std::vector<double> ubounds;
    lbounds.reserve(normalized_bounds.size());
    ubounds.reserve(normalized_bounds.size());
    for (const auto& bound : normalized_bounds) {
        lbounds.push_back(bound.first);
        ubounds.push_back(bound.second);
    }

    GenoPhenoT gp(lbounds.data(), ubounds.data(), static_cast<int>(bounds.size()));
    libcmaes::CMAParameters<GenoPhenoT> params(
        x0,
        0.3,
        population_size > 0 ? population_size : -1,
        static_cast<std::uint64_t>(seed),
        gp);
    params.set_str_algo(algo);
    params.set_max_fevals(max_evals);
    params.set_quiet(true);

    libcmaes::FitFunc fitfunc = [&problem, logger, bounds](const double* x, const int& n) -> double {
        std::vector<double> normalized_candidate(x, x + n);
        const std::vector<double> candidate = denormalize_point(normalized_candidate, bounds);
        const double value = problem.evaluate(candidate);
        if (logger) {
            logger->update(std::vector<double>{value});
        }
        return value;
    };

    return libcmaes::cmaes(fitfunc, params);
}

} // namespace

double minimize_bbob_functions(int function_number,
                               const minion::BBOB2009Problem& problem,
                               int population_size,
                               int max_evals,
                               std::string algo = "acmaes",
                               int seed = -1,
                               MinEvLogger* logger = nullptr) {
    auto start = std::chrono::high_resolution_clock::now();
    if (logger) {
        logger->best = std::numeric_limits<double>::infinity();
        logger->evals = 0;
        logger->next_index = 0;
    }

    auto solutions = run_libcmaes_optimizer(
        problem,
        population_size,
        max_evals,
        algo,
        seed,
        logger);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (end - start);

    const auto best = solutions.best_candidate();
    const double ret = best.get_fvalue();
    const double error = ret - problem.bestValue();

    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "\tAlgo : libcmaes_" << algo << ". Best Value: " << ret << "\n";
    std::cout << "\tBest Error : " << error << "\n";
    std::cout << "\tNevals : " << solutions.nevals() << "\n";
    std::cout << "\tElapsed time: " << duration.count() << " seconds\n";

    if (logger) {
        logger->finalize();
    }
    return ret;
}

int main(int argc, char* argv[]) {
    int numRuns = 1;
    int dimension = 10;
    std::string algo = "acmaes";
    int popsize = 0;
    int year = 2009;
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
    std::cout << "Algo     : libcmaes_" << algo << "\n";
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

                        const double fval = minimize_bbob_functions(
                            function_number,
                            problem,
                            popsize,
                            maxevals,
                            algo,
                            run,
                            logger ? logger.get() : nullptr);
                        result_per_run.push_back(fval);

                        if (log_min_ev && logger) {
                            auto it = min_ev_logs.find(function_number);
                            if (it != min_ev_logs.end()) {
                                auto& matrix = it->second;
                                for (size_t row = 0; row < logger->samples.size(); ++row) {
                                    matrix[row][static_cast<size_t>(run)] = logger->samples[row];
                                }
                            }
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

    const std::string dump_algo = "libcmaes_" + algo;
    dumpResultsToFile(results, "results_" + std::to_string(year) + "_" + dump_algo + "_" +
                                 std::to_string(dimension) + "_" + std::to_string(maxevals) + ".txt");
    if (log_min_ev) {
        const std::string out_dir = dump_algo;
        std::filesystem::create_directories(out_dir);
        for (const auto& entry : min_ev_logs) {
            const int func = entry.first;
            const auto& matrix = entry.second;
            dumpMatrixToFile(matrix, out_dir + "/" + dump_algo + "_F" + std::to_string(func) + "_Min_EV.txt");
        }
    }

    return 0;
}
