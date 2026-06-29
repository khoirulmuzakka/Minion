#define minimize_cec_functions minimize_cec_functions_minion
#define main main_cec_multi_minion
#include "main_cec_multi.cpp"
#undef main
#undef minimize_cec_functions

#ifdef CMAES_H
#undef CMAES_H
#endif
#include <libcmaes/cmaes.h>

namespace {

libcmaes::CMASolutions run_libcmaes_optimizer(
    minion::CECBase* cecfunc,
    int effective_dimension,
    const std::vector<std::pair<double, double>>& bounds,
    int population_size,
    int max_evals,
    const std::string& algo,
    int seed,
    double sigma0,
    MinEvLogger* logger) {
    std::vector<double> x0;
    x0.reserve(bounds.size());
    for (const auto& bound : bounds) {
        x0.push_back(0.5 * (bound.first + bound.second));
    }

    std::vector<double> lbounds;
    std::vector<double> ubounds;
    lbounds.reserve(bounds.size());
    ubounds.reserve(bounds.size());
    for (const auto& bound : bounds) {
        lbounds.push_back(bound.first);
        ubounds.push_back(bound.second);
    }

    using GenoPhenoT = libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;
    GenoPhenoT gp(lbounds.data(), ubounds.data(), effective_dimension);
    libcmaes::CMAParameters<GenoPhenoT> params(
        x0,
        sigma0 > 0.0 ? sigma0 : 0.3,
        population_size > 0 ? population_size : -1,
        static_cast<std::uint64_t>(seed),
        gp);
    params.set_str_algo(algo);
    params.set_max_fevals(max_evals);
    params.set_quiet(true);

    libcmaes::FitFunc fitfunc = [cecfunc, logger](const double* x, const int& n) -> double {
        std::vector<std::vector<double>> candidate(1, std::vector<double>(x, x + n));
        auto values = cecfunc->operator()(candidate);
        if (logger) {
            logger->update(values);
        }
        return values.empty() ? std::numeric_limits<double>::infinity() : values.front();
    };

    return libcmaes::cmaes(fitfunc, params);
}

} // namespace

double minimize_cec_functions(int function_number,
                              int dimension,
                              int population_size,
                              int max_evals,
                              int year = 2022,
                              std::string algo = "acmaes",
                              int seed = -1,
                              MinEvLogger* logger = nullptr) {
    minion::CECBase* cecfunc;
    std::vector<std::pair<double, double>> bounds;
    int effective_dimension = get_effective_dimension(function_number, dimension, year);
    if (year == 2019) {
        for (int i = 0; i < effective_dimension; i++) {
            if (function_number == 1) bounds.push_back(std::make_pair(-8192, 8192));
            else if (function_number == 2) bounds.push_back(std::make_pair(-16384, 16384));
            else if (function_number == 3) bounds.push_back(std::make_pair(-4, 4));
            else bounds.push_back(std::make_pair(-100, 100));
        }
    } else if (year == 2011) {
        const auto& problem = get_cec2011_problem(function_number);
        effective_dimension = problem.dimension;
        bounds = problem.bounds;
    } else {
        bounds = std::vector<std::pair<double, double>>(effective_dimension, std::make_pair(-100.0, 100.0));
    }

    if (year == 2020) cecfunc = new minion::CEC2020Functions(function_number, effective_dimension);
    else if (year == 2022) cecfunc = new minion::CEC2022Functions(function_number, effective_dimension);
    else if (year == 2017) cecfunc = new minion::CEC2017Functions(function_number, effective_dimension);
    else if (year == 2019) cecfunc = new minion::CEC2019Functions(function_number, effective_dimension);
    else if (year == 2014) cecfunc = new minion::CEC2014Functions(function_number, effective_dimension);
    else if (year == 2011) cecfunc = new minion::CEC2011Functions(function_number, effective_dimension);
    else throw std::runtime_error("Invalid year.");

    auto start = std::chrono::high_resolution_clock::now();
    if (logger) {
        logger->best = std::numeric_limits<double>::infinity();
        logger->evals = 0;
        logger->next_index = 0;
    }

    auto solutions = run_libcmaes_optimizer(
        cecfunc,
        effective_dimension,
        bounds,
        population_size,
        max_evals,
        algo,
        seed,
        0.3,
        logger);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (end - start);

    const auto best = solutions.best_candidate();
    const double ret = best.get_fvalue();

    std::cout << "Optimization Results for Function " << function_number << ":\n";
    std::cout << "\tAlgo : libcmaes_" << algo << ". Best Value: " << ret << "\n";
    std::cout << "\tReal Ncalls : " << cecfunc->Ncalls << "\n";
    std::cout << "\tElapsed time: " << duration.count() << " seconds\n";

    delete cecfunc;
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
    int year = 2017;
    int Nmaxevals = int(1e+4 * dimension);
    int Nthreads = 1;
    int log_min_ev = 0;
    if (argc > 1) {
        numRuns = std::atoi(argv[1]);
    }
    if (argc > 2) {
        dimension = std::atoi(argv[2]);
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
        Nmaxevals = std::atoi(argv[6]);
    }
    if (argc > 7) {
        Nthreads = std::atoi(argv[7]);
    }
    if (argc > 8) {
        log_min_ev = std::atoi(argv[8]);
    }

    Nthreads = std::max(1, Nthreads);
    if (numRuns > 0) {
        Nthreads = std::min(Nthreads, numRuns);
    }

    std::vector<int> funcnums;
    if (year == 2017 || year == 2014) funcnums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    else if (year == 2020 || year == 2019) funcnums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    else if (year == 2022) funcnums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    else if (year == 2011) funcnums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    else throw std::runtime_error("Year invalid.");

    std::vector<std::vector<double>> results(numRuns);
    std::unordered_map<int, std::vector<std::vector<double>>> min_ev_logs;
    if (log_min_ev) {
        for (int func : funcnums) {
            int eff_dim = get_effective_dimension(func, dimension, year);
            size_t step = std::max<size_t>(1, 10 * static_cast<size_t>(eff_dim));
            size_t count = static_cast<size_t>(Nmaxevals) / step + 1;
            min_ev_logs.emplace(func, std::vector<std::vector<double>>(count, std::vector<double>(numRuns, std::numeric_limits<double>::infinity())));
        }
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(Nthreads));

    int base = (Nthreads > 0) ? numRuns / Nthreads : numRuns;
    int remainder = (Nthreads > 0) ? numRuns % Nthreads : 0;
    int start_index = 0;

    for (int t = 0; t < Nthreads; ++t) {
        int count = base + (t < remainder ? 1 : 0);
        int end_index = start_index + count;
        workers.emplace_back([&, start_index, end_index]() {
            for (int i = start_index; i < end_index; ++i) {
                std::cout << "========================\n";
                std::cout << "\nRun : " << i + 1 << "\n";
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> result_per_run;
                for (auto& num : funcnums) {
                    try {
                        std::unique_ptr<MinEvLogger> logger;
                        if (log_min_ev) {
                            logger = std::make_unique<MinEvLogger>(
                                static_cast<size_t>(get_effective_dimension(num, dimension, year)),
                                static_cast<size_t>(Nmaxevals),
                                get_global_optimum(num, year));
                        }
                        double fval = minimize_cec_functions(num, dimension, popsize, Nmaxevals, year, algo, i,
                                                             logger ? logger.get() : nullptr);
                        result_per_run.push_back(fval);
                        if (log_min_ev) {
                            auto it = min_ev_logs.find(num);
                            if (it != min_ev_logs.end()) {
                                auto& matrix = it->second;
                                for (size_t row = 0; row < logger->samples.size(); ++row) {
                                    matrix[row][static_cast<size_t>(i)] = logger->samples[row];
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error optimizing function " << num << ": " << e.what() << std::endl;
                        continue;
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = (end - start);
                std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
                results[i] = std::move(result_per_run);
                std::cout << "========================\n";
            }
        });
        start_index = end_index;
    }

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    const std::string dump_algo = "libcmaes_" + algo;
    dumpResultsToFile(results, "results_" + std::to_string(year) + "_" + dump_algo + "_" +
                                 std::to_string(dimension) + "_" + std::to_string(Nmaxevals) + ".txt");
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
