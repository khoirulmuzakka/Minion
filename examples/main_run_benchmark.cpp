#include <iostream>
#include <vector>
#include <chrono>
#include <array>
#include <unordered_map>
#include "minion.h"
#include "bbob2009.h"
#include "minion_cec.h"
#include "utility.h"
#include <fstream>
#include <cmath>
#include <thread>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <atomic>
#include <mutex>
#include <sstream>

namespace {

enum class BenchmarkMode {
    Cec,
    Bbob
};

constexpr double kPi = 3.14159265358979323846;

bool is_supported_bbob_dimension(int dimension) {
    return dimension == 2 || dimension == 3 || dimension == 5 || dimension == 10 || dimension == 20 || dimension == 40;
}

std::vector<int> get_function_numbers(BenchmarkMode mode, int year) {
    if (mode == BenchmarkMode::Bbob) {
        std::vector<int> functions;
        functions.reserve(24);
        for (int func = 1; func <= 24; ++func) {
            functions.push_back(func);
        }
        return functions;
    }

    if (year == 2017 || year == 2014) {
        std::vector<int> functions;
        functions.reserve(30);
        for (int func = 1; func <= 30; ++func) {
            functions.push_back(func);
        }
        return functions;
    }
    if (year == 2020 || year == 2019) {
        return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    }
    if (year == 2022) {
        return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    }
    if (year == 2011) {
        return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    }
    throw std::runtime_error("Year invalid.");
}

} // namespace

std::vector <double> objective_function (const std::vector<std::vector<double>> & x, void* data){
     minion::CECBase* func = static_cast<minion::CECBase* > (data);
    return func->operator()(x); // Call the operator with a single vector
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

/**
 * Format a floating-point value in scientific notation with a fixed number
 * of digits after the decimal point.
 */
std::string format_scientific(double value, int acc) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(std::max(0, acc)) << value;
    return oss.str();
}

struct CecLogContext {
    minion::CECBase* func = nullptr;
    MinEvLogger* logger = nullptr;
};

struct BbobLogContext {
    minion::BBOB2009Problem* problem = nullptr;
    MinEvLogger* logger = nullptr;
};

struct Job {
    int run_index = 0;
    size_t function_index = 0;
    int function_number = 0;
};

std::string format_eta(std::chrono::seconds seconds) {
    const auto total_seconds = seconds.count();
    const auto hours = total_seconds / 3600;
    const auto minutes = (total_seconds % 3600) / 60;
    const auto secs = total_seconds % 60;

    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hours << ':'
        << std::setw(2) << std::setfill('0') << minutes << ':'
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

void render_progress_bar(
    size_t completed,
    size_t total,
    const std::chrono::steady_clock::time_point& start_time) {
    constexpr size_t kBarWidth = 40;
    if (total == 0) {
        std::cout << "\rProgress: [----------------------------------------] 0/0 (0%) ETA --:--:--" << std::flush;
        return;
    }

    const double ratio = static_cast<double>(completed) / static_cast<double>(total);
    const size_t filled = std::min(kBarWidth, static_cast<size_t>(ratio * static_cast<double>(kBarWidth)));
    std::ostringstream percent;
    percent << std::fixed << std::setprecision(1) << (ratio * 100.0);

    std::cout << "\rProgress: [";
    for (size_t i = 0; i < kBarWidth; ++i) {
        std::cout << (i < filled ? '#' : '-');
    }

    std::string eta_text = "--:--:--";
    if (completed > 0 && completed < total) {
        const auto elapsed = std::chrono::steady_clock::now() - start_time;
        const double elapsed_sec = std::chrono::duration<double>(elapsed).count();
        const double sec_per_job = elapsed_sec / static_cast<double>(completed);
        const double remaining_sec = sec_per_job * static_cast<double>(total - completed);
        eta_text = format_eta(std::chrono::seconds(static_cast<long long>(std::max(0.0, std::ceil(remaining_sec)))));
    } else if (completed >= total) {
        eta_text = "00:00:00";
    }

    std::cout << "] " << completed << "/" << total << " (" << percent.str() << "%) ETA " << eta_text << std::flush;
}

std::vector<double> objective_function_logged(const std::vector<std::vector<double>>& x, void* data) {
    auto* ctx = static_cast<CecLogContext*>(data);
    auto values = ctx->func->operator()(x);
    if (ctx->logger) {
        ctx->logger->update(values);
    }
    return values;
}

std::vector<double> objective_function_logged_bbob(const std::vector<std::vector<double>>& candidates, void* data) {
    auto* ctx = static_cast<BbobLogContext*>(data);
    auto values = ctx->problem->evaluateBatch(candidates);
    if (ctx->logger) {
        ctx->logger->update(values);
    }
    return values;
}

void callBack(minion::MinionResult* res) {
    //std::cout << "Best fitness " << res->fun << "\n";
};

/**
 * Parse the benchmark family from the first command-line argument.
 */
BenchmarkMode parse_mode(const std::string& arg) {
    if (arg == "cec") {
        return BenchmarkMode::Cec;
    }
    if (arg == "bbob") {
        return BenchmarkMode::Bbob;
    }
    throw std::runtime_error("First argument must be either 'cec' or 'bbob'.");
}

/**
 * Return the problem dimension used by the selected CEC benchmark family.
 */
int get_effective_dimension(int function_number, int dimension, int year) {
    if (year == 2019) { 
        if (function_number == 1) return 9;
        if (function_number == 2) return 16;
        if (function_number == 3) return 18;
        return 10;
    }
    if (year == 2011) {
        return minion::CEC2011::problemDimension(function_number);
    }
    return dimension;
}

/**
 * Return the known global optimum offset used by the min-EV logger.
 */
double get_global_optimum(int function_number, int year) {
    if (year == 2022) {
        static const std::array<double, 12> kCEC2022 = {300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700};
        if (function_number >= 1 && function_number <= static_cast<int>(kCEC2022.size())) {
            return kCEC2022[static_cast<size_t>(function_number - 1)];
        }
    } else if (year == 2020) {
        static const std::array<double, 10> kCEC2020 = {100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500};
        if (function_number >= 1 && function_number <= static_cast<int>(kCEC2020.size())) {
            return kCEC2020[static_cast<size_t>(function_number - 1)];
        }
    } else if (year == 2017 || year == 2014) {
        if (function_number >= 1 && function_number <= 30) {
            return 100.0 * static_cast<double>(function_number);
        }
    } else if (year == 2019) {
        return 1.0;
    }
    throw std::runtime_error("Global optimum not defined for the given function number and year.");
}

double minimize_cec_functions(int function_number,
                              int dimension,
                              int population_size,
                              int max_evals,
                              int year = 2022,
                              std::string algo = "ARRDE",
                              int seed = -1,
                              MinEvLogger* logger = nullptr,
                              std::ostream* out = &std::cout) {
    minion::CECBase* cecfunc;
    std::vector<std::pair<double, double>> bounds;
    int effective_dimension = get_effective_dimension(function_number, dimension, year);
    if (year==2019) { 
        for (int i=0; i<effective_dimension; i++) {
            if (function_number ==1) bounds.push_back(std::make_pair(-8192, 8192)); 
            else if (function_number==2) bounds.push_back(std::make_pair(-16384, 16384)); 
            else if (function_number==3) bounds.push_back(std::make_pair(-4, 4)); 
            else bounds.push_back(std::make_pair(-100, 100));
        }
    } else if (year==2011) {
        const auto& problem = minion::CEC2011::problemDefinition(function_number);
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
    settings["convergence_tol"] = 0.0;
    std::vector<std::vector<double>> x0={};
    if (algo == "NelderMead" || algo == "L_BFGS_B" || algo == "DA"){
        std::vector<double> x00;
        for (auto& el : bounds) x00.push_back(0.5*(el.first+el.second));
        x0 = {x00};
    };

    CecLogContext ctx{cecfunc, logger};
    minion::Minimizer optimizer(logger ? objective_function_logged : objective_function,
                                bounds,
                                x0,
                                logger ? static_cast<void*>(&ctx) : static_cast<void*>(cecfunc),
                                callBack,
                                algo,
                                max_evals,
                                seed,
                                settings);
    // Optimize and get the result
    minion::MinionResult result = optimizer();
    double ret = result.fun;

    // Output the results
    if (false) {
        (*out) << "Optimization Results for Function " << function_number << ":\n";
        (*out) << "\tAlgo : "<< algo << ". Best Value: " << result.fun << "\n";
        (*out) << "\tReal Ncalls : " << cecfunc->Ncalls << "\n";
    }

    delete cecfunc;
    if (logger) {
        logger->finalize();
    }
    return ret;
}

double minimize_bbob_functions(minion::BBOB2009Problem& problem,
                               int population_size,
                               int max_evals,
                               std::string algo = "ARRDE",
                               int seed = -1,
                               MinEvLogger* logger = nullptr,
                               std::ostream* out = &std::cout) {
    auto settings = minion::DefaultSettings().getDefaultSettings(algo);
    settings["population_size"] = population_size;
    settings["rel_initial_step"] = 0.3;
    settings["bound_strategy"] = std::string("reflect-random");

    const auto& bounds = problem.bounds();
    const std::vector<std::vector<double>> x0 = {problem.initialSolution()};

    BbobLogContext ctx{&problem, logger};
    minion::MinionFunction objective;
    void* objective_data = nullptr;
    if (logger) {
        objective = objective_function_logged_bbob;
        objective_data = &ctx;
    } else {
        objective = [&problem](const std::vector<std::vector<double>>& candidates, void*) {
            return problem.evaluateBatch(candidates);
        };
        objective_data = nullptr;
    }

    minion::Minimizer optimizer(
        objective,
        bounds,
        x0,
        objective_data,
        nullptr,
        algo,
        static_cast<size_t>(max_evals),
        seed,
        settings);

    const minion::MinionResult result = optimizer.optimize();
    const double ret = result.fun;

    if (false) {
        (*out) << "\tAlgo : " << algo << ". Best Value: " << result.fun << "\n";
    }

    if (logger) {
        logger->finalize();
    }
    return ret;
}

/**
 * Dump the run-by-function result table using the configured precision.
 */
void dumpResultsToFile(const std::vector<std::vector<double>>& results,
                       const std::string& filename,
                       int acc) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << format_scientific(row[i], acc);
            if (i < row.size() - 1) {
                file << "\t";
            }
        }
        file << "\n";
    }

    file.close();
}

/**
 * Dump the optional min-EV trace matrix using the configured precision.
 */
void dumpMatrixToFile(const std::vector<std::vector<double>>& matrix,
                      const std::string& filename,
                      int acc) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << format_scientific(row[i], acc);
            if (i < row.size() - 1) {
                file << "\t";
            }
        }
        file << "\n";
    }
    file.close();
}

/**
 * Combined benchmark driver.
 *
 * CLI:
 *   cec|bbob Nruns dim algo popsize year maxevals nthreads acc
 *
 * The final argument controls scientific formatting precision for both
 * terminal output and dumped result tables.
 */
int main(int argc, char* argv[]) {
    BenchmarkMode mode = BenchmarkMode::Cec;
    int arg_offset = 0;
    if (argc > 1) {
        const std::string first_arg = argv[1];
        if (first_arg == "cec" || first_arg == "bbob") {
            mode = parse_mode(first_arg);
            arg_offset = 1;
        }
    }

    int numRuns = 1;
    int dimension = 10;
    std::string algo = "ARRDE";
    int popsize = 0;
    int year = (mode == BenchmarkMode::Bbob) ? 2018 : 2017;
    int Nmaxevals = -1;
    int Nthreads = 1;
    int acc = 8;

    if (argc > 1 + arg_offset) {
        numRuns = std::max(1, std::atoi(argv[1 + arg_offset]));
    }
    if (argc > 2 + arg_offset) {
        dimension = std::max(1, std::atoi(argv[2 + arg_offset]));
    }
    if (argc > 3 + arg_offset) {
        algo = argv[3 + arg_offset];
    }
    if (argc > 4 + arg_offset) {
        popsize = std::atoi(argv[4 + arg_offset]);
    }
    if (argc > 5 + arg_offset) {
        year = std::atoi(argv[5 + arg_offset]);
    }
    if (argc > 6 + arg_offset) {
        Nmaxevals = std::max(1, std::atoi(argv[6 + arg_offset]));
    }
    if (argc > 7 + arg_offset) {
        Nthreads = std::max(1, std::atoi(argv[7 + arg_offset]));
    }
    if (argc > 8 + arg_offset) {
        acc = std::max(0, std::atoi(argv[8 + arg_offset]));
    }

    if (Nmaxevals < 0) {
        Nmaxevals = static_cast<int>(1e4 * dimension);
    }

    if (mode == BenchmarkMode::Bbob && !is_supported_bbob_dimension(dimension)) {
        throw std::runtime_error("BBOB2009 only supports dimensions 2, 3, 5, 10, 20, and 40.");
    }

    const std::vector<int> funcnums = get_function_numbers(mode, year);

    std::vector<std::vector<double>> results(
        static_cast<size_t>(numRuns),
        std::vector<double>(funcnums.size(), std::numeric_limits<double>::infinity()));
    // Min-EV logging stays compiled in, but is disabled by default for now.
    const bool log_min_ev = false;
    std::unordered_map<int, std::vector<std::vector<double>>> min_ev_logs;
    if (log_min_ev) {
        for (int func : funcnums) {
            const size_t logger_dimension = (mode == BenchmarkMode::Cec)
                                                ? static_cast<size_t>(get_effective_dimension(func, dimension, year))
                                                : static_cast<size_t>(dimension);
            const size_t step = std::max<size_t>(1, 10 * logger_dimension);
            const size_t count = static_cast<size_t>(Nmaxevals) / step + 1;
            min_ev_logs.emplace(
                func,
                std::vector<std::vector<double>>(
                    count,
                    std::vector<double>(static_cast<size_t>(numRuns), std::numeric_limits<double>::infinity())));
        }
    }

    std::vector<Job> jobs;
    jobs.reserve(static_cast<size_t>(numRuns) * funcnums.size());
    for (int run = 0; run < numRuns; ++run) {
        for (size_t func_index = 0; func_index < funcnums.size(); ++func_index) {
            jobs.push_back(Job{run, func_index, funcnums[func_index]});
        }
    }

    Nthreads = std::max(1, Nthreads);
    if (!jobs.empty()) {
        Nthreads = std::min(Nthreads, static_cast<int>(jobs.size()));
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(Nthreads));
    std::atomic<size_t> nextJob{0};
    std::atomic<size_t> completedJobs{0};
    std::mutex coutMutex;
    std::mutex cocoMutex;
    const auto progress_start = std::chrono::steady_clock::now();

    for (int t = 0; t < Nthreads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                const size_t job_index = nextJob.fetch_add(1, std::memory_order_relaxed);
                if (job_index >= jobs.size()) {
                    break;
                }

                const Job job = jobs[job_index];
                std::unique_ptr<MinEvLogger> logger;
                std::ostringstream job_output;
                double fval = std::numeric_limits<double>::infinity();

                try {
                    if (mode == BenchmarkMode::Cec) {
                        if (log_min_ev) {
                            logger = std::make_unique<MinEvLogger>(
                                static_cast<size_t>(get_effective_dimension(job.function_number, dimension, year)),
                                static_cast<size_t>(Nmaxevals),
                                get_global_optimum(job.function_number, year));
                        }

                        const auto start = std::chrono::high_resolution_clock::now();
                        fval = minimize_cec_functions(
                            job.function_number,
                            dimension,
                            popsize,
                            Nmaxevals,
                            year,
                            algo,
                            job.run_index,
                            logger ? logger.get() : nullptr,
                            &job_output);
                        const auto end = std::chrono::high_resolution_clock::now();
                        const std::chrono::duration<double> duration = end - start;
                        (void)duration;

                        if (log_min_ev && logger) {
                            auto it = min_ev_logs.find(job.function_number);
                            if (it != min_ev_logs.end()) {
                                auto& matrix = it->second;
                                for (size_t row = 0; row < logger->samples.size(); ++row) {
                                    matrix[row][static_cast<size_t>(job.run_index)] = logger->samples[row];
                                }
                            }
                        }
                    } else {
                        std::unique_ptr<minion::BBOB2009Problem> problem;
                        {
                            std::lock_guard<std::mutex> lock(cocoMutex);
                            problem = std::make_unique<minion::BBOB2009Problem>(job.function_number, dimension, year);
                        }

                        if (log_min_ev) {
                            logger = std::make_unique<MinEvLogger>(
                                static_cast<size_t>(dimension),
                                static_cast<size_t>(Nmaxevals),
                                problem->bestValue());
                        }

                        const auto start = std::chrono::high_resolution_clock::now();
                        fval = minimize_bbob_functions(
                            *problem,
                            popsize,
                            Nmaxevals,
                            algo,
                            job.run_index,
                            logger ? logger.get() : nullptr,
                            &job_output);
                        const auto end = std::chrono::high_resolution_clock::now();
                        const std::chrono::duration<double> duration = end - start;
                        (void)duration;

                        if (log_min_ev && logger) {
                            auto it = min_ev_logs.find(job.function_number);
                            if (it != min_ev_logs.end()) {
                                auto& matrix = it->second;
                                for (size_t row = 0; row < logger->samples.size(); ++row) {
                                    matrix[row][static_cast<size_t>(job.run_index)] = logger->samples[row];
                                }
                            }
                        }
                    }

                    results[static_cast<size_t>(job.run_index)][job.function_index] = fval;
                    job_output << "  Best F" << job.function_number << " : "
                               << format_scientific(fval, acc) << "\n";
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(coutMutex);
                    std::cerr << "Error optimizing function F" << job.function_number
                              << " in run " << (job.run_index + 1) << ": " << e.what() << std::endl;
                    std::cout << "========================\n";
                }

                {
                    std::lock_guard<std::mutex> lock(coutMutex);
                    std::cout << job_output.str();
                }

                const size_t done = completedJobs.fetch_add(1, std::memory_order_relaxed) + 1;
                {
                    std::lock_guard<std::mutex> lock(coutMutex);
                    render_progress_bar(done, jobs.size(), progress_start);
                    if (done == jobs.size()) {
                        std::cout << std::endl;
                    }
                }
            }
        });
    }

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    const std::string results_prefix = (mode == BenchmarkMode::Bbob) ? "results_bbob_" : "results_cec_";
    dumpResultsToFile(
        results,
        results_prefix + std::to_string(year) + "_" + algo + "_" + std::to_string(dimension) + "_" +
            std::to_string(Nmaxevals) + "_popsize_" + std::to_string(popsize) + ".txt",
        acc);
    if (log_min_ev) {
        const std::string out_dir = algo;
        std::filesystem::create_directories(out_dir);
        for (const auto& entry : min_ev_logs) {
            const int func = entry.first;
            const auto& matrix = entry.second;
            dumpMatrixToFile(matrix, out_dir + "/" + algo + "_F" + std::to_string(func) + "_Min_EV.txt", acc);
        }
    }

    return 0;
}
