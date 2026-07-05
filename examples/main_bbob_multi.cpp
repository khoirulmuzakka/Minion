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
#include <sstream>

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

struct Job {
    int run_index = 0;
    size_t function_index = 0;
    int function_number = 0;
};

std::vector<double> objective_function_logged(const std::vector<std::vector<double>>& candidates, void* data) {
    auto* ctx = static_cast<LogContext*>(data);
    auto values = ctx->problem->evaluateBatch(candidates);
    if (ctx->logger) {
        ctx->logger->update(values);
    }
    return values;
}

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

    std::vector<std::vector<double>> results(static_cast<size_t>(numRuns), std::vector<double>(functions.size(), std::numeric_limits<double>::infinity()));
    std::map<int, std::vector<std::vector<double>>> min_ev_logs;
    if (log_min_ev) {
        for (int func : functions) {
            size_t step = std::max<size_t>(1, 10 * static_cast<size_t>(dimension));
            size_t count = static_cast<size_t>(maxevals) / step + 1;
            min_ev_logs.emplace(func, std::vector<std::vector<double>>(
                                          count, std::vector<double>(static_cast<size_t>(numRuns), std::numeric_limits<double>::infinity())));
        }
    }
    std::vector<Job> jobs;
    jobs.reserve(static_cast<size_t>(numRuns) * functions.size());
    for (int run = 0; run < numRuns; ++run) {
        for (size_t func_index = 0; func_index < functions.size(); ++func_index) {
            jobs.push_back(Job{run, func_index, functions[func_index]});
        }
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
                std::ostringstream job_output;
                try {
                    std::unique_ptr<minion::BBOB2009Problem> problem;
                    {
                        std::lock_guard<std::mutex> lock(cocoMutex);
                        problem = std::make_unique<minion::BBOB2009Problem>(job.function_number, dimension, year);
                    }
                    const auto function_start = std::chrono::high_resolution_clock::now();
                    std::unique_ptr<MinEvLogger> logger;
                    if (log_min_ev) {
                        logger = std::make_unique<MinEvLogger>(
                            static_cast<size_t>(dimension),
                            static_cast<size_t>(maxevals),
                            problem->bestValue());
                    }

                    job_output << "========================\n";
                    job_output << "Run : " << (job.run_index + 1) << "\n";
                    job_output << "F" << job.function_number << "\n";
                    job_output << "  name : " << problem->name() << "\n";

                    LogContext ctx{problem.get(), logger ? logger.get() : nullptr};
                    minion::MinionFunction objective;
                    void* objective_data = nullptr;
                    if (logger) {
                        objective = objective_function_logged;
                        objective_data = &ctx;
                    } else {
                        objective = [problem_ptr = problem.get()](const std::vector<std::vector<double>>& candidates, void*) {
                            return problem_ptr->evaluateBatch(candidates);
                        };
                        objective_data = nullptr;
                    }

                    const std::vector<std::pair<double, double>>& bounds = problem->bounds();
                    const std::vector<std::vector<double>> x0 = {problem->initialSolution()};
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
                        job.run_index,
                        options);
                    const minion::MinionResult result = optimizer.optimize();
                    const double error = result.fun - problem->bestValue();

                    job_output << "  nfev = " << result.nfev << "\n";
                    job_output << "  err  = " << error << "\n";

                    results[static_cast<size_t>(job.run_index)][job.function_index] = result.fun;
                    if (log_min_ev && logger) {
                        auto it = min_ev_logs.find(job.function_number);
                        if (it != min_ev_logs.end()) {
                            auto& matrix = it->second;
                            for (size_t row = 0; row < logger->samples.size(); ++row) {
                                matrix[row][static_cast<size_t>(job.run_index)] = logger->samples[row];
                            }
                        }
                        logger->finalize();
                    }

                    const auto function_end = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double> function_elapsed = function_end - function_start;
                    job_output << "  time = " << std::fixed << std::setprecision(3)
                               << function_elapsed.count() << "s\n\n";

                } catch (const std::exception& ex) {
                    std::lock_guard<std::mutex> lock(coutMutex);
                    std::cerr << "Error optimizing function F" << job.function_number
                              << " in run " << (job.run_index + 1) << ": " << ex.what() << std::endl;
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
