#ifndef MINION_CEC_BENCHMARK_H
#define MINION_CEC_BENCHMARK_H

#include <filesystem>
#include <string>
#include <vector>

namespace minion {

enum class BenchmarkMode {
    Cec,
    Bbob,
};

struct BenchmarkConfig {
    BenchmarkMode mode = BenchmarkMode::Cec;
    int num_runs = 1;
    int dimension = 10;
    std::string algo = "ARRDE";
    int population_size = 0;
    int year = 2017;
    int max_evals = -1;
    int nthreads = 1;
    int acc = 8;
    bool dump_results = true;
    bool log_min_ev = false;
    std::string results_folder = ".";
};

struct BenchmarkResult {
    std::vector<std::vector<double>> results;
    std::string results_file;
};

class Benchmark {
public:
    explicit Benchmark(BenchmarkConfig config);

    BenchmarkResult run();
    const BenchmarkConfig& config() const;

private:
    BenchmarkConfig config_;
};

BenchmarkResult run_benchmark(const BenchmarkConfig& config);

} // namespace minion

#endif
