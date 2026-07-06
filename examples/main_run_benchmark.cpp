#include <algorithm>
#include <cstdlib>
#include <string>

#include "benchmark.h"

namespace {

minion::BenchmarkMode parse_mode(const std::string& arg) {
    if (arg == "cec") {
        return minion::BenchmarkMode::Cec;
    }
    if (arg == "bbob") {
        return minion::BenchmarkMode::Bbob;
    }
    throw std::runtime_error("First argument must be either 'cec' or 'bbob'.");
}

/**
 * CLI layout:
 *   cec|bbob Nruns dim algo popsize year maxevals nthreads acc
 *
 * The benchmark implementation itself lives in `minion::Benchmark`.
 */
minion::BenchmarkConfig parse_cli(int argc, char* argv[]) {
    minion::BenchmarkConfig config;
    int arg_offset = 0;

    if (argc > 1) {
        const std::string first_arg = argv[1];
        if (first_arg == "cec" || first_arg == "bbob") {
            config.mode = parse_mode(first_arg);
            arg_offset = 1;
        }
    }

    config.num_runs = 1;
    config.dimension = 10;
    config.algo = "ARRDE";
    config.population_size = 0;
    config.year = (config.mode == minion::BenchmarkMode::Bbob) ? 2009 : 2017;
    config.max_evals = -1;
    config.nthreads = 1;
    config.acc = 8;

    if (argc > 1 + arg_offset) config.num_runs = std::max(1, std::atoi(argv[1 + arg_offset]));
    if (argc > 2 + arg_offset) config.dimension = std::max(1, std::atoi(argv[2 + arg_offset]));
    if (argc > 3 + arg_offset) config.algo = argv[3 + arg_offset];
    if (argc > 4 + arg_offset) config.population_size = std::atoi(argv[4 + arg_offset]);
    if (argc > 5 + arg_offset) config.year = std::atoi(argv[5 + arg_offset]);
    if (argc > 6 + arg_offset) config.max_evals = std::max(1, std::atoi(argv[6 + arg_offset]));
    if (argc > 7 + arg_offset) config.nthreads = std::max(1, std::atoi(argv[7 + arg_offset]));
    if (argc > 8 + arg_offset) config.acc = std::max(0, std::atoi(argv[8 + arg_offset]));

    return config;
}

} // namespace

int main(int argc, char* argv[]) {
    const minion::BenchmarkConfig config = parse_cli(argc, argv);
    minion::Benchmark benchmark(config);
    (void)benchmark.run();
    return 0;
}
