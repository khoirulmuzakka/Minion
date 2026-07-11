#include "minion_bench.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    constexpr int dimension = 50;
    constexpr int num_hybrid = 250;
    constexpr int num_composition = 250;
    constexpr int instance = 1;
    constexpr bool use_rotation = true;
    constexpr double tolerance = 1.0e-8;

    const int basic_count = static_cast<int>(minion::list_basic_functions().size());
    const int total_functions = basic_count + num_hybrid + num_composition;

    bool all_ok = true;
    std::cout << "Checking all MinionBenchmark optima in D=" << dimension
              << " with total_functions=" << total_functions
              << " (basic=" << basic_count
              << ", hybrid=" << num_hybrid
              << ", composition=" << num_composition
              << ", instance=" << instance
              << ", use_rotation=" << (use_rotation ? "1" : "0") << ")\n";
    std::cout << std::left << std::setw(8) << "fn"
              << std::setw(20) << "inner"
              << std::right << std::setw(18) << "stored f_opt"
              << std::setw(18) << "f(x_opt)"
              << std::setw(18) << "abs diff"
              << std::setw(18) << "max |x_opt|"
              << std::setw(8) << "ok" << "\n";

    for (int fn = 1; fn <= total_functions; ++fn) {
        minion::MinionBenchmark benchmark(
            fn,
            dimension,
            num_hybrid,
            num_composition,
            instance,
            use_rotation);

        const double value = benchmark.evaluate_point(benchmark.x_opt.data());
        const double diff = std::fabs(value - benchmark.f_opt);
        double max_abs_xopt = 0.0;
        for (double xi : benchmark.x_opt) {
            max_abs_xopt = std::max(max_abs_xopt, std::fabs(xi));
        }
        const bool ok = diff <= tolerance;
        all_ok = all_ok && ok;

        std::cout << std::left << std::setw(8) << fn
                  << std::setw(20) << benchmark.function()->name
                  << std::right << std::setw(18) << std::setprecision(10) << benchmark.f_opt
                  << std::setw(18) << std::setprecision(10) << value
                  << std::setw(18) << std::setprecision(10) << diff
                  << std::setw(18) << std::setprecision(10) << max_abs_xopt
                  << std::setw(8) << (ok ? "yes" : "no") << "\n";
    }

    std::cout << (all_ok ? "ALL_CHECKS_PASSED" : "CHECKS_FAILED") << "\n";
    return all_ok ? 0 : 1;
}
