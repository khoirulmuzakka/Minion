#ifndef MINION_MINION_BENCHMARK_H
#define MINION_MINION_BENCHMARK_H

#include "compositionf.h"
#include "hybridf.h"

#include <memory>
#include <utility>
#include <vector>

namespace minion {

/**
 * @brief High-level custom benchmark suite wrapper.
 *
 * `MinionBenchmark` is the suite-level API built on top of the low-level
 * function objects `BasicF`, `HybridF`, and `CompositionF`.
 *
 * Responsibilities of this wrapper:
 * - instantiate one reproducible benchmark function using `function_number`
 *   and `instance`;
 * - own the recommended optimization bounds;
 * - apply suite-level shift and, optionally, suite-level rotation;
 * - shift the final optimum value so that the benchmark minimum is
 *   `100.0 + function()->f_opt`.
 *
 * Internally, generated hybrids and compositions are created directly from a
 * deterministic per-function seed, so construction cost does not scale with
 * the total suite size.
 *
 * The low-level function objects remain canonical, typically with `f_opt=0.0`,
 * while `MinionBenchmark` is the layer that turns them into a benchmark
 * problem instance.
 */
class MinionBenchmark : public FunctionBase {
public:
    MinionBenchmark(
        int function_number,
        int dimension,
        int Nhybrid,
        int Ncomposition,
        int instance = 1,
        bool useRotation = true);

    double evaluate_point(const double* x) const override;
    void evaluate_batch_raw(const double* xs, std::size_t count, double* out) const override;

    int function_number() const;
    int instance() const;
    const std::vector<std::pair<double, double>>& bounds() const;
    const std::shared_ptr<FunctionBase>& function() const;

private:
    int function_number_ = 0;
    int instance_ = 1;
    bool use_rotation_ = true;
    std::vector<std::pair<double, double>> bounds_;
    std::shared_ptr<FunctionBase> function_;
    std::vector<double> shift_;
    std::vector<double> rotation_;
};

} // namespace minion

#endif // MINION_MINION_BENCHMARK_H
