#ifndef MINION_COMPOSITIONF_H
#define MINION_COMPOSITIONF_H

#include "basicf.h"

#include <memory>
#include <vector>

namespace minion {

/**
 * @brief Composition benchmark function built from arbitrary function objects.
 *
 * A `CompositionF` blends several full-dimensional component functions using
 * distance-based weights. Components are passed as `FunctionBase` pointers, so
 * a composition may contain:
 * - `BasicF`
 * - `HybridF`
 * - another `CompositionF`
 *
 * The constructor arguments define only the composition structure:
 * - `dimension`
 * - component functions
 * - `style`
 * - optional `deltas`
 * - optional additive `biases`
 * - optional positive `scales`
 * - optional per-component rotations
 * - optional per-component centers
 *
 * Style convention:
 * - `1..10`: common-optimum compositions `f(x)=\sum_i w_i(x) g_i(x)`
 * - `11..20`: shifted-component compositions
 *   `f(x)=\sum_i w_i(x) (b_i + s_i g_i(x-c_i))`
 *
 * Current defaults:
 * - `style=1`: Gaussian common-optimum weights
 * - `style=11`: distance-based Gaussian shifted weights
 *
 * The function metadata is inferred automatically:
 * - `x_opt` is inferred from the component definitions and centers;
 * - `f_opt` is inferred by evaluating the constructed composition at `x_opt`;
 * - `lambda` is derived from the component scales;
 * - `properties` is a human-readable summary string.
 *
 * As with `BasicF` and `HybridF`, this class represents a low-level canonical
 * function object. Benchmark-protocol choices such as bounds, instance-specific
 * shift, or a suite-level target value are handled by `MinionBenchmark`.
 */
class CompositionF : public FunctionBase {
public:
    CompositionF(
        int dimension,
        std::vector<std::shared_ptr<FunctionBase>> components,
        int style = 1,
        std::vector<double> deltas = {},
        std::vector<double> biases = {},
        std::vector<double> scales = {},
        std::vector<std::vector<double>> rotations = {},
        std::vector<std::vector<double>> centers = {});

    double evaluate_point(const double* x) const override;
    void evaluate_batch_raw(const double* xs, std::size_t count, double* out) const override;

    const std::vector<std::shared_ptr<FunctionBase>>& components() const;
    int style() const;
    const std::vector<double>& deltas() const;
    const std::vector<double>& biases() const;
    const std::vector<double>& scales() const;
    const std::vector<std::vector<double>>& rotations() const;
    const std::vector<std::vector<double>>& centers() const;

private:
    int style_ = 1;
    std::vector<std::shared_ptr<FunctionBase>> components_;
    std::vector<double> deltas_;
    std::vector<double> biases_;
    std::vector<double> scales_;
    std::vector<std::vector<double>> rotations_;
    std::vector<std::vector<double>> centers_;
};

} // namespace minion

#endif // MINION_COMPOSITIONF_H
