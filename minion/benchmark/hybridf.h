#ifndef MINION_HYBRIDF_H
#define MINION_HYBRIDF_H

#include "basicf.h"

#include <vector>

namespace minion {

/**
 * @brief Hybrid benchmark function built from basic components.
 *
 * A `HybridF` partitions the decision vector into blocks and evaluates a
 * different `BasicF` on each block. The constructor describes only the
 * structure of the hybrid:
 * - `dimension`
 * - the component list `basicFcomps`
 * - optional block proportions
 * - optional global rotation
 * - optional permutation
 *
 * The function metadata is inferred automatically:
 * - `x_opt` is the canonical hybrid optimum and is stored as the zero vector;
 * - `f_opt` is the sum of component optima and is therefore `0.0` for the
 *   built-in normalized basic functions;
 * - `lambda` is derived from the component scales;
 * - `properties` is a human-readable description of the hybrid.
 *
 * This is a low-level function object. Any suite-level optimum shift, bounds,
 * or fixed benchmark offset belongs in `MinionBenchmark`, not in `HybridF`.
 */
class HybridF : public FunctionBase {
public:
    HybridF(
        int dimension,
        std::vector<BasicFunctionId> basicFcomps,
        std::vector<double> proportions = {},
        std::vector<double> rotation = {},
        std::vector<int> permutation = {});

    double evaluate_point(const double* x) const override;
    void evaluate_batch_raw(const double* xs, std::size_t count, double* out) const override;

    const std::vector<BasicFunctionId>& components() const;
    const std::vector<int>& block_sizes() const;
    const std::vector<int>& permutation() const;
    const std::vector<double>& rotation() const;

private:
    std::vector<BasicFunctionId> component_ids_;
    std::vector<std::shared_ptr<BasicF>> components_;
    std::vector<double> proportions_;
    std::vector<int> block_sizes_;
    std::vector<int> permutation_;
    std::vector<double> rotation_;
};

} // namespace minion

#endif // MINION_HYBRIDF_H
