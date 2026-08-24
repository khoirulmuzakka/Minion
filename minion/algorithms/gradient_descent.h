#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "blackbox_gradient.h"
#include "default_options.h"
#include "minimizer_base.h"

namespace minion {

/**
 * @class GradientDescent
 * @brief Zeroth-order gradient-descent family for bounded black-box optimization.
 *
 * This optimizer estimates gradients from objective evaluations and then applies
 * one of three update rules:
 *
 * - ``"gd"``: full gradient descent using the chosen estimator.
 * - ``"sgd"``: stochastic / subsampled updates.
 * - ``"adam"``: Adam-style first and second moment adaptation.
 *
 * Supported gradient estimators:
 *
 * - ``"coordinate_fd"``: coordinate-wise finite differences. With
 *   ``coordinate_batch_size > 0`` it uses a coordinate subset per iteration.
 * - ``"random_direction_fd"``: random directional finite differences averaged
 *   over ``gradient_samples`` directions.
 *
 * The public step-size option is ``base_learning_rate``. It is the constant step
 * size used by ``"gd"`` and ``"sgd"`` when ``lr_decay == 1``, and the global
 * scaling factor for Adam before coordinate-wise normalization. The legacy option
 * name ``learning_rate`` is still accepted for backward compatibility.
 *
 * Optional Armijo backtracking line search can be enabled with
 * ``use_line_search``. In that mode, Minion scales the proposed update by
 * ``1, rho, rho^2, ...`` until it finds sufficient decrease or exhausts the
 * configured ``max_linesearch`` trials.
 */
class GradientDescent : public MinimizerBase {
public:
    size_t Nevals = 0;
    std::vector<double> best;
    double best_f = std::numeric_limits<double>::infinity();

    /**
     * @brief Construct a gradient-descent optimizer.
     *
     * @param func Vectorized black-box objective.
     * @param bounds Box constraints for every variable.
     * @param x0 Optional initial guesses. If multiple points are supplied, the
     * best one is selected before the first update.
     * @param data Optional opaque pointer forwarded to ``func``.
     * @param callback Optional stop callback receiving the current result.
     * @param maxevals Maximum number of objective evaluations.
     * @param seed Random seed used by stochastic estimators.
     * @param options Algorithm options. See ``default_options.h`` for defaults.
     */
    GradientDescent(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<bool(MinionResult*)> callback = nullptr,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>())
        : MinimizerBase(func, bounds, x0, data, callback, maxevals, seed, options) {}

    /**
     * @brief Run the optimization loop.
     *
     * Stopping can be triggered by ``maxevals``, ``maxiters``, the callback, the
     * gradient norm tolerance ``g_tol``, the step tolerance ``x_tol``, the
     * relative objective tolerance ``f_tol``, or stagnation after projection to
     * the bound constraints.
     */
    MinionResult optimize() override;
    void initialize() override;

private:
    double last_f = 1.0;
    double finite_diff_rel_step = std::sqrt(std::numeric_limits<double>::epsilon());
};

}  // namespace minion

#endif
