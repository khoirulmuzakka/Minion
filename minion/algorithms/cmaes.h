#ifndef CMAES_H
#define CMAES_H

#include "cmaes_base.h"

namespace minion {

/**
 * @class CMAES
 * @brief Class implementing the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
 * Reference : N. Hansen and A. Ostermeier, "Adapting arbitrary normal mutation distributions in evolution strategies: 
 *              the covariance matrix adaptation," Proceedings of IEEE International Conference on Evolutionary Computation, 
 *              Nagoya, Japan, 1996, pp. 312-317, doi: 10.1109/ICEC.1996.542381.
 *
 * CMA-ES maintains a multivariate normal search distribution whose mean and
 * covariance are adapted from successful samples, enabling efficient search in
 * non-separable, ill-conditioned landscapes.
 */
class CMAES : public CMAESBase {
public:
    /**
     * @brief Construct a CMAES optimizer.
     *
     * @param func Objective function to minimise.
     * @param bounds Search-space bounds for each decision variable.
     * @param x0 Optional collection of initial guesses. When multiple
     *           candidates are provided, the best according to @p func is used
     *           to seed the distribution mean.
     * @param data Additional opaque data forwarded to @p func.
     * @param callback Optional callback invoked with intermediate results.
     * @param maxevals Maximum number of function evaluations.
     * @param seed Seed for the pseudo-random number generator (``-1`` keeps the global setting).
     * @param options Algorithm-specific configuration overrides.
     */
    CMAES(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>());

    /**
     * @brief Prepare internal state using the supplied configuration.
     */
    void initialize() override;

    /**
     * @brief Run the optimisation loop until a stopping criterion is met.
     * @return Best-known result collected during the search.
     */
    MinionResult optimize() override;

private:
};

}

#endif
