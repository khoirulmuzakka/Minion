#ifndef CMAES_H
#define CMAES_H

#include "minimizer_base.h"
#include <Eigen/Dense>

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
class CMAES : public MinimizerBase {
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
     * @param tol Relative convergence tolerance (see stopping criterion).
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
        double tol = 0.0001,
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
    size_t lambda = 0;
    size_t mu = 0;
    double muEff = 0.0;

    double sigma = 0.3;
    double cc = 0.0;
    double cs = 0.0;
    double c1 = 0.0;
    double cmu = 0.0;
    double damps = 0.0;
    double chiN = 0.0;

    std::vector<double> diversity;
    std::vector<double> best;
    double best_fitness;
    size_t Nevals=0;

    std::vector<double> weights;

    Eigen::VectorXd mean;
    Eigen::MatrixXd C;
    Eigen::MatrixXd B;
    Eigen::VectorXd D;
    Eigen::VectorXd ps;
    Eigen::VectorXd pc;

    bool useBounds = false;
    size_t dimension = 0;
    bool support_tol = true;

    /**
     * @brief Refresh the eigen decomposition of the covariance matrix.
     */
    void updateEigenDecomposition();

    /**
     * @brief Clamp a candidate to the feasible domain when bounds are used.
     */
    std::vector<double> ensureBounds(std::vector<double> candidate) const;
};

}

#endif
