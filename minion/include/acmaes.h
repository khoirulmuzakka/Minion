#ifndef ACMAES_H
#define ACMAES_H

#include "minimizer_base.h"
#include <Eigen/Dense>

namespace minion {

/**
 * @class ACMAES
 * @brief Class implementing active CMA-ES with the libcmaes-style active
 *        covariance update, but without the restart logic.
 */
class ACMAES : public MinimizerBase {
public:
    ACMAES(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>());

    void initialize() override;
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
    size_t Nevals = 0;

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

    void updateEigenDecomposition();
    std::vector<double> ensureBounds(std::vector<double> candidate) const;
};

}

#endif
