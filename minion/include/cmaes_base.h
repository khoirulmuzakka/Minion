#ifndef CMAES_BASE_H
#define CMAES_BASE_H

#include "minimizer_base.h"
#include <Eigen/Dense>

#include <limits>
#include <string>
#include <vector>

namespace minion {

class CMAESBase : public MinimizerBase {
public:
    CMAESBase(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>());

protected:
    Options buildOptions(const std::string& algorithm_name) const;
    void initializeCommon(const std::string& algorithm_name, double damps_extra_term);
    void initializeMean();
    void updateEigenDecomposition();
    std::vector<double> applyBounds(std::vector<double> candidate) const;
    std::vector<double> sampleCandidate(
        const Eigen::VectorXd& meanState,
        const Eigen::MatrixXd& BState,
        const Eigen::VectorXd& DState,
        double sigmaState) const;
    std::vector<double> denormalizePoint(const std::vector<double>& candidate) const;
    double computeRelativeRange(const std::vector<double>& fitness) const;
    void recordIteration(size_t generation, size_t evaluations, double relRange);

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
    double best_fitness = std::numeric_limits<double>::infinity();
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
    std::vector<std::pair<double, double>> original_bounds;
};

}  // namespace minion

#endif
