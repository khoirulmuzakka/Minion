#ifndef RCMAES_H
#define RCMAES_H

#include "cmaes_base.h"

#include <Eigen/Dense>

#include <limits>
#include <string>
#include <vector>

namespace minion {

/**
 * @class RCMAES
 * @brief Restart CMA-ES in normalized coordinates with active covariance updates.
 */
class RCMAES : public CMAESBase {
public:
    RCMAES(
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
    struct ExclusionBox {
        std::vector<double> low;
        std::vector<double> high;
    };

    std::vector<double> applyBounds(const std::vector<double>& candidate) const;
    std::vector<double> eigenToStd(const Eigen::VectorXd& vec) const;
    std::vector<double> denormalizePoint(const std::vector<double>& candidate) const;
    Eigen::VectorXd buildCustomActiveWeights(size_t lambdaValue, size_t muValue) const;
    void configurePopulationParameters(size_t lambdaValue);
    void resetRegimeState(const Eigen::Ref<const Eigen::VectorXd>& startMean, double startSigma);
    void checkStoppingCriteria(bool& shouldStop) const;
    void recordHistory(double relRange);
    ExclusionBox buildExclusionBox(const std::vector<double>& best) const;
    bool isExcludedPoint(const std::vector<double>& candidate) const;

    size_t lambda_base = 0;
    size_t lambda_min = 0;
    double mu_ratio = 0.5;
    double sigma0 = 0.0;
    bool useCustomActive = false;

    Eigen::VectorXd initialMean;
    std::vector<std::pair<double, double>> original_bounds;

    size_t generation = 0;
    std::vector<double> currentFitness;

    std::vector<std::vector<double>> restart_bests;
    std::vector<ExclusionBox> exclusion_boxes;
    size_t exclusion_max_attempts = 50;
};

}  // namespace minion

#endif
