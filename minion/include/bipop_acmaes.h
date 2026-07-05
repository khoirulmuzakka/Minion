#ifndef BIPOP_ACMAES_H
#define BIPOP_ACMAES_H

#include "cmaes_base.h"

#include <Eigen/Dense>

#include <limits>
#include <string>
#include <vector>

namespace minion {

/**
 * @class BIPOP_aCMAES
 * @brief BIPOP restart strategy combined with the standard CMA-ES covariance update.
 *
 * Reference: Nikolaus Hansen. 2009. Benchmarking a BI-population CMA-ES on the
 * BBOB-2009 function testbed. In Proceedings of the 11th Annual Conference
 * Companion on Genetic and Evolutionary Computation Conference: Late Breaking
 * Papers (GECCO '09). Association for Computing Machinery, New York, NY, USA,
 * 2389-2396. https://doi.org/10.1145/1570256.1570333
 */
class BIPOP_aCMAES : public CMAESBase {
public:
    BIPOP_aCMAES(
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
    std::vector<double> eigenToStd(const Eigen::VectorXd& vec) const;
    Eigen::VectorXd sampleRandomMean() const;
    void configureRegime(const Eigen::Ref<const Eigen::VectorXd>& startMean, double startSigma, size_t lambdaValue);
    size_t runRegime(const Eigen::Ref<const Eigen::VectorXd>& startMean, double startSigma, size_t lambdaValue);

    size_t lambda0 = 0;

    double sigma0 = 0.0;
    double minRelStep = 1e-8;

    Eigen::VectorXd initialMean;

    size_t globalGeneration = 0;
};

}  // namespace minion

#endif
