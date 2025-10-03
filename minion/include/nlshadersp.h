#ifndef NLSHADERSP_H
#define NLSHADERSP_H

#include "de.h"

namespace minion {

/**
 * @file nlshader_sp.h
 * @brief Header file for the NLSHADE_RSP class, which implements a minimization algorithm.
 * 
 * My own implementation of the NLSHADE_RSP algorithm.
 * Reference : V. Stanovov, S. Akhmedova and E. Semenkin, "NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for CEC 2021 Numerical Optimization," 2021 IEEE Congress on Evolutionary Computation (CEC), Kraków, Poland, 2021, pp. 809-816, doi: 10.1109/CEC45853.2021.9504959.
 */


/**
 * @class NLSHADE_RSP
 * @brief A class implementing the NLSHADE_RSP optimization algorithm.
 * 
 * The NLSHADE_RSP class inherits from MinimizerBase and provides functionality to perform
 * optimization using a modified version of the L-SHADE algorithm with an archive and a
 * memory mechanism to store previous successes.
 */
class NLSHADE_RSP : public Differential_Evolution {
public:
    NLSHADE_RSP(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>())
        : Differential_Evolution(func, bounds, x0, data, callback, 0.0, maxevals, seed, options) {
        support_tol = true;
    }

    void initialize() override;
    MinionResult optimize() override;

private:
    void recordSuccess(double cr, double f, double delta);
    void updateParameterMemory();
    double weightedLehmerMean(const std::vector<double>& values, const std::vector<double>& weights, double gp, double gm) const;
    void updateArchive(const std::vector<double>& parent);
    void adjustPopulation(size_t newSize);
    void trimArchive(size_t capacity);

    int memorySize = 0;
    size_t memoryIndex = 0;
    double archiveSizeRatio = 0.0;
    size_t minPopulationSize = 4;
    size_t maxPopulationSize = 0;
    double archiveProbability = 0.5;
    size_t archiveCapacity = 0;

    std::vector<double> parameterMemoryCR;
    std::vector<double> parameterMemoryF;
    std::vector<double> successfulCR;
    std::vector<double> successfulF;
    std::vector<double> successDelta;
};

}

#endif
