#ifndef LSRTDE_H
#define LSRTDE_H

#include "de.h"

namespace minion {

/**
 * @file lsrtde.h
 * @brief Header file for the LSRTDE class, which implements a minimization algorithm.
 * 
 * Implementation of the LSRTDE algorithm.
 * Reference : V. Stanovov and E. Semenkin, "Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition," 2024 IEEE Congress on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: 10.1109/CEC60901.2024.10611907.
 */


/**
 * @class LSRTDE
 * @brief A class implementing the LSRTDE optimization algorithm.
 * 
 * The LSRTDE class inherits from MinimizerBase and provides functionality to perform
 * optimization using a modified version of the L-SHADE algorithm with an archive and a
 * memory mechanism to store previous successes.
 */
class LSRTDE : public Differential_Evolution {
public:
    LSRTDE(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>())
        : Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options) {
        support_tol = true;
    }

    void initialize() override;
    MinionResult optimize() override;

private:
    void setup_population();
    void update_memory_cr(const std::vector<double>& successCR, const std::vector<double>& deltas);
    double weighted_lehmer(const std::vector<double>& values, const std::vector<double>& weights) const;

    size_t memorySize = 0;
    size_t memoryIndex = 0;
    double successRate = 0.5;

    size_t frontSize = 0;
    size_t frontSizeMax = 0;

    std::vector<std::vector<double>> bufferPopulation;
    std::vector<double> bufferFitness;
    std::vector<double> memoryCR;
    std::vector<double> successCRBuffer;
    std::vector<double> successDeltaBuffer;
};

}

#endif