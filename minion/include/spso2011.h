#ifndef SPSO2011_H
#define SPSO2011_H

#include "pso.h"
#include <cmath>
#include <limits>

namespace minion {

/**
 * @class SPSO2011
 * @brief Implementation of the stochastic PSO 2011 variant built on top of the base PSO.
 */
class SPSO2011 : public PSO {
public:
    /**
     * @brief Construct the SPSO2011 variant.
     *
     * @param func Objective function to minimize.
     * @param bounds Search-space bounds.
     * @param x0 Optional initial swarm positions.
     * @param data User payload for the objective.
     * @param callback Progress callback invoked with the best-so-far state.
     * @param tol Diversity tolerance that governs early stopping.
     * @param maxevals Maximum number of objective evaluations.
     * @param seed RNG seed (negative -> random seed).
     * @param options Configuration map (phi values, neighbourhood size, etc.).
     */
    SPSO2011(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
    ) :
        PSO(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

    void initialize() override;

protected:
    void init() override;
    void updateVelocitiesAndPositions() override;

public:
    MinionResult optimize() override;

private:
    double c1 = 0.5 + std::log(2.0);
    double c2 = 0.5 + std::log(2.0);
    double inertia = 1.0 / (2.0 * std::log(2.0));
    size_t informantDegree = 3;
    bool normalizeSpace = false;

    std::vector<std::vector<size_t>> informants;
    bool topologyDirty = true;
    size_t stagnationCounter = 0;
    double lastBestFitness = std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>> normPositions;
    std::vector<std::vector<double>> normPersonalBest;

    void randomizeInformants();
    std::vector<double> samplePointInSphere(const std::vector<double>& center, double radius) const;
};

}

#endif
