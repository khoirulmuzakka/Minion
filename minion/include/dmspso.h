#ifndef DMSPSO_H
#define DMSPSO_H

#include "pso.h"

namespace minion {

/**
 * @class DMSPSO
 * @brief Dynamic Multi-Swarm Particle Swarm Optimization built on top of the base PSO implementation.
 */
class DMSPSO : public PSO {
public:
    /**
     * @brief Construct the dynamic multi-swarm PSO variant.
     *
     * @param func Objective function to minimize.
     * @param bounds Search-space bounds.
     * @param x0 Optional set of initial particles.
     * @param data User payload forwarded to the objective.
     * @param callback Per-iteration callback receiving the best solution.
     * @param tol Diversity tolerance used by the base stop criterion.
     * @param maxevals Maximum number of objective evaluations.
     * @param seed RNG seed (negative -> random seed).
     * @param options Configuration map (sub-swarm count, regroup period, etc.).
     */
    DMSPSO(
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

private:
    size_t subswarmCount = 4;
    size_t regroupPeriod = 5;
    double localCoefficient = 1.4;
    double globalCoefficient = 0.8;

    size_t iterationCounter = 0;

    std::vector<std::vector<size_t>> subswarms;
    std::vector<size_t> subswarmBestIndices;
    std::vector<size_t> particleToSubswarm;

    void rebuildSubswarmAssignments();
    void updateSubswarmBests();
};

}

#endif
