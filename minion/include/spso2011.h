#ifndef SPSO2011_H
#define SPSO2011_H

#include "pso.h"

namespace minion {

/**
 * @class SPSO2011
 * @brief Implementation of the stochastic PSO 2011 variant built on top of the base PSO.
 */
class SPSO2011 : public PSO {
public:
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

private:
    double phiPersonal = 2.05;
    double phiSocial = 2.05;
    double constriction = 0.72984;
    size_t neighborhoodSize = 3;

    std::vector<std::vector<size_t>> neighborhoods;
    std::vector<size_t> neighborhoodBestIndices;

    void rebuildNeighborhoods();
    void updateNeighborhoodBests();
};

}

#endif
