#ifndef PSO_H
#define PSO_H

#include "minimizer_base.h"
#include "default_options.h"
#include <limits>

namespace minion {

/**
 * @class PSO
 * @brief Basic particle swarm optimizer that serves as the foundation for other PSO variants.
 */
class PSO : public MinimizerBase {
public:
    std::vector<std::vector<double>> population;
    std::vector<std::vector<double>> velocities;
    std::vector<std::vector<double>> personalBestPositions;
    std::vector<double> personalBestFitness;
    std::vector<double> fitness;
    std::vector<double> best;
    double best_fitness = std::numeric_limits<double>::infinity();
    size_t populationSize = 0;
    size_t Nevals = 0;

    double inertiaWeight = 0.7;
    double cognitiveCoeff = 1.5;
    double socialCoeff = 1.5;
    double velocityClamp = 0.2;
    bool useLatin = true;
    bool support_tol = true;

    std::vector<double> diversity;
    std::vector<double> spatialDiversity;

protected:
    virtual void init();
    virtual void updateVelocitiesAndPositions();
    virtual void recordMetrics();
    virtual bool checkStopping() const;

    void configureFromOptions(const Options& options);

public:
    /**
     * @brief Construct a PSO minimizer with the standard global-best topology.
     *
     * @param func Objective function to minimize.
     * @param bounds Search-space bounds for each variable.
     * @param x0 Optional initial particle locations.
     * @param data User data forwarded to the objective.
     * @param callback Per-iteration callback invoked with the current best.
     * @param tol Relative tolerance used in the diversity-based stop test.
     * @param maxevals Maximum number of objective evaluations.
     * @param seed RNG seed (negative -> random seed).
     * @param options Configuration map (inertia weight, coefficients, etc.).
     */
    PSO(
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
        MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

    void initialize() override;
    MinionResult optimize() override;
};

}

#endif
