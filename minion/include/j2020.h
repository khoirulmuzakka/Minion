#ifndef J2020_H
#define J2020_H

#include "de.h"
#include <limits>

namespace minion {

/**
 * @file j2020.h
 * @brief Header file for the j2020 class.
 *
 * Implementation of the j2020 algorithm.
 * Reference : J. Brest, M. S. Maučec and B. Bošković, "Differential Evolution Algorithm for Single Objective Bound-Constrained Optimization: Algorithm j2020," 2020 IEEE Congress on Evolutionary Computation (CEC), Glasgow, UK, 2020, pp. 1-8, doi: 10.1109/CEC48606.2020.9185551.
 */

/**
 * @class j2020
 * @brief A class implementing a differential evolution optimization algorithm.
 *
 * This class derives from the MinimizerBase class and implements the j2020 algorithm
 * for global optimization. It includes methods for initializing the population, 
 * computing distances, crowding mechanisms, and optimizing a given objective function.
 */
class j2020 : public Differential_Evolution {
public:
    j2020(
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
        support_tol = false;
    }

    void initialize() override;
    MinionResult optimize() override;

private:
    double distance_squared(const std::vector<double>& a, const std::vector<double>& b) const;
    size_t crowding_index(const std::vector<double>& candidate) const;
    bool too_many_equals(const std::vector<double>& costs, size_t count, double bestCost) const;
    void reinitialize_range(size_t start, size_t end, size_t skipIndex = std::numeric_limits<size_t>::max());

    size_t desiredPopulation = 0;
    size_t bigPopulationSize = 0;
    size_t smallPopulationSize = 0;

    double tau1 = 0.1;
    double tau2 = 0.1;
    double myEqs = 0.25;

    const double baseF = 0.5;
    const double baseCR = 0.9;
    const double eps = 1e-12;
    const double Fu = 1.1;

    long nReset = 0;
    long sReset = 0;
    long age = 0;
};

} // namespace minion

#endif
