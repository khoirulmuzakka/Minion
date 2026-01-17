#ifndef ALGOTEST_H
#define ALGOTEST_H

#include "de.h"

namespace minion {

/**
 * @class NJADE
 * @brief Plain DE with linear population reduction.
 */
class NJADE : public Differential_Evolution {
  private:
    size_t minPopSize = 4;
    bool popreduce = false;
    size_t memorySize = 5;
    size_t memoryIndex = 0;
    std::vector<double> M_CR;
    std::vector<double> M_F;
    std::vector<double> compare_fitness;
    double pe = 1.0;

  public:
    NJADE(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>())
        : Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

    void initialize() override;
    void adaptParameters() override;
    MinionResult optimize() override;
};

}  // namespace minion

#endif
