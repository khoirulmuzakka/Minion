#ifndef ABC_H
#define ABC_H

#include "minimizer_base.h"
#include "default_options.h"

namespace minion {
/**
 * @class ABC
 * @brief A class for performing Artificial Bee Colony (ABC) optimization.
 * 
 * This class implements the ABC algorithm for optimization. It inherits from the MinimizerBase class.
 */
class ABC : public MinimizerBase {
public:
    std::vector<std::vector<double>> population;
    std::vector<double> fitness;
    std::vector<double> best;
    double best_fitness;
    size_t populationSize;
    size_t Nevals = 0;
    size_t limit = 100;

protected:
    /**
     * @brief Initializes the population and other parameters.
     */
    virtual void init();

public:
    /**
     * @brief Constructor for ABC.
     * @param func The objective function to minimize.
     * @param bounds The bounds for the variables.
     * @param x0 The initial solution. Note that Minion assumes multiple initial guesses, thus, x0 is an std::vector<std::vector<double>> object. These guesses will be used for population initialization.
     * @param data Additional data for the objective function.
     * @param callback Callback function for intermediate results.
     * @param tol The tolerance for stopping criteria.
     * @param maxevals The maximum number of evaluations.
     * @param seed The seed for random number generation.
     * @param options Option map that specifies further configurational settings for the algorithm.
     */
    ABC(
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

    /**
     * @brief Optimizes the objective function.
     * @return The result of the optimization.
     */
    MinionResult optimize() override;

    /**
     * @brief Initialize the algorithm given the input settings.
     */
    void initialize() override;

private:
    std::vector<size_t> trialCounters;
};

}

#endif
