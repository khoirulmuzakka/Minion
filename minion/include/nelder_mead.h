#ifndef ADAPTIVE_NELDER_MEAD_H
#define ADAPTIVE_NELDER_MEAD_H

#include "minimizer_base.h"

namespace minion {
/**
 * @class NelderMead
 * @brief Implements the Nelder-Mead optimization algorithm.
 *
 * This class derives from MinimizerBase and implements the Nelder-Mead method
 * for function optimization over a bounded domain.
 */
class NelderMead : public MinimizerBase {
public:
    
    /**
     * @brief Constructor for Differential_Evolution.
     * @param func The objective function to minimize.
     * @param bounds The bounds for the variables.
     * @param x0 The initial solution.
     * @param data Additional data for the objective function.
     * @param callback Callback function for intermediate results.
     * @param tol The tolerance for stopping criteria.
     * @param maxevals The maximum number of evaluations.
     * @param seed The seed for random number generation.
     * @param options Option map that specifies further configurational settings for the algorithm.
     */
    NelderMead(
        MinionFunction func, 
        const std::vector<std::pair<double, double>>& bounds, 
        const std::vector<double>& x0 = {},
        void* data = nullptr, 
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001, 
        size_t maxevals = 100000, 
        int seed=-1, 
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
        ) :  
        MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, seed, options){};

    /**
     * @brief Performs optimization using the Nelder-Mead method.
     *
     * @return MinionResult containing the optimized point, function value, and optimization statistics.
     */
    MinionResult optimize() override;

    /**
     * @brief Initialize the algorithm given the input settings.
     */
    void initialize  () override;

private : 
    std::vector<std::vector<double>> xtemp;
    std::vector<double> best; 
    double fbest; 
    int no_improve_counter=0;
    size_t bestIndex;
};

}

#endif // ADAPTIVE_NELDER_MEAD_H
