#ifndef ADAPTIVE_NELDER_MEAD_H
#define ADAPTIVE_NELDER_MEAD_H

#include "minimizer_base.h"

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
     * @brief Constructor for NelderMead.
     *
     * @param func The objective function to minimize.
     * @param bounds The bounds (constraints) for each variable of the function.
     * @param x0 The initial guess for the minimizer.
     * @param data Additional data passed to the objective function.
     * @param callback A callback function to monitor or handle optimization progress.
     * @param relTol Relative tolerance for convergence criteria.
     * @param maxevals Maximum number of function evaluations allowed.
     * @param boundStrategy Strategy for handling boundary constraints ("reflect-random" by default).
     * @param seed Seed for the random number generator (default is -1, which means use a random seed).
     */
    NelderMead(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0 = {},
                       void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                       double relTol = 0.0001, int maxevals = 100000, std::string boundStrategy = "reflect-random", int seed=-1);

    /**
     * @brief Performs optimization using the Nelder-Mead method.
     *
     * @return MinionResult containing the optimized point, function value, and optimization statistics.
     */
    MinionResult optimize() override;

private : 
    std::vector<std::vector<double>> xtemp;
    std::vector<double> best; 
    double fbest; 
    int no_improve_counter=0;
    size_t bestIndex;
};

#endif // ADAPTIVE_NELDER_MEAD_H
