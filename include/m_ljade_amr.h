#ifndef M_LJADE_AMR_H
#define M_LJADE_AMR_H

#include "de_base.h"

/**
 * @class M_LJADE_AMR
 * @brief Class implementing of the modified JADE with linear population size reduction with adaptive mutation rate (M-LJADE-AMR) algorithm.
 */
class M_LJADE_AMR : public DE_Base {
    public:
        double meanCR;
        double meanF;
        double c;
        std::vector<double> muCR, muF, stdCR, stdF;

    public :
        /**
         * @brief Constructor for M_LJADE_AMR.
         * @param func The objective function to minimize.
         * @param bounds The bounds for the decision variables.
         * @param data Additional data to pass to the objective function.
         * @param x0 The initial guess for the solution.
         * @param population_size The population size.
         * @param maxevals The maximum number of function evaluations.
         * @param strategy The DE strategy to use.
         * @param relTol The relative tolerance for convergence.
         * @param minPopSize The minimum population size.
         * @param c The control parameter for M-LJADE-AMR.
         * @param callback A callback function to call after each iteration.
         * @param boundStrategy Strategy when bounds are violated. Available strategy : "random", "reflect", "reflect-random", "clip".
         * @param seed The seed for the random number generator.
         */
        M_LJADE_AMR(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, void* data = nullptr, 
                    const std::vector<double>& x0 = {}, int population_size = 30, int maxevals = 100000, 
                    std::string strategy = "current_to_pbest1bin", double relTol = 0.00001, int minPopSize = 10, 
                    double c = 0.5, std::function<void(MinionResult*)> callback = nullptr, std::string boundStrategy = "reflect-random", int seed = -1);

        /**
         * @brief Method to adapt parameters during optimization.
         */
        void _adapt_parameters();

        /**
         * @brief Override of the optimize method from the base class.
         * @return The result of the optimization process.
         */
        MinionResult optimize() override;
};


#endif