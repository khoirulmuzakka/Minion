#ifndef DE_BASE_H 
#define DE_BASE_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <functional>
#include "utility.h" 
#include "minimizer_base.h"

/**
 * @class DE_Base
 * @brief A base class for Differential Evolution (DE)-based optimization algorithms.
 */
class DE_Base : public MinimizerBase {
    public:
        size_t original_popsize;
        size_t popsize;
        size_t minPopSize;
        bool popDecrease;
        size_t maxiter;
        size_t Nevals;
        double rangeScale;
        std::string strategy;
        std::vector<std::vector<double>> population;
        std::vector<double> fitness;
        size_t best_idx;
        std::vector<double> best;
        double best_fitness;
        bool use_clip;
        std::vector<double> F, CR;

    public:
        /**
         * @brief Constructor for DE_Base.
         * @param func The objective function to minimize.
         * @param bounds The bounds for the decision variables.
         * @param data Additional data to pass to the objective function.
         * @param x0 The initial guess for the solution.
         * @param population_size The population size.
         * @param maxevals The maximum number of function evaluations.
         * @param strategy The DE strategy to use.
         * @param relTol The relative tolerance for convergence.
         * @param minPopSize The minimum population size.
         * @param callback A callback function to call after each iteration.
         * @param boundStrategy Strategy when bounds are violated. Available strategy : "random", "reflect", "reflect-random", "clip".
         * @param seed The seed for the random number generator.
         */
        DE_Base(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, void* data = nullptr, 
                const std::vector<double>& x0 = {}, int population_size = 20, int maxevals = 1000000,
                std::string strategy = "current_to_pbest1bin", double relTol = 0.0001, int minPopSize = 10,
                std::function<void(MinionResult*)> callback = nullptr, std::string boundStrategy = "reflect-random", int seed = -1);
        
        /**
         * @brief Destructor for DE_Base.
         */
        ~DE_Base(){
            for (auto minRes : history){
                delete minRes;
            };
        };

        /**
         * @brief Get the maximum number of iterations based on the current settings.
         * @return The maximum number of iterations.
         */
        size_t getMaxIter();

        /**
         * @brief Initialize the population of candidate solutions.
         */
        void _initialize_population();

        /**
         * @brief Perform mutation on a candidate solution.
         * @param idx The index of the candidate solution to mutate.
         * @return The mutant vector.
         */
        std::vector<double> _mutate(int idx);

        /**
         * @brief Perform binomial crossover between a target and mutant vector.
         * @param target The target vector.
         * @param mutant The mutant vector.
         * @param CR The crossover probability.
         * @return The trial vector after crossover.
         */
        std::vector<double> _crossover_bin(const std::vector<double>& target, const std::vector<double>& mutant, double CR);

        /**
         * @brief Perform exponential crossover between a target and mutant vector.
         * @param target The target vector.
         * @param mutant The mutant vector.
         * @param CR The crossover probability.
         * @return The trial vector after crossover.
         */
        std::vector<double> _crossover_exp(const std::vector<double>& target, const std::vector<double>& mutant, double CR);

        /**
         * @brief Perform crossover between a target and mutant vector using the selected strategy.
         * @param target The target vector.
         * @param mutant The mutant vector.
         * @param CR The crossover probability.
         * @return The trial vector after crossover.
         */
        std::vector<double> _crossover(const std::vector<double>& target, const std::vector<double>& mutant, double CR);    
};


#endif