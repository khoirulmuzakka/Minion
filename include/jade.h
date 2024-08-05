#ifndef JADE_H
#define JADE_H

#include "de.h"

/**
 * @class JADE 
 * @brief Class implementing the JADE algorithm.
 * 
 * The JADE class is an extension of the Differential Evolution algorithm 
 * with mechanisms for self-adaptation of control parameters.
 */
class JADE : public Differential_Evolution {
    public:
        JADE_Settings settings;
        double c=0.0;
        double muCR=0.5; 
        double muF=0.5;

    private : 
        double archive_size_ratio;
        size_t minPopSize;
        std::string reduction_strategy;
        bool popreduce;

    public :
        /**
         * @brief Constructor for JADE.
         * 
         * @param func The objective function to minimize.
         * @param bounds The bounds for the variables.
         * @param options A map of configuration options.
         * @param x0 The initial solution.
         * @param data Additional data for the objective function.
         * @param callback Callback function for intermediate results.
         * @param tol The tolerance for stopping criteria.
         * @param maxevals The maximum number of evaluations.
         * @param boundStrategy The strategy for handling bounds.
         * @param seed The seed for random number generation.
         * @param populationSize The size of the population.
         */
        JADE(
            MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::map<std::string, ConfigValue>& options, 
                    const std::vector<double>& x0 = {}, void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                    double tol = 0.0001, size_t maxevals = 100000, std::string boundStrategy = "reflect-random",  int seed=-1, 
                    size_t populationSize=30
        );

        /**
         * @brief Adapts parameters of the JADE algorithm.
         * 
         * This function overrides the adaptParameters function in the Differential_Evolution class.
         */
        void adaptParameters() override;
};

#endif