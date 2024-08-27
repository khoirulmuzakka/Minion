#ifndef LSHADE_H
#define LSHADE_H

#include "de.h"

/**
 * @class LSHADE 
 * @brief Class implementing the LSHADE algorithm.
 * Reference : R. Tanabe and A. S. Fukunaga, "Improving the search performance of SHADE using linear population size reduction," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, China, 2014, pp. 1658-1665, doi: 10.1109/CEC.2014.6900380.
 * 
 * The LSHADE class is an extension of the Differential Evolution algorithm 
 * with mechanisms for self-adaptation of control parameters.
 */
class LSHADE : public Differential_Evolution {
    public:
        std::vector<double> M_CR, M_F;
        size_t memorySize;
        LSHADE_Settings settings;

    private : 
        size_t memoryIndex=0;
        double archive_size_ratio;
        size_t minPopSize;
        std::string reduction_strategy;
        bool popreduce;

    public :
        /**
         * @brief Constructor for LSHADE.
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
        LSHADE(
            MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::map<std::string, ConfigValue>& options, 
                    const std::vector<double>& x0 = {}, void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
                    double tol = 0.0001, size_t maxevals = 100000, std::string boundStrategy = "reflect-random",  int seed=-1, 
                    size_t populationSize=30
        );

        /**
         * @brief Adapts parameters of the LSHADE algorithm.
         * 
         * This function overrides the adaptParameters function in the Differential_Evolution class.
         */
        void adaptParameters() override;
};

#endif