#ifndef jSO_H
#define jSO_H

#include "de.h"

namespace minion {

/**
 * @class jSO 
 * @brief Class implementing the jSO algorithm.
 * 
 * The jSO class is an extension of the Differential Evolution algorithm 
 * with mechanisms for self-adaptation of control parameters.
 */
class jSO : public Differential_Evolution {
    public:
        std::vector<double> M_CR, M_F;
        size_t memorySize;

    private : 
        size_t memoryIndex=0;
        double archive_size_ratio;
        size_t minPopSize;
        std::string reduction_strategy;
        bool popreduce;

    public :
        /**
         * @brief Constructor for jSO.
         * 
         * @param func The objective function to minimize.
         * @param bounds The bounds for the variables.
         * @param options A map of configuration options.
         * @param x0 The initial solution.
         * @param data Additional data for the objective function.
         * @param callback Callback function for intermediate results.
         * @param tol The tolerance for stopping criteria.
         * @param maxevals The maximum number of evaluations.
         * @param seed The seed for random number generation.
         * @param options Option map that specifies further configurational settings for the algorithm.
         */
        jSO(
            MinionFunction func, 
            const std::vector<std::pair<double, double>>& bounds, 
            const std::vector<double>& x0 = {},
            void* data = nullptr, 
            std::function<void(MinionResult*)> callback = nullptr,
            double tol = 0.0001, 
            size_t maxevals = 10000, 
            int seed=-1, 
            std::map<std::string, std::any> options = std::map<std::string, std::any>()
        ) :  
            Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options){};

        /**
         * @brief Adapts parameters of the jSO algorithm.
         * 
         * This function overrides the adaptParameters function in the Differential_Evolution class.
         */
        void adaptParameters() override;

        /**
         * @brief Initialize the algorithm given the input settings.
         */
        void initialize  () override;
};

}

#endif