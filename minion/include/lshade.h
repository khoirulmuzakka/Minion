#ifndef LSHADE_H
#define LSHADE_H

#include "de.h"

namespace minion {
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
         * @param x0 The initial guesses for the solution. Note that Minion assumes multiple initial guesses, thus, x0 is an std::vector<std::vector<double>> object. These guesses will be used for population initialization.
         * @param data Additional data for the objective function.
         * @param callback Callback function for intermediate results.
         * @param tol The tolerance for stopping criteria.
         * @param maxevals The maximum number of evaluations.
         * @param seed The seed for random number generation.
         * @param options Option map that specifies further configurational settings for the algorithm.
         */
        LSHADE(
            MinionFunction func, 
            const std::vector<std::pair<double, double>>& bounds, 
            const std::vector<std::vector<double>>& x0 = {},
            void* data = nullptr, 
            std::function<void(MinionResult*)> callback = nullptr,
            double tol = 0.0001, 
            size_t maxevals = 100000, 
            int seed=-1, 
            std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
        ) :  
            Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options){};

        /**
         * @brief Adapts parameters of the LSHADE algorithm.
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