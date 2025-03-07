#ifndef ARRDE_H
#define ARRDE_H

#include "de.h"
#include "utility.h"

namespace minion {

/**
 * @class ARRDE : Adaptive Restart-Refine - Differential Evolution 
 * @brief Class implementing the ARRDE algorithm.
 * 
 */
class ARRDE : public Differential_Evolution {
    public:
        std::vector<double> M_CR, M_F;
        size_t memorySize=50;
        std::vector<std::vector<double>> population_records, archive_records;
        std::vector<double> fitness_records;
        std::vector<double> MCR_records, MF_records;
        double memorySizeRatio=2.0;
        int minPopSize= 4;

    private : 
        double archive_size_ratio;
        size_t memoryIndex=0;
        size_t Neval_stratrefine=0;
        std::string reduction_strategy;
        bool popreduce;

        bool refine = false;
        bool restart = false;
        bool final_refine = false;
        bool first_run = true;
        double bestOverall =  std::numeric_limits<double>::max();
        double decrease=0.9;
        double reltol;
        double restartRelTol;
        double refineRelTol;
        double strartRefine=0.8;
        size_t Nrestart=1; //nitially set to 1, first run is consederee a restart
        std::vector<std::vector<std::pair<double, double>>> locals; 
        bool update_records = false;
        double maxRestart =1;
        size_t maxPopSize_finalRefine;
        bool init_final_refine=false;

    private :

        /**
         * @brief Checks if a value is between two bounds.
         * @param x The value to check.
         * @param low The lower bound.
         * @param high The upper bound.
         * @return True if x is between low and high, false otherwise.
         */
        bool checkIsBetween(double x, double low, double high);

        /**
         * @brief Checks if a value is outside of all given local intervals.
         * @param x The value to check.
         * @param local The list of local intervals.
         * @return True if x is outside all intervals, false otherwise.
         */
        bool checkOutsideLocals(double x, std::vector<std::pair<double, double>> local);

        /**
         * @brief Merges overlapping intervals.
         * @param intervals The list of intervals to merge.
         * @return A list of merged intervals.
         */
        std::vector<std::pair<double, double>> merge_intervals(const std::vector<std::pair<double, double>>& intervals);

        /**
         * @brief Merges overlapping intervals for multiple variables.
         * @param intervals The list of interval lists to merge.
         * @return A list of merged interval lists.
         */
        std::vector<std::vector<std::pair<double, double>>> merge_intervals(std::vector<std::vector<std::pair<double, double>>>& intervals);

        /**
         * @brief Samples a value outside of the given local bounds.
         * @param low The lower bound of the entire range.
         * @param high The upper bound of the entire range.
         * @param local_bounds The local bounds to avoid.
         * @return A sampled value outside of the local bounds.
         */
        double sample_outside_local_bounds(double low, double high, const std::vector<std::pair<double, double>>& local_bounds);

        /**
         * @brief Applies local constraints to a given solution.
         * @param p The solution to constrain.
         * @return The constrained solution.
         */
        std::vector<double> applyLocalConstraints(const std::vector<double>& p);

        /**
         * @brief Updates the local constraints based on the current population.
         */
        void update_locals();

        /**
         * @brief Remove element from a avector
         */
        void removeElement(std::vector<size_t>& vec, size_t x);

    public :

         /**
         * @brief Constructor 
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
        ARRDE(
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
            Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options){support_tol=false;};

        /**
         * @brief Adapts parameters of the algorithm.
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