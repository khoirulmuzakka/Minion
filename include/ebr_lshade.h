#ifndef EBR_LSHADE_H
#define EBR_LSHADE_H

#include "de_base.h"

/**
 * @class EBR_LSHADE : Exclusion-Based Restart LSHADE algorithm
 * @brief Class implementing the EBR_LSHADE algorithm.
 */
class EBR_LSHADE : public DE_Base {
    public:
        size_t memorySize, max_restarts;
        std::vector<double> M_CR, M_F;
        std::vector<double> muCR, muF, stdCR, stdF;
        std::vector<std::vector<std::vector<double>>> population_records;
        std::vector<std::vector<double>> archive_records;
        std::vector<std::vector<double>> fitness_records;
        std::vector<std::vector<std::pair<double, double>>> locals; 
        double strartRefine=0.75;
        size_t maxPopSize;
        double relTol_firstRun=0.01;

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

        
    public :

         /**
         * @brief Constructs the EBR_LSHADE object.
         * @param func The objective function to optimize.
         * @param bounds The bounds for each variable.
         * @param data Optional data to pass to the objective function.
         * @param x0 Optional initial solution.
         * @param population_size Initial population size.
         * @param maxevals Maximum number of evaluations.
         * @param relTol_firstRun Relative tolerance for the first run.
         * @param minPopSize Minimum population size.
         * @param memorySize Size of the memory for storing CR and F values.
         * @param callback Optional callback function for intermediate results.
         * @param max_restarts Maximum number of restarts.
         * @param startRefine Fraction of evaluations after which to start refinement.
         * @param boundStrategy Strategy for handling bounds.
         * @param seed Random seed.
         */
        EBR_LSHADE(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, void* data = nullptr, 
                    const std::vector<double>& x0 = {}, size_t population_size = 0, int maxevals = 100000, double relTol_firstRun = 0.01, size_t minPopSize = 5, 
                     size_t memorySize=50, std::function<void(MinionResult*)> callback = nullptr, size_t max_restarts=0, 
                     double startRefine=0.75, std::string boundStrategy = "reflect-random", int seed = -1);

        /**
         * @brief Adapts the algorithm parameters (CR and F) based on the memory.
         */
        void _adapt_parameters();

        /**
         * @brief update Archive from the archive records
         */
        void updateArchive(size_t size);

        /**
         * @brief Initializes the population using Latin Hypercube Sampling.
         */
        void _initialize_population() override;

        /**
         * @brief Performs the search process of the algorithm.
         * @param refine Flag indicating if the search is in the refinement phase.
         * @param firstRun Flag indicating if this is the first run of the algorithm.
         */
        void _do_search(bool refine=false, bool firstRun=false);

        /**
         * @brief Performs the refinement phase of the algorithm.
         */
        void _do_refinement();
        
        /**
         * @brief Runs the optimization process.
         * @return The best result found by the algorithm.
         */
        MinionResult optimize() override;
};


#endif