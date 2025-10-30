#ifndef ARRDE_H
#define ARRDE_H

#include <algorithm>
#include <cmath>
#include <iostream>

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
        std::vector<double> fitness_records, archive_fitness_records;
        std::vector<double> MCR_records, MF_records;
        std::vector<std::vector<double>> first_run_archive;
        std::vector<double> first_run_archive_fitness;
        double memorySizeRatio=2.0;
        int minPopSize= 4;
        static constexpr size_t archiveRecordMaxSize = 1000;
        size_t first_run_archive_max_size = 0;

    private : 
        double archive_size_ratio;
        size_t memoryIndex=0;
        size_t Neval_stratrefine=0;
        std::string reduction_strategy;
        bool popreduce;
        bool do_refine=false;

        bool refine = false;
        bool restart = false;
        bool first_run = true;
        bool population_converged = false;
        double bestOverall =  std::numeric_limits<double>::max();
        double decrease=0.9;
        double reltol;
        double restartRelTol;
        double refineRelTol;
        size_t Nrestart=1; //nitially set to 1, first run is consedered a restart
        std::vector<std::vector<std::pair<double, double>>> locals; 
        bool update_records = false;
        double maxRestart =2;

        size_t newPopulationSize = 0;
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
        bool checkOutsideLocals(double x, const std::vector<std::pair<double, double>>& local);

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

        void adjustPopulationSize();
        void adjustArchiveSize();
        void processRestartCycle();
        void executeRestart(size_t targetSize);
        void executeRefine(size_t targetSize);
        void updateParameterMemory();
        void resampleControlParameters();
        void addToFirstRunArchive(const std::vector<double>& candidate, double fitnessValue);


    public :

         /**
         * @brief Constructor 
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
        ARRDE(
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

    protected:
        void onBestUpdated(const std::vector<double>& candidate, double fitnessValue, bool improved) override;
};

}
#endif
