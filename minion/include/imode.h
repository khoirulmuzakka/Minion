#ifndef IMODE_H
#define IMODE_H

#include "de.h"

namespace minion {

/**
 * @class IMODE
 * @brief Improved Multi-Operator Differential Evolution.
 *
 * Reference:
 * K. Sallam, "Improved Multi-Operator Differential Evolution Algorithm (IMODE)."
 *
 * IMODE mixes multiple mutation operators with success-history based
 * parameter adaptation and linear population size reduction.
 */
class IMODE : public Differential_Evolution {
public:
    IMODE(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
    ) :
        Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

    void initialize() override;
    void adaptParameters() override;
        void doDE_operation(std::vector<std::vector<double>>& trials) override;

    protected:
        void postEvaluation(const std::vector<std::vector<double>>&, const std::vector<double>&) override;
        void onBestUpdated(const std::vector<double>& candidate, double fitnessValue, bool improved) override;

    private:
        size_t memorySize = 0;
        std::vector<double> memoryF;
        std::vector<double> memoryCR;
    size_t memoryIndex = 0;
    double archive_size_ratio = 2.6;
        size_t minPopulationSize = 4;
        size_t initialPopulationSize = 0;
        std::vector<double> operatorProbabilities = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        std::vector<int> operatorAssignment;
        double probLocalSearch = 0.1;
        double localSearchStartFraction = 0.85;
        double localSearchBudgetFraction = 0.02;
        size_t lastLocalSearchEval = 0;
        bool pendingLocalSearchAttempt = false;

        void reducePopulation();
        void trimArchive();
        void sortPopulationByFitness();
        void updateOperatorProbabilities(const std::vector<double>& rewards);
    void updateParameterMemory(const std::vector<double>& goodF,
                               const std::vector<double>& goodCR,
                               const std::vector<double>& improvement);
    double sampleScalingFactor(double meanF) const;
        double sampleCrossover(double meanCR) const;
        void applyHanBoundary(std::vector<std::vector<double>>& mutants,
                              const std::vector<std::vector<double>>& parents);
        void maybeRunLocalSearch();
        bool shouldRunLocalSearch() const;
        bool runLocalSearch();
};

} // namespace minion

#endif
