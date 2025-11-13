#ifndef AGSK_H
#define AGSK_H

#include "de.h"
#include <array>
#include <tuple>

namespace minion {

/**
 * @class AGSK
 * @brief Adaptive Gaining-Sharing Knowledge-based algorithm (AGSK).
 *
 * Reference:
 * Ali W. Mohamed, Anas A. Hadi, Ali K. Mohamed, and Noor H. Awad,
 * "Evaluation of Adaptive Gaining-Sharing Knowledge Based Algorithm on CEC 2020,"
 * IEEE CEC 2020.
 */
class AGSK : public Differential_Evolution {
public:
    AGSK(
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
    void postEvaluation(const std::vector<std::vector<double>>&, const std::vector<double>&) override;

private:
    void initializeKnowledgeParameters();
    void ensurePopulationReduction();
    void updateKnowledgeWeights();
    void assignKnowledgeControls();

    std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
    generateJuniorTriplets(const std::vector<size_t>& sorted_indices) const;

    std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
    generateSeniorTriplets(const std::vector<size_t>& sorted_indices) const;

    std::vector<size_t> sampleFromPool(size_t popSize, const std::vector<size_t>& pool) const;

private:
    size_t minPopulationSize = 12;
    size_t maxPopulationSize = 0;
    std::vector<double> knowledgeParameterK;
    std::vector<double> KF_values;
    std::vector<double> KR_values;
    std::vector<int> knowledgeAssignment;
    std::array<double, 4> knowledgeWeights = {0.85, 0.05, 0.05, 0.05};
    std::array<double, 4> improvementShare = {0.25, 0.25, 0.25, 0.25};
    const std::array<double, 4> KF_pool = {0.1, 1.0, 0.5, 1.0};
    const std::array<double, 4> KR_pool = {0.2, 0.1, 0.9, 0.9};
};

}

#endif
