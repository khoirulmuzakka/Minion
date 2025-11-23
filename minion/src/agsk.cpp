#include "agsk.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace minion {

void AGSK::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("AGSK");
    for (const auto& option : optionMap) {
        defaults[option.first] = option.second;
    }
    Options options(defaults);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    const std::vector<std::string> allowedBoundStrategies = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(allowedBoundStrategies.begin(), allowedBoundStrategies.end(), boundStrategy) == allowedBoundStrategies.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    const size_t dimension = bounds.size();
    const int configuredPopulation = options.get<int>("population_size", 0);
    if (configuredPopulation > 0) {
        populationSize = static_cast<size_t>(configuredPopulation);
    } else {
        populationSize = (dimension > 5) ? 40 * dimension : size_t(100);
    }

    minPopulationSize = static_cast<size_t>(std::max(4, options.get<int>("minimum_population_size", 12)));
    populationSize = std::max(populationSize, minPopulationSize);
    maxPopulationSize = populationSize;

    F = std::vector<double>(populationSize, 0.5);
    CR = std::vector<double>(populationSize, 0.9);
    p = std::vector<size_t>(populationSize, 1);

    KF_values = std::vector<double>(populationSize, KF_pool[0]);
    KR_values = std::vector<double>(populationSize, KR_pool[0]);
    knowledgeAssignment = std::vector<int>(populationSize, 0);

    initializeKnowledgeParameters();
    support_tol = false;
    hasInitialized = true;
}

void AGSK::initializeKnowledgeParameters() {
    knowledgeParameterK.resize(populationSize);
    for (size_t i = 0; i < populationSize; ++i) {
        if (rand_gen() < 0.5) {
            knowledgeParameterK[i] = rand_gen();
        } else {
            knowledgeParameterK[i] = std::ceil(20.0 * rand_gen());
        }
    }
}

void AGSK::adaptParameters() {
    ensurePopulationReduction();
    updateKnowledgeWeights();
    assignKnowledgeControls();
}

void AGSK::ensurePopulationReduction() {
    if (population.empty()) return;

    const double progress = (maxevals == 0) ? 1.0 : std::min(1.0, double(Nevals) / double(maxevals));
    double targetSize = double(minPopulationSize) - double(maxPopulationSize);
    targetSize *= std::pow(progress, 1.0 - progress);
    targetSize += double(maxPopulationSize);

    size_t planPopSize = static_cast<size_t>(std::round(targetSize));
    planPopSize = std::clamp(planPopSize, minPopulationSize, maxPopulationSize);

    if (population.size() <= planPopSize) return;

    auto sortedIndex = argsort(fitness, true);
    std::vector<std::vector<double>> trimmedPopulation(planPopSize);
    std::vector<double> trimmedFitness(planPopSize);
    std::vector<double> trimmedK(planPopSize);

    for (size_t i = 0; i < planPopSize; ++i) {
        const size_t idx = sortedIndex[i];
        trimmedPopulation[i] = population[idx];
        trimmedFitness[i] = fitness[idx];
        trimmedK[i] = knowledgeParameterK[idx];
    }

    population = std::move(trimmedPopulation);
    fitness = std::move(trimmedFitness);
    knowledgeParameterK = std::move(trimmedK);

    KF_values.assign(population.size(), KF_pool[0]);
    KR_values.assign(population.size(), KR_pool[0]);
    knowledgeAssignment.assign(population.size(), 0);
    F.assign(population.size(), 0.5);
    CR.assign(population.size(), 0.9);
    p.assign(population.size(), 1);
}

void AGSK::updateKnowledgeWeights() {
    const double progress = (maxevals == 0) ? 1.0 : std::min(1.0, double(Nevals) / double(maxevals));
    if (progress < 0.1) {
        knowledgeWeights = {0.85, 0.05, 0.05, 0.05};
        return;
    }

    for (size_t i = 0; i < knowledgeWeights.size(); ++i) {
        knowledgeWeights[i] = 0.95 * knowledgeWeights[i] + 0.05 * improvementShare[i];
    }

    double sum = std::accumulate(knowledgeWeights.begin(), knowledgeWeights.end(), 0.0);
    if (sum <= 0.0 || !std::isfinite(sum)) {
        knowledgeWeights = {0.25, 0.25, 0.25, 0.25};
        return;
    }
    for (double& value : knowledgeWeights) {
        value /= sum;
    }
}

void AGSK::assignKnowledgeControls() {
    const size_t popSize = population.size();
    if (popSize == 0) return;

    KF_values.resize(popSize);
    KR_values.resize(popSize);
    knowledgeAssignment.resize(popSize);

    std::array<double, 4> cumulative{};
    cumulative[0] = knowledgeWeights[0];
    for (size_t i = 1; i < cumulative.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + knowledgeWeights[i];
    }
    cumulative.back() = 1.0;

    for (size_t i = 0; i < popSize; ++i) {
        const double r = rand_gen();
        size_t idx = 0;
        while (idx + 1 < cumulative.size() && r > cumulative[idx]) {
            ++idx;
        }
        knowledgeAssignment[i] = static_cast<int>(idx);
        KF_values[i] = KF_pool[idx];
        KR_values[i] = KR_pool[idx];
    }
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
AGSK::generateJuniorTriplets(const std::vector<size_t>& sorted_indices) const {
    const size_t popSize = sorted_indices.size();
    std::vector<size_t> ranks(popSize);
    for (size_t pos = 0; pos < popSize; ++pos) {
        ranks[sorted_indices[pos]] = pos;
    }

    std::vector<size_t> r1(popSize), r2(popSize), r3(popSize);
    for (size_t i = 0; i < popSize; ++i) {
        const size_t rank = ranks[i];
        if (rank == 0) {
            r1[i] = sorted_indices[1 % popSize];
            r2[i] = sorted_indices[2 % popSize];
        } else if (rank == popSize - 1) {
            r1[i] = sorted_indices[popSize - 3];
            r2[i] = sorted_indices[popSize - 2];
        } else {
            r1[i] = sorted_indices[rank - 1];
            r2[i] = sorted_indices[rank + 1];
        }

        size_t candidate;
        size_t attempts = 0;
        do {
            candidate = rand_int(popSize);
            attempts++;
        } while ((candidate == i || candidate == r1[i] || candidate == r2[i]) && attempts < 1000);

        if (candidate == i || candidate == r1[i] || candidate == r2[i]) {
            candidate = sorted_indices[(rank + 2) % popSize];
        }
        r3[i] = candidate;
    }
    return {r1, r2, r3};
}

std::vector<size_t> AGSK::sampleFromPool(size_t popSize, const std::vector<size_t>& pool) const {
    std::vector<size_t> indices(popSize);
    const std::vector<size_t>* source = &pool;
    if (pool.empty()) {
        source = nullptr;
    }
    for (size_t i = 0; i < popSize; ++i) {
        if (source == nullptr) {
            indices[i] = rand_int(popSize);
        } else {
            indices[i] = (*source)[rand_int(source->size())];
        }
    }
    return indices;
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
AGSK::generateSeniorTriplets(const std::vector<size_t>& sorted_indices) const {
    const size_t popSize = sorted_indices.size();
    if (popSize == 0) {
        return {std::vector<size_t>(), std::vector<size_t>(), std::vector<size_t>()};
    }

    size_t bestCount = std::max<size_t>(1, static_cast<size_t>(std::round(popSize * 0.05)));
    bestCount = std::min(bestCount, popSize);
    size_t worstCount = std::max<size_t>(1, static_cast<size_t>(std::round(popSize * 0.05)));
    worstCount = std::min(worstCount, popSize - std::min(popSize, bestCount));

    if (bestCount + worstCount >= popSize) {
        worstCount = std::max<size_t>(1, popSize > bestCount ? popSize - bestCount : 1);
    }

    size_t midCount = popSize - bestCount - worstCount;
    if (midCount == 0) {
        if (bestCount > 1) {
            --bestCount;
        } else if (worstCount > 1) {
            --worstCount;
        }
        midCount = popSize - bestCount - worstCount;
    }

    std::vector<size_t> bestPool(sorted_indices.begin(), sorted_indices.begin() + bestCount);
    std::vector<size_t> midPool(sorted_indices.begin() + bestCount, sorted_indices.begin() + bestCount + midCount);
    std::vector<size_t> worstPool(sorted_indices.begin() + bestCount + midCount, sorted_indices.end());

    if (midPool.empty()) midPool = bestPool;
    if (worstPool.empty()) worstPool = midPool;

    auto r1 = sampleFromPool(popSize, bestPool);
    auto r2 = sampleFromPool(popSize, midPool);
    auto r3 = sampleFromPool(popSize, worstPool);
    return {r1, r2, r3};
}

void AGSK::doDE_operation(std::vector<std::vector<double>>& trials) {
    const size_t popSize = population.size();
    if (popSize == 0) return;
    const size_t dim = bounds.size();

    auto sorted = argsort(fitness, true);
    auto [junior_r1, junior_r2, junior_r3] = generateJuniorTriplets(sorted);
    auto [senior_r1, senior_r2, senior_r3] = generateSeniorTriplets(sorted);

    std::vector<std::vector<double>> junior(popSize, std::vector<double>(dim));
    std::vector<std::vector<double>> senior(popSize, std::vector<double>(dim));

    for (size_t i = 0; i < popSize; ++i) {
        const auto& xi = population[i];
        const auto& xr1_j = population[junior_r1[i]];
        const auto& xr2_j = population[junior_r2[i]];
        const auto& xr3_j = population[junior_r3[i]];
        const auto& xr1_s = population[senior_r1[i]];
        const auto& xr2_s = population[senior_r2[i]];
        const auto& xr3_s = population[senior_r3[i]];
        const bool juniorCond = fitness[i] > fitness[junior_r3[i]];
        const bool seniorCond = fitness[i] > fitness[senior_r2[i]];

        for (size_t d = 0; d < dim; ++d) {
            double termJ = xr1_j[d] - xr2_j[d];
            double balanceJ = juniorCond ? (xr3_j[d] - xi[d]) : (xi[d] - xr3_j[d]);
            junior[i][d] = xi[d] + KF_values[i] * (termJ + balanceJ);

            double termS = xr1_s[d] - (seniorCond ? xi[d] : xr2_s[d]);
            double balanceS = seniorCond ? (xr2_s[d] - xr3_s[d]) : (xi[d] - xr3_s[d]);
            senior[i][d] = xi[d] + KF_values[i] * (termS + balanceS);
        }
    }

    enforce_bounds(junior, bounds, boundStrategy);
    enforce_bounds(senior, bounds, boundStrategy);

    const double progress = (maxevals == 0) ? 1.0 : std::min(1.0, double(Nevals) / double(maxevals));
    const double baseShare = std::clamp(1.0 - progress, 0.0, 1.0);

    for (size_t i = 0; i < popSize; ++i) {
        trials[i] = population[i];
        double desired = 0.0;
        if (!knowledgeParameterK.empty()) {
            desired = std::pow(baseShare, knowledgeParameterK[i]);
        }
        desired = std::clamp(desired, 0.0, 1.0);
        const double ratio = (dim == 0) ? 0.0 : std::clamp(std::ceil(desired * double(dim)) / double(dim), 0.0, 1.0);
        for (size_t d = 0; d < dim; ++d) {
            bool takeJunior = rand_gen() <= ratio;
            if (takeJunior) {
                takeJunior = rand_gen() <= KR_values[i];
            }
            bool takeSenior = false;
            if (!takeJunior) {
                takeSenior = rand_gen() <= KR_values[i];
            }

            if (takeJunior) {
                trials[i][d] = junior[i][d];
            } else if (takeSenior) {
                trials[i][d] = senior[i][d];
            }
        }
    }
}

void AGSK::postEvaluation(const std::vector<std::vector<double>>&, const std::vector<double>& trial_fitness) {
    if (population.empty() || trial_fitness.empty()) {
        improvementShare = {0.25, 0.25, 0.25, 0.25};
        return;
    }

    std::array<double, 4> raw = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < population.size() && i < trial_fitness.size(); ++i) {
        if (trial_fitness[i] < fitness[i]) {
            const int idx = std::clamp(knowledgeAssignment[i], 0, 3);
            raw[idx] += std::fabs(fitness[i] - trial_fitness[i]);
        }
    }

    double sum = std::accumulate(raw.begin(), raw.end(), 0.0);
    if (sum <= 0.0 || !std::isfinite(sum)) {
        improvementShare = {0.25, 0.25, 0.25, 0.25};
        return;
    }

    std::array<double, 4> normalized;
    for (size_t i = 0; i < raw.size(); ++i) {
        normalized[i] = raw[i] / sum;
    }

    std::array<size_t, 4> order = {0, 1, 2, 3};
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return normalized[a] < normalized[b];
    });

    double remaining = 1.0;
    for (size_t i = 0; i < order.size() - 1; ++i) {
        const size_t idx = order[i];
        normalized[idx] = std::max(normalized[idx], 0.05);
        remaining -= normalized[idx];
    }
    normalized[order.back()] = std::max(0.0, remaining);

    double finalSum = std::accumulate(normalized.begin(), normalized.end(), 0.0);
    if (finalSum <= 0.0) {
        improvementShare = {0.25, 0.25, 0.25, 0.25};
    } else {
        for (double& value : normalized) {
            value /= finalSum;
        }
        improvementShare = normalized;
    }
}

}
