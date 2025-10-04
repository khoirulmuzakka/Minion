#include "abc.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minion {

namespace {
inline double nectar_quality(double fitness_value) {
    if (fitness_value >= 0.0) {
        return 1.0 / (1.0 + fitness_value);
    }
    return 1.0 + std::fabs(fitness_value);
}
} // namespace

void ABC::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("ABC");
    for (const auto& kv : optionMap) {
        defaults[kv.first] = kv.second;
    }
    Options options(defaults);

    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));
    std::vector<std::string> allowed = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(allowed.begin(), allowed.end(), boundStrategy) == allowed.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    populationSize = static_cast<size_t>(options.get<int>("population_size", 0));
    if (populationSize == 0) {
        populationSize = std::max<size_t>(5 * std::max<size_t>(bounds.size(), size_t(1)), size_t(5));
    }

    limit = static_cast<size_t>(std::max<int>(options.get<int>("limit", 100), 1));
    hasInitialized = true;
}

void ABC::init() {
    population = random_sampling(bounds, populationSize);
    if (!x0.empty()) {
        for (size_t i = 0; i < x0.size() && i < population.size(); ++i) {
            population[i] = x0[i];
        }
    }

    fitness = func(population, data);
    Nevals += population.size();

    trialCounters.assign(population.size(), 0);

    size_t best_idx = findArgMin(fitness);
    best = population[best_idx];
    best_fitness = fitness[best_idx];

    history.push_back(MinionResult(best, best_fitness, 0, Nevals, false, ""));
}

std::vector<double> ABC::generateNeighbor(size_t sourceIndex, size_t partnerIndex, size_t dimensionIndex) const {
    std::vector<double> candidate = population[sourceIndex];
    double phi = rand_gen(-1.0, 1.0);
    candidate[dimensionIndex] = candidate[dimensionIndex] + phi * (candidate[dimensionIndex] - population[partnerIndex][dimensionIndex]);
    return candidate;
}

double ABC::evaluateCandidate(const std::vector<double>& candidate) {
    std::vector<std::vector<double>> wrapper(1, candidate);
    auto result = func(wrapper, data);
    Nevals += 1;
    return result.front();
}

std::vector<double> ABC::computeProbabilities() const {
    std::vector<double> probabilities(population.size(), 0.0);
    std::vector<double> qualities(population.size(), 0.0);
    for (size_t i = 0; i < population.size(); ++i) {
        qualities[i] = nectar_quality(fitness[i]);
    }
    double sum = std::accumulate(qualities.begin(), qualities.end(), 0.0);
    if (sum <= 0.0) {
        double uniform = 1.0 / static_cast<double>(probabilities.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
    } else {
        for (size_t i = 0; i < probabilities.size(); ++i) {
            probabilities[i] = qualities[i] / sum;
        }
    }
    return probabilities;
}

bool ABC::employedPhase() {
    bool improved = false;
    size_t dim = bounds.size();
    for (size_t i = 0; i < population.size(); ++i) {
        size_t partner;
        do {
            partner = rand_int(population.size());
        } while (partner == i);
        size_t dimIndex = rand_int(dim == 0 ? 1 : dim);

        auto candidate = generateNeighbor(i, partner, dimIndex);
        enforce_bounds(candidate, bounds, boundStrategy);
        double candidateFitness = evaluateCandidate(candidate);

        if (candidateFitness < fitness[i]) {
            population[i] = std::move(candidate);
            fitness[i] = candidateFitness;
            trialCounters[i] = 0;
            improved = true;
        } else {
            trialCounters[i] += 1;
        }
        if (Nevals >= maxevals) {
            break;
        }
    }
    return improved;
}

bool ABC::onlookerPhase() {
    bool improved = false;
    auto probabilities = computeProbabilities();
    std::vector<size_t> indices(population.size());
    std::iota(indices.begin(), indices.end(), 0);
    size_t dim = bounds.size();

    size_t onlookers = 0;
    while (onlookers < population.size()) {
        auto chosen = random_choice(indices, 1, probabilities);
        size_t i = chosen.front();

        size_t partner;
        do {
            partner = rand_int(population.size());
        } while (partner == i);
        size_t dimIndex = rand_int(dim == 0 ? 1 : dim);

        auto candidate = generateNeighbor(i, partner, dimIndex);
        enforce_bounds(candidate, bounds, boundStrategy);
        double candidateFitness = evaluateCandidate(candidate);

        if (candidateFitness < fitness[i]) {
            population[i] = std::move(candidate);
            fitness[i] = candidateFitness;
            trialCounters[i] = 0;
            improved = true;
        } else {
            trialCounters[i] += 1;
        }
        ++onlookers;
        if (Nevals >= maxevals) {
            break;
        }
    }
    return improved;
}

bool ABC::scoutPhase() {
    auto maxIt = std::max_element(trialCounters.begin(), trialCounters.end());
    if (maxIt == trialCounters.end() || static_cast<size_t>(*maxIt) < limit) {
        return false;
    }
    size_t index = static_cast<size_t>(std::distance(trialCounters.begin(), maxIt));
    population[index] = random_sampling(bounds, 1).front();
    fitness[index] = evaluateCandidate(population[index]);
    trialCounters[index] = 0;
    return true;
}

MinionResult ABC::optimize() {
    if (!hasInitialized) initialize();
    try {
        history.clear();
        Nevals = 0;
        init();
        size_t iter = 1;

        while (Nevals < maxevals) {
            employedPhase();
            if (Nevals >= maxevals) break;
            onlookerPhase();
            if (Nevals >= maxevals) break;
            scoutPhase();

            size_t best_idx = findArgMin(fitness);
            if (fitness[best_idx] < best_fitness) {
                best_fitness = fitness[best_idx];
                best = population[best_idx];
            }

            minionResult = MinionResult(best, best_fitness, iter, Nevals, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) callback(&minionResult);
            ++iter;
        }

        return getBestFromHistory();
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

}
