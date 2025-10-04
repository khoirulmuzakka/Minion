#include "abc.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minion {

namespace {
inline double nectar_quality(double f) {
    if (f >= 0.0) {
        return 1.0 / (1.0 + f);
    }
    return 1.0 - f;
}

inline double clamp_value(double value, double lower, double upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}
}

void ABC::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("ABC");
    for (const auto& el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy) == all_boundStrategy.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    populationSize = static_cast<size_t>(options.get<int>("population_size", 0));
    if (populationSize == 0) {
        populationSize = std::max<size_t>(3 * std::max<size_t>(bounds.size(), size_t(1)), size_t(5));
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

MinionResult ABC::optimize() {
    if (!hasInitialized) initialize();
    try {
        history.clear();
        Nevals = 0;
        init();

        size_t dim = bounds.size();
        if (population.empty()) {
            throw std::runtime_error("Population is empty in ABC optimizer");
        }

        size_t iter = 1;
        while (Nevals < maxevals) {
            // Employed bees phase
            std::vector<std::vector<double>> employedCandidates(population.size());
            for (size_t i = 0; i < population.size(); ++i) {
                employedCandidates[i] = population[i];
                if (dim == 0) {
                    continue;
                }
                size_t compIndex = rand_int(dim);
                size_t partner;
                do {
                    partner = rand_int(population.size());
                } while (partner == i && population.size() > 1);
                double phi = rand_gen(-1.0, 1.0);
                employedCandidates[i][compIndex]
                    += phi * (employedCandidates[i][compIndex] - population[partner][compIndex]);
                employedCandidates[i][compIndex]
                    = clamp_value(employedCandidates[i][compIndex], bounds[compIndex].first, bounds[compIndex].second);
            }

            auto employedFitness = func(employedCandidates, data);
            Nevals += employedFitness.size();
            for (size_t i = 0; i < population.size(); ++i) {
                if (employedFitness[i] < fitness[i]) {
                    population[i] = employedCandidates[i];
                    fitness[i] = employedFitness[i];
                    trialCounters[i] = 0;
                } else {
                    trialCounters[i] += 1;
                }
            }
            if (Nevals >= maxevals) break;

            // Scout phase
            size_t maxTrialIndex = 0;
            for (size_t i = 1; i < trialCounters.size(); ++i) {
                if (trialCounters[i] > trialCounters[maxTrialIndex]) {
                    maxTrialIndex = i;
                }
            }
            if (!trialCounters.empty() && trialCounters[maxTrialIndex] >= limit) {
                auto scoutSolution = random_sampling(bounds, 1).front();
                auto scoutFitness = func({scoutSolution}, data);
                Nevals += scoutFitness.size();
                population[maxTrialIndex] = std::move(scoutSolution);
                fitness[maxTrialIndex] = scoutFitness.front();
                trialCounters[maxTrialIndex] = 0;
                if (Nevals >= maxevals) break;
            }

            // Onlooker phase
            std::vector<double> probabilities(population.size(), 0.0);
            double sumProb = 0.0;
            for (size_t i = 0; i < population.size(); ++i) {
                probabilities[i] = nectar_quality(fitness[i]);
                sumProb += probabilities[i];
            }
            if (sumProb <= 0.0) {
                double uniform = 1.0 / static_cast<double>(population.size());
                std::fill(probabilities.begin(), probabilities.end(), uniform);
            } else {
                for (double& prob : probabilities) {
                    prob /= sumProb;
                }
            }

            std::vector<size_t> acceptedIndices;
            std::vector<std::vector<double>> onlookerCandidates;
            acceptedIndices.reserve(population.size());
            onlookerCandidates.reserve(population.size());

            size_t accepted = 0;
            size_t s = 0;
            while (accepted < population.size()) {
                double r = rand_gen();
                if (r < probabilities[s]) {
                    ++accepted;
                    std::vector<double> candidate = population[s];
                    if (dim > 0) {
                        size_t compIndex = rand_int(dim);
                        size_t partner;
                        do {
                            partner = rand_int(population.size());
                        } while (partner == s && population.size() > 1);
                        double phi = rand_gen(-1.0, 1.0);
                        candidate[compIndex]
                            += phi * (candidate[compIndex] - population[partner][compIndex]);
                        candidate[compIndex]
                            = clamp_value(candidate[compIndex], bounds[compIndex].first, bounds[compIndex].second);
                    }
                    acceptedIndices.push_back(s);
                    onlookerCandidates.push_back(std::move(candidate));
                }
                s = (s + 1) % population.size();
            }

            auto onlookerFitness = func(onlookerCandidates, data);
            Nevals += onlookerFitness.size();
            for (size_t j = 0; j < acceptedIndices.size(); ++j) {
                size_t idx = acceptedIndices[j];
                double newFit = onlookerFitness[j];
                if (newFit < fitness[idx]) {
                    fitness[idx] = newFit;
                    population[idx] = onlookerCandidates[j];
                    trialCounters[idx] = 0;
                } else {
                    trialCounters[idx] += 1;
                }
            }

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

