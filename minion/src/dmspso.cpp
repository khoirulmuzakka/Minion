#include "dmspso.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>

namespace minion {

void DMSPSO::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("DMSPSO");
    for (const auto& el : optionMap) {
        defaultKey[el.first] = el.second;
    }
    Options options(defaultKey);

    configureFromOptions(options);

    subswarmCount = static_cast<size_t>(options.get<int>("subswarm_count", 4));
    if (subswarmCount == 0) {
        subswarmCount = 1;
    }
    regroupPeriod = static_cast<size_t>(options.get<int>("regroup_period", 5));
    if (regroupPeriod == 0) {
        regroupPeriod = 1;
    }
    localCoefficient = options.get<double>("local_coefficient", 1.4);
    globalCoefficient = options.get<double>("global_coefficient", 0.8);

    hasInitialized = true;
}

void DMSPSO::init() {
    PSO::init();
    iterationCounter = 0;
    rebuildSubswarmAssignments();
    updateSubswarmBests();
}

void DMSPSO::rebuildSubswarmAssignments() {
    size_t popSize = population.size();
    if (popSize == 0) {
        subswarms.clear();
        particleToSubswarm.clear();
        subswarmBestIndices.clear();
        return;
    }

    size_t effectiveSubswarm = std::min(popSize, subswarmCount);
    subswarms.assign(effectiveSubswarm, {});
    particleToSubswarm.assign(popSize, 0);

    std::vector<size_t> indices(popSize);
    std::iota(indices.begin(), indices.end(), 0);
    auto& rng = get_rng();
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t i = 0; i < popSize; ++i) {
        size_t group = i % effectiveSubswarm;
        size_t particleIndex = indices[i];
        subswarms[group].push_back(particleIndex);
        particleToSubswarm[particleIndex] = group;
    }
}

void DMSPSO::updateSubswarmBests() {
    size_t groupCount = subswarms.size();
    subswarmBestIndices.assign(groupCount, 0);
    for (size_t g = 0; g < groupCount; ++g) {
        if (subswarms[g].empty()) {
            continue;
        }
        size_t bestIdx = subswarms[g].front();
        double bestFitness = personalBestFitness[bestIdx];
        for (size_t idx : subswarms[g]) {
            if (personalBestFitness[idx] < bestFitness) {
                bestFitness = personalBestFitness[idx];
                bestIdx = idx;
            }
        }
        subswarmBestIndices[g] = bestIdx;
    }
}

void DMSPSO::updateVelocitiesAndPositions() {
    if (regroupPeriod > 0 && (iterationCounter % regroupPeriod == 0)) {
        rebuildSubswarmAssignments();
    }
    updateSubswarmBests();

    size_t dim = bounds.size();
    for (size_t i = 0; i < population.size(); ++i) {
        size_t group = particleToSubswarm.empty() ? 0 : particleToSubswarm[i];
        size_t bestIdx = (group < subswarmBestIndices.size()) ? subswarmBestIndices[group] : i;
        const auto& localBest = personalBestPositions[bestIdx];

        for (size_t d = 0; d < dim; ++d) {
            double range = bounds[d].second - bounds[d].first;
            if (range <= 0.0) {
                velocities[i][d] = 0.0;
                population[i][d] = bounds[d].first;
                continue;
            }

            double r1 = rand_gen();
            double r2 = rand_gen();
            double r3 = rand_gen();
            double cognitive = cognitiveCoeff * r1 * (personalBestPositions[i][d] - population[i][d]);
            double localInfluence = localCoefficient * r2 * (localBest[d] - population[i][d]);
            double globalInfluence = globalCoefficient * r3 * (best[d] - population[i][d]);

            double newVelocity = inertiaWeight * velocities[i][d] + cognitive + localInfluence + globalInfluence;
            double maxVelocity = velocityClamp > 0.0 ? velocityClamp * range : range;
            velocities[i][d] = clamp(newVelocity, -maxVelocity, maxVelocity);
            population[i][d] += velocities[i][d];
        }
    }

    std::vector<std::vector<double>> beforeEnforce = population;
    enforce_bounds(population, bounds, boundStrategy);
    for (size_t i = 0; i < population.size(); ++i) {
        for (size_t d = 0; d < bounds.size(); ++d) {
            if (beforeEnforce[i][d] != population[i][d]) {
                velocities[i][d] = 0.0;
            }
        }
    }

    iterationCounter++;
}

}
