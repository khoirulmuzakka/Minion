#include "spso2011.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace minion {

void SPSO2011::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("SPSO2011");
    for (const auto& el : optionMap) {
        defaultKey[el.first] = el.second;
    }
    Options options(defaultKey);

    configureFromOptions(options);

    phiPersonal = options.get<double>("phi_personal", 2.05);
    phiSocial = options.get<double>("phi_social", 2.05);
    neighborhoodSize = static_cast<size_t>(options.get<int>("neighborhood_size", 3));
    if (neighborhoodSize == 0) {
        neighborhoodSize = 1;
    }

    double phi = phiPersonal + phiSocial;
    if (phi <= 4.0) {
        phi = 4.0001;
    }
    double discriminant = std::max(0.0, phi * phi - 4.0 * phi);
    double denominator = std::fabs(2.0 - phi - std::sqrt(discriminant));
    if (denominator == 0.0) {
        constriction = 1.0;
    } else {
        constriction = 2.0 / denominator;
    }

    inertiaWeight = constriction;
    cognitiveCoeff = phiPersonal;
    socialCoeff = phiSocial;

    hasInitialized = true;
}

void SPSO2011::init() {
    PSO::init();
    neighborhoods.assign(population.size(), {});
    neighborhoodBestIndices.assign(population.size(), 0);
    rebuildNeighborhoods();
    updateNeighborhoodBests();
}

void SPSO2011::rebuildNeighborhoods() {
    size_t popSize = population.size();
    if (popSize == 0) {
        return;
    }
    size_t effectiveSize = std::min(popSize, neighborhoodSize);
    neighborhoods.assign(popSize, {});
    for (size_t i = 0; i < popSize; ++i) {
        neighborhoods[i].push_back(i);
        if (effectiveSize > 1) {
            std::vector<size_t> candidates(popSize);
            std::iota(candidates.begin(), candidates.end(), 0);
            candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(i));
            auto selected = random_choice(candidates, effectiveSize - 1, false);
            neighborhoods[i].insert(neighborhoods[i].end(), selected.begin(), selected.end());
        }
    }
}

void SPSO2011::updateNeighborhoodBests() {
    if (neighborhoods.empty()) {
        return;
    }
    neighborhoodBestIndices.resize(neighborhoods.size());
    for (size_t i = 0; i < neighborhoods.size(); ++i) {
        size_t bestIdx = neighborhoods[i].front();
        double bestFit = personalBestFitness[bestIdx];
        for (size_t idx : neighborhoods[i]) {
            if (personalBestFitness[idx] < bestFit) {
                bestFit = personalBestFitness[idx];
                bestIdx = idx;
            }
        }
        neighborhoodBestIndices[i] = bestIdx;
    }
}

void SPSO2011::updateVelocitiesAndPositions() {
    rebuildNeighborhoods();
    updateNeighborhoodBests();

    size_t dim = bounds.size();
    for (size_t i = 0; i < population.size(); ++i) {
        size_t nbIdx = neighborhoodBestIndices.empty() ? i : neighborhoodBestIndices[i];
        const auto& bestNeighbor = personalBestPositions[nbIdx];
        for (size_t d = 0; d < dim; ++d) {
            double range = bounds[d].second - bounds[d].first;
            if (range <= 0.0) {
                velocities[i][d] = 0.0;
                population[i][d] = bounds[d].first;
                continue;
            }

            double r1 = rand_gen();
            double r2 = rand_gen();
            double cognitive = phiPersonal * r1 * (personalBestPositions[i][d] - population[i][d]);
            double social = phiSocial * r2 * (bestNeighbor[d] - population[i][d]);
            double newVelocity = constriction * (velocities[i][d] + cognitive + social);

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
}

}
