#include "pso.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace minion {

void PSO::configureFromOptions(const Options& options) {
    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));
    std::vector<std::string> allowedStrategies = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(allowedStrategies.begin(), allowedStrategies.end(), boundStrategy) == allowedStrategies.end()) {
        std::cerr << "Bound strategy '" << boundStrategy << "' is not recognized. 'reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    size_t suggestedSize = bounds.empty() ? 0 : 40;
    populationSize = static_cast<size_t>(options.get<int>("population_size", 0));
    if (populationSize == 0) {
        populationSize = std::max<size_t>(suggestedSize, size_t(10));
    }
    if (populationSize == 0) {
        throw std::runtime_error("Population size must be greater than zero for PSO.");
    }

    inertiaWeight = options.get<double>("inertia_weight", 0.7);
    cognitiveCoeff = options.get<double>("cognitive_coefficient", 1.5);
    socialCoeff = options.get<double>("social_coefficient", 1.5);
    velocityClamp = options.get<double>("velocity_clamp", 0.2);
    if (velocityClamp < 0.0) {
        velocityClamp = 0.0;
    }
    useLatin = options.get<bool>("use_latin", true);
    support_tol = options.get<bool>("support_tolerance", true);
}

void PSO::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("PSO");
    for (const auto& el : optionMap) {
        defaultKey[el.first] = el.second;
    }
    Options options(defaultKey);

    configureFromOptions(options);

    hasInitialized = true;
}

void PSO::init() {
    if (bounds.empty()) {
        throw std::runtime_error("PSO requires bound constraints.");
    }
    size_t dim = bounds.size();

    population = useLatin ? latin_hypercube_sampling(bounds, populationSize)
                          : random_sampling(bounds, populationSize);
    if (!x0.empty()) {
        size_t limit = std::min(x0.size(), population.size());
        for (size_t i = 0; i < limit; ++i) {
            population[i] = x0[i];
        }
    }

    velocities.assign(populationSize, std::vector<double>(dim, 0.0));
    for (size_t i = 0; i < population.size(); ++i) {
        for (size_t d = 0; d < dim; ++d) {
            double range = bounds[d].second - bounds[d].first;
            if (range <= 0.0) {
                velocities[i][d] = 0.0;
                continue;
            }
            double limit = velocityClamp > 0.0 ? velocityClamp * range : range;
            velocities[i][d] = rand_gen(-limit, limit);
        }
    }

    fitness = func(population, data);
    std::replace_if(fitness.begin(), fitness.end(), [](double value) { return std::isnan(value); }, 1e+100);

    personalBestPositions = population;
    personalBestFitness = fitness;

    size_t bestIdx = findArgMin(fitness);
    best = population[bestIdx];
    best_fitness = fitness[bestIdx];

    Nevals += population.size();

    diversity.clear();
    spatialDiversity.clear();
    recordMetrics();

    history.push_back(MinionResult(best, best_fitness, 0, Nevals, false, ""));
}

void PSO::updateVelocitiesAndPositions() {
    size_t dim = bounds.size();
    for (size_t i = 0; i < population.size(); ++i) {
        for (size_t d = 0; d < dim; ++d) {
            double range = bounds[d].second - bounds[d].first;
            if (range <= 0.0) {
                velocities[i][d] = 0.0;
                population[i][d] = bounds[d].first;
                continue;
            }

            double r1 = rand_gen();
            double r2 = rand_gen();
            double vel = inertiaWeight * velocities[i][d]
                       + cognitiveCoeff * r1 * (personalBestPositions[i][d] - population[i][d])
                       + socialCoeff * r2 * (best[d] - population[i][d]);

            double maxVelocity = velocityClamp > 0.0 ? velocityClamp * range : range;
            velocities[i][d] = clamp(vel, -maxVelocity, maxVelocity);
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

void PSO::recordMetrics() {
    if (population.empty() || fitness.empty()) {
        return;
    }
    double fmax = findMax(fitness);
    double fmin = findMin(fitness);
    double mean = calcMean(fitness);
    double denom = std::fabs(mean);
    double relRange = denom > 1e-12 ? (fmax - fmin) / denom : (fmax - fmin);
    diversity.push_back(relRange);
    spatialDiversity.push_back(averageEuclideanDistance(population));
}

bool PSO::checkStopping() const {
    if (diversity.empty()) {
        return false;
    }
    return diversity.back() <= stoppingTol;
}

MinionResult PSO::optimize() {
    if (!hasInitialized) {
        initialize();
    }
    try {
        history.clear();
        diversity.clear();
        spatialDiversity.clear();
        Nevals = 0;
        init();

        size_t iter = 1;
        while (Nevals < maxevals) {
            updateVelocitiesAndPositions();
            auto newFitness = func(population, data);
            Nevals += newFitness.size();
            std::replace_if(newFitness.begin(), newFitness.end(), [](double value) { return std::isnan(value); }, 1e+100);

            for (size_t i = 0; i < population.size(); ++i) {
                if (newFitness[i] < personalBestFitness[i]) {
                    personalBestFitness[i] = newFitness[i];
                    personalBestPositions[i] = population[i];
                }
            }

            fitness = newFitness;
            size_t bestIdx = findArgMin(fitness);
            if (fitness[bestIdx] < best_fitness) {
                best = population[bestIdx];
                best_fitness = fitness[bestIdx];
            }

            recordMetrics();

            minionResult = MinionResult(best, best_fitness, iter, Nevals, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) {
                callback(&minionResult);
            }

            if (support_tol && checkStopping()) {
                break;
            }

            if (Nevals >= maxevals) {
                break;
            }
            iter++;
        }

        return getBestFromHistory();

    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

}
