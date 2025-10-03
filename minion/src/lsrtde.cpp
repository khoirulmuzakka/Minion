#include "lsrtde.h"
#include "default_options.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace minion {

namespace {
inline double clamp01(double value) {
    return clamp(value, 0.0, 1.0);
}
}

void LSRTDE::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("LSRTDE");
    for (const auto& el : optionMap) {
        defaultKey[el.first] = el.second;
    }
    Options options(defaultKey);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    std::vector<std::string> allowedStrategies = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(allowedStrategies.begin(), allowedStrategies.end(), boundStrategy) == allowedStrategies.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    frontSize = options.get<int>("population_size", 0);
    if (frontSize == 0) {
        frontSize = std::max<size_t>(static_cast<size_t>(20 * bounds.size()), 4);
    }
    populationSize = frontSize;
    frontSizeMax = frontSize;

    memorySize = static_cast<size_t>(std::max(1, options.get<int>("memory_size", 6)));
    memoryCR.assign(memorySize, 1.0);
    memoryIndex = 0;

    successRate = clamp(options.get<double>("success_rate", 0.5), 0.0, 1.0);

    setup_population();
    hasInitialized = true;
}

void LSRTDE::setup_population() {
    size_t bufferSize = std::max(frontSize * 2, frontSize + 4);
    bufferPopulation = random_sampling(bounds, bufferSize);
    if (!x0.empty()) {
        for (size_t i = 0; i < x0.size() && i < bufferPopulation.size(); ++i) {
            if (x0[i].size() == bounds.size()) {
                bufferPopulation[i] = x0[i];
            }
        }
    }

    population.assign(bufferPopulation.begin(), bufferPopulation.begin() + frontSize);
    bufferFitness.assign(bufferPopulation.size(), std::numeric_limits<double>::max());
    fitness.assign(frontSize, std::numeric_limits<double>::max());

    successCRBuffer.clear();
    successDeltaBuffer.clear();
}

double LSRTDE::weighted_lehmer(const std::vector<double>& values, const std::vector<double>& weights) const {
    if (values.empty() || weights.empty() || values.size() != weights.size()) {
        return 1.0;
    }
    double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weightSum <= 0.0) {
        return 1.0;
    }
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        double w = weights[i] / weightSum;
        numerator += w * values[i] * values[i];
        denominator += w * values[i];
    }
    if (std::fabs(denominator) <= 1e-8) {
        return 1.0;
    }
    return numerator / denominator;
}

void LSRTDE::update_memory_cr(const std::vector<double>& successCR, const std::vector<double>& deltas) {
    if (successCR.empty() || deltas.empty()) {
        return;
    }
    double wl = weighted_lehmer(successCR, deltas);
    double updated = 0.5 * (wl + memoryCR[memoryIndex]);
    memoryCR[memoryIndex] = clamp(updated, 0.0, 1.0);
    memoryIndex = (memoryIndex + 1) % memorySize;
}

MinionResult LSRTDE::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();

        // Evaluate initial front
        fitness = func(population, data);
        for (size_t i = 0; i < frontSize; ++i) {
            bufferPopulation[i] = population[i];
            bufferFitness[i] = fitness[i];
        }
        Nevals = population.size();

        size_t bestIndex = findArgMin(fitness);
        best = population[bestIndex];
        best_fitness = fitness[bestIndex];
        minionResult = MinionResult(best, best_fitness, 0, Nevals, false, "");
        history.push_back(minionResult);
        if (callback != nullptr) {
            callback(&minionResult);
        }

        std::mt19937& rng = get_rng();
        size_t generation = 1;

        while (Nevals < maxevals) {
            size_t currentFront = population.size();
            if (currentFront == 0) {
                break;
            }

            // Sort current front by fitness for selection pressure
            std::vector<size_t> sortedFront = argsort(fitness, true);

            std::vector<double> weights(currentFront);
            for (size_t i = 0; i < currentFront; ++i) {
                weights[i] = std::exp(-static_cast<double>(i) / static_cast<double>(currentFront) * 3.0);
            }
            std::discrete_distribution<size_t> frontSelector(weights.begin(), weights.end());

            size_t selectionPool = static_cast<size_t>(std::max<double>(2.0, std::floor(static_cast<double>(currentFront) * 0.7 * std::exp(-successRate * 7.0))));
            selectionPool = std::min(selectionPool, currentFront);

            std::vector<std::vector<double>> trials(currentFront, std::vector<double>(bounds.size()));
            std::vector<double> actualCR(currentFront, 0.0);
            std::vector<double> usedF(currentFront, 0.0);
            std::vector<size_t> chosen(currentFront, 0);

            double meanF = 0.4 + std::tanh(successRate * 5.0) * 0.25;
            double sigmaF = 0.02;

            successCRBuffer.clear();
            successDeltaBuffer.clear();

            for (size_t i = 0; i < currentFront; ++i) {
                size_t target = rand_int(currentFront);
                chosen[i] = target;

                size_t memorySlot = rand_int(memorySize);

                size_t prandIdx;
                do {
                    prandIdx = sortedFront[rand_int(selectionPool)];
                } while (prandIdx == target);

                size_t rand1Idx;
                do {
                    rand1Idx = sortedFront[frontSelector(rng)];
                } while (rand1Idx == prandIdx);

                size_t rand2Idx;
                do {
                    rand2Idx = sortedFront[rand_int(currentFront)];
                } while (rand2Idx == prandIdx || rand2Idx == rand1Idx);

                double sampledF;
                do {
                    sampledF = rand_norm(meanF, sigmaF);
                } while (sampledF < 0.0 || sampledF > 1.0);
                usedF[i] = sampledF;

                double sampledCR = clamp01(rand_norm(memoryCR[memorySlot], 0.05));

                size_t forceIndex = bounds.empty() ? 0 : rand_int(bounds.size());
                size_t crossoverCount = 0;
                for (size_t d = 0; d < bounds.size(); ++d) {
                    double donor = population[target][d]
                        + sampledF * (bufferPopulation[prandIdx][d] - population[target][d])
                        + sampledF * (population[rand1Idx][d] - bufferPopulation[rand2Idx][d]);

                    if (rand_gen() < sampledCR || d == forceIndex) {
                        trials[i][d] = donor;
                        crossoverCount++;
                    } else {
                        trials[i][d] = population[target][d];
                    }
                }

                actualCR[i] = bounds.empty() ? 0.0 : static_cast<double>(crossoverCount) / static_cast<double>(bounds.size());
                enforce_bounds(trials[i], bounds, "random");
            }

            auto trialFitness = func(trials, data);
            Nevals += trials.size();
            std::replace_if(trialFitness.begin(), trialFitness.end(), [](double value) { return std::isnan(value); }, 1e+100);

            size_t bestTrialIndex = findArgMin(trialFitness);
            minionResult = MinionResult(trials[bestTrialIndex], trialFitness[bestTrialIndex], generation, Nevals, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) {
                callback(&minionResult);
            }

            std::vector<std::vector<double>> candidatePopulation = population;
            std::vector<double> candidateFitness = fitness;
            size_t successCount = 0;

            for (size_t i = 0; i < currentFront; ++i) {
                size_t target = chosen[i];
                double trialFit = trialFitness[i];
                double targetFit = fitness[target];
                if (trialFit <= targetFit) {
                    candidatePopulation.push_back(trials[i]);
                    candidateFitness.push_back(trialFit);
                    successCRBuffer.push_back(actualCR[i]);
                    successDeltaBuffer.push_back(std::fabs(targetFit - trialFit));
                    successCount++;
                }
            }

            successRate = currentFront == 0 ? 0.0 : static_cast<double>(successCount) / static_cast<double>(currentFront);

            update_memory_cr(successCRBuffer, successDeltaBuffer);

            size_t newFrontSize = frontSizeMax;
            if (maxevals > 0) {
                double ratio = static_cast<double>(Nevals) / static_cast<double>(maxevals);
                double projected = static_cast<double>(frontSizeMax) + (4.0 - static_cast<double>(frontSizeMax)) * ratio;
                newFrontSize = static_cast<size_t>(std::round(projected));
            }
            newFrontSize = std::clamp(newFrontSize, static_cast<size_t>(4), frontSizeMax);
            newFrontSize = std::min(newFrontSize, candidatePopulation.size());

            auto order = argsort(candidateFitness, true);
            population.clear();
            fitness.clear();
            for (size_t i = 0; i < newFrontSize; ++i) {
                size_t idx = order[i];
                population.push_back(candidatePopulation[idx]);
                fitness.push_back(candidateFitness[idx]);
                bufferPopulation[i] = candidatePopulation[idx];
                bufferFitness[i] = candidateFitness[idx];
            }
            frontSize = newFrontSize;
            populationSize = frontSize;

            double avgF = usedF.empty() ? 0.0 : calcMean(usedF);
            double stdFVal = usedF.empty() ? 0.0 : calcStdDev(usedF);
            double avgCR = actualCR.empty() ? 0.0 : calcMean(actualCR);
            double stdCRVal = actualCR.empty() ? 0.0 : calcStdDev(actualCR);

            F.assign(frontSize, avgF);
            CR.assign(frontSize, avgCR);
           // meanF.push_back(avgF);
           // stdF.push_back(stdFVal);
           // meanCR.push_back(avgCR);
           // stdCR.push_back(stdCRVal);

            bestIndex = findArgMin(fitness);
            best = population[bestIndex];
            best_fitness = fitness[bestIndex];

            generation++;
            if (Nevals >= maxevals) {
                break;
            }
        }

        return getBestFromHistory();
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

} // namespace minion
