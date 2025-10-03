#include "nlshadersp.h"
#include "default_options.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>

namespace minion {

void NLSHADE_RSP::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("NLSHADE_RSP");
    for (const auto& el : optionMap) {
        defaultKey[el.first] = el.second;
    }
    Options options(defaultKey);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy) == all_boundStrategy.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    int optPopulation = options.get<int>("population_size", 0);
    if (optPopulation > 0) {
        populationSize = static_cast<size_t>(optPopulation);
    } else {
        populationSize = std::max(static_cast<size_t>(30 * bounds.size()), static_cast<size_t>(10));
    }

    memorySize = options.get<int>("memory_size", 100);
    if (memorySize <= 0) {
        memorySize = 1;
    }

    archiveSizeRatio = options.get<double>("archive_size_ratio", 2.6);
    if (archiveSizeRatio < 0.0) {
        archiveSizeRatio = 2.6;
    }

    int optMinPopulation = options.get<int>("minimum_population_size", 4);
    minPopulationSize = optMinPopulation > 0 ? static_cast<size_t>(optMinPopulation) : static_cast<size_t>(4);
    minPopulationSize = std::max<size_t>(2, minPopulationSize);

    maxPopulationSize = populationSize;
    archiveCapacity = static_cast<size_t>(std::round(static_cast<double>(populationSize) * archiveSizeRatio));
    archiveProbability = 0.5;
    memoryIndex = 0;

    parameterMemoryCR.assign(static_cast<size_t>(memorySize), 0.2);
    parameterMemoryF.assign(static_cast<size_t>(memorySize), 0.2);
    successfulCR.clear();
    successfulF.clear();
    successDelta.clear();

    F = std::vector<double>(populationSize, 0.2);
    CR = std::vector<double>(populationSize, 0.2);
    p = std::vector<size_t>(populationSize, 1);

    hasInitialized = true;
}

void NLSHADE_RSP::recordSuccess(double cr, double f, double delta) {
    successfulCR.push_back(cr);
    successfulF.push_back(f);
    successDelta.push_back(delta);
}

double NLSHADE_RSP::weightedLehmerMean(const std::vector<double>& values, const std::vector<double>& weights, double gp, double gm) const {
    if (values.empty() || weights.empty() || values.size() != weights.size()) {
        return 0.5;
    }
    double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weightSum <= 0.0) {
        return 0.5;
    }
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        double normalizedWeight = weights[i] / weightSum;
        numerator += normalizedWeight * std::pow(values[i], gp);
        denominator += normalizedWeight * std::pow(values[i], gp - gm);
    }
    if (std::fabs(denominator) <= 1e-6) {
        return 0.5;
    }
    return numerator / denominator;
}

void NLSHADE_RSP::updateParameterMemory() {
    if (parameterMemoryCR.empty() || parameterMemoryF.empty()) {
        return;
    }

    if (!successDelta.empty()) {
        parameterMemoryCR[memoryIndex] = weightedLehmerMean(successfulCR, successDelta, 2.0, 1.0);
        parameterMemoryF[memoryIndex] = weightedLehmerMean(successfulF, successDelta, 2.0, 1.0);
    } else {
        parameterMemoryCR[memoryIndex] = 0.5;
        parameterMemoryF[memoryIndex] = 0.5;
    }

    memoryIndex = (memoryIndex + 1) % parameterMemoryCR.size();
    successfulCR.clear();
    successfulF.clear();
    successDelta.clear();
}

void NLSHADE_RSP::updateArchive(const std::vector<double>& parent) {
    if (archiveCapacity == 0) {
        return;
    }

    if (archive.size() < archiveCapacity) {
        archive.push_back(parent);
    } else if (!archive.empty()) {
        size_t randomIndex = rand_int(archive.size());
        archive[randomIndex] = parent;
    }
}

void NLSHADE_RSP::trimArchive(size_t capacity) {
    if (archive.size() <= capacity) {
        return;
    }
    while (archive.size() > capacity && !archive.empty()) {
        size_t randomIndex = rand_int(archive.size());
        archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(randomIndex));
    }
}

void NLSHADE_RSP::adjustPopulation(size_t newSize) {
    if (population.size() <= newSize) {
        return;
    }
    auto sortedIndex = argsort(fitness, true);
    std::vector<std::vector<double>> newPopulation;
    std::vector<double> newFitness;
    newPopulation.reserve(newSize);
    newFitness.reserve(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        size_t idx = sortedIndex[i];
        newPopulation.push_back(population[idx]);
        newFitness.push_back(fitness[idx]);
    }
    population.swap(newPopulation);
    fitness.swap(newFitness);
}

MinionResult NLSHADE_RSP::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();
        init();

        if (!population.empty()) {
            Nevals = population.size();
        }
        maxPopulationSize = population.size();
        archive.clear();
        archiveCapacity = static_cast<size_t>(std::round(static_cast<double>(population.size()) * archiveSizeRatio));
        trimArchive(archiveCapacity);

        size_t generation = history.empty() ? 0 : history.back().nit;

        while (Nevals < maxevals) {
            size_t popSize = population.size();
            if (popSize == 0) {
                break;
            }
            size_t dimension = population[0].size();

            auto sortedIdx = argsort(fitness, true);
            std::vector<size_t> ranks(popSize);
            for (size_t r = 0; r < sortedIdx.size(); ++r) {
                ranks[sortedIdx[r]] = r;
            }

            std::vector<double> candidateF(popSize);
            std::vector<double> candidateCR(popSize);
            for (size_t i = 0; i < popSize; ++i) {
                size_t memIndex = static_cast<size_t>(rand_int(static_cast<size_t>(memorySize)));
                double crSample = clamp(rand_norm(parameterMemoryCR[memIndex], 0.1), 0.0, 1.0);
                double fSample;
                do {
                    fSample = rand_cauchy(parameterMemoryF[memIndex], 0.1);
                } while (fSample <= 0.0);
                candidateF[i] = clamp(fSample, 0.0, 1.0);
                candidateCR[i] = crSample;
            }

            std::vector<double> sortedCR = candidateCR;
            std::sort(sortedCR.begin(), sortedCR.end());

            double evalRatio = maxevals == 0 ? 0.0 : static_cast<double>(Nevals) / static_cast<double>(maxevals);
            size_t topCount = static_cast<size_t>(std::round(static_cast<double>(popSize) * (0.2 * evalRatio + 0.2)));
            topCount = std::clamp<size_t>(topCount, 2, popSize);

            std::vector<double> selectionWeights(popSize);
            for (size_t i = 0; i < popSize; ++i) {
                selectionWeights[i] = std::exp(-static_cast<double>(i) / static_cast<double>(popSize));
            }
            std::discrete_distribution<size_t> selectionDistribution(selectionWeights.begin(), selectionWeights.end());
            std::mt19937& rng = get_rng();

            bool useExponential = rand_gen() < 0.5;
            std::vector<std::vector<double>> trials(popSize, std::vector<double>(dimension));
            std::vector<double> appliedCR(popSize, 0.0);
            std::vector<double> appliedF(popSize, 0.0);
            std::vector<bool> usedArchive(popSize, false);

            auto isForbidden = [](size_t value, const std::vector<size_t>& used) {
                return std::find(used.begin(), used.end(), value) != used.end();
            };

            auto chooseFromTop = [&](size_t target, const std::vector<size_t>& forbidden) {
                for (int attempt = 0; attempt < 25; ++attempt) {
                    size_t candidate = sortedIdx[rand_int(topCount)];
                    if (candidate == target || isForbidden(candidate, forbidden)) {
                        continue;
                    }
                    return candidate;
                }
                for (size_t candidate : sortedIdx) {
                    if (candidate != target && !isForbidden(candidate, forbidden)) {
                        return candidate;
                    }
                }
                return target;
            };

            auto chooseFromPopulation = [&](size_t target, const std::vector<size_t>& forbidden) {
                for (int attempt = 0; attempt < 25; ++attempt) {
                    size_t candidate = rand_int(popSize);
                    if (candidate == target || isForbidden(candidate, forbidden)) {
                        continue;
                    }
                    return candidate;
                }
                for (size_t candidate = 0; candidate < popSize; ++candidate) {
                    if (candidate != target && !isForbidden(candidate, forbidden)) {
                        return candidate;
                    }
                }
                return target;
            };

            auto chooseFromArchive = [&](const std::vector<size_t>& forbidden) {
                if (archive.empty()) {
                    return population.size();
                }
                for (int attempt = 0; attempt < 25; ++attempt) {
                    size_t candidate = population.size() + rand_int(archive.size());
                    if (!isForbidden(candidate, forbidden)) {
                        return candidate;
                    }
                }
                for (size_t idx = 0; idx < archive.size(); ++idx) {
                    size_t candidate = population.size() + idx;
                    if (!isForbidden(candidate, forbidden)) {
                        return candidate;
                    }
                }
                return population.size();
            };

            auto chooseFromDistribution = [&](size_t target, const std::vector<size_t>& forbidden) {
                for (int attempt = 0; attempt < 25; ++attempt) {
                    size_t candidate = sortedIdx[selectionDistribution(rng)];
                    if (candidate == target || isForbidden(candidate, forbidden)) {
                        continue;
                    }
                    return candidate;
                }
                for (size_t candidate : sortedIdx) {
                    if (candidate != target && !isForbidden(candidate, forbidden)) {
                        return candidate;
                    }
                }
                return target;
            };

            auto getIndividual = [&](size_t combinedIndex) -> const std::vector<double>& {
                if (combinedIndex < population.size()) {
                    return population[combinedIndex];
                }
                size_t archiveIndex = combinedIndex - population.size();
                return archive[archiveIndex];
            };

            for (size_t i = 0; i < popSize; ++i) {
                std::vector<size_t> forbidden = {i};

                size_t r0 = chooseFromTop(i, forbidden);
                forbidden.push_back(r0);

                size_t r1 = chooseFromPopulation(i, forbidden);
                forbidden.push_back(r1);

                bool takeFromArchive = archiveProbability > 0.0 && rand_gen() <= archiveProbability && !archive.empty();
                size_t r2 = population.size();
                if (takeFromArchive) {
                    r2 = chooseFromArchive(forbidden);
                    if (r2 >= population.size() && (r2 - population.size()) < archive.size()) {
                        forbidden.push_back(r2);
                        usedArchive[i] = true;
                    } else {
                        takeFromArchive = false;
                    }
                }
                if (!takeFromArchive) {
                    r2 = chooseFromDistribution(i, forbidden);
                    forbidden.push_back(r2);
                    usedArchive[i] = false;
                }

                const std::vector<double>& target = population[i];
                const std::vector<double>& vecR0 = getIndividual(r0);
                const std::vector<double>& vecR1 = getIndividual(r1);
                const std::vector<double>& vecR2 = getIndividual(r2);

                std::vector<double> donor(dimension, 0.0);
                for (size_t d = 0; d < dimension; ++d) {
                    donor[d] = target[d] + candidateF[i] * (vecR0[d] - target[d]) + candidateF[i] * (vecR1[d] - vecR2[d]);
                }

                appliedCR[i] = sortedCR[ranks[i]];
                appliedF[i] = candidateF[i];

                std::vector<double> trial = target;
                if (!useExponential) {
                    double crossoverProbability = 0.0;
                    if (Nevals > static_cast<size_t>(0.5 * maxevals)) {
                        crossoverProbability = (static_cast<double>(Nevals) / static_cast<double>(maxevals) - 0.5) * 2.0;
                    }
                    crossoverProbability = clamp(crossoverProbability, 0.0, 1.0);
                    size_t forceIndex = dimension > 0 ? rand_int(dimension) : 0;
                    for (size_t d = 0; d < dimension; ++d) {
                        if (rand_gen() < crossoverProbability || d == forceIndex) {
                            trial[d] = donor[d];
                        }
                    }
                } else {
                    size_t start = dimension > 0 ? rand_int(dimension) : 0;
                    size_t L = start + 1;
                    double crValue = clamp(appliedCR[i], 0.0, 1.0);
                    while (rand_gen() < crValue && L < dimension) {
                        ++L;
                    }
                    for (size_t d = start; d < L && d < dimension; ++d) {
                        trial[d] = donor[d];
                    }
                }

                trials[i] = std::move(trial);
            }

            enforce_bounds(trials, bounds, boundStrategy);

            F = appliedF;
            CR = appliedCR;
            meanCR.push_back(calcMean(CR));
            meanF.push_back(calcMean(F));
            stdCR.push_back(calcStdDev(CR));
            stdF.push_back(calcStdDev(F));

            auto trialFit = func(trials, data);
            Nevals += trials.size();

            std::replace_if(trialFit.begin(), trialFit.end(), [](double value) { return std::isnan(value); }, 1e+100);
            fitness_before = fitness;

            double archiveSuccess = 0.0;
            double populationSuccess = 0.0;
            size_t archiveUseCount = 0;

            for (size_t i = 0; i < popSize; ++i) {
                if (trialFit[i] < fitness[i]) {
                    double improvement = fitness[i] - trialFit[i];
                    recordSuccess(appliedCR[i], appliedF[i], std::fabs(improvement));

                    double denom = fitness[i];
                    if (std::fabs(denom) < 1e-12) {
                        denom = denom >= 0.0 ? 1e-12 : -1e-12;
                    }
                    double normalizedGain = improvement / denom;

                    if (usedArchive[i]) {
                        archiveSuccess += normalizedGain;
                        archiveUseCount++;
                    } else {
                        populationSuccess += normalizedGain;
                    }

                    updateArchive(population[i]);
                    population[i] = trials[i];
                    fitness[i] = trialFit[i];
                }
            }

            if (archiveUseCount > 0) {
                double archAvg = archiveSuccess / static_cast<double>(archiveUseCount);
                double popAvg = 0.0;
                size_t nonArchiveCount = popSize > archiveUseCount ? popSize - archiveUseCount : 0;
                if (nonArchiveCount > 0) {
                    popAvg = populationSuccess / static_cast<double>(nonArchiveCount);
                }
                if (archAvg > 0.0 || popAvg > 0.0) {
                    double denom = archAvg + popAvg;
                    if (denom <= 0.0) {
                        archiveProbability = 0.5;
                    } else {
                        archiveProbability = clamp(archAvg / denom, 0.1, 0.9);
                    }
                    if (archAvg == 0.0) {
                        archiveProbability = 0.5;
                    }
                } else {
                    archiveProbability = 0.5;
                }
            } else {
                archiveProbability = 0.5;
            }

            updateParameterMemory();

            auto bestIndex = findArgMin(fitness);
            best = population[bestIndex];
            best_fitness = fitness[bestIndex];

            minionResult = MinionResult(best, best_fitness, generation, Nevals, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) {
                callback(&minionResult);
            }

            if (support_tol && checkStopping()) {
                break;
            }

            double ratio = maxevals == 0 ? 0.0 : static_cast<double>(Nevals) / static_cast<double>(maxevals);
            ratio = std::clamp(ratio, 0.0, 1.0);
            size_t newPopulationSize = static_cast<size_t>(std::round((static_cast<double>(minPopulationSize) - static_cast<double>(maxPopulationSize)) * std::pow(ratio, 1.0 - ratio) + static_cast<double>(maxPopulationSize)));
            newPopulationSize = std::clamp(newPopulationSize, minPopulationSize, maxPopulationSize);

            adjustPopulation(newPopulationSize);
            populationSize = population.size();
            F.resize(populationSize, 0.5);
            CR.resize(populationSize, 0.5);
            p = std::vector<size_t>(populationSize, 1);

            archiveCapacity = static_cast<size_t>(std::round(static_cast<double>(populationSize) * archiveSizeRatio));
            trimArchive(archiveCapacity);
            generation++;
        }

        return getBestFromHistory();
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

} // namespace minion
