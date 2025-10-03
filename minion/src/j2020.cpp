#include "j2020.h"
#include "default_options.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace minion {

void j2020::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("j2020");
    for (const auto& entry : optionMap) {
        defaults[entry.first] = entry.second;
    }
    Options options(defaults);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    std::vector<std::string> supportedBounds = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(supportedBounds.begin(), supportedBounds.end(), boundStrategy) == supportedBounds.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    desiredPopulation = options.get<int>("population_size", 0);
    if (desiredPopulation == 0 || desiredPopulation < 32) {
        size_t dim = bounds.size();
        desiredPopulation = std::max<size_t>(std::min<size_t>(1000, 8 * dim), 32);
    }

    bigPopulationSize = static_cast<size_t>(std::round(0.875 * static_cast<double>(desiredPopulation)));
    if (bigPopulationSize == 0) {
        bigPopulationSize = 1;
    }
    smallPopulationSize = desiredPopulation > bigPopulationSize ? desiredPopulation - bigPopulationSize : 1;
    bigPopulationSize = desiredPopulation - smallPopulationSize;
    populationSize = bigPopulationSize + smallPopulationSize;

    tau1 = options.get<double>("tau1", 0.1);
    tau2 = options.get<double>("tau2", 0.1);
    myEqs = options.get<double>("myEqs", 0.25);

    hasInitialized = true;
}

double j2020::distance_squared(const std::vector<double>& a, const std::vector<double>& b) const {
    double acc = 0.0;
    for (size_t d = 0; d < a.size(); ++d) {
        double diff = a[d] - b[d];
        acc += diff * diff;
    }
    return acc;
}

size_t j2020::crowding_index(const std::vector<double>& candidate) const {
    size_t bestIdx = 0;
    double bestDist = distance_squared(population[0], candidate);
    for (size_t i = 1; i < bigPopulationSize; ++i) {
        double dist = distance_squared(population[i], candidate);
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }
    return bestIdx;
}

bool j2020::too_many_equals(const std::vector<double>& costs, size_t count, double bestCost) const {
    size_t equalCount = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(costs[i] - bestCost) < eps) {
            ++equalCount;
        }
    }
    return equalCount > static_cast<size_t>(myEqs * static_cast<double>(count)) && equalCount > 2;
}

void j2020::reinitialize_range(size_t start, size_t end, size_t skipIndex) {
    for (size_t idx = start; idx < end; ++idx) {
        if (idx == skipIndex) {
            continue;
        }
        for (size_t d = 0; d < bounds.size(); ++d) {
            population[idx][d] = rand_gen(bounds[d].first, bounds[d].second);
        }
        fitness[idx] = std::numeric_limits<double>::max();
        F[idx] = baseF;
        CR[idx] = baseCR;
    }
}

MinionResult j2020::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();

        if (bounds.empty()) {
            throw std::runtime_error("j2020 requires problem dimension greater than zero.");
        }

        population = random_sampling(bounds, populationSize);
        if (!x0.empty()) {
            for (size_t i = 0; i < x0.size() && i < population.size(); ++i) {
                if (x0[i].size() == bounds.size()) {
                    population[i] = x0[i];
                }
            }
        }

        fitness = func(population, data);
        Nevals = fitness.size();

        F = std::vector<double>(populationSize, baseF);
        CR = std::vector<double>(populationSize, baseCR);

        size_t bestIndex = findArgMin(fitness);
        best = population[bestIndex];
        best_fitness = fitness[bestIndex];
        minionResult = MinionResult(best, best_fitness, 0, Nevals, false, "");
        history.push_back(minionResult);
        if (callback != nullptr) {
            callback(&minionResult);
        }

        size_t maxFES = maxevals;
        size_t cycle = 0;
        age = 0;
        nReset = 0;
        sReset = 0;

        while (Nevals < maxFES) {
            size_t idx = cycle % (2 * bigPopulationSize);
            bool inBig = idx < bigPopulationSize;

            if (idx == 0) {
                if (too_many_equals(fitness, bigPopulationSize, fitness[bestIndex]) || age > maxFES / 10) {
                    ++nReset;
                    reinitialize_range(0, bigPopulationSize);
                    age = 0;
                    bestIndex = bigPopulationSize;
                    for (size_t k = bigPopulationSize + 1; k < populationSize; ++k) {
                        if (fitness[k] < fitness[bestIndex]) {
                            bestIndex = k;
                        }
                    }
                    best = population[bestIndex];
                    best_fitness = fitness[bestIndex];
                }
            }

            if (idx == bigPopulationSize) {
                std::vector<double> smallFitness(fitness.begin() + bigPopulationSize, fitness.end());
                if (bestIndex >= bigPopulationSize && too_many_equals(smallFitness, smallPopulationSize, fitness[bestIndex])) {
                    ++sReset;
                    reinitialize_range(bigPopulationSize, populationSize, bestIndex);
                }

                if (bestIndex < bigPopulationSize && smallPopulationSize > 0) {
                    size_t target = bigPopulationSize;
                    population[target] = population[bestIndex];
                    fitness[target] = fitness[bestIndex];
                    F[target] = F[bestIndex];
                    CR[target] = CR[bestIndex];
                    bestIndex = target;
                }
            }

            size_t individualIndex;
            if (inBig) {
                individualIndex = idx;
            } else {
                size_t localIdx = (idx - bigPopulationSize) % smallPopulationSize;
                individualIndex = bigPopulationSize + localIdx;
            }

            double Fl = inBig ? 0.01 : 0.17;
            double CRl = inBig ? 0.0 : 0.1;
            double CRu = inBig ? 1.0 : 0.7;

            int mig = 0;
            if (inBig) {
                if (Nevals < maxFES / 3) {
                    mig = 1;
                } else if (Nevals < (2 * maxFES) / 3) {
                    mig = 2;
                } else {
                    mig = 3;
                }
            }

            size_t r1, r2, r3;
            if (inBig) {
                do {
                    r1 = rand_int(bigPopulationSize + 1);
                } while (r1 == individualIndex && r1 == bestIndex);
                do {
                    r2 = rand_int(bigPopulationSize + static_cast<size_t>(mig));
                } while (r2 == individualIndex || r2 == r1);
                do {
                    r3 = rand_int(bigPopulationSize + static_cast<size_t>(mig));
                } while (r3 == individualIndex || r3 == r2 || r3 == r1);
            } else {
                size_t offset = bigPopulationSize;
                size_t localIdx = individualIndex - offset;
                do {
                    r1 = rand_int(smallPopulationSize);
                } while (r1 == localIdx);
                do {
                    r2 = rand_int(smallPopulationSize);
                } while (r2 == localIdx || r2 == r1);
                do {
                    r3 = rand_int(smallPopulationSize);
                } while (r3 == localIdx || r3 == r2 || r3 == r1);
                r1 += offset;
                r2 += offset;
                r3 += offset;
            }

            double mutatedF = F[individualIndex];
            if (rand_gen() < tau1) {
                mutatedF = Fl + rand_gen() * Fu;
            }

            double mutatedCR = CR[individualIndex];
            if (rand_gen() < tau2) {
                mutatedCR = CRl + rand_gen() * (CRu - CRl);
            }

            size_t jrand = rand_int(bounds.size());
            std::vector<double> trial = population[individualIndex];
            for (size_t d = 0; d < bounds.size(); ++d) {
                if (rand_gen() < mutatedCR || d == jrand) {
                    trial[d] = population[r1][d] + mutatedF * (population[r2][d] - population[r3][d]);
                }
            }
            enforce_bounds(trial, bounds, boundStrategy);

            double trialCost = func({trial}, data)[0];
            ++Nevals;

            if (inBig) {
                ++age;
                individualIndex = crowding_index(trial);
            }

            bool improvesCurrent = trialCost <= fitness[individualIndex];

            if (improvesCurrent) {
                population[individualIndex] = trial;
                fitness[individualIndex] = trialCost;
                F[individualIndex] = mutatedF;
                CR[individualIndex] = mutatedCR;
            }

            if (trialCost < best_fitness) {
                age = 0;
                bestIndex = individualIndex;
                best = trial;
                best_fitness = trialCost;
                minionResult = MinionResult(best, best_fitness, cycle + 1, Nevals, false, "");
                history.push_back(minionResult);
                if (callback != nullptr) {
                    callback(&minionResult);
                }
            }

            double meanFVal = calcMean(F);
            double stdFVal = calcStdDev(F);
            double meanCRVal = calcMean(CR);
            double stdCRVal = calcStdDev(CR);
            meanF.push_back(meanFVal);
            stdF.push_back(stdFVal);
            meanCR.push_back(meanCRVal);
            stdCR.push_back(stdCRVal);

            ++cycle;
            if (Nevals >= maxFES) {
                break;
            }
        }

        return getBestFromHistory();
    } catch (const std::exception& ex) {
        throw std::runtime_error(ex.what());
    }
}

} // namespace minion
