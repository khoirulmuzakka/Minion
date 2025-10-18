#include "lshadecnepsin.h"
#include "default_options.h"
#include "utility.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace minion {

namespace {
constexpr double PI_CONST = 3.14159265358979323846;
constexpr double CONDITION_LIMIT = 1e20;
}

void LSHADE_cnEpSin::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("LSHADE_cnEpSin");
    for (const auto& kv : optionMap) {
        defaults[kv.first] = kv.second;
    }
    optionMap = defaults;

    Options options(defaults);

    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy) == all_boundStrategy.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    size_t dimension = bounds.size();
    populationSize = options.get<int>("population_size", 0);
    if (populationSize == 0) {
        populationSize = std::max<size_t>(18 * std::max<size_t>(dimension, size_t(1)), size_t(5));
    }
    maxPopulationSize = populationSize;
    minPopulationSize = static_cast<size_t>(std::max<int>(options.get<int>("minimum_population_size", 4), 4));

    archiveRate = options.get<double>("archive_rate", 1.4);
    pBestRate = options.get<double>("p_best_fraction", 0.11);
    rotationProbability = options.get<double>("rotation_probability", 0.4);
    neighbourhoodFraction = options.get<double>("neighborhood_fraction", 0.5);
    freqInit = options.get<double>("freq_init", 0.5);
    sinFreqBase = options.get<double>("sin_freq_base", 0.5);
    memorySize = static_cast<size_t>(std::max<int>(options.get<int>("memory_size", 5), 1));
    learningPeriod = static_cast<size_t>(std::max<int>(options.get<int>("learning_period", 20), 1));
    epsilon = options.get<double>("epsilon", 1e-8);

    memorySF.assign(memorySize, 0.5);
    memoryCR.assign(memorySize, 0.5);
    memoryFreq.assign(memorySize, freqInit);
    memoryPos = 0;

    successHistory1.clear();
    failureHistory1.clear();
    successHistory2.clear();
    failureHistory2.clear();

    lastCR.clear();
    lastF.clear();
    lastFreq.clear();

    generationCounter = 0;
    estimatedMaxGenerations = std::max<size_t>(1, computeGmax(dimension));

    hasInitialized = true;
}

size_t LSHADE_cnEpSin::computeGmax(size_t dimension) const {
    switch (dimension) {
        case 10: return 2163;
        case 30: return 2745;
        case 50: return 3022;
        case 100: return 3401;
        default:
            if (maxPopulationSize == 0) return 1;
            return std::max<size_t>(1, static_cast<size_t>(std::ceil(static_cast<double>(maxevals) / static_cast<double>(maxPopulationSize))));
    }
}

void LSHADE_cnEpSin::sampleParameters(const std::vector<size_t>& /*sortedIndices*/,
                                      std::vector<double>& muSF,
                                      std::vector<double>& muCR,
                                      std::vector<double>& muFreq,
                                      std::vector<size_t>& memIndices,
                                      std::vector<double>& CRvec,
                                      std::vector<double>& Fvec,
                                      std::vector<double>& freqVec,
                                      bool& usedStrategy1,
                                      bool& usedStrategy2) {
    size_t popSize = population.size();
    muSF.resize(popSize);
    muCR.resize(popSize);
    muFreq.resize(popSize);
    memIndices.resize(popSize);
    CRvec.resize(popSize);
    Fvec.resize(popSize);
    freqVec.resize(popSize);

    for (size_t i = 0; i < popSize; ++i) {
        size_t idx = rand_int(memorySize);
        memIndices[i] = idx;
        muSF[i] = memorySF[idx];
        muCR[i] = memoryCR[idx];
        muFreq[i] = memoryFreq[idx];
    }

    for (size_t i = 0; i < popSize; ++i) {
        double value;
        do {
            value = muSF[i] + 0.1 * std::tan(PI_CONST * (rand_gen() - 0.5));
        } while (value <= 0.0);
        value = std::min(1.0, value);
        Fvec[i] = value;
    }

    for (size_t i = 0; i < popSize; ++i) {
        double value;
        do {
            value = muFreq[i] + 0.1 * std::tan(PI_CONST * (rand_gen() - 0.5));
        } while (value <= 0.0);
        value = std::min(1.0, value);
        freqVec[i] = value;
    }

    for (size_t i = 0; i < popSize; ++i) {
        double value = rand_norm(muCR[i], 0.1);
        value = clamp(value, 0.0, 1.0);
        if (muCR[i] < 0.0) value = 0.0;
        CRvec[i] = value;
    }

    usedStrategy1 = false;
    usedStrategy2 = false;

    bool inSinusoidalPhase = (Nevals <= maxevals / 2);
    if (!inSinusoidalPhase) {
        return;
    }

    double p1 = 0.5;
    double p2 = 0.5;

    if (generationCounter > learningPeriod) {
        double success1 = 0.0;
        double failure1 = 0.0;
        for (size_t v : successHistory1) success1 += static_cast<double>(v);
        for (size_t v : failureHistory1) failure1 += static_cast<double>(v);

        double success2 = 0.0;
        double failure2 = 0.0;
        for (size_t v : successHistory2) success2 += static_cast<double>(v);
        for (size_t v : failureHistory2) failure2 += static_cast<double>(v);

        double sumS1 = ((success1 + failure1) > 0.0 ? (success1 / (success1 + failure1)) : 0.0) + 0.01;
        double sumS2 = ((success2 + failure2) > 0.0 ? (success2 / (success2 + failure2)) : 0.0) + 0.01;
        double denom = sumS1 + sumS2;
        if (denom <= epsilon) {
            p1 = 0.5;
            p2 = 0.5;
        } else {
            p1 = sumS1 / denom;
            p2 = sumS2 / denom;
        }
    }

    double pick = rand_gen();
    if (pick < p1) {
        usedStrategy1 = true;
    } else {
        usedStrategy2 = true;
    }

    if (usedStrategy1) {
        double progress = static_cast<double>(generationCounter) / static_cast<double>(std::max<size_t>(1, estimatedMaxGenerations));
        double amplitude = std::max(0.0, static_cast<double>(estimatedMaxGenerations - generationCounter)) / static_cast<double>(std::max<size_t>(1, estimatedMaxGenerations));
        double baseValue = 0.5 * (std::sin(2.0 * PI_CONST * sinFreqBase * generationCounter + PI_CONST) * amplitude + 1.0);
        baseValue = clamp(baseValue, 0.01, 1.0);
        std::fill(Fvec.begin(), Fvec.end(), baseValue);
    } else if (usedStrategy2) {
        for (size_t i = 0; i < popSize; ++i) {
            double progress = static_cast<double>(generationCounter) / static_cast<double>(std::max<size_t>(1, estimatedMaxGenerations));
            double value = 0.5 * (std::sin(2.0 * PI_CONST * freqVec[i] * generationCounter) * progress + 1.0);
            Fvec[i] = clamp(value, 0.01, 1.0);
        }
    }
}

void LSHADE_cnEpSin::updateHistories(bool usedStrategy1, bool usedStrategy2, size_t good1, size_t bad1, size_t good2, size_t bad2) {
    if (usedStrategy1) {
        successHistory1.push_back(good1);
        failureHistory1.push_back(bad1);
        successHistory2.push_back(1);
        failureHistory2.push_back(1);
    } else if (usedStrategy2) {
        successHistory1.push_back(1);
        failureHistory1.push_back(1);
        successHistory2.push_back(good2);
        failureHistory2.push_back(bad2);
    } else {
        successHistory1.push_back(1);
        failureHistory1.push_back(1);
        successHistory2.push_back(1);
        failureHistory2.push_back(1);
    }

    while (successHistory1.size() > learningPeriod) successHistory1.pop_front();
    while (failureHistory1.size() > learningPeriod) failureHistory1.pop_front();
    while (successHistory2.size() > learningPeriod) successHistory2.pop_front();
    while (failureHistory2.size() > learningPeriod) failureHistory2.pop_front();
}

void LSHADE_cnEpSin::updateMemories(const std::vector<size_t>& successIndices,
                                    const std::vector<double>& difValues,
                                    const std::vector<double>& CRvec,
                                    const std::vector<double>& Fvec,
                                    const std::vector<double>& freqVec) {
    if (successIndices.empty()) {
        return;
    }

    std::vector<double> weights = difValues;
    double sumDif = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumDif <= 0.0) {
        double uniform = 1.0 / static_cast<double>(weights.size());
        std::fill(weights.begin(), weights.end(), uniform);
    } else {
        for (double& w : weights) {
            w /= sumDif;
        }
    }

    auto lehmerMean = [&](const std::vector<double>& values) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t i = 0; i < values.size(); ++i) {
            numerator += weights[i] * values[i] * values[i];
            denominator += weights[i] * values[i];
        }
        if (denominator <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return numerator / denominator;
    };

    std::vector<double> goodF;
    std::vector<double> goodCR;
    std::vector<double> goodFreq;
    goodF.reserve(successIndices.size());
    goodCR.reserve(successIndices.size());
    goodFreq.reserve(successIndices.size());
    for (size_t idx : successIndices) {
        goodF.push_back(Fvec[idx]);
        goodCR.push_back(CRvec[idx]);
        goodFreq.push_back(freqVec[idx]);
    }

    double newSF = lehmerMean(goodF);
    if (std::isnan(newSF) || newSF <= 0.0) newSF = 0.5;

    double newCR;
    double maxCR = *std::max_element(goodCR.begin(), goodCR.end());
    if (maxCR == 0.0) {
        newCR = -1.0;
    } else {
        newCR = lehmerMean(goodCR);
        if (std::isnan(newCR)) newCR = 0.5;
        newCR = clamp(newCR, 0.0, 1.0);
    }

    double newFreq;
    double maxFreq = *std::max_element(goodFreq.begin(), goodFreq.end());
    if (maxFreq == 0.0) {
        newFreq = -1.0;
    } else {
        newFreq = lehmerMean(goodFreq);
        if (std::isnan(newFreq) || newFreq <= 0.0) newFreq = freqInit;
        newFreq = std::max(1e-6, newFreq);
    }

    memorySF[memoryPos] = newSF;
    memoryCR[memoryPos] = newCR;
    memoryFreq[memoryPos] = newFreq;
    memoryPos = (memoryPos + 1) % memorySize;
}

void LSHADE_cnEpSin::enforceArchiveLimit() {
    if (archive.empty()) return;
    size_t capacity = static_cast<size_t>(std::round(archiveRate * population.size()));
    capacity = std::max<size_t>(capacity, 1);
    while (archive.size() > capacity) {
        size_t idx = rand_int(archive.size());
        archive.erase(archive.begin() + static_cast<long>(idx));
        if (idx < archive_fitness.size()) {
            archive_fitness.erase(archive_fitness.begin() + static_cast<long>(idx));
        }
    }
}

void LSHADE_cnEpSin::reducePopulationIfNeeded() {
    size_t currentPop = population.size();
    if (currentPop <= minPopulationSize) {
        populationSize = currentPop;
        lastCR.resize(populationSize, 0.5);
        lastF.resize(populationSize, 0.5);
        lastFreq.resize(populationSize, freqInit);
        return;
    }

    if (maxevals == 0) {
        populationSize = currentPop;
        return;
    }

    double nfes = static_cast<double>(Nevals);
    double plan = (((static_cast<double>(minPopulationSize) - static_cast<double>(maxPopulationSize)) / static_cast<double>(maxevals)) * nfes) + static_cast<double>(maxPopulationSize);
    size_t target = static_cast<size_t>(std::round(plan));
    if (target < minPopulationSize) target = minPopulationSize;

    if (currentPop > target) {
        auto sorted = argsort(fitness, true);
        std::vector<size_t> worstIndices(sorted.begin() + target, sorted.end());
        std::sort(worstIndices.rbegin(), worstIndices.rend());
        for (size_t idx : worstIndices) {
            population.erase(population.begin() + static_cast<long>(idx));
            fitness.erase(fitness.begin() + static_cast<long>(idx));
        }
        populationSize = population.size();
        enforceArchiveLimit();
    } else {
        populationSize = currentPop;
    }

    lastCR.resize(populationSize, 0.5);
    lastF.resize(populationSize, 0.5);
    lastFreq.resize(populationSize, freqInit);
}

void LSHADE_cnEpSin::buildNeighbourCovariance(const std::vector<size_t>& sortedIndices,
                                              Eigen::MatrixXd& eigenBasis,
                                              Eigen::MatrixXd& eigenBasisT,
                                              size_t currentPopSize) {
    size_t dim = bounds.size();
    if (currentPopSize < 2 || dim == 0) {
        eigenBasis = Eigen::MatrixXd::Identity(dim, dim);
        eigenBasisT = eigenBasis.transpose();
        return;
    }

    size_t sel = static_cast<size_t>(std::round(neighbourhoodFraction * static_cast<double>(currentPopSize)));
    sel = std::min(std::max<size_t>(sel, 2), currentPopSize);

    const auto& bestVec = population[sortedIndices.front()];
    std::vector<std::pair<double, size_t>> distances;
    distances.reserve(currentPopSize);
    for (size_t i = 0; i < currentPopSize; ++i) {
        double dist = euclideanDistance(bestVec, population[i]);
        distances.emplace_back(dist, i);
    }
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    Eigen::MatrixXd data(sel, dim);
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
    for (size_t row = 0; row < sel; ++row) {
        size_t idx = distances[row].second;
        for (size_t col = 0; col < dim; ++col) {
            double value = population[idx][col];
            data(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = value;
            mean[static_cast<Eigen::Index>(col)] += value;
        }
    }
    mean /= static_cast<double>(sel);

    for (size_t row = 0; row < sel; ++row) {
        for (size_t col = 0; col < dim; ++col) {
            data(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) -= mean[static_cast<Eigen::Index>(col)];
        }
    }

    Eigen::MatrixXd cov = (data.transpose() * data) / std::max<double>(1.0, static_cast<double>(sel - 1));
    cov = 0.5 * (cov + cov.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    if (solver.info() != Eigen::Success) {
        eigenBasis = Eigen::MatrixXd::Identity(dim, dim);
        eigenBasisT = eigenBasis.transpose();
        return;
    }

    Eigen::VectorXd eigenValues = solver.eigenvalues();
    double minVal = eigenValues.minCoeff();
    double maxVal = eigenValues.maxCoeff();
    if (minVal <= 0.0 || maxVal / std::max(minVal, 1e-30) > CONDITION_LIMIT) {
        double adjustment = maxVal / CONDITION_LIMIT - minVal;
        cov += adjustment * Eigen::MatrixXd::Identity(dim, dim);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solverAdjusted(cov);
        if (solverAdjusted.info() != Eigen::Success) {
            eigenBasis = Eigen::MatrixXd::Identity(dim, dim);
            eigenBasisT = eigenBasis.transpose();
            return;
        }
        eigenBasis = solverAdjusted.eigenvectors();
    } else {
        eigenBasis = solver.eigenvectors();
    }
    eigenBasisT = eigenBasis.transpose();
}

MinionResult LSHADE_cnEpSin::optimize() {
    if (!hasInitialized) initialize();
    try {
        history.clear();
        diversity.clear();
        meanCR.clear();
        meanF.clear();
        stdCR.clear();
        stdF.clear();

        Nevals = 0;
        archive.clear();
        archive_fitness.clear();

        init();

        maxPopulationSize = population.size();
        populationSize = population.size();
        lastCR.assign(populationSize, 0.5);
        lastF.assign(populationSize, 0.5);
        lastFreq.assign(populationSize, freqInit);

        size_t dim = bounds.size();

        while (Nevals < maxevals) {
            ++generationCounter;
            size_t currentPopSize = population.size();
            if (currentPopSize == 0) {
                break;
            }

            std::vector<std::vector<double>> populationOld = population;
            std::vector<double> fitnessOld = fitness;
            auto sortedIndices = argsort(fitness, true);

            std::vector<double> muSF, muCR, muFreq;
            std::vector<size_t> memIndices;
            std::vector<double> CRvec, Fvec, freqVec;
            bool usedStrategy1 = false;
            bool usedStrategy2 = false;
            sampleParameters(sortedIndices, muSF, muCR, muFreq, memIndices, CRvec, Fvec, freqVec, usedStrategy1, usedStrategy2);

            lastCR = CRvec;
            lastF = Fvec;
            lastFreq = freqVec;

            meanCR.push_back(calcMean(CRvec));
            meanF.push_back(calcMean(Fvec));
            stdCR.push_back(calcStdDev(CRvec));
            stdF.push_back(calcStdDev(Fvec));

            std::vector<std::vector<double>> mutants(currentPopSize, std::vector<double>(dim, 0.0));
            std::vector<std::vector<double>> trials(currentPopSize, std::vector<double>(dim, 0.0));

            std::vector<std::vector<double>> combined = populationOld;
            combined.insert(combined.end(), archive.begin(), archive.end());
            size_t combinedSize = combined.size();

            for (size_t i = 0; i < currentPopSize; ++i) {
                size_t r1;
                do {
                    r1 = rand_int(currentPopSize);
                } while (r1 == i);

                size_t r2;
                if (combinedSize > 1) {
                    do {
                        r2 = rand_int(combinedSize);
                        if (r2 < currentPopSize) {
                            if (r2 == i || r2 == r1) continue;
                        }
                        break;
                    } while (true);
                } else {
                    r2 = r1;
                }

                size_t pCount = std::max<size_t>(2, static_cast<size_t>(std::round(pBestRate * currentPopSize)));
                size_t pick = rand_int(pCount);
                pick = std::min(pick, pCount - 1);
                size_t pbestIdx = sortedIndices[pick];

                const auto& xi = populationOld[i];
                const auto& xpbest = populationOld[pbestIdx];
                const auto& xr1 = populationOld[r1];
                const auto& xr2 = (r2 < currentPopSize) ? populationOld[r2] : archive[r2 - currentPopSize];

                for (size_t d = 0; d < dim; ++d) {
                    mutants[i][d] = xi[d] + Fvec[i] * ((xpbest[d] - xi[d]) + (xr1[d] - xr2[d]));
                }
            }

            enforce_bounds(mutants, bounds, boundStrategy);

            bool useRotation = (rand_gen() < rotationProbability);
            Eigen::MatrixXd eigenBasis;
            Eigen::MatrixXd eigenBasisT;
            if (useRotation) {
                buildNeighbourCovariance(sortedIndices, eigenBasis, eigenBasisT, currentPopSize);
            }

            for (size_t i = 0; i < currentPopSize; ++i) {
                size_t jRand = rand_int(dim == 0 ? 1 : dim);
                if (useRotation) {
                    Eigen::VectorXd target(dim);
                    Eigen::VectorXd mutant(dim);
                    for (size_t d = 0; d < dim; ++d) {
                        target[static_cast<Eigen::Index>(d)] = populationOld[i][d];
                        mutant[static_cast<Eigen::Index>(d)] = mutants[i][d];
                    }
                    Eigen::VectorXd targetEig = eigenBasisT * target;
                    Eigen::VectorXd mutantEig = eigenBasisT * mutant;
                    Eigen::VectorXd trialEig = targetEig;
                    for (size_t d = 0; d < dim; ++d) {
                        double randVal = rand_gen();
                        if (randVal < CRvec[i] || d == jRand) {
                            trialEig[static_cast<Eigen::Index>(d)] = mutantEig[static_cast<Eigen::Index>(d)];
                        }
                    }
                    Eigen::VectorXd trialVec = eigenBasis * trialEig;
                    for (size_t d = 0; d < dim; ++d) {
                        trials[i][d] = trialVec[static_cast<Eigen::Index>(d)];
                    }
                } else {
                    trials[i] = populationOld[i];
                    for (size_t d = 0; d < dim; ++d) {
                        double randVal = rand_gen();
                        if (randVal < CRvec[i] || d == jRand) {
                            trials[i][d] = mutants[i][d];
                        }
                    }
                }
            }

            enforce_bounds(trials, bounds, boundStrategy);

            trial_fitness = func(trials, data);
            Nevals += trials.size();
            for (double& val : trial_fitness) {
                if (std::isnan(val)) val = 1e100;
            }

            std::vector<size_t> successIndices;
            std::vector<double> difValues;
            successIndices.reserve(currentPopSize);
            difValues.reserve(currentPopSize);

            std::vector<std::vector<double>> parentsToArchive;
            std::vector<double> parentFitnessToArchive;
            parentsToArchive.reserve(currentPopSize);
            parentFitnessToArchive.reserve(currentPopSize);

            for (size_t i = 0; i < currentPopSize; ++i) {
                double parentFit = fitnessOld[i];
                double childFit = trial_fitness[i];
                double diff = std::fabs(parentFit - childFit);
                if (childFit < parentFit) {
                    population[i] = trials[i];
                    fitness[i] = childFit;
                    parentsToArchive.push_back(populationOld[i]);
                    parentFitnessToArchive.push_back(parentFit);
                    successIndices.push_back(i);
                    difValues.push_back(diff);
                } else {
                    population[i] = populationOld[i];
                    fitness[i] = parentFit;
                }
            }

            for (size_t k = 0; k < parentsToArchive.size(); ++k) {
                archive.push_back(parentsToArchive[k]);
                archive_fitness.push_back(parentFitnessToArchive[k]);
            }
            enforceArchiveLimit();

            size_t successCount = successIndices.size();
            size_t failureCount = currentPopSize - successCount;
            size_t good1 = usedStrategy1 ? successCount : 0;
            size_t bad1 = usedStrategy1 ? failureCount : 0;
            size_t good2 = usedStrategy2 ? successCount : 0;
            size_t bad2 = usedStrategy2 ? failureCount : 0;
            updateHistories(usedStrategy1, usedStrategy2, good1, bad1, good2, bad2);

            updateMemories(successIndices, difValues, CRvec, Fvec, freqVec);

            if (!fitness.empty()) {
                size_t bestIdx = findArgMin(fitness);
                best = population[bestIdx];
                best_fitness = fitness[bestIdx];
            }

            minionResult = MinionResult(best, best_fitness, generationCounter, Nevals, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) callback(&minionResult);

            reducePopulationIfNeeded();

            if (support_tol && checkStopping()) {
                break;
            }
            if (Nevals >= maxevals) {
                break;
            }
        }

        return getBestFromHistory();
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

}
