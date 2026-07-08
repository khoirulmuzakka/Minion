#include "rdex.h"

#include <algorithm>
#include <array>

#include "default_options.h"

namespace minion {

void RDEX::initialize() {
    auto defaultKey = DefaultSettings().getDefaultSettings("RDEX");
    for (const auto& el : optionMap) {
        defaultKey[el.first] = el.second;
    }
    Options options(defaultKey);

    boundStrategy = options.get<std::string>("bound_strategy", "random");
    const std::vector<std::string> allBoundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(allBoundStrategy.begin(), allBoundStrategy.end(), boundStrategy) == allBoundStrategy.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'random' will be used.\n";
        boundStrategy = "random";
    }

    populSize = options.get<int>("population_size", 0);
    if (populSize == 0) {
        populSize = int(18 * bounds.size());
    }

    memorySize = options.get<int>("memory_size", 5);
    successRate = options.get<double>("success_rate", 0.5);
    ebHybridRateInit = options.get<double>("eb_hybrid_rate_init", 0.7);
    ebHybridRate = ebHybridRateInit;
    perturbationRate = options.get<double>("perturbation_rate", 0.4);
    maxFEval = int(maxevals);

    initialize_population(populSize, int(bounds.size()));
    hasInitialized = true;
}

void RDEX::qSort2int(double* mass, int* mass2, int low, int high) {
    int i = low;
    int j = high;
    const double x = mass[(low + high) >> 1];
    do {
        while (mass[i] < x) {
            ++i;
        }
        while (mass[j] > x) {
            --j;
        }
        if (i <= j) {
            std::swap(mass[i], mass[j]);
            std::swap(mass2[i], mass2[j]);
            ++i;
            --j;
        }
    } while (i <= j);
    if (low < j) {
        qSort2int(mass, mass2, low, j);
    }
    if (i < high) {
        qSort2int(mass, mass2, i, high);
    }
}

void RDEX::initialize_population(int newNInds, int newNVars) {
    nVars = newNVars;
    nIndsCurrent = newNInds;
    nIndsFront = newNInds;
    nIndsFrontMax = newNInds;
    populSize = newNInds * 2;
    generation = 0;
    chosenOne = 0;
    memoryIter = 0;
    successFilled = 0;
    globalbestinit = false;
    nfeval = 0;
    pfIndex = 0;

    popul.assign(populSize, std::vector<double>(nVars, 0.0));
    populFront.assign(nIndsFront, std::vector<double>(nVars, 0.0));
    populTemp.assign(populSize, std::vector<double>(nVars, 0.0));
    fitArr.assign(populSize, 0.0);
    fitArrCopy.assign(populSize, 0.0);
    fitArrFront.assign(nIndsFront, 0.0);
    weights.assign(populSize, 0.0);
    tempSuccessCr.assign(populSize, 0.0);
    tempSuccessF.assign(populSize, 0.0);
    fitDelta.assign(populSize, 0.0);
    memoryCr.assign(memorySize, 1.0);
    memoryF.assign(memorySize, 1.0);
    trial.assign(nVars, 0.0);
    indices.assign(populSize, 0);
    indices2.assign(populSize, 0);
    fitMass.assign(nIndsFrontMax, 0.0);

    popul = random_sampling(bounds, populSize);
    if (!x0.empty()) {
        for (size_t i = 0; i < x0.size() && i < popul.size(); ++i) {
            popul[i] = x0[i];
        }
    }
}

void RDEX::UpdateMemory() {
    if (successFilled != 0) {
        memoryCr[memoryIter] = 0.5 * (MeanWL(tempSuccessCr, fitDelta) + memoryCr[memoryIter]);
        memoryF[memoryIter] = MeanWL(tempSuccessF, fitDelta);
        memoryIter = (memoryIter + 1) % memorySize;
    }
}

double RDEX::MeanWL(const std::vector<double>& values, const std::vector<double>& tempWeights) const {
    double sumWeight = 0.0;
    double sumSquare = 0.0;
    double sum = 0.0;

    for (int i = 0; i != successFilled; ++i) {
        sumWeight += tempWeights[i];
    }
    if (sumWeight <= 0.0) {
        return 1.0;
    }

    for (int i = 0; i != successFilled; ++i) {
        const double weight = tempWeights[i] / sumWeight;
        sumSquare += weight * values[i] * values[i];
        sum += weight * values[i];
    }
    if (std::fabs(sum) > 1e-8) {
        return sumSquare / sum;
    }
    return 1.0;
}

void RDEX::FindNSaveBest(bool init, int index) {
    if (fitArr[index] <= bestfit || init) {
        bestfit = fitArr[index];
    }
    if (bestfit < globalbest || init) {
        globalbest = bestfit;
    }
}

void RDEX::RemoveWorst(int currentSize, int newSize) {
    const int pointsToRemove = currentSize - newSize;
    for (int remove = 0; remove != pointsToRemove; ++remove) {
        double worstFit = fitArrFront[0];
        int worstNum = 0;
        for (int i = 1; i != currentSize; ++i) {
            if (fitArrFront[i] > worstFit) {
                worstFit = fitArrFront[i];
                worstNum = i;
            }
        }
        for (int i = worstNum; i != currentSize - 1; ++i) {
            populFront[i] = populFront[i + 1];
            fitArrFront[i] = fitArrFront[i + 1];
            fitMass[i] = fitMass[i + 1];
        }
    }
}

void RDEX::EBOrder(
    int prand,
    int rand1,
    int rand2,
    const std::vector<double>*& bestVec,
    const std::vector<double>*& mediumVec,
    const std::vector<double>*& worstVec) const {
    struct Candidate {
        const std::vector<double>* vec;
        double fit;
    };

    std::array<Candidate, 3> ordered = {{
        {&popul[prand], fitArr[prand]},
        {&populFront[rand1], fitArrFront[rand1]},
        {&popul[rand2], fitArr[rand2]},
    }};

    std::sort(ordered.begin(), ordered.end(), [](const Candidate& a, const Candidate& b) {
        return a.fit < b.fit;
    });

    bestVec = ordered[0].vec;
    mediumVec = ordered[1].vec;
    worstVec = ordered[2].vec;
}

void RDEX::UpdateEBHybridParam(
    const std::vector<int>& hybridFlags,
    const std::vector<double>& previousFit,
    const std::vector<double>& trialFit) {
    double sumEbDeltaFit = 0.0;
    double sumOriginDeltaFit = 0.0;

    for (size_t i = 0; i < hybridFlags.size(); ++i) {
        if (trialFit[i] > previousFit[i]) {
            continue;
        }
        const double delta = previousFit[i] - trialFit[i];
        if (hybridFlags[i] == 1) {
            sumEbDeltaFit += delta;
        } else {
            sumOriginDeltaFit += delta;
        }
    }

    if (sumEbDeltaFit != 0.0 && sumOriginDeltaFit != 0.0) {
        ebHybridRate = sumEbDeltaFit / (sumEbDeltaFit + sumOriginDeltaFit);
        ebHybridRate = std::clamp(ebHybridRate, 0.0, 1.0);
    } else {
        ebHybridRate = ebHybridRateInit;
    }
}

void RDEX::MainCycle() {
    resetBestSoFar();

    std::vector<std::vector<double>> pop;
    pop.reserve(nIndsFront);
    for (int i = 0; i < nIndsFront; ++i) {
        pop.push_back(popul[i]);
    }

    std::vector<double> funPop = func(pop, data);
    nfeval += int(funPop.size());
    for (int i = 0; i < nIndsFront; ++i) {
        fitArr[i] = funPop[i];
        FindNSaveBest(i == 0, i);
        if (!globalbestinit || bestfit < globalbest) {
            globalbest = bestfit;
            globalbestinit = true;
        }
    }

    size_t bestIndex = findArgMin(funPop);
    minionResult = MinionResult(pop[bestIndex], funPop[bestIndex], generation, nfeval, false, "");
    updateBestSoFar(minionResult);

    double minfit = fitArr[0];
    double maxfit = fitArr[0];
    for (int i = 0; i < nIndsFront; ++i) {
        fitArrCopy[i] = fitArr[i];
        indices[i] = i;
        maxfit = std::max(maxfit, fitArr[i]);
        minfit = std::min(minfit, fitArr[i]);
    }
    if (minfit != maxfit) {
        qSort2int(fitArrCopy.data(), indices.data(), 0, nIndsFront - 1);
    }
    for (int i = 0; i < nIndsFront; ++i) {
        populFront[i] = popul[indices[i]];
        fitArrFront[i] = fitArrCopy[i];
        fitMass[i] = fitArrFront[i];
    }

    while (nfeval < maxFEval) {
        const double progress = static_cast<double>(nfeval) / static_cast<double>(maxFEval);
        const double meanF = 0.4 + std::tanh(successRate * 5.0) * 0.25;
        const double sigmaF = 0.02;

        minfit = fitArr[0];
        maxfit = fitArr[0];
        for (int i = 0; i < nIndsFront; ++i) {
            fitArrCopy[i] = fitArr[i];
            indices[i] = i;
            maxfit = std::max(maxfit, fitArr[i]);
            minfit = std::min(minfit, fitArr[i]);
        }
        if (minfit != maxfit) {
            qSort2int(fitArrCopy.data(), indices.data(), 0, nIndsFront - 1);
        }

        minfit = fitArrFront[0];
        maxfit = fitArrFront[0];
        for (int i = 0; i < nIndsFront; ++i) {
            fitArrCopy[i] = fitArrFront[i];
            indices2[i] = i;
            maxfit = std::max(maxfit, fitArrFront[i]);
            minfit = std::min(minfit, fitArrFront[i]);
        }
        if (minfit != maxfit) {
            qSort2int(fitArrCopy.data(), indices2.data(), 0, nIndsFront - 1);
        }

        std::vector<double> fitTempFront(nIndsFront, 0.0);
        for (int i = 0; i < nIndsFront; ++i) {
            fitTempFront[i] = std::exp(-static_cast<double>(i) / static_cast<double>(nIndsFront) * 3.0);
        }
        std::discrete_distribution<int> componentSelectorFront(fitTempFront.begin(), fitTempFront.end());

        std::vector<double> fitTempPrand(nIndsFront, 0.0);
        for (int i = 0; i < nIndsFront; ++i) {
            fitTempPrand[i] = 3.0 * static_cast<double>(nIndsFront - i);
        }

        int psizeval = std::max(2, int(nIndsFront * 0.7 * std::exp(-successRate * 7.0)));
        int psizeval2 = int(nIndsFront * 0.17 * (1.0 - 0.5 * progress));
        if (psizeval2 <= 1) {
            psizeval2 = 2;
        }
        psizeval2 = std::min(psizeval2, nIndsFront);
        std::discrete_distribution<int> componentSelectorFront2(fitTempPrand.begin(), fitTempPrand.begin() + psizeval2);
        std::discrete_distribution<int> componentSelectorFront3(fitTempPrand.begin(), fitTempPrand.end());

        pop.clear();
        std::vector<int> targetIndices;
        std::vector<int> hybridFlags;
        std::vector<double> actualCrs;
        std::vector<double> usedFs;
        std::vector<double> previousFit;
        targetIndices.reserve(nIndsFront);
        hybridFlags.reserve(nIndsFront);
        actualCrs.reserve(nIndsFront);
        usedFs.reserve(nIndsFront);
        previousFit.reserve(nIndsFront);

        for (int indIter = 0; indIter < nIndsFront; ++indIter) {
            chosenOne = int(rand_int(nIndsFront));
            memoryCurrentIndex = int(rand_int(memorySize));
            memoryCurrentIndex2 = int(rand_int(memorySize + 1));

            int prand = 0;
            int rand1 = 0;
            int rand2 = 0;

            do {
                prand = indices[int(rand_int(psizeval))];
            } while (prand == chosenOne);

            do {
                rand1 = indices2[componentSelectorFront(get_rng())];
            } while (rand1 == prand);

            do {
                rand2 = indices[int(rand_int(nIndsFront))];
            } while (rand2 == prand || rand2 == rand1);

            do {
                F = rand_norm(meanF, sigmaF);
            } while (F < 0.0 || F > 1.0);

            Cr = rand_norm(memoryCr[memoryCurrentIndex], 0.05);
            Cr = std::clamp(Cr, 0.0, 1.0);

            double actualCr = 0.0;
            const int willCrossover = int(rand_int(nVars));

            int useHybrid = 0;
            if (progress >= 0.7) {
                const double randEb = rand_gen(0.0, 1.0);
                if (randEb * (1.0 - progress) < ebHybridRate) {
                    useHybrid = 1;
                }
            }

            const bool perturbation = rand_gen() < perturbationRate;
            if (useHybrid == 1) {
                do {
                    prand = indices[componentSelectorFront2(get_rng())];
                } while (prand == chosenOne);
                do {
                    rand1 = indices2[componentSelectorFront3(get_rng())];
                } while (rand1 == prand);
                do {
                    rand2 = indices[componentSelectorFront3(get_rng())];
                } while (rand2 == prand || rand2 == rand1);

                const std::vector<double>* orderedBest = nullptr;
                const std::vector<double>* orderedMedium = nullptr;
                const std::vector<double>* orderedWorst = nullptr;
                EBOrder(prand, rand1, rand2, orderedBest, orderedMedium, orderedWorst);

                do {
                    if (memoryCurrentIndex2 < memorySize) {
                        F = rand_cauchy(memoryF[memoryCurrentIndex2], 0.1);
                    } else {
                        F = rand_cauchy(0.9, 0.1);
                    }
                } while (F < 0.0);
                if (F > 1.0) {
                    F = 1.0;
                }
                if (progress < 0.6 && F > 0.7) {
                    F = 0.7;
                }

                if (memoryCurrentIndex2 < memorySize) {
                    if (memoryCr[memoryCurrentIndex2] < 0.0) {
                        Cr = 0.0;
                    } else {
                        Cr = rand_norm(memoryCr[memoryCurrentIndex2], 0.1);
                    }
                } else {
                    Cr = rand_norm(0.9, 0.1);
                }
                Cr = std::clamp(Cr, 0.0, 1.0);
                if (progress < 0.25) {
                    Cr = std::max(Cr, 0.7);
                }
                if (progress < 0.5) {
                    Cr = std::max(Cr, 0.6);
                }

                for (int j = 0; j < nVars; ++j) {
                    if (rand_gen(0.0, 1.0) < Cr || willCrossover == j) {
                        trial[j] = populFront[chosenOne][j]
                            + F * ((*orderedBest)[j] - populFront[chosenOne][j])
                            + F * ((*orderedMedium)[j] - (*orderedWorst)[j]);
                        if (trial[j] < bounds[j].first || trial[j] > bounds[j].second) {
                            trial[j] = rand_gen(bounds[j].first, bounds[j].second);
                        }
                        actualCr += 1.0;
                    } else {
                        trial[j] = perturbation ? rand_cauchy(populFront[chosenOne][j], 0.1) : populFront[chosenOne][j];
                    }
                }
            } else {
                for (int j = 0; j < nVars; ++j) {
                    if (rand_gen(0.0, 1.0) < Cr || willCrossover == j) {
                        trial[j] = populFront[chosenOne][j]
                            + F * (popul[prand][j] - populFront[chosenOne][j])
                            + F * (populFront[rand1][j] - popul[rand2][j]);
                        if (trial[j] < bounds[j].first || trial[j] > bounds[j].second) {
                            trial[j] = rand_gen(bounds[j].first, bounds[j].second);
                        }
                        actualCr += 1.0;
                    } else {
                        trial[j] = perturbation ? rand_cauchy(populFront[chosenOne][j], 0.1) : populFront[chosenOne][j];
                    }
                }
            }

            pop.push_back(trial);
            targetIndices.push_back(chosenOne);
            hybridFlags.push_back(useHybrid);
            actualCrs.push_back(actualCr / static_cast<double>(nVars));
            usedFs.push_back(F);
            previousFit.push_back(fitArrFront[chosenOne]);
        }

        funPop = func(pop, data);
        nfeval += int(funPop.size());

        bestIndex = findArgMin(funPop);
        minionResult = MinionResult(pop[bestIndex], funPop[bestIndex], generation, nfeval, false, "");
        updateBestSoFar(minionResult);

        for (int indIter = 0; indIter < nIndsFront; ++indIter) {
            const double tempFit = funPop[indIter];
            chosenOne = targetIndices[indIter];
            if (tempFit <= fitArrFront[chosenOne]) {
                popul[nIndsCurrent + successFilled] = pop[indIter];
                populFront[pfIndex] = pop[indIter];
                fitArr[nIndsCurrent + successFilled] = tempFit;
                fitArrFront[pfIndex] = tempFit;
                FindNSaveBest(false, nIndsCurrent + successFilled);
                tempSuccessCr[successFilled] = actualCrs[indIter];
                tempSuccessF[successFilled] = usedFs[indIter];
                fitDelta[successFilled] = std::fabs(fitArrFront[chosenOne] - tempFit);
                ++successFilled;
                pfIndex = (pfIndex + 1) % nIndsFront;
            }
        }

        UpdateEBHybridParam(hybridFlags, previousFit, funPop);
        for (int i = 0; i < nIndsFront; ++i) {
            fitMass[i] = fitArrFront[i];
        }

        successRate = static_cast<double>(successFilled) / static_cast<double>(nIndsFront);
        newNIndsFront = int(double(4 - nIndsFrontMax) / double(maxFEval) * nfeval + nIndsFrontMax);
        RemoveWorst(nIndsFront, newNIndsFront);
        nIndsFront = newNIndsFront;
        UpdateMemory();
        nIndsCurrent = nIndsFront + successFilled;
        successFilled = 0;
        ++generation;

        if (nIndsCurrent > nIndsFront) {
            minfit = fitArr[0];
            maxfit = fitArr[0];
            for (int i = 0; i < nIndsCurrent; ++i) {
                indices[i] = i;
                maxfit = std::max(maxfit, fitArr[i]);
                minfit = std::min(minfit, fitArr[i]);
            }
            if (minfit != maxfit) {
                qSort2int(fitArr.data(), indices.data(), 0, nIndsCurrent - 1);
            }
            nIndsCurrent = nIndsFront;
            for (int i = 0; i < nIndsCurrent; ++i) {
                populTemp[i] = popul[indices[i]];
            }
            for (int i = 0; i < nIndsCurrent; ++i) {
                popul[i] = populTemp[i];
            }
        }

        if (callback != nullptr) {
            callback(&minionResult);
        }
    }
}

}  // namespace minion
