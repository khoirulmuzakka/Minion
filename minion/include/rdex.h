#ifndef RDEX_H
#define RDEX_H

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "minimizer_base.h"

namespace minion {

/**
 * @file rdex.h
 * @brief Header file for the RDEX minimizer.
 *
 * This implementation adapts the standalone RDEx reference code to the Minion
 * optimizer API.
 */
class RDEX : public MinimizerBase {
private:
    int memorySize = 5;
    int memoryIter = 0;
    int successFilled = 0;
    int memoryCurrentIndex = 0;
    int memoryCurrentIndex2 = 0;
    int nVars = 0;
    int nIndsCurrent = 0;
    int nIndsFront = 0;
    int nIndsFrontMax = 0;
    int newNIndsFront = 0;
    int populSize = 0;
    int chosenOne = 0;
    int generation = 0;
    int pfIndex = 0;

    int nfeval = 0;
    int maxFEval = 0;

    double bestfit = 0.0;
    double successRate = 0.5;
    double F = 0.0;
    double Cr = 0.0;
    double globalbest = 0.0;
    double ebHybridRate = 0.7;
    double ebHybridRateInit = 0.7;
    double perturbationRate = 0.4;

    bool globalbestinit = false;

    std::vector<std::vector<double>> popul;
    std::vector<std::vector<double>> populFront;
    std::vector<std::vector<double>> populTemp;
    std::vector<double> fitArr;
    std::vector<double> fitArrCopy;
    std::vector<double> fitArrFront;
    std::vector<double> trial;
    std::vector<double> tempSuccessCr;
    std::vector<double> tempSuccessF;
    std::vector<double> memoryCr;
    std::vector<double> memoryF;
    std::vector<double> fitDelta;
    std::vector<double> weights;
    std::vector<double> fitMass;
    std::vector<int> indices;
    std::vector<int> indices2;

private:
    void initialize_population(int newNInds, int newNVars);
    void MainCycle();
    void FindNSaveBest(bool init, int index);
    void UpdateMemory();
    double MeanWL(const std::vector<double>& values, const std::vector<double>& tempWeights) const;
    void RemoveWorst(int currentSize, int newSize);
    void qSort2int(double* mass, int* mass2, int low, int high);
    void EBOrder(
        int prand,
        int rand1,
        int rand2,
        const std::vector<double>*& bestVec,
        const std::vector<double>*& mediumVec,
        const std::vector<double>*& worstVec) const;
    void UpdateEBHybridParam(
        const std::vector<int>& hybridFlags,
        const std::vector<double>& previousFit,
        const std::vector<double>& trialFit);

public:
    RDEX(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>())
        : MinimizerBase(func, bounds, x0, data, callback, maxevals, seed, options) {}

    ~RDEX() override = default;

    MinionResult optimize() override {
        if (!hasInitialized) {
            initialize();
        }
        MainCycle();
        return getBestFromHistory();
    }

    void initialize() override;
};

}  // namespace minion

#endif
