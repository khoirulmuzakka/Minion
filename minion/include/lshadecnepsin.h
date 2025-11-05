#ifndef LSHADE_CN_EPSIN_H
#define LSHADE_CN_EPSIN_H

#include "de.h"
#include <array>
#include <deque>
#include <Eigen/Dense>

namespace minion {

/**
 * @class LSHADE_cnEpSin
 * @brief Ensemble sinusoidal L-SHADE with covariance learning and Euclidean neighbourhoods.
 *
 * This implementation mirrors the reference MATLAB code proposed by Awad et al.
 * for CEC 2017.  Two sinusoidal strategies compete during the first half of the
 * run, frequency memories are updated via Lehmer means, population size is
 * linearly reduced, and crossover can be performed in a locally learned eigen
 * space.
 */
class LSHADE_cnEpSin : public Differential_Evolution {
public:
    /**
     * @brief Construct the LSHADE-cnEpSin optimizer.
     *
     * @param func Objective function to minimize.
     * @param bounds Variable bounds for the DE population.
     * @param x0 Optional set of initial individuals.
     * @param data User payload passed to the objective.
     * @param callback Progress callback invoked with the best-so-far state.
     * @param tol Relative tolerance used by the inherited stop criterion.
     * @param maxevals Maximum number of objective evaluations.
     * @param seed RNG seed (negative -> random seed).
     * @param options Configuration map (population multiplier, archive rate, etc.).
     */
    LSHADE_cnEpSin(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
    ) :
        Differential_Evolution(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

    void initialize() override;
    MinionResult optimize() override;

private:
    size_t learningPeriod = 20;
    double sinFreqBase = 0.5;
    double epsilon = 1e-8;
    double pBestRate = 0.11;
    double archiveRate = 1.4;
    double rotationProbability = 0.4;
    double neighbourhoodFraction = 0.5;
    double freqInit = 0.5;
    size_t memorySize = 5;

    size_t maxPopulationSize = 0;
    size_t minPopulationSize = 4;
    size_t generationCounter = 0;
    size_t estimatedMaxGenerations = 1;

    std::vector<double> memorySF;
    std::vector<double> memoryCR;
    std::vector<double> memoryFreq;
    size_t memoryPos = 0;

    std::deque<size_t> successHistory1;
    std::deque<size_t> failureHistory1;
    std::deque<size_t> successHistory2;
    std::deque<size_t> failureHistory2;

    std::vector<double> lastCR;
    std::vector<double> lastF;
    std::vector<double> lastFreq;

    size_t computeGmax(size_t dimension) const;
    void sampleParameters(const std::vector<size_t>& sortedIndices,
                          std::vector<double>& muSF,
                          std::vector<double>& muCR,
                          std::vector<double>& muFreq,
                          std::vector<size_t>& memIndices,
                          std::vector<double>& CRvec,
                          std::vector<double>& Fvec,
                          std::vector<double>& freqVec,
                          bool& usedStrategy1,
                          bool& usedStrategy2);

    void updateHistories(bool usedStrategy1, bool usedStrategy2, size_t good1, size_t bad1, size_t good2, size_t bad2);

    void updateMemories(const std::vector<size_t>& successIndices,
                        const std::vector<double>& difValues,
                        const std::vector<double>& CRvec,
                        const std::vector<double>& Fvec,
                        const std::vector<double>& freqVec);

    void updateArchive(const std::vector<std::vector<double>>& newEntries,
                       const std::vector<double>& newFitness);
    void reducePopulationIfNeeded();

    void buildNeighbourCovariance(const std::vector<size_t>& sortedIndices,
                                  Eigen::MatrixXd& eigenBasis,
                                  Eigen::MatrixXd& eigenBasisT,
                                  size_t currentPopSize);
};

}

#endif
