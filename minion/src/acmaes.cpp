#include "acmaes.h"
#include "utility.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minion {

ACMAES::ACMAES(
    MinionFunction func,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<std::vector<double>>& x0,
    void* data,
    std::function<void(MinionResult*)> callback,
    size_t maxevals,
    int seed,
    std::map<std::string, ConfigValue> options)
    : CMAESBase(func, bounds, x0, data, callback, maxevals, seed, std::move(options)) {}

void ACMAES::initialize() {
    initializeCommon("ACMAES", 0.0);
}

MinionResult ACMAES::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();

        size_t generation = 0;

        std::vector<std::vector<double>> population(lambda, std::vector<double>(dimension, 0.0));
        std::vector<Eigen::VectorXd> zs(lambda, Eigen::VectorXd::Zero(dimension));
        std::vector<double> fitness(lambda, std::numeric_limits<double>::infinity());

        while (Nevals < maxevals) {
            ++generation;

            for (size_t k = 0; k < lambda; ++k) {
                population[k] = sampleCandidate(mean, B, D, sigma);
            }

            auto fitVals = func(population, data);
            size_t evalCount = std::min(fitVals.size(), population.size());
            Nevals += evalCount;
            for (size_t i = 0; i < evalCount; ++i) {
                fitness[i] = std::isnan(fitVals[i]) ? 1e+100 : fitVals[i];
            }
            for (size_t i = evalCount; i < lambda; ++i) {
                fitness[i] = std::numeric_limits<double>::infinity();
            }

            std::vector<size_t> order = argsort(fitness, true);
            if (!order.empty() && fitness[order[0]] < best_fitness) {
                best_fitness = fitness[order[0]];
                best = population[order[0]];
            }

            Eigen::VectorXd meanOld = mean;
            Eigen::VectorXd newMean = Eigen::VectorXd::Zero(dimension);
            for (size_t i = 0; i < mu; ++i) {
                size_t idx = order[i];
                for (size_t d = 0; d < dimension; ++d) {
                    newMean(static_cast<Eigen::Index>(d)) += weights[i] * population[idx][d];
                }
            }

            Eigen::VectorXd y_w = (newMean - meanOld) / sigma;
            mean = newMean;

            Eigen::VectorXd CinvSqrt_y = B * (D.cwiseInverse().asDiagonal() * (B.transpose() * y_w));
            ps = (1.0 - cs) * ps + std::sqrt(cs * (2.0 - cs) * muEff) * CinvSqrt_y;

            double psNorm = ps.norm();
            double hsigCond = psNorm / std::sqrt(1.0 - std::pow(1.0 - cs, 2.0 * static_cast<double>(generation))) / chiN;
            double hsig = hsigCond < (1.4 + 2.0 / (static_cast<double>(dimension) + 1.0)) ? 1.0 : 0.0;

            pc = (1.0 - cc) * pc + hsig * std::sqrt(cc * (2.0 - cc) * muEff) * y_w;

            Eigen::MatrixXd CinvSqrt = B * D.cwiseInverse().asDiagonal() * B.transpose();
            ActiveCMAUpdate activeUpdate = buildActiveCMAUpdate(population, order, meanOld, sigma, weights, CinvSqrt, cmu, muEff);

            const double alphaminusold = 0.5;
            Eigen::MatrixXd previousC = C;
            Eigen::MatrixXd spc = pc * pc.transpose();
            C = (1.0 - c1 - cmu + activeUpdate.cminus * alphaminusold) * previousC +
                c1 * spc +
                (cmu + activeUpdate.cminus * (1.0 - alphaminusold)) * activeUpdate.cmu_plus -
                activeUpdate.cminus * activeUpdate.cmu_minus;
            C = 0.5 * (C + C.transpose());

            sigma *= std::exp(cs / damps * (psNorm / chiN - 1.0));

            updateEigenDecomposition();

            const double relRange = computeRelativeRange(fitness);
            recordIteration(generation, Nevals, relRange);

            if (support_tol && relRange <= stoppingTol) {
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
