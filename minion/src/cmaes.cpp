#include "cmaes.h"
#include "utility.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minion {

CMAES::CMAES(
    MinionFunction func,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<std::vector<double>>& x0,
    void* data,
    std::function<void(MinionResult*)> callback,
    size_t maxevals,
    int seed,
    std::map<std::string, ConfigValue> options)
    : CMAESBase(func, bounds, x0, data, callback, maxevals, seed, std::move(options)) {}

void CMAES::initialize() {
    initializeCommon("CMAES", cs);
}

MinionResult CMAES::optimize() {
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
            generation++;

            for (size_t k = 0; k < lambda; ++k) {
                bool valid = false;
                std::vector<double> candidate(dimension, 0.0);
                Eigen::VectorXd z = Eigen::VectorXd::Zero(dimension);
                int retries = 0;
                while (!valid && retries < 20) {
                    for (size_t d = 0; d < dimension; ++d) {
                        z(static_cast<Eigen::Index>(d)) = rand_norm(0.0, 1.0);
                    }
                    Eigen::VectorXd step = B * (D.asDiagonal() * z);
                    Eigen::VectorXd x = mean + sigma * step;
                    for (size_t d = 0; d < dimension; ++d) {
                        candidate[d] = x(static_cast<Eigen::Index>(d));
                    }
                    if (useBounds) {
                        bool inside = true;
                        for (size_t d = 0; d < dimension; ++d) {
                            if (candidate[d] < bounds[d].first || candidate[d] > bounds[d].second) {
                                inside = false;
                                break;
                            }
                        }
                        if (!inside) {
                            ++retries;
                            continue;
                        }
                    }
                    valid = true;
                }
                if (!valid) {
                    candidate = applyBounds(candidate);
                }
                population[k] = candidate;
                zs[k] = z;
            }

            auto fitVals = func(population, data);
            Nevals += fitVals.size();
            for (size_t i = 0; i < lambda; ++i) {
                if (std::isnan(fitVals[i])) {
                    fitVals[i] = 1e+100;
                }
                fitness[i] = fitVals[i];
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
            double hsigCond = psNorm / std::sqrt(1.0 - std::pow(1.0 - cs, 2.0 * generation)) / chiN;
            double hsig = hsigCond < (1.4 + 2.0 / (static_cast<double>(dimension) + 1.0)) ? 1.0 : 0.0;

            pc = (1.0 - cc) * pc + hsig * std::sqrt(cc * (2.0 - cc) * muEff) * y_w;

            Eigen::MatrixXd rankOne = pc * pc.transpose();
            C = (1.0 - c1 - cmu) * C + c1 * (rankOne + (1.0 - hsig) * cc * (2.0 - cc) * C);

            for (size_t i = 0; i < mu; ++i) {
                size_t idx = order[i];
                Eigen::VectorXd diff(dimension);
                for (size_t d = 0; d < dimension; ++d) {
                    diff(static_cast<Eigen::Index>(d)) = (population[idx][d] - meanOld(static_cast<Eigen::Index>(d))) / sigma;
                }
                C += cmu * weights[i] * (diff * diff.transpose());
            }

            sigma *= std::exp((cs / damps) * (psNorm / chiN - 1.0));

            //if (generation % (dimension == 0 ? 1 : (7 + static_cast<int>(3.0 * dimension))) == 0) {
            
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
