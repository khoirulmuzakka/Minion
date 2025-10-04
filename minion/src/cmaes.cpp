#include "cmaes.h"
#include "default_options.h"
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
    double tol,
    size_t maxevals,
    int seed,
    std::map<std::string, ConfigValue> options)
    : MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, seed, options) {}

void CMAES::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("CMAES");
    for (const auto& item : optionMap) {
        defaults[item.first] = item.second;
    }
    Options options(defaults);

    dimension = bounds.size();
    if (dimension == 0) {
        throw std::runtime_error("CMAES requires bounded variables");
    }

    useBounds = !bounds.empty();

    lambda = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambda == 0) {
        lambda = 2*(4 + static_cast<size_t>(std::floor(3.0 * std::log(static_cast<double>(dimension)))) );
    }
    lambda = std::max<size_t>(lambda, 5);

    mu = static_cast<size_t>(options.get<int>("mu", 0));
    if (mu == 0) {
        mu = lambda / 2;
    }
    mu = std::max<size_t>(mu, 1);
    mu = std::min(mu, lambda);

    weights.resize(mu);
    for (size_t i = 0; i < mu; ++i) {
        weights[i] = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i) + 1.0);
    }
    double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (double& w : weights) {
        w /= weightSum;
    }
    muEff = 0.0;
    for (double w : weights) {
        muEff += w * w;
    }
    muEff = 1.0 / muEff;

    sigma = options.get<double>("initial_step", 0.3);
    if (sigma <= 0.0) {
        sigma = 0.3;
    }

    double avgRange = 0.0;
    for (const auto& b : bounds) {
        avgRange += (b.second - b.first);
    }
    avgRange = (dimension > 0) ? avgRange / static_cast<double>(dimension) : 1.0;

    sigma *= avgRange;

    double ccDefault = (4.0 + muEff / static_cast<double>(dimension)) / (dimension + 4.0 + 2.0 * muEff / static_cast<double>(dimension));
    double csDefault = (muEff + 2.0) / (static_cast<double>(dimension) + muEff + 5.0);
    double c1Default = 2.0 / ((dimension + 1.3) * (dimension + 1.3) + muEff);
    double cmuDefault = std::min(1.0 - c1Default, 2.0 * (muEff - 2.0 + 1.0 / muEff) / ((dimension + 2.0) * (dimension + 2.0) + muEff));

    cc = options.get<double>("cc", ccDefault);
    if (cc <= 0.0) cc = ccDefault;
    cs = options.get<double>("cs", csDefault);
    if (cs <= 0.0) cs = csDefault;
    c1 = options.get<double>("c1", c1Default);
    if (c1 <= 0.0) c1 = c1Default;
    cmu = options.get<double>("cmu", cmuDefault);
    if (cmu <= 0.0) cmu = cmuDefault;
    damps = options.get<double>("damps", 0.0);
    if (damps <= 0.0) {
        damps = 1.0 + cs + 2.0 * std::max(0.0, std::sqrt((muEff - 1.0) / (dimension + 1.0)) - 1.0) + cs;
    }

    chiN = std::sqrt(static_cast<double>(dimension)) * (1.0 - 1.0 / (4.0 * static_cast<double>(dimension)) + 1.0 / (21.0 * static_cast<double>(dimension) * static_cast<double>(dimension)));

    mean = Eigen::VectorXd::Zero(dimension);
    if (!x0.empty() && x0[0].size() == dimension) {
        for (size_t i = 0; i < dimension; ++i) {
            mean(static_cast<Eigen::Index>(i)) = x0[0][i];
        }
    } else {
        auto initial = random_sampling(bounds, 1).front();
        for (size_t i = 0; i < dimension; ++i) {
            mean(static_cast<Eigen::Index>(i)) = initial[i];
        }
    }

    C = Eigen::MatrixXd::Identity(dimension, dimension);
    B = Eigen::MatrixXd::Identity(dimension, dimension);
    D = Eigen::VectorXd::Ones(dimension);
    ps = Eigen::VectorXd::Zero(dimension);
    pc = Eigen::VectorXd::Zero(dimension);

    hasInitialized = true;
}

std::vector<double> CMAES::ensureBounds(std::vector<double> candidate) const {
    if (!useBounds) {
        return candidate;
    }
    for (size_t d = 0; d < dimension; ++d) {
        candidate[d] = clamp(candidate[d], bounds[d].first, bounds[d].second);
    }
    return candidate;
}

void CMAES::updateEigenDecomposition() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C);
    Eigen::VectorXd evals = solver.eigenvalues();
    Eigen::MatrixXd evecs = solver.eigenvectors();

    for (Eigen::Index i = 0; i < evals.size(); ++i) {
        if (evals(i) < 1e-30) {
            evals(i) = 1e-30;
        }
    }
    B = evecs;
    D = evals.cwiseSqrt();
    Eigen::MatrixXd Dmat = D.asDiagonal();
    C = B * Dmat * Dmat * B.transpose();
}

MinionResult CMAES::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();
        diversity.clear();

        size_t evalsPerIter = lambda;
        size_t generation = 0;
        Nevals = 0;

        std::vector<std::vector<double>> population(lambda, std::vector<double>(dimension, 0.0));
        std::vector<Eigen::VectorXd> zs(lambda, Eigen::VectorXd::Zero(dimension));
        std::vector<double> fitness(lambda, std::numeric_limits<double>::infinity());

        std::vector<double> bestVec(mean.data(), mean.data() + dimension);
        best = bestVec;
        best_fitness = std::numeric_limits<double>::infinity();

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
                    candidate = ensureBounds(candidate);
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

            if (generation % (dimension == 0 ? 1 : (7 + static_cast<int>(3.0 * dimension))) == 0) {
                updateEigenDecomposition();
            }

            double fmax = *std::max_element(fitness.begin(), fitness.end());
            double fmin = *std::min_element(fitness.begin(), fitness.end());
            double fmean = std::accumulate(fitness.begin(), fitness.end(), 0.0) / static_cast<double>(fitness.size());
            double relRange = 0.0;
            if (std::fabs(fmean) > 1e-12) {
                relRange = (fmax - fmin) / std::fabs(fmean);
            }
            diversity.push_back(relRange);

            minionResult = MinionResult(best, best_fitness, generation, Nevals, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) {
                callback(&minionResult);
            }

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
