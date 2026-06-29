#include "acmaes.h"
#include "default_options.h"
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
    : MinimizerBase(func, bounds, x0, data, callback, maxevals, seed, options) {}

void ACMAES::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("ACMAES");
    for (const auto& item : optionMap) {
        defaults[item.first] = item.second;
    }
    Options options(defaults);

    dimension = bounds.size();
    if (dimension == 0) {
        throw std::runtime_error("ACMAES requires bounded variables");
    }

    useBounds = !bounds.empty();

    lambda = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambda == 0) {
        lambda = 4 + static_cast<size_t>(std::floor(3.0 * std::log(static_cast<double>(dimension))));
    }
    lambda = std::max<size_t>(lambda, 5);

    mu = static_cast<size_t>(options.get<int>("mu", 0));
    if (mu == 0) {
        mu = lambda / 2;
    }
    mu = std::max<size_t>(mu, 1);
    mu = std::min(mu, lambda);

    weights = makeLogWeights(mu);
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
        damps = 1.0 + cs + 2.0 * std::max(0.0, std::sqrt((muEff - 1.0) / (dimension + 1.0)) - 1.0);
    }

    chiN = std::sqrt(static_cast<double>(dimension)) *
           (1.0 - 1.0 / (4.0 * static_cast<double>(dimension)) +
            1.0 / (21.0 * static_cast<double>(dimension) * static_cast<double>(dimension)));

    mean = Eigen::VectorXd::Zero(dimension);
    if (!x0.empty() && x0[0].size() == dimension) {
        std::vector<double> initialGuess;
        if (x0.size() > 1) {
            initialGuess = findBestPoint(x0);
        } else {
            initialGuess = x0.front();
        }
        for (size_t i = 0; i < dimension; ++i) {
            mean(static_cast<Eigen::Index>(i)) = initialGuess[i];
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

std::vector<double> ACMAES::ensureBounds(std::vector<double> candidate) const {
    if (!useBounds) {
        return candidate;
    }
    for (size_t d = 0; d < dimension; ++d) {
        candidate[d] = clamp(candidate[d], bounds[d].first, bounds[d].second);
    }
    return candidate;
}

void ACMAES::updateEigenDecomposition() {
    Eigen::VectorXd evals;
    safeSelfAdjointEigenDecomposition(C, B, evals);
    D = evals.cwiseSqrt();
    Eigen::MatrixXd Dmat = D.asDiagonal();
    C = B * Dmat * Dmat * B.transpose();
}

MinionResult ACMAES::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();
        diversity.clear();

        Nevals = 0;
        size_t generation = 0;
        best = std::vector<double>(mean.data(), mean.data() + mean.size());
        best_fitness = std::numeric_limits<double>::infinity();

        std::vector<std::vector<double>> population(lambda, std::vector<double>(dimension, 0.0));
        std::vector<Eigen::VectorXd> zs(lambda, Eigen::VectorXd::Zero(dimension));
        std::vector<double> fitness(lambda, std::numeric_limits<double>::infinity());

        while (Nevals < maxevals) {
            ++generation;

            for (size_t k = 0; k < lambda; ++k) {
                for (size_t d = 0; d < dimension; ++d) {
                    zs[k](static_cast<Eigen::Index>(d)) = rand_norm(0.0, 1.0);
                }
                Eigen::VectorXd step = B * (D.asDiagonal() * zs[k]);
                Eigen::VectorXd x = mean + sigma * step;
                for (size_t d = 0; d < dimension; ++d) {
                    population[k][d] = x(static_cast<Eigen::Index>(d));
                }
                if (useBounds) {
                    population[k] = ensureBounds(population[k]);
                }
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

            double fmax = *std::max_element(fitness.begin(), fitness.end());
            double fmin = *std::min_element(fitness.begin(), fitness.end());
            double fmean = std::accumulate(fitness.begin(), fitness.end(), 0.0) / static_cast<double>(fitness.size());
            double denom = std::fabs(fmean);
            if (denom <= 1e-12) {
                denom = std::max({std::fabs(fmax), std::fabs(fmin), 1.0});
            }
            double relRange = (fmax - fmin) / denom;
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
