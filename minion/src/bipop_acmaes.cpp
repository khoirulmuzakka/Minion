#include "bipop_acmaes.h"

#include "default_options.h"
#include "utility.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace minion {

BIPOP_aCMAES::BIPOP_aCMAES(
    MinionFunction func,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<std::vector<double>>& x0,
    void* data,
    std::function<void(MinionResult*)> callback,
    size_t maxevals,
    int seed,
    std::map<std::string, ConfigValue> options)
    : CMAESBase(func, bounds, x0, data, callback, maxevals, seed, std::move(options)) {}

std::vector<double> BIPOP_aCMAES::applyBounds(const std::vector<double>& candidate) const {
    if (!useBounds) {
        return candidate;
    }
    std::vector<std::vector<double>> wrapper = {candidate};
    enforce_bounds(wrapper, bounds, boundStrategy);
    return wrapper.front();
}

std::vector<double> BIPOP_aCMAES::eigenToStd(const Eigen::VectorXd& vec) const {
    return std::vector<double>(vec.data(), vec.data() + vec.size());
}

void BIPOP_aCMAES::applyCovarianceScale() {
    if (cov_scale.empty()) {
        return;
    }

    D = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(cov_scale.size()));
    for (size_t i = 0; i < cov_scale.size(); ++i) {
        D(static_cast<Eigen::Index>(i)) = cov_scale[i];
    }
    B = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(cov_scale.size()), static_cast<Eigen::Index>(cov_scale.size()));
    C = D.array().square().matrix().asDiagonal();
}

void BIPOP_aCMAES::initialize() {
    const Options options = buildOptions("BIPOP_aCMAES");

    dimension = bounds.size();
    if (dimension == 0) {
        throw std::runtime_error("BIPOP aCMA-ES requires bounded variables");
    }

    useBounds = !bounds.empty();
    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));
    support_tol = true;
    stoppingTol = getConvergenceTolerance(options, 1e-4);

    lambda0 = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambda0 == 0) {
        const double logDim = std::log(static_cast<double>(dimension));
        lambda0 = static_cast<size_t>(4.0 + std::floor(3.0 * logDim));
    }
    lambda0 = std::max<size_t>(lambda0, 4);

    maxIterations = static_cast<size_t>(options.get<int>("max_iterations", 100000));

    sigma0 = options.getSilent<double>("rel_initial_step", 0.3);
    if (sigma0 <= 0.0) {
        sigma0 = 0.3;
    }

    cov_scale.clear();
    cov_scale.reserve(dimension);
    double avgRange = 0.0;
    for (const auto& b : bounds) {
        const double range = b.second - b.first;
        avgRange += range;
        cov_scale.push_back(range);
    }
    avg_range = (dimension > 0) ? avgRange / static_cast<double>(dimension) : 1.0;
    sigma0 *= avg_range;
    if (avg_range > 0.0) {
        for (double& v : cov_scale) {
            v /= avg_range;
            if (v <= 0.0) {
                v = 1.0;
            }
        }
    }

    mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    initializeMean();
    initialMean = mean;

    best = eigenToStd(initialMean);
    best_fitness = std::numeric_limits<double>::infinity();
    diversity.clear();
    Nevals = 0;
    hasInitialized = true;
}

void BIPOP_aCMAES::configureRegime(const Eigen::Ref<const Eigen::VectorXd>& startMean, double startSigma, size_t lambdaValue) {
    lambda = std::max<size_t>(lambdaValue, 4);
    mu = std::max<size_t>(lambda / 2, 1);
    mu = std::min(mu, lambda);

    weights = makeLogWeights(mu);
    muEff = 0.0;
    for (double w : weights) {
        muEff += w * w;
    }
    muEff = 1.0 / std::max(muEff, 1e-12);

    sigma = startSigma;
    mean = startMean;

    const double dim = static_cast<double>(dimension);
    cc = (4.0 + muEff / dim) / (dim + 4.0 + 2.0 * muEff / dim);
    cs = (muEff + 2.0) / (dim + muEff + 5.0);
    c1 = 2.0 / (std::pow(dim + 1.3, 2.0) + muEff);
    cmu = std::min(
        1.0 - c1,
        2.0 * (muEff - 2.0 + 1.0 / muEff) / (std::pow(dim + 2.0, 2.0) + muEff));
    damps = 1.0 + cs + 2.0 * std::max(0.0, std::sqrt((muEff - 1.0) / (dim + 1.0)) - 1.0);
    chiN = std::sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim));

    ps = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    pc = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    B = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
    D = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(dimension));
    C = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
    applyCovarianceScale();
}

void BIPOP_aCMAES::checkStoppingCriteria(bool& shouldStopRun) const {
    if (D.size() == 0) {
        return;
    }

    const Eigen::VectorXd eigenvalues = D.array().square().matrix();
    double eigvalMin = eigenvalues.minCoeff();
    double eigvalMax = eigenvalues.maxCoeff();
    if (eigvalMin <= 0.0) {
        eigvalMin = 1e-30;
    }
    if (eigvalMax / eigvalMin > 1e14) {
        shouldStopRun = true;
        return;
    }

    const double sigmaFac = sigma0 > 0.0 ? sigma / sigma0 : sigma;
    const double sigmaUpThresh = 1e20 * std::sqrt(eigvalMax);
    if (sigmaFac > sigmaUpThresh) {
        shouldStopRun = true;
        return;
    }

    for (size_t i = 0; i < dimension; ++i) {
        const double stepScale = 0.1 * sigma * eigenvalues(static_cast<Eigen::Index>(i));
        const Eigen::VectorXd moved = mean + stepScale * B.col(static_cast<Eigen::Index>(i));
        if ((moved.array() == mean.array()).all()) {
            shouldStopRun = true;
            return;
        }
    }

    for (size_t i = 0; i < dimension; ++i) {
        const double delta = 0.2 * sigma * std::sqrt(C(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)));
        if (mean(static_cast<Eigen::Index>(i)) + delta == mean(static_cast<Eigen::Index>(i))) {
            shouldStopRun = true;
            return;
        }
    }
}

size_t BIPOP_aCMAES::runRegime(
    const Eigen::Ref<const Eigen::VectorXd>& startMean,
    double startSigma,
    size_t lambdaValue) {
    configureRegime(startMean, startSigma, lambdaValue);

    bool shouldStopRun = false;
    size_t regimeEvaluations = 0;
    size_t iteration = 0;

    std::vector<std::vector<double>> population(lambda, std::vector<double>(dimension, 0.0));
    std::vector<double> fitness(lambda, std::numeric_limits<double>::infinity());

    while (!shouldStopRun && Nevals < maxevals && iteration < maxIterations) {
        ++iteration;

        for (size_t k = 0; k < lambda; ++k) {
            Eigen::VectorXd z = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
            for (size_t d = 0; d < dimension; ++d) {
                z(static_cast<Eigen::Index>(d)) = rand_norm(0.0, 1.0);
            }

            const Eigen::VectorXd step = B * (D.asDiagonal() * z);
            const Eigen::VectorXd x = mean + sigma * step;
            population[k] = applyBounds(eigenToStd(x));
        }

        std::fill(fitness.begin(), fitness.end(), std::numeric_limits<double>::infinity());
        const size_t remaining = maxevals - Nevals;
        const size_t evalCount = std::min(remaining, lambda);
        if (evalCount == 0) {
            break;
        }

        std::vector<std::vector<double>> evaluatedPopulation(population.begin(), population.begin() + static_cast<std::ptrdiff_t>(evalCount));
        const auto fitVals = func(evaluatedPopulation, data);
        if (fitVals.size() != evalCount) {
            throw std::runtime_error("Objective function returned unexpected number of values");
        }

        for (size_t i = 0; i < evalCount; ++i) {
            fitness[i] = std::isnan(fitVals[i]) ? std::numeric_limits<double>::infinity() : fitVals[i];
        }

        Nevals += evalCount;
        regimeEvaluations += evalCount;

        const std::vector<size_t> order = argsort(fitness, true);
        if (!order.empty() && fitness[order[0]] < best_fitness) {
            best_fitness = fitness[order[0]];
            best = population[order[0]];
        }

        const Eigen::VectorXd meanOld = mean;
        Eigen::VectorXd newMean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
        for (size_t i = 0; i < mu; ++i) {
            const size_t idx = order[i];
            for (size_t d = 0; d < dimension; ++d) {
                newMean(static_cast<Eigen::Index>(d)) += weights[i] * population[idx][d];
            }
        }

        const Eigen::VectorXd y_w = (newMean - meanOld) / sigma;
        mean = newMean;

        const Eigen::VectorXd CinvSqrt_y = B * (D.cwiseInverse().asDiagonal() * (B.transpose() * y_w));
        ps = (1.0 - cs) * ps + std::sqrt(cs * (2.0 - cs) * muEff) * CinvSqrt_y;

        const double psNorm = ps.norm();
        const double hsigCond = psNorm / std::sqrt(1.0 - std::pow(1.0 - cs, 2.0 * static_cast<double>(iteration))) / chiN;
        const double hsig = hsigCond < (1.4 + 2.0 / (static_cast<double>(dimension) + 1.0)) ? 1.0 : 0.0;

        pc = (1.0 - cc) * pc + hsig * std::sqrt(cc * (2.0 - cc) * muEff) * y_w;

        const Eigen::MatrixXd CinvSqrt = B * D.cwiseInverse().asDiagonal() * B.transpose();
        const ActiveCMAUpdate activeUpdate = buildActiveCMAUpdate(population, order, meanOld, sigma, weights, CinvSqrt, cmu, muEff);

        const double alphaminusold = 0.5;
        const Eigen::MatrixXd previousC = C;
        const Eigen::MatrixXd spc = pc * pc.transpose();
        C = (1.0 - c1 - cmu + activeUpdate.cminus * alphaminusold) * previousC +
            c1 * spc +
            (cmu + activeUpdate.cminus * (1.0 - alphaminusold)) * activeUpdate.cmu_plus -
            activeUpdate.cminus * activeUpdate.cmu_minus;
        C = 0.5 * (C + C.transpose());

        sigma *= std::exp(cs / damps * (psNorm / chiN - 1.0));
        updateEigenDecomposition();

        const std::vector<double> evaluatedFitness(fitness.begin(), fitness.begin() + static_cast<std::ptrdiff_t>(evalCount));
        const double relRange = computeRelativeRange(evaluatedFitness);
        diversity.push_back(relRange);
        ++globalGeneration;
        recordIteration(globalGeneration, Nevals, relRange);

        if (support_tol && relRange <= stoppingTol) {
            shouldStopRun = true;
        } else {
            checkStoppingCriteria(shouldStopRun);
        }

        if (evalCount < lambda) {
            break;
        }
    }

    return regimeEvaluations;
}

MinionResult BIPOP_aCMAES::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();
        diversity.clear();
        best = eigenToStd(initialMean);
        best_fitness = std::numeric_limits<double>::infinity();
        Nevals = 0;
        globalGeneration = 0;

        size_t budgetLarge = runRegime(initialMean, sigma0, lambda0);
        size_t budgetSmall = 0;
        size_t restart = 1;

        while (Nevals < maxevals) {
            size_t lambdaLarge = 0;
            if (restart >= (8 * sizeof(size_t))) {
                lambdaLarge = std::numeric_limits<size_t>::max() / 2;
            } else {
                lambdaLarge = lambda0 * (1ull << restart);
            }
            if (lambdaLarge < lambda0) {
                lambdaLarge = std::numeric_limits<size_t>::max() / 2;
            }
            if (maxevals > 0) {
                lambdaLarge = std::min(lambdaLarge, maxevals);
            }

            while (budgetLarge > budgetSmall && Nevals < maxevals) {
                const double u1 = rand_gen(0.0, 1.0);
                const double u2 = rand_gen(0.0, 1.0);
                size_t lambdaSmall = static_cast<size_t>(
                    std::round(lambda0 * std::pow(0.5 * static_cast<double>(lambdaLarge) / static_cast<double>(lambda0), u1 * u1)));
                lambdaSmall = std::max<size_t>(lambdaSmall, 4);
                const double sigmaSmall = sigma0 * 2.0 * std::pow(10.0, -2.0 * u2);

                Eigen::Map<const Eigen::VectorXd> bestEigen(best.data(), static_cast<Eigen::Index>(best.size()));
                budgetSmall += runRegime(bestEigen, sigmaSmall, lambdaSmall);
                if (Nevals >= maxevals) {
                    break;
                }
            }

            if (Nevals >= maxevals) {
                break;
            }

            budgetLarge += runRegime(initialMean, sigma0, lambdaLarge);
            ++restart;
        }

        if (history.empty()) {
            recordIteration(globalGeneration, Nevals, 0.0);
        }

        return getBestFromHistory();
    } catch (const std::exception& ex) {
        throw std::runtime_error(ex.what());
    }
}

}  // namespace minion
