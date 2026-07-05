#include "rcmaes.h"

#include "default_options.h"
#include "utility.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

std::vector<std::pair<double, double>> merge_intervals_1d(std::vector<std::pair<double, double>> intervals) {
    if (intervals.empty()) {
        return intervals;
    }

    std::sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::vector<std::pair<double, double>> merged;
    merged.reserve(intervals.size());
    double currentLow = intervals.front().first;
    double currentHigh = intervals.front().second;
    for (size_t i = 1; i < intervals.size(); ++i) {
        const auto& interval = intervals[i];
        if (interval.first <= currentHigh) {
            currentHigh = std::max(currentHigh, interval.second);
        } else {
            merged.emplace_back(currentLow, currentHigh);
            currentLow = interval.first;
            currentHigh = interval.second;
        }
    }
    merged.emplace_back(currentLow, currentHigh);
    return merged;
}

double sample_outside_local_bounds(
    double low,
    double high,
    const std::vector<std::pair<double, double>>& local_bounds) {
    std::vector<std::pair<double, double>> merged = merge_intervals_1d(local_bounds);
    std::vector<std::pair<double, double>> valid_intervals;
    double previous_high = low;
    for (const auto& bound : merged) {
        if (bound.first > previous_high) {
            valid_intervals.emplace_back(previous_high, bound.first);
        }
        previous_high = std::min(bound.second, high);
    }
    if (previous_high < high) {
        valid_intervals.emplace_back(previous_high, high);
    }

    std::vector<double> lengths;
    lengths.reserve(valid_intervals.size());
    for (const auto& interval : valid_intervals) {
        lengths.push_back(interval.second - interval.first);
    }

    if (lengths.empty()) {
        valid_intervals.emplace_back(low, high);
        lengths.push_back(high - low);
    }

    std::discrete_distribution<size_t> interval_dist(lengths.begin(), lengths.end());
    const size_t chosen_interval = interval_dist(minion::get_rng());
    return minion::rand_gen(valid_intervals[chosen_interval].first, valid_intervals[chosen_interval].second);
}

}  // namespace

namespace minion {

RCMAES::RCMAES(
    MinionFunction func,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<std::vector<double>>& x0,
    void* data,
    std::function<void(MinionResult*)> callback,
    size_t maxevals,
    int seed,
    std::map<std::string, ConfigValue> options)
    : CMAESBase(func, bounds, x0, data, callback, maxevals, seed, std::move(options)) {}

std::vector<double> RCMAES::applyBounds(const std::vector<double>& candidate) const {
    if (!useBounds) {
        return candidate;
    }

    std::vector<std::vector<double>> wrapper = {candidate};
    enforce_bounds(wrapper, bounds, boundStrategy);
    return wrapper.front();
}

std::vector<double> RCMAES::eigenToStd(const Eigen::VectorXd& vec) const {
    return std::vector<double>(vec.data(), vec.data() + vec.size());
}

std::vector<double> RCMAES::denormalizePoint(const std::vector<double>& candidate) const {
    return CMAESBase::denormalizePoint(candidate);
}

Eigen::VectorXd RCMAES::buildCustomActiveWeights(size_t lambdaValue, size_t muValue) const {
    Eigen::VectorXd signedWeights = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(lambdaValue));

    double wNegSum = 0.0;
    double wPosSum = 0.0;
    for (size_t i = 0; i < lambdaValue; ++i) {
        const double wi =
            std::log((static_cast<double>(lambdaValue) + 1.0) / 2.0) - std::log(static_cast<double>(i) + 1.0);
        signedWeights(static_cast<Eigen::Index>(i)) = wi;
        if (wi >= 0.0) {
            wPosSum += wi;
        } else {
            wNegSum += wi;
        }
    }

    double wSumParent = 0.0;
    double wSqSumParent = 0.0;
    for (size_t i = 0; i < muValue; ++i) {
        const double wi = signedWeights(static_cast<Eigen::Index>(i));
        wSumParent += wi;
        wSqSumParent += wi * wi;
    }
    const double muEffSigned = (wSqSumParent > 0.0) ? ((wSumParent * wSumParent) / wSqSumParent) : 1.0;

    const double dim = static_cast<double>(dimension);
    const double aMu = 1.0 + c1 / std::max(1e-12, cmu);
    const double aMueff = 1.0 + 2.0 * muEffSigned;
    const double aPosdef = (1.0 - c1 - cmu) / (dim * std::max(1e-12, cmu));
    const double aMin = std::min({aMu, aMueff, aPosdef});

    for (size_t i = 0; i < lambdaValue; ++i) {
        const double wi = signedWeights(static_cast<Eigen::Index>(i));
        if (wi >= 0.0) {
            signedWeights(static_cast<Eigen::Index>(i)) = wi / std::max(1e-12, wPosSum);
        } else {
            signedWeights(static_cast<Eigen::Index>(i)) = aMin * wi / std::max(1e-12, std::abs(wNegSum));
        }
    }

    return signedWeights;
}

void RCMAES::configurePopulationParameters(size_t lambdaValue) {
    lambda = std::min<size_t>(2000, std::max<size_t>(lambdaValue, 4));
    mu = std::max<size_t>(static_cast<size_t>(std::ceil(mu_ratio * static_cast<double>(lambda))), 1);
    mu = std::min(mu, lambda);

    weights = makeLogWeights(mu);
    muEff = 0.0;
    for (double w : weights) {
        muEff += w * w;
    }
    muEff = 1.0 / std::max(muEff, 1e-12);

    const double dim = static_cast<double>(dimension);
    cc = (4.0 + muEff / dim) / (dim + 4.0 + 2.0 * muEff / dim);
    cs = (muEff + 2.0) / (dim + muEff + 5.0);
    c1 = 2.0 / (std::pow(dim + 1.3, 2.0) + muEff);
    cmu = std::min(
        1.0 - c1,
        2.0 * (muEff - 2.0 + 1.0 / muEff) / (std::pow(dim + 2.0, 2.0) + muEff));
    damps = 1.0 + cs + 2.0 * std::max(0.0, std::sqrt((muEff - 1.0) / (dim + 1.0)) - 1.0);
    chiN = std::sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim));
}

void RCMAES::resetRegimeState(const Eigen::Ref<const Eigen::VectorXd>& startMean, double startSigma) {
    mean = startMean;
    sigma = startSigma;
    ps = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    pc = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    B = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
    D = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(dimension));
    C = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
}

void RCMAES::initialize() {
    Options options = buildOptions("RCMAES");

    dimension = bounds.size();
    if (dimension == 0) {
        throw std::runtime_error("RCMAES requires bounded variables");
    }

    useBounds = !bounds.empty();
    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));
    support_tol = false;

    const double logDim = dimension > 0 ? std::log(static_cast<double>(dimension)) : 1.0;
    lambda_min = static_cast<size_t>(4.0 + std::ceil(3.0 * logDim));
    lambda_min = std::max<size_t>(lambda_min, 4);

    size_t lambdaInit = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambdaInit == 0) {
        const double dim = static_cast<double>(bounds.size());
        const double eta = dim > 0.0 ? double(maxevals) / dim : 1.0;
        const double logeta = std::log10(std::max(eta, 1e-12));
        //const double multiplier = 10.0 * logeta - 20.0;
        const double multiplier = std::min(std::pow(20.0, (logeta - 3.0)), 30.0); // Limit the multiplier to prevent excessively large population sizes
        const double suggested = std::clamp(dim * multiplier, double(lambda_min), 2000.0);
        lambdaInit = static_cast<size_t>(suggested);
    }

    lambda_base = lambdaInit;
    mu_ratio = 0.5;
    configurePopulationParameters(lambda_base);

    sigma0 = options.getSilent<double>("rel_initial_step", 0.3);
    if (sigma0 <= 0.0) {
        sigma0 = 0.3;
    }
    minRelStep = options.getSilent<double>("min_rel_step", 1e-8);
    if (minRelStep <= 0.0) {
        minRelStep = 1e-8;
    }
    useCustomActive = options.getSilent<bool>("useCustomActive", true);

    mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    initializeMean();
    initialMean = mean;

    best = eigenToStd(initialMean);
    best_fitness = std::numeric_limits<double>::infinity();
    diversity.clear();
    Nevals = 0;
    hasInitialized = true;
}

void RCMAES::recordHistory(double relRange) {
    const bool success = support_tol && relRange <= stoppingTol;
    minionResult = MinionResult(denormalizePoint(best), best_fitness, generation, Nevals, success,
                                success ? "stopping tolerance reached" : "");
    history.push_back(minionResult);
    if (callback != nullptr) {
        callback(&minionResult);
    }
}

RCMAES::ExclusionBox RCMAES::buildExclusionBox(const std::vector<double>& bestPoint) const {
    ExclusionBox box;
    const size_t dim = bounds.size();
    box.low.resize(dim);
    box.high.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
        const double range = bounds[i].second - bounds[i].first;
        const double delta = 0.1 * range;
        box.low[i] = std::max(bounds[i].first, bestPoint[i] - delta);
        box.high[i] = std::min(bounds[i].second, bestPoint[i] + delta);
    }
    return box;
}

bool RCMAES::isExcludedPoint(const std::vector<double>& candidate) const {
    for (const auto& box : exclusion_boxes) {
        bool inside = true;
        for (size_t i = 0; i < candidate.size(); ++i) {
            if (candidate[i] < box.low[i] || candidate[i] > box.high[i]) {
                inside = false;
                break;
            }
        }
        if (inside) {
            return true;
        }
    }
    return false;
}

MinionResult RCMAES::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();
        diversity.clear();
        currentFitness.clear();
        best = eigenToStd(initialMean);
        best_fitness = std::numeric_limits<double>::infinity();
        Nevals = 0;
        generation = 0;
        restart_bests.clear();
        exclusion_boxes.clear();

        bool useRestartSamples = false;
        std::vector<std::vector<double>> restartSamples;
        double sigmaEff = sigma0;

        configurePopulationParameters(lambda_base);
        resetRegimeState(initialMean, sigmaEff);

        while (Nevals < maxevals) {
            double progress = (maxevals > 0) ? (double(Nevals) / double(maxevals)) : 1.0;
            progress = std::min(progress, 1.0);

            const double dim = static_cast<double>(bounds.size());
            const double a = static_cast<double>(lambda_base);
            const double c =  lambda_min; //std::max(static_cast<double>(lambda_min), dim);
            const double pp = std::max(0.5, 2.5 -  0.02 * dim);
            const double value = a - (a - c) * (1.0 - std::pow(1.0 - progress, pp));
            const size_t lambdaTarget = std::max<size_t>(c, static_cast<size_t>(std::round(value)));
            if (lambdaTarget != lambda) {
                configurePopulationParameters(lambdaTarget);
            }

            std::vector<std::vector<double>> population(lambda, std::vector<double>(dimension, 0.0));
            if (useRestartSamples) {
                for (size_t j = 0; j < lambda; ++j) {
                    population[j] = restartSamples[j];
                }
                useRestartSamples = false;
            } else {
                for (size_t k = 0; k < lambda; ++k) {
                    population[k] = sampleCandidate(mean, B, D, sigma);
                }
            }

            std::fill(currentFitness.begin(), currentFitness.end(), std::numeric_limits<double>::infinity());
            currentFitness.assign(lambda, std::numeric_limits<double>::infinity());

            const size_t remaining = maxevals - Nevals;
            const size_t evalCount = std::min(remaining, lambda);
            if (evalCount == 0) {
                break;
            }

            std::vector<std::vector<double>> evaluatedPopulation(
                population.begin(),
                population.begin() + static_cast<std::ptrdiff_t>(evalCount));
            const auto fitVals = func(evaluatedPopulation, data);
            if (fitVals.size() != evalCount) {
                throw std::runtime_error("Objective function returned unexpected number of values");
            }

            std::vector<double> fitness(lambda, std::numeric_limits<double>::infinity());
            for (size_t i = 0; i < evalCount; ++i) {
                const double valueFit = std::isnan(fitVals[i]) ? std::numeric_limits<double>::infinity() : fitVals[i];
                fitness[i] = valueFit;
                currentFitness[i] = valueFit;
            }

            Nevals += evalCount;

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
            const double hsigCond = psNorm / std::sqrt(1.0 - std::pow(1.0 - cs, 2.0 * static_cast<double>(generation + 1))) / chiN;
            const double hsig = hsigCond < (1.4 + 2.0 / (static_cast<double>(dimension) + 1.0)) ? 1.0 : 0.0;

            pc = (1.0 - cc) * pc + hsig * std::sqrt(cc * (2.0 - cc) * muEff) * y_w;

            const Eigen::MatrixXd CinvSqrt = B * D.cwiseInverse().asDiagonal() * B.transpose();
            const Eigen::MatrixXd previousC = C;
            const Eigen::MatrixXd spc = pc * pc.transpose();
            // Custom active mode uses signed rank weights over all offspring:
            //   w_i^raw = log((lambda + 1) / 2) - log(i)
            // with negative ranks rescaled as
            //   w_i <- w_i * n / ||C^{-1/2} y_i||^2
            // and then updates
            //   C <- (1 + c1*h1 - c1 - cmu*sum_i w_i) C
            //        + c1 * pc*pc^T
            //        + cmu * sum_i w_i y_i y_i^T.
            // The alternate branch matches ACMAES/libcmaes-style active CMA:
            // it builds explicit positive and negative covariance terms,
            //   Cmu+ from the best mu offspring and Cmu- from the worst mu,
            // and applies
            //   C <- (1 - c1 - cmu + cminus*alphaminusold) C
            //        + c1 * pc*pc^T
            //        + (cmu + cminus*(1-alphaminusold)) Cmu+
            //        - cminus * Cmu-.
            if (useCustomActive) {
                const Eigen::VectorXd signedWeights = buildCustomActiveWeights(lambda, mu);
                Eigen::VectorXd covarianceWeights = signedWeights;
                Eigen::MatrixXd rankedSteps(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(lambda));

                for (size_t rank = 0; rank < lambda; ++rank) {
                    const size_t idx = order[rank];
                    Eigen::VectorXd diff(static_cast<Eigen::Index>(dimension));
                    for (size_t d = 0; d < dimension; ++d) {
                        diff(static_cast<Eigen::Index>(d)) =
                            (population[idx][d] - meanOld(static_cast<Eigen::Index>(d))) / sigma;
                    }
                    rankedSteps.col(static_cast<Eigen::Index>(rank)) = diff;

                    if (signedWeights(static_cast<Eigen::Index>(rank)) < 0.0) {
                        const Eigen::VectorXd adjusted = CinvSqrt * diff;
                        const double denom = adjusted.squaredNorm();
                        covarianceWeights(static_cast<Eigen::Index>(rank)) =
                            (denom > 0.0)
                                ? signedWeights(static_cast<Eigen::Index>(rank)) * static_cast<double>(dimension) / denom
                                : 0.0;
                    }
                }

                const double h1 = (1.0 - hsig) * cc * (2.0 - cc);
                const double wSum = covarianceWeights.sum();
                C = (1.0 + c1 * h1 - c1 - cmu * wSum) * previousC +
                    c1 * spc +
                    cmu * rankedSteps * covarianceWeights.asDiagonal() * rankedSteps.transpose();
            } else {
                const ActiveCMAUpdate activeUpdate =
                    buildActiveCMAUpdate(population, order, meanOld, sigma, weights, CinvSqrt, cmu, muEff);
                const double alphaminusold = 0.5;
                C = (1.0 - c1 - cmu + activeUpdate.cminus * alphaminusold) * previousC +
                    c1 * spc +
                    (cmu + activeUpdate.cminus * (1.0 - alphaminusold)) * activeUpdate.cmu_plus -
                    activeUpdate.cminus * activeUpdate.cmu_minus;
            }
            C = 0.5 * (C + C.transpose());

            sigma *= std::exp(cs / damps * (psNorm / chiN - 1.0));
            updateEigenDecomposition();

            const std::vector<double> evaluatedFitness(
                currentFitness.begin(),
                currentFitness.begin() + static_cast<std::ptrdiff_t>(evalCount));
            const double relRange = computeRelativeRange(evaluatedFitness);
            diversity.push_back(relRange);
            ++generation;
            recordHistory(relRange);

            const double sqrtMaxEigenvalue = D.size() > 0 ? D.maxCoeff() : 0.0;
            const double effectiveStep = sigma * sqrtMaxEigenvalue;
            double minRelStep_eff = minRelStep + 99*minRelStep*(1.0-double(Nevals)/double(maxevals));
            bool restartRequested = effectiveStep < minRelStep_eff ;
            if (restartRequested) {
                if (false ){
                    std::cerr << "[RCMAES] restart at generation " << generation
                            << ", evals " << Nevals
                            << ", sigma " << sigma
                            << ", sqrt(max_eigenvalue(C)) " << sqrtMaxEigenvalue
                            << ", effective_step " << effectiveStep
                            << ", relRange " << relRange
                            << std::endl;
                }
                if (!best.empty()) {
                    restart_bests.push_back(best);
                    exclusion_boxes.push_back(buildExclusionBox(best));
                }

                std::vector<std::vector<std::pair<double, double>>> locals(bounds.size());
                for (const auto& box : exclusion_boxes) {
                    for (size_t dimIdx = 0; dimIdx < bounds.size(); ++dimIdx) {
                        locals[dimIdx].emplace_back(box.low[dimIdx], box.high[dimIdx]);
                    }
                }

                restartSamples.clear();
                restartSamples.reserve(lambda);
                for (size_t j = 0; j < lambda; ++j) {
                    std::vector<double> candidate(bounds.size(), 0.0);
                    for (size_t dimIdx = 0; dimIdx < bounds.size(); ++dimIdx) {
                        candidate[dimIdx] = sample_outside_local_bounds(
                            bounds[dimIdx].first,
                            bounds[dimIdx].second,
                            locals[dimIdx]);
                    }

                    if (isExcludedPoint(candidate)) {
                        size_t attempts = 0;
                        while (attempts < exclusion_max_attempts && isExcludedPoint(candidate)) {
                            for (size_t dimIdx = 0; dimIdx < bounds.size(); ++dimIdx) {
                                candidate[dimIdx] = sample_outside_local_bounds(
                                    bounds[dimIdx].first,
                                    bounds[dimIdx].second,
                                    locals[dimIdx]);
                            }
                            ++attempts;
                        }
                    }
                    restartSamples.push_back(candidate);
                }

                if (!restartSamples.empty()) {
                    Eigen::VectorXd restartMean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
                    for (const auto& sample : restartSamples) {
                        restartMean += Eigen::Map<const Eigen::VectorXd>(
                            sample.data(),
                            static_cast<Eigen::Index>(sample.size()));
                    }
                    restartMean /= static_cast<double>(restartSamples.size());
                    sigmaEff = sigma0;
                    resetRegimeState(restartMean, sigmaEff);
                    useRestartSamples = true;
                    continue;
                }
            }

            if (evalCount < lambda) {
                break;
            }
        }

        if (history.empty()) {
            recordHistory(0.0);
        }

        return getBestFromHistory();
    } catch (const std::exception& ex) {
        throw std::runtime_error(ex.what());
    }
}

}  // namespace minion
