#include "acmaes.h"

#include "default_options.h"
#include "utility.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

namespace minion {

ACMAES::ACMAES(
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

void ACMAES::Parameter::reserve(size_t n_offsprings_reserve_, size_t n_parents_reserve_, size_t n_params_) {
    n_params = n_params_;
    n_offsprings_reserve = n_offsprings_reserve_;
    n_parents_reserve = n_parents_reserve_;

    x_offsprings.resize(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_offsprings_reserve));
    x_parents_ranked.resize(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_parents_reserve));
    z_offsprings.resize(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_offsprings_reserve));
    y_offsprings.resize(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_offsprings_reserve));
    y_offsprings_ranked.resize(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_offsprings_reserve));
    f_offsprings = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_offsprings_reserve), std::numeric_limits<double>::infinity());
    w = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_offsprings_reserve));
    w_var = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_offsprings_reserve));
    y_mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_params));
    x_mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_params));
    x_mean_old = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_params));
    p_c = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_params));
    p_s = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_params));
    eigvals_C = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(n_params));
    C = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_params));
    C_invsqrt = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_params));
    B = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_params));
    D = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n_params), static_cast<Eigen::Index>(n_params));
    keys_offsprings.resize(n_offsprings_reserve);
}

void ACMAES::Parameter::reinit(size_t n_offsprings_, size_t n_parents_, size_t n_params_, const Eigen::VectorXd& x_mean_, double sigma_) {
    n_params = n_params_;
    n_offsprings = std::min(n_offsprings_, n_offsprings_reserve);
    n_parents = std::min(n_parents_, n_parents_reserve);
    i_iteration = 0;
    i_func_eval = 0;

    x_mean = x_mean_;
    x_mean_old = x_mean_;

    double w_neg_sum = 0.0;
    double w_pos_sum = 0.0;
    for (size_t i = 0; i < n_offsprings; ++i) {
        double wi = std::log((static_cast<double>(n_offsprings) + 1.0) / 2.0) - std::log(static_cast<double>(i) + 1.0);
        w(static_cast<Eigen::Index>(i)) = wi;
        if (wi >= 0.0) {
            w_pos_sum += wi;
        } else {
            w_neg_sum += wi;
        }
    }

    double w_sum_parent = 0.0;
    double w_sq_sum_parent = 0.0;
    for (size_t i = 0; i < n_parents; ++i) {
        double wi = w(static_cast<Eigen::Index>(i));
        w_sum_parent += wi;
        w_sq_sum_parent += wi * wi;
    }
    double denom = w_sq_sum_parent <= 0.0 ? 1e-12 : w_sq_sum_parent;
    n_mu_eff = (w_sum_parent * w_sum_parent) / denom;

    p_s.setZero();
    p_c.setZero();

    double dim = static_cast<double>(n_params);
    c_s = (n_mu_eff + 2.0) / (dim + n_mu_eff + 5.0);
    c_c = (4.0 + n_mu_eff / dim) / (dim + 4.0 + 2.0 * n_mu_eff / dim);
    c_1 = 2.0 / (std::pow(dim + 1.3, 2.0) + n_mu_eff);
    c_mu = 2.0 * (n_mu_eff - 2.0 + 1.0 / n_mu_eff) / (std::pow(dim + 2.0, 2.0) + n_mu_eff);
    c_mu = std::min(1.0 - c_1, c_mu);
    d_s = 1.0 + c_s + 2.0 * std::max(0.0, std::sqrt((n_mu_eff - 1.0) / (dim + 1.0)) - 1.0);
    chi = std::sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim));
    p_s_fact = std::sqrt(c_s * (2.0 - c_s) * n_mu_eff);
    p_c_fact = std::sqrt(c_c * (2.0 - c_c) * n_mu_eff);

    double a_mu = 1.0 + c_1 / std::max(1e-12, c_mu);
    double a_mueff = 1.0 + 2.0 * n_mu_eff;
    double a_posdef = (1.0 - c_1 - c_mu) / (dim * std::max(1e-12, c_mu));
    double a_min = std::min({a_mu, a_mueff, a_posdef});
    for (size_t i = 0; i < n_offsprings; ++i) {
        double wi = w(static_cast<Eigen::Index>(i));
        if (wi >= 0.0) {
            w(static_cast<Eigen::Index>(i)) = wi / std::max(1e-12, w_pos_sum);
        } else {
            w(static_cast<Eigen::Index>(i)) = a_min * wi / std::max(1e-12, std::abs(w_neg_sum));
        }
    }

    w_var.head(static_cast<Eigen::Index>(n_offsprings)) = w.head(static_cast<Eigen::Index>(n_offsprings));
    y_mean.setZero();

    sigma = sigma_;
    C.setIdentity();
    C_invsqrt.setIdentity();
    B.setIdentity();
    D.setIdentity();
    eigvals_C.setOnes();
    f_offsprings.head(static_cast<Eigen::Index>(n_offsprings)) = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_offsprings), std::numeric_limits<double>::infinity());
}

std::vector<double> ACMAES::applyBounds(const std::vector<double>& candidate) const {
    if (!useBounds) {
        return candidate;
    }
    std::vector<std::vector<double>> wrapper = {candidate};
    enforce_bounds(wrapper, bounds, boundStrategy);
    return wrapper.front();
}

std::vector<double> ACMAES::eigenToStd(const Eigen::VectorXd& vec) const {
    return std::vector<double>(vec.data(), vec.data() + vec.size());
}

void ACMAES:: initialize() {
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
    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));

    lambda = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambda == 0) {
        double logDim = dimension > 0 ? std::log(static_cast<double>(dimension)) : 1.0;
        lambda = static_cast<size_t>(4.0 + std::floor(3.0 * logDim));
    }
    lambda = std::max<size_t>(lambda, 4);

    mu = static_cast<size_t>(options.get<int>("mu", 0));
    if (mu == 0) {
        mu = std::max<size_t>(lambda / 2, 1);
    }
    mu = std::min(mu, lambda);
    mu_ratio = lambda > 0 ? static_cast<double>(mu) / static_cast<double>(lambda) : 0.5;

    maxIterations = static_cast<size_t>(options.get<int>("max_iterations", 5000));
    support_tol = false;

    sigma0 = options.get<double>("initial_step", 0.3);
    if (sigma0 <= 0.0) {
        sigma0 = 0.3;
    }

    double avgRange = 0.0;
    for (const auto& b : bounds) {
        avgRange += (b.second - b.first);
    }
    avgRange = dimension > 0 ? avgRange / static_cast<double>(dimension) : 1.0;
    sigma0 *= avgRange;

    initialMean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
    if (!x0.empty()) {
        std::vector<double> initialGuess;
        if (x0.size() > 1) {
            initialGuess = findBestPoint(x0);
        } else {
            initialGuess = x0.front();
        }
        if (initialGuess.size() != dimension) {
            throw std::runtime_error("Initial guess does not match problem dimension");
        }
        for (size_t i = 0; i < dimension; ++i) {
            initialMean(static_cast<Eigen::Index>(i)) = initialGuess[i];
        }
    } else {
        auto initial = random_sampling(bounds, 1).front();
        for (size_t i = 0; i < dimension; ++i) {
            initialMean(static_cast<Eigen::Index>(i)) = initial[i];
        }
    }

    era.reserve(lambda, mu, dimension);

    best = eigenToStd(initialMean);
    best_fitness = std::numeric_limits<double>::infinity();

    hasInitialized = true;
}

void ACMAES::sampleOffsprings() {
    for (size_t j = 0; j < era.n_offsprings; ++j) {
        for (size_t i = 0; i < era.n_params; ++i) {
            era.z_offsprings(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = rand_norm(0.0, 1.0);
        }
    }
    Eigen::MatrixXd BD = era.B * era.D;
    era.y_offsprings.block(0, 0, static_cast<Eigen::Index>(era.n_params), static_cast<Eigen::Index>(era.n_offsprings)) =
        BD * era.z_offsprings.block(0, 0, static_cast<Eigen::Index>(era.n_params), static_cast<Eigen::Index>(era.n_offsprings));

    for (size_t j = 0; j < era.n_offsprings; ++j) {
        Eigen::VectorXd candidate = era.x_mean + era.sigma * era.y_offsprings.col(static_cast<Eigen::Index>(j));
        auto candidateVec = eigenToStd(candidate);
        candidateVec = applyBounds(candidateVec);
        Eigen::VectorXd bounded = Eigen::Map<const Eigen::VectorXd>(candidateVec.data(), static_cast<Eigen::Index>(candidateVec.size()));
        era.x_offsprings.col(static_cast<Eigen::Index>(j)) = bounded;
        era.y_offsprings.col(static_cast<Eigen::Index>(j)) = (bounded - era.x_mean) / era.sigma;
    }
}

size_t ACMAES::evaluatePopulation() {
    currentFitness.assign(era.n_offsprings, std::numeric_limits<double>::infinity());
    if (Nevals >= maxevals) {
        should_stop = true;
        return 0;
    }

    size_t remaining = maxevals - Nevals;
    size_t evalCount = std::min(remaining, era.n_offsprings);
    if (evalCount == 0) {
        should_stop = true;
        return 0;
    }

    std::vector<std::vector<double>> population;
    population.reserve(evalCount);
    for (size_t j = 0; j < evalCount; ++j) {
        population.push_back(eigenToStd(era.x_offsprings.col(static_cast<Eigen::Index>(j))));
    }

    auto values = func(population, data);
    if (values.size() != evalCount) {
        throw std::runtime_error("Objective function returned unexpected number of values");
    }

    for (size_t j = 0; j < evalCount; ++j) {
        double val = values[j];
        if (std::isnan(val)) {
            val = std::numeric_limits<double>::infinity();
        }
        currentFitness[j] = val;
        era.f_offsprings(static_cast<Eigen::Index>(j)) = val;
    }

    if (evalCount < era.n_offsprings) {
        for (size_t j = evalCount; j < era.n_offsprings; ++j) {
            era.f_offsprings(static_cast<Eigen::Index>(j)) = std::numeric_limits<double>::infinity();
        }
        should_stop = true;
    }

    era.i_func_eval += evalCount;
    Nevals += evalCount;
    return evalCount;
}

void ACMAES::rankAndSort() {
    std::vector<double> fitness(era.n_offsprings);
    for (size_t i = 0; i < era.n_offsprings; ++i) {
        fitness[i] = era.f_offsprings(static_cast<Eigen::Index>(i));
    }

    auto order = argsort(fitness, true);
    era.keys_offsprings = order;

    for (size_t i = 0; i < era.n_parents; ++i) {
        size_t idx = order[i];
        era.x_parents_ranked.col(static_cast<Eigen::Index>(i)) = era.x_offsprings.col(static_cast<Eigen::Index>(idx));
    }

    for (size_t i = 0; i < era.n_offsprings; ++i) {
        size_t idx = order[i];
        era.y_offsprings_ranked.col(static_cast<Eigen::Index>(i)) = era.y_offsprings.col(static_cast<Eigen::Index>(idx));
    }

    era.w_var.head(static_cast<Eigen::Index>(era.n_offsprings)) = era.w.head(static_cast<Eigen::Index>(era.n_offsprings));
    currentFitness = fitness;
}

void ACMAES::updateBest() {
    if (era.keys_offsprings.empty()) {
        return;
    }
    size_t bestIdx = era.keys_offsprings.front();
    double candidateFitness = era.f_offsprings(static_cast<Eigen::Index>(bestIdx));
    if (!std::isfinite(candidateFitness)) {
        return;
    }
    if (candidateFitness < best_fitness) {
        best_fitness = candidateFitness;
        best = eigenToStd(era.x_offsprings.col(static_cast<Eigen::Index>(bestIdx)));
    }
}

void ACMAES::assignNewMean() {
    era.x_mean_old = era.x_mean;
    Eigen::VectorXd weights = era.w.head(static_cast<Eigen::Index>(era.n_parents));
    era.y_mean = era.y_offsprings_ranked.block(0, 0, static_cast<Eigen::Index>(era.n_params), static_cast<Eigen::Index>(era.n_parents)) * weights;
    era.x_mean = era.x_mean + era.sigma * era.y_mean;
}

void ACMAES::updateEvolutionPaths() {
    Eigen::VectorXd CinvSqrt_y = era.C_invsqrt * era.y_mean;
    era.p_s = (1.0 - era.c_s) * era.p_s + era.p_s_fact * CinvSqrt_y;

    double norm_ps = era.p_s.norm();
    double threshold = (1.4 + 2.0 / (static_cast<double>(era.n_params) + 1.0)) *
                       std::sqrt(1.0 - std::pow(1.0 - era.c_s, 2.0 * static_cast<double>(era.i_iteration + 1))) *
                       era.chi;
    era.h_sig = norm_ps < threshold;

    era.p_c = (1.0 - era.c_c) * era.p_c + era.h_sig * era.p_c_fact * era.y_mean;
}

void ACMAES::updateWeights() {
    for (size_t i = 0; i < era.n_offsprings; ++i) {
        if (era.w(static_cast<Eigen::Index>(i)) < 0.0) {
            Eigen::VectorXd adjusted = era.C_invsqrt * era.y_offsprings_ranked.col(static_cast<Eigen::Index>(i));
            double denom = adjusted.squaredNorm();
            if (denom > 0.0) {
                era.w_var(static_cast<Eigen::Index>(i)) = era.w(static_cast<Eigen::Index>(i)) * static_cast<double>(era.n_params) / denom;
            } else {
                era.w_var(static_cast<Eigen::Index>(i)) = 0.0;
            }
        }
    }
}

void ACMAES::updateCovarianceMatrix() {
    double h1 = (1.0 - era.h_sig) * era.c_c * (2.0 - era.c_c);
    double w_sum = era.w.head(static_cast<Eigen::Index>(era.n_offsprings)).sum();

    Eigen::MatrixXd yWeighted = era.y_offsprings_ranked.block(0, 0, static_cast<Eigen::Index>(era.n_params), static_cast<Eigen::Index>(era.n_offsprings)) *
                                era.w_var.head(static_cast<Eigen::Index>(era.n_offsprings)).asDiagonal() *
                                era.y_offsprings_ranked.block(0, 0, static_cast<Eigen::Index>(era.n_params), static_cast<Eigen::Index>(era.n_offsprings)).transpose();

    era.C = (1.0 + era.c_1 * h1 - era.c_1 - era.c_mu * w_sum) * era.C +
             era.c_1 * (era.p_c * era.p_c.transpose()) +
             era.c_mu * yWeighted;
    era.C = 0.5 * (era.C + era.C.transpose());
}

void ACMAES::updateStepsize() {
    double norm_ps = era.p_s.norm();
    era.sigma *= std::exp(era.c_s / era.d_s * (norm_ps / era.chi - 1.0));
}

void ACMAES::updateEigenDecomposition() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(era.C);
    if (solver.info() != Eigen::Success) {
        era.C = 0.5 * (era.C + era.C.transpose());
        era.C += 1e-12 * Eigen::MatrixXd::Identity(era.C.rows(), era.C.cols());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> retry(era.C);
        if (retry.info() != Eigen::Success) {
            era.C = Eigen::MatrixXd::Identity(era.C.rows(), era.C.cols());
            era.B = Eigen::MatrixXd::Identity(era.B.rows(), era.B.cols());
            era.D = Eigen::MatrixXd::Identity(era.D.rows(), era.D.cols());
            era.eigvals_C = Eigen::VectorXd::Ones(era.eigvals_C.size());
            era.C_invsqrt = Eigen::MatrixXd::Identity(era.C_invsqrt.rows(), era.C_invsqrt.cols());
            return;
        }
        solver = std::move(retry);
    }
    Eigen::VectorXd evals = solver.eigenvalues();
    for (Eigen::Index i = 0; i < evals.size(); ++i) {
        if (evals(i) < 1e-30) {
            evals(i) = 1e-30;
        }
    }
    era.eigvals_C = evals;
    era.B = solver.eigenvectors();
    Eigen::VectorXd sqrtVals = evals.cwiseSqrt();
    era.D = sqrtVals.asDiagonal();
    Eigen::VectorXd invSqrt = sqrtVals.cwiseInverse();
    era.C_invsqrt = era.B * invSqrt.asDiagonal() * era.B.transpose();
}

void ACMAES::checkStoppingCriteria() {
    if (era.eigvals_C.size() == 0) {
        return;
    }

    double eigval_min = era.eigvals_C.minCoeff();
    double eigval_max = era.eigvals_C.maxCoeff();
    if (eigval_min <= 0.0) {
        eigval_min = 1e-30;
    }
    if (eigval_max / eigval_min > 1e14) {
        should_stop = true;
        return;
    }

    if (best_fitness < 1e-10) {
        should_stop = true;
        return;
    }

    double sigma_fac = sigma0 > 0.0 ? era.sigma / sigma0 : era.sigma;
    double sigma_up_thresh = 1e20 * std::sqrt(eigval_max);
    if (sigma_fac > sigma_up_thresh) {
        should_stop = true;
        return;
    }

    for (size_t i = 0; i < era.n_params; ++i) {
        double ei = 0.1 * era.sigma * era.eigvals_C(static_cast<Eigen::Index>(i));
        Eigen::VectorXd axis = ei * era.B.col(static_cast<Eigen::Index>(i));
        Eigen::VectorXd moved = era.x_mean + axis;
        if ((moved.array() == era.x_mean.array()).all()) {
            should_stop = true;
            return;
        }
    }

    for (size_t i = 0; i < era.n_params; ++i) {
        double delta = 0.2 * era.sigma * std::sqrt(era.C(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)));
        double moved = era.x_mean(static_cast<Eigen::Index>(i)) + delta;
        if (moved == era.x_mean(static_cast<Eigen::Index>(i))) {
            should_stop = true;
            return;
        }
    }
}

void ACMAES::recordHistory(double relRange) {
    bool success = support_tol && relRange <= stoppingTol;
    minionResult = MinionResult(best, best_fitness, generation, Nevals, success, success ? "stopping tolerance reached" : "");
    history.push_back(minionResult);
    if (callback != nullptr) {
        callback(&minionResult);
    }
}

MinionResult ACMAES::optimize() {
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
        should_stop = false;

        size_t restart_factor = 1;
        const size_t lambda_base = lambda;
        Eigen::VectorXd restart_mean = initialMean;
        bool first_run = true;

        while (!should_stop && Nevals < maxevals) {
            size_t lambda_current = lambda_base * restart_factor;
            if (lambda_current < 4) lambda_current = 4;
            size_t mu_current = static_cast<size_t>(std::round(mu_ratio * static_cast<double>(lambda_current)));
            if (mu_current < 1) mu_current = 1;
            if (mu_current > lambda_current) mu_current = lambda_current;

            if (lambda_current > era.n_offsprings_reserve || mu_current > era.n_parents_reserve) {
                era.reserve(lambda_current, mu_current, dimension);
            }
            auto lhs_init = latin_hypercube_sampling(bounds, lambda_current);
            if (!lhs_init.empty()) {
                Eigen::VectorXd mean = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dimension));
                for (const auto& sample : lhs_init) {
                    mean += Eigen::Map<const Eigen::VectorXd>(sample.data(), static_cast<Eigen::Index>(sample.size()));
                }
                mean /= static_cast<double>(lhs_init.size());
                restart_mean = mean;
            }
            era.reinit(lambda_current, mu_current, dimension, restart_mean, sigma0);

            bool restart = false;
            bool use_lhs = true;
            while (!should_stop && !restart && era.i_iteration < maxIterations) {
                if (use_lhs) {
                    for (size_t j = 0; j < era.n_offsprings; ++j) {
                        Eigen::VectorXd candidate = Eigen::Map<const Eigen::VectorXd>(lhs_init[j].data(),
                                                                                     static_cast<Eigen::Index>(lhs_init[j].size()));
                        era.x_offsprings.col(static_cast<Eigen::Index>(j)) = candidate;
                        era.y_offsprings.col(static_cast<Eigen::Index>(j)) = (candidate - era.x_mean) / era.sigma;
                    }
                    if (!best.empty()) {
                        Eigen::VectorXd elite = Eigen::Map<const Eigen::VectorXd>(best.data(), static_cast<Eigen::Index>(best.size()));
                        era.x_offsprings.col(0) = elite;
                        era.y_offsprings.col(0) = (elite - era.x_mean) / era.sigma;
                    }
                    use_lhs = false;
                } else {
                    sampleOffsprings();
                }
                size_t evaluated = evaluatePopulation();
                if (evaluated == 0) {
                    should_stop = true;
                    break;
                }

                rankAndSort();
                updateBest();
                assignNewMean();
                updateEvolutionPaths();
                updateWeights();
                updateCovarianceMatrix();
                updateEigenDecomposition();
                updateStepsize();

                double fmax = *std::max_element(currentFitness.begin(), currentFitness.end());
                double fmin = *std::min_element(currentFitness.begin(), currentFitness.end());
                double fmean = std::accumulate(currentFitness.begin(), currentFitness.end(), 0.0) /
                               static_cast<double>(currentFitness.size());
                double relRange = 0.0;
                if (std::fabs(fmean) > 1e-12) {
                    relRange = (fmax - fmin) / std::fabs(fmean);
                }
                diversity.push_back(relRange);
                recordHistory(relRange);
              //  std::cout << "Generation " << generation
               //           << ", Best Fitness = " << best_fitness << "\n";

                if (relRange < 1e-8) {
                    restart = true;
                }

                ++era.i_iteration;
                ++generation;
            }

            if (should_stop || Nevals >= maxevals) {
                break;
            }

            if (!restart) {
                restart = true;
            }

            ++restart_factor;
            first_run = false;
        }

        if (history.empty()) {
            recordHistory(0.0);
        }

        return getBestFromHistory();
    } catch (const std::exception& ex) {
        throw std::runtime_error(ex.what());
    }
}

}
