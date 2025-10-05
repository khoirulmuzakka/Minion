#include "abipop_cmaes.h"

#include "default_options.h"
#include "utility.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace minion {

ABIPOP_CMAES::ABIPOP_CMAES(
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

void ABIPOP_CMAES::Parameter::reserve(size_t n_offsprings_reserve_, size_t n_parents_reserve_, size_t n_params_) {
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

void ABIPOP_CMAES::Parameter::reinit(size_t n_offsprings_, size_t n_parents_, size_t n_params_, const Eigen::VectorXd& x_mean_, double sigma_) {
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

std::vector<double> ABIPOP_CMAES::applyBounds(const std::vector<double>& candidate) const {
    if (!useBounds) {
        return candidate;
    }
    std::vector<std::vector<double>> wrapper = {candidate};
    enforce_bounds(wrapper, bounds, boundStrategy);
    return wrapper.front();
}

std::vector<double> ABIPOP_CMAES::eigenToStd(const Eigen::VectorXd& vec) const {
    return std::vector<double>(vec.data(), vec.data() + vec.size());
}

void ABIPOP_CMAES::initialize() {
    auto defaults = DefaultSettings().getDefaultSettings("ABIPOP_CMAES");
    for (const auto& item : optionMap) {
        defaults[item.first] = item.second;
    }
    Options options(defaults);

    dimension = bounds.size();
    if (dimension == 0) {
        throw std::runtime_error("ABIPOP CMA-ES requires bounded variables");
    }

    useBounds = !bounds.empty();
    boundStrategy = options.get<std::string>("bound_strategy", std::string("reflect-random"));

    lambda0 = static_cast<size_t>(options.get<int>("population_size", 0));
    if (lambda0 == 0) {
        double logDim = dimension > 0 ? std::log(static_cast<double>(dimension)) : 1.0;
        lambda0 = static_cast<size_t>(4.0 + std::floor(3.0 * logDim));
    }
    lambda0 = std::max<size_t>(lambda0, 4);
    mu0 = std::max<size_t>(lambda0 / 2, 1);

    maxRestarts = static_cast<size_t>(options.get<int>("max_restarts", 8));
    maxIterations = static_cast<size_t>(options.get<int>("max_iterations", 5000));
    support_tol = options.get<bool>("support_tol", true);

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

    size_t lambda_max = lambda0;
    if (maxRestarts > 0) {
        size_t limit = std::min<size_t>(maxRestarts, 20);
        lambda_max = lambda0 * (1ull << limit);
    }
    lambda_max = std::max<size_t>(lambda_max, lambda0);
    size_t mu_max = std::max<size_t>(lambda_max / 2, 1);
    era.reserve(lambda_max, mu_max, dimension);

    best = eigenToStd(initialMean);
    best_fitness = std::numeric_limits<double>::infinity();

    hasInitialized = true;
}

void ABIPOP_CMAES::sampleOffsprings() {
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

size_t ABIPOP_CMAES::evaluatePopulation() {
    currentFitness.assign(era.n_offsprings, std::numeric_limits<double>::infinity());
    if (Nevals >= maxevals) {
        should_stop_run = true;
        should_stop_optimization = true;
        return 0;
    }

    size_t remaining = maxevals - Nevals;
    size_t evalCount = std::min(remaining, era.n_offsprings);
    if (evalCount == 0) {
        should_stop_run = true;
        should_stop_optimization = true;
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
        should_stop_run = true;
        should_stop_optimization = true;
    }

    era.i_func_eval += evalCount;
    Nevals += evalCount;
    return evalCount;
}

void ABIPOP_CMAES::rankAndSort() {
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

void ABIPOP_CMAES::updateBest() {
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

void ABIPOP_CMAES::assignNewMean() {
    era.x_mean_old = era.x_mean;
    Eigen::VectorXd weights = era.w.head(static_cast<Eigen::Index>(era.n_parents));
    era.y_mean = era.y_offsprings_ranked.block(0, 0, static_cast<Eigen::Index>(era.n_params), static_cast<Eigen::Index>(era.n_parents)) * weights;
    era.x_mean = era.x_mean + era.sigma * era.y_mean;
}

void ABIPOP_CMAES::updateEvolutionPaths() {
    Eigen::VectorXd CinvSqrt_y = era.C_invsqrt * era.y_mean;
    era.p_s = (1.0 - era.c_s) * era.p_s + era.p_s_fact * CinvSqrt_y;

    double norm_ps = era.p_s.norm();
    double threshold = (1.4 + 2.0 / (static_cast<double>(era.n_params) + 1.0)) *
                       std::sqrt(1.0 - std::pow(1.0 - era.c_s, 2.0 * static_cast<double>(era.i_iteration + 1))) *
                       era.chi;
    era.h_sig = norm_ps < threshold;

    era.p_c = (1.0 - era.c_c) * era.p_c + era.h_sig * era.p_c_fact * era.y_mean;
}

void ABIPOP_CMAES::updateWeights() {
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

void ABIPOP_CMAES::updateCovarianceMatrix() {
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

void ABIPOP_CMAES::updateStepsize() {
    double norm_ps = era.p_s.norm();
    era.sigma *= std::exp(era.c_s / era.d_s * (norm_ps / era.chi - 1.0));
}

void ABIPOP_CMAES::updateEigenDecomposition() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(era.C);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed in ABIPOP CMA-ES");
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

void ABIPOP_CMAES::checkStoppingCriteria() {
    if (era.eigvals_C.size() == 0) {
        return;
    }

    double eigval_min = era.eigvals_C.minCoeff();
    double eigval_max = era.eigvals_C.maxCoeff();
    if (eigval_min <= 0.0) {
        eigval_min = 1e-30;
    }
    if (eigval_max / eigval_min > 1e14) {
        should_stop_run = true;
        return;
    }

    if (best_fitness < 1e-10) {
        should_stop_run = true;
        should_stop_optimization = true;
        return;
    }

    double sigma_fac = sigma0 > 0.0 ? era.sigma / sigma0 : era.sigma;
    double sigma_up_thresh = 1e20 * std::sqrt(eigval_max);
    if (sigma_fac > sigma_up_thresh) {
        should_stop_run = true;
        return;
    }

    for (size_t i = 0; i < era.n_params; ++i) {
        double ei = 0.1 * era.sigma * era.eigvals_C(static_cast<Eigen::Index>(i));
        Eigen::VectorXd axis = ei * era.B.col(static_cast<Eigen::Index>(i));
        Eigen::VectorXd moved = era.x_mean + axis;
        if ((moved.array() == era.x_mean.array()).all()) {
            should_stop_run = true;
            return;
        }
    }

    for (size_t i = 0; i < era.n_params; ++i) {
        double delta = 0.2 * era.sigma * std::sqrt(era.C(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)));
        double moved = era.x_mean(static_cast<Eigen::Index>(i)) + delta;
        if (moved == era.x_mean(static_cast<Eigen::Index>(i))) {
            should_stop_run = true;
            return;
        }
    }
}

void ABIPOP_CMAES::recordHistory(double relRange) {
    bool success = support_tol && relRange <= stoppingTol;
    minionResult = MinionResult(best, best_fitness, globalGeneration, Nevals, success, success ? "stopping tolerance reached" : "");
    history.push_back(minionResult);
    if (callback != nullptr) {
        callback(&minionResult);
    }

    if (success) {
        should_stop_optimization = true;
    }
}

MinionResult ABIPOP_CMAES::optimize() {
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
        globalGeneration = 0;
        should_stop_optimization = false;

    auto runRegime = [&](const Eigen::Ref<const Eigen::VectorXd>& startMean, double startSigma, size_t lambda) {
            era.reinit(lambda, std::max<size_t>(lambda / 2, 1), dimension, startMean, startSigma);
            should_stop_run = false;

            while (!should_stop_run && !should_stop_optimization && era.i_iteration < maxIterations) {
                sampleOffsprings();
                size_t evaluated = evaluatePopulation();
                if (evaluated == 0) {
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

                checkStoppingCriteria();
                ++era.i_iteration;
                ++globalGeneration;
            }

            return era.i_func_eval;
        };

        size_t budgetLarge = 0;
        size_t budgetSmall = 0;

        budgetLarge += runRegime(initialMean, sigma0, lambda0);

        for (size_t restart = 1; restart <= maxRestarts && !should_stop_optimization; ++restart) {
            size_t lambdaLarge = lambda0 * (1ull << restart);
            while (budgetLarge > budgetSmall && !should_stop_optimization) {
                double u1 = rand_gen(0.0, 1.0);
                double u2 = rand_gen(0.0, 1.0);
                size_t lambdaSmall = static_cast<size_t>(std::round(lambda0 * std::pow(0.5 * static_cast<double>(lambdaLarge) / lambda0, u1 * u1)));
                lambdaSmall = std::max<size_t>(lambdaSmall, 4);
                double sigmaSmall = sigma0 * 2.0 * std::pow(10.0, -2.0 * u2);
                Eigen::Map<const Eigen::VectorXd> bestVecEigen(best.data(), static_cast<Eigen::Index>(best.size()));
                size_t evals = runRegime(bestVecEigen, sigmaSmall, lambdaSmall);
                budgetSmall += evals;
                if (should_stop_optimization) {
                    break;
                }
            }

            if (should_stop_optimization) {
                break;
            }

            size_t evals = runRegime(initialMean, sigma0, lambdaLarge);
            budgetLarge += evals;
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
