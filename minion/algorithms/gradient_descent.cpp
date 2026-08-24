#include "gradient_descent.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace minion {

namespace {

std::vector<double> select_initial_point(
    MinionFunction func,
    void* data,
    const std::vector<std::pair<double, double>>& bounds,
    std::vector<std::vector<double>> x0,
    size_t& nfev,
    std::vector<double>& best,
    double& best_f)
{
    if (x0.empty()) {
        x0 = latin_hypercube_sampling(bounds, 1);
    }
    const std::vector<double> initial = (x0.size() == 1) ? x0.front() : [&]() {
        const auto values = func(x0, data);
        nfev += values.size();
        const size_t index = findArgMin(values);
        best = x0[index];
        best_f = values[index];
        return x0[index];
    }();
    if (best.empty()) {
        best = initial;
    }
    return initial;
}

bool check_scalar_convergence(double previous, double current, double tolerance)
{
    if (tolerance < 0.0) {
        return false;
    }
    const double denom = std::max(std::max(std::fabs(previous), std::fabs(current)), 1.0);
    return std::fabs(current - previous) / denom <= tolerance;
}

double vector_step_norm(const std::vector<double>& a, const std::vector<double>& b)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

bool points_equal(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

void GradientDescent::initialize() {
    hasInitialized = true;
}

MinionResult GradientDescent::optimize() {
    resetBestSoFar();

    auto settings = DefaultSettings().getDefaultSettings("GradientDescent");
    for (const auto& item : optionMap) {
        settings[item.first] = item.second;
    }
    Options options(settings);
    configureConvergenceTolerances(options, 1e-8, -1.0);

    const std::string update_rule = options.get<std::string>("update_rule", std::string("adam"));
    std::string estimator = options.get<std::string>("gradient_estimator", std::string("coordinate_fd"));
    const bool has_legacy_learning_rate = optionMap.find("learning_rate") != optionMap.end();
    const double base_learning_rate = has_legacy_learning_rate
        ? options.get<double>("learning_rate", 1e-2)
        : options.get<double>("base_learning_rate", 1e-2);
    const double lr_decay = options.get<double>("lr_decay", 1.0);
    const double beta1 = options.get<double>("beta1", 0.9);
    const double beta2 = options.get<double>("beta2", 0.999);
    const double epsilon = options.get<double>("epsilon", 1e-8);
    const double momentum = options.get<double>("momentum", 0.0);
    const bool use_line_search = options.get<bool>("use_line_search", false);
    const int max_linesearch = std::max(options.get<int>("max_linesearch", 8), 0);
    const double line_search_c1 = options.get<double>("line_search_c1", 1e-4);
    const double line_search_rho = std::min(std::max(options.get<double>("line_search_rho", 0.5), 1e-6), 0.999);
    const double g_tol = options.get<double>("g_tol", 1e-6);
    const int N_points = options.get<int>("N_points_derivative", 2);
    const double func_noise_ratio = options.get<double>("func_noise_ratio", 1e-10);
    const double fd_epsilon = options.get<double>("fd_epsilon", 0.0);
    size_t gradient_samples = static_cast<size_t>(std::max(options.get<int>("gradient_samples", 8), 1));
    size_t coordinate_batch_size = static_cast<size_t>(std::max(options.get<int>("coordinate_batch_size", 0), 0));
    boundStrategy = options.get<std::string>("bound_strategy", std::string("clip"));

    if (update_rule == "sgd") {
        if (estimator == "coordinate_fd" && coordinate_batch_size == 0) {
            coordinate_batch_size = 1;
        }
        if (estimator == "random_direction_fd" && gradient_samples == 0) {
            gradient_samples = 1;
        }
    }

    Nevals = 0;
    best.clear();
    best_f = std::numeric_limits<double>::infinity();

    std::vector<double> x = select_initial_point(func, data, bounds, x0, Nevals, best, best_f);
    std::vector<double> m(x.size(), 0.0);
    std::vector<double> v(x.size(), 0.0);
    std::vector<double> velocity(x.size(), 0.0);

    double current_lr = base_learning_rate;
    size_t iterations = 0;

    if (maxiters == 0) {
        minionResult = MinionResult(x, best_f, 0, Nevals, TerminationStatus::MaxIterationsReached, "Maximum number of iterations reached.");
        updateBestSoFar(minionResult);
        return getBestSoFar();
    }

    try {
        while (true) {
            BlackBoxGradientOptions gradient_options;
            gradient_options.N_points = N_points;
            gradient_options.func_noise_ratio = func_noise_ratio;
            gradient_options.last_f = last_f;
            gradient_options.fd_epsilon = fd_epsilon;
            gradient_options.finite_diff_rel_step = finite_diff_rel_step;
            gradient_options.estimator = estimator;
            gradient_options.gradient_samples = gradient_samples;
            gradient_options.coordinate_batch_size = coordinate_batch_size;
            gradient_options.bounded = true;
            gradient_options.bounds = bounds;

            const BlackBoxGradientResult gradient_result = estimateBlackBoxGradient(func, data, x, gradient_options);
            Nevals += gradient_result.evaluations;
            last_f = gradient_result.value;

            const size_t best_index = findArgMin(gradient_result.sampled_values);
            if (gradient_result.sampled_values[best_index] < best_f) {
                best_f = gradient_result.sampled_values[best_index];
                best = gradient_result.sampled_points[best_index];
            }

            minionResult = MinionResult(best, best_f, iterations, Nevals, TerminationStatus::Running, "");
            updateBestSoFar(minionResult);
            if (shouldStopFromCallback(minionResult)) {
                return getBestSoFar();
            }

            double grad_norm = 0.0;
            for (double value : gradient_result.gradient) {
                grad_norm += value * value;
            }
            grad_norm = std::sqrt(grad_norm);
            if (g_tol >= 0.0 && grad_norm <= g_tol) {
                return finalizeBestSoFar(
                    TerminationStatus::Converged,
                    "Gradient norm is below convergence tolerance.",
                    Nevals,
                    iterations);
            }
            if (Nevals >= maxevals) {
                return finalizeBestSoFar(
                    TerminationStatus::MaxEvaluationsReached,
                    "Maximum number of function evaluations reached.",
                    Nevals,
                    iterations);
            }
            if (reachedMaxIterations(iterations)) {
                return finalizeBestSoFar(
                    TerminationStatus::MaxIterationsReached,
                    "Maximum number of iterations reached.",
                    Nevals,
                    iterations);
            }

            std::vector<double> x_previous = x;
            std::vector<double> update(x.size(), 0.0);
            if (update_rule == "adam") {
                const double t = static_cast<double>(iterations + 1);
                for (size_t i = 0; i < x.size(); ++i) {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * gradient_result.gradient[i];
                    v[i] = beta2 * v[i] + (1.0 - beta2) * gradient_result.gradient[i] * gradient_result.gradient[i];
                    const double m_hat = m[i] / (1.0 - std::pow(beta1, t));
                    const double v_hat = v[i] / (1.0 - std::pow(beta2, t));
                    update[i] = -current_lr * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            } else {
                for (size_t i = 0; i < x.size(); ++i) {
                    velocity[i] = momentum * velocity[i] - current_lr * gradient_result.gradient[i];
                    update[i] = velocity[i];
                }
            }

            std::vector<double> x_candidate = x;
            bool accepted_step = false;
            if (use_line_search) {
                std::vector<std::vector<double>> line_search_points;
                std::vector<double> line_search_scales;
                line_search_points.reserve(static_cast<size_t>(max_linesearch) + 1);
                line_search_scales.reserve(static_cast<size_t>(max_linesearch) + 1);

                double scale = 1.0;
                for (int ls = 0; ls <= max_linesearch; ++ls) {
                    std::vector<double> candidate = x;
                    for (size_t i = 0; i < candidate.size(); ++i) {
                        candidate[i] += scale * update[i];
                    }
                    enforce_bounds(candidate, bounds, boundStrategy);
                    line_search_points.push_back(std::move(candidate));
                    line_search_scales.push_back(scale);
                    scale *= line_search_rho;
                }

                const auto line_search_values = func(line_search_points, data);
                Nevals += line_search_values.size();
                if (line_search_values.size() != line_search_points.size() ||
                    !std::all_of(line_search_values.begin(), line_search_values.end(), [](double value) { return std::isfinite(value); })) {
                    throw std::runtime_error("Objective function returned non-finite value during backtracking line search.");
                }

                const double directional_derivative = std::inner_product(
                    gradient_result.gradient.begin(),
                    gradient_result.gradient.end(),
                    update.begin(),
                    0.0);

                for (size_t i = 0; i < line_search_points.size(); ++i) {
                    if (line_search_values[i] < best_f) {
                        best_f = line_search_values[i];
                        best = line_search_points[i];
                    }
                }

                for (size_t i = 0; i < line_search_points.size(); ++i) {
                    if (points_equal(line_search_points[i], x_previous)) {
                        continue;
                    }
                    const bool armijo_ok = (directional_derivative < 0.0)
                        ? line_search_values[i] <= gradient_result.value + line_search_c1 * line_search_scales[i] * directional_derivative
                        : line_search_values[i] < gradient_result.value;
                    if (armijo_ok) {
                        x_candidate = line_search_points[i];
                        accepted_step = true;
                        break;
                    }
                }

                if (!accepted_step) {
                    size_t best_improving_index = line_search_points.size();
                    double best_improving_value = gradient_result.value;
                    for (size_t i = 0; i < line_search_points.size(); ++i) {
                        if (points_equal(line_search_points[i], x_previous)) {
                            continue;
                        }
                        if (line_search_values[i] < best_improving_value) {
                            best_improving_value = line_search_values[i];
                            best_improving_index = i;
                        }
                    }
                    if (best_improving_index < line_search_points.size()) {
                        x_candidate = line_search_points[best_improving_index];
                        accepted_step = true;
                    }
                }
            } else {
                for (size_t i = 0; i < x_candidate.size(); ++i) {
                    x_candidate[i] += update[i];
                }
                enforce_bounds(x_candidate, bounds, boundStrategy);
                accepted_step = true;
            }

            x = x_candidate;
            ++iterations;

            if (xTol >= 0.0 && vector_step_norm(x_previous, x) <= xTol) {
                minionResult = MinionResult(best, best_f, iterations, Nevals, TerminationStatus::Converged, "Step size is below convergence tolerance.");
                updateBestSoFar(minionResult);
                return getBestSoFar();
            }
            if (fTol >= 0.0 && check_scalar_convergence(gradient_result.value, best_f, fTol)) {
                minionResult = MinionResult(best, best_f, iterations, Nevals, TerminationStatus::Converged, "Relative objective improvement is below convergence tolerance.");
                updateBestSoFar(minionResult);
                return getBestSoFar();
            }
            if (x == x_previous) {
                return finalizeBestSoFar(
                    TerminationStatus::Stagnated,
                    use_line_search
                        ? "Backtracking line search failed to find a productive step."
                        : "Projected gradient step produced no movement.",
                    Nevals,
                    iterations);
            }

            current_lr *= lr_decay;
        }
    } catch (const std::exception& e) {
        minionResult = MinionResult(best, best_f, iterations, Nevals, TerminationStatus::NumericalError, e.what());
        updateBestSoFar(minionResult);
        return getBestSoFar();
    }
}

}  // namespace minion
