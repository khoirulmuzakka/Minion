#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <minion.h>
#include <minion_cec.h>

namespace {

struct CheckRunner {
    using Check = std::pair<std::string, bool (*)()>;

    int run(const std::vector<Check>& checks) const {
        const size_t total = checks.size();
        for (size_t i = 0; i < total; ++i) {
            const auto& [name, check] = checks[i];
            if (!check()) {
                std::cerr << "Test " << (i + 1) << "/" << total << ": " << name << ": failed\n";
                return 1;
            }
            std::cout << "Test " << (i + 1) << "/" << total << ": " << name << ": passed\n";
        }
        std::cout << total << "/" << total << " Passed.\n";
        return 0;
    }
};

std::vector<double> sphere_batch(const std::vector<std::vector<double>>& X, void*) {
    std::vector<double> out(X.size(), 0.0);
    for (size_t i = 0; i < X.size(); ++i) {
        double f = 0.0;
        for (double value : X[i]) {
            f += value * value;
        }
        out[i] = f;
    }
    return out;
}

std::vector<double> rosenbrock_batch(const std::vector<std::vector<double>>& X, void*) {
    std::vector<double> out(X.size(), 0.0);
    for (size_t i = 0; i < X.size(); ++i) {
        double f = 0.0;
        for (size_t j = 0; j + 1 < X[i].size(); ++j) {
            const double a = X[i][j + 1] - X[i][j] * X[i][j];
            const double b = 1.0 - X[i][j];
            f += 100.0 * a * a + b * b;
        }
        out[i] = f;
    }
    return out;
}

std::vector<double> cec2017_batch(const std::vector<std::vector<double>>& X, void* data) {
    auto* cec = static_cast<minion::CECBase*>(data);
    return (*cec)(X);
}

const std::vector<std::string>& algorithms() {
    static const std::vector<std::string> values = {
        "DE",
        "LSHADE",
        "AGSK",
        "JADE",
        "j2020",
        "NLSHADE_RSP",
        "NLSHADE_LBC",
        "LSRTDE",
        "RDEX",
        "jSO",
        "IMODE",
        "ARRDE",
        "NelderMead",
        "ABC",
        "PSO",
        "SPSO2011",
        "DMSPSO",
        "LSHADE_cnEpSin",
        "CMAES",
        "ACMAES",
        "RCMAES",
        "BIPOP_aCMAES",
        "DA",
        "L_BFGS_B",
        "L_BFGS",
        "GradientDescent",
        "GD",
    };
    return values;
}

std::map<std::string, minion::ConfigValue> settings_for(const std::string& algo) {
    auto settings = minion::DefaultSettings().getDefaultSettings(algo);
    settings["x_tol"] = 1e-8;
    settings["f_tol"] = 1e-8;
    if (algo == "ARRDE") {
        settings["minimum_population_size"] = 4;
    }
    return settings;
}

bool is_valid_result(const minion::MinionResult& result, size_t maxevals, size_t slack) {
    return std::isfinite(result.fun) &&
           result.nfev <= maxevals + slack &&
           result.status != minion::TerminationStatus::Running &&
           result.status != minion::TerminationStatus::RuntimeError;
}

bool check_status_streaming() {
    std::ostringstream status_stream;
    status_stream << minion::TerminationStatus::MaxEvaluationsReached;

    minion::MinionResult result(
        {0.0},
        0.0,
        1,
        2,
        minion::TerminationStatus::Converged,
        "done");
    std::ostringstream result_stream;
    result_stream << result;

    return status_stream.str() == "max_evaluations_reached" &&
           result_stream.str().find("status=converged") != std::string::npos &&
           result.status == minion::TerminationStatus::Converged;
}

bool run_suite(
    const std::string& suite_name,
    minion::MinionFunction objective,
    void* data,
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<double>& x0,
    size_t maxevals,
    size_t slack,
    double quality_upper,
    bool check_quality) {
    bool passed = true;
    std::cout << suite_name << " benchmark\n";
    std::cout << std::left << std::setw(18) << "Algorithm"
              << std::right << std::setw(16) << "best_f"
              << std::setw(12) << "nfev"
              << std::setw(24) << "status" << '\n';
    std::cout << std::string(70, '-') << '\n';

    for (const auto& algo : algorithms()) {
        try {
            auto settings = settings_for(algo);
            minion::Minimizer optimizer(objective, bounds, x0, data, nullptr, algo, maxevals, 42, settings);
            const minion::MinionResult result = optimizer.optimize();

            std::cout << std::left << std::setw(18) << algo
                      << std::right << std::setw(16) << std::setprecision(8) << std::scientific << result.fun
                      << std::setw(12) << result.nfev
                      << std::setw(24) << result.status << '\n';

            const bool valid = is_valid_result(result, maxevals, slack);
            const bool quality_ok = !check_quality || result.fun <= quality_upper;
            if (!valid || !quality_ok) {
                passed = false;
                std::cerr << "[FAIL][" << suite_name << "] " << algo
                          << " finite=" << std::isfinite(result.fun)
                          << " nfev=" << result.nfev << " limit=" << (maxevals + slack)
                          << " status=" << result.status
                          << " fun=" << result.fun;
                if (check_quality) {
                    std::cerr << " quality_limit=" << quality_upper;
                }
                std::cerr << '\n';
            }
        } catch (const std::exception& e) {
            passed = false;
            std::cerr << "[FAIL][" << suite_name << "] " << algo << " exception: " << e.what() << '\n';
        }
    }

    std::cout << '\n';
    return passed;
}

bool check_sphere_benchmark() {
    const size_t dim = 2;
    const size_t maxevals = 1200;
    const size_t slack = 120;
    const std::vector<std::pair<double, double>> bounds(dim, {-5.0, 5.0});
    const std::vector<double> x0(dim, 0.5);

    return run_suite("Sphere", sphere_batch, nullptr, bounds, x0, maxevals, slack, 1e-2, true);
}

bool check_rosenbrock_benchmark() {
    const size_t dim = 2;
    const size_t maxevals = 2000;
    const size_t slack = 120;
    const std::vector<std::pair<double, double>> bounds(dim, {-5.0, 5.0});
    const std::vector<double> x0(dim, 0.5);

    return run_suite("Rosenbrock", rosenbrock_batch, nullptr, bounds, x0, maxevals, slack, 5.0, true);
}

bool check_cec2017_benchmarks() {
    const int dim = 10;
    const size_t maxevals = 600;
    const size_t slack = 160;
    const std::vector<std::pair<double, double>> bounds(static_cast<size_t>(dim), {-100.0, 100.0});
    const std::vector<double> x0(static_cast<size_t>(dim), 0.0);
    bool passed = true;

    for (int function_number : {1, 3}) {
        minion::CEC2017Functions cec(function_number, dim);
        const std::string suite_name = "CEC2017 F" + std::to_string(function_number);
        passed = run_suite(suite_name, cec2017_batch, &cec, bounds, x0, maxevals, slack, 0.0, false) && passed;
    }

    return passed;
}

bool callback_stop(minion::MinionResult*) {
    return true;
}

bool check_callback_stop() {
    const size_t dim = 2;
    const size_t maxevals = 300;
    const std::vector<std::pair<double, double>> bounds(dim, {-1.0, 1.0});
    const std::vector<double> x0(dim, 0.3);

    auto settings = settings_for("DE");
    minion::Minimizer optimizer(sphere_batch, bounds, x0, nullptr, callback_stop, "DE", maxevals, 7, settings);
    const minion::MinionResult result = optimizer.optimize();

    return result.status == minion::TerminationStatus::CallbackStopped &&
           result.message == "Callback requested optimization stop.";
}

bool check_arrde_spatial_phase_tolerances() {
    struct Probe : minion::ARRDE {
        using minion::ARRDE::ARRDE;
        using minion::ARRDE::normalizedPopulationSpread;
    };
    Probe metric(sphere_batch, {{0, 10}, {0, 1000}});
    metric.population = {{1, 100}, {9, 900}, {1, 900}, {9, 100}};
    if (std::fabs(metric.normalizedPopulationSpread() - 0.4) > 1e-12) return false;
    Probe scaled(sphere_batch, {{0, 100}, {-100, 900}});
    scaled.population = {{10, 0}, {90, 800}, {10, 800}, {90, 0}};
    if (std::fabs(scaled.normalizedPopulationSpread() - metric.normalizedPopulationSpread()) > 1e-12) return false;
    for (double refineTol : {-1.0, 1.0}) {
        size_t calls = 0;
        minion::MinionFunction constant = [&calls](const auto& points, void*) {
            ++calls;
            return std::vector<double>(points.size(), 0.0);
        };
        std::map<std::string, minion::ConfigValue> options = {
            {"population_size", 8}, {"x_tol_restart", 1.0}, {"x_tol_refine", refineTol}
        };
        Probe solver(constant, {{-1, 1}, {-1, 1}}, {}, nullptr, nullptr, 1000, 3, options);
        solver.initialize();
        solver.population = {{-0.8,-0.8},{0.8,0.8},{-0.6,0.6},{0.6,-0.6},
                             {-0.4,0.4},{0.4,-0.4},{-0.2,0.2},{0.2,-0.2}};
        solver.fitness.assign(8, 0.0);
        solver.best = solver.population.front();
        solver.best_fitness = 0.0;
        solver.Nevals = 8;
        solver.adaptParameters(); // Initial spatial convergence -> restart.
        if (calls != 1 || solver.population_records.empty()) return false;
        solver.adaptParameters(); // Restart allowance exhausted -> refinement.
        if (calls != 1) return false; // Refinement uses cached fitness.
        solver.adaptParameters(); // Only the enabled refine tolerance restarts.
        if (calls != (refineTol < 0 ? 1u : 2u)) return false;
    }
    return true;
}

bool check_lshade_population_reduction() {
    for (const std::string strategy : {"linear", "exponential", "agsk"}) {
        for (const int initialSize : {0, 20}) {
            for (int mode = 0; mode < 3; ++mode) {
                std::map<std::string, minion::ConfigValue> options = {
                    {"population_size", initialSize}, {"reduction_strategy", strategy},
                    {"x_tol", -1.0}, {"f_tol", -1.0}
                };
                // Default and explicit true reduce; explicit false stays constant.
                if (mode != 0) options["reducePop"] = (mode == 2);
                minion::LSHADE solver(sphere_batch, {{-5, 5}, {-5, 5}}, {},
                                     nullptr, nullptr, 720, 17, options);
                solver.optimize();
                const size_t initial = initialSize == 0 ? 36 : 20;
                const size_t size = solver.population.size();
                if (mode == 1 ? size != initial : (size < 4 || size >= initial)) return false;
                if (solver.F.size() != size || solver.CR.size() != size || solver.meanF.size() < 2) return false;
                if (solver.M_F == std::vector<double>(solver.memorySize, 0.5)) return false;
            }
        }
    }
    return true;
}

bool check_hessian_uncertainty() {
    minion::MinionFunction objective = [](const auto& points, void*) {
        std::vector<double> values;
        for (const auto& p : points) {
            values.push_back(0.5 * (4*p[0]*p[0] + 2*p[0]*p[1] + 3*p[1]*p[1]));
        }
        return values;
    };
    minion::uncertainty::Uncertainty analysis(objective, {0, 0}, {{-3, 3}, {-3, 3}});
    auto result = analysis.hessian(1);
    if (!result.covariance_valid || result.std.size() != 2) return false;
    Eigen::Matrix2d hessian, covariance;
    hessian << 4, 1, 1, 3;
    covariance << 3.0/11, -1.0/11, -1.0/11, 4.0/11;
    return (result.hessian-hessian).norm() < 1e-6 &&
           (result.covariance-covariance).norm() < 1e-6 &&
           std::fabs(result.std[0]-std::sqrt(3.0/11)) < 1e-6 &&
           std::fabs(result.std[1]-std::sqrt(4.0/11)) < 1e-6;
}

bool check_bootstrap_uncertainty() {
    minion::uncertainty::Uncertainty analysis(sphere_batch, {0}, {{-3, 3}});
    minion::uncertainty::BootstrapOptions options;
    options.n_samples = 3;
    options.optimizer = "NelderMead";
    options.maxevals = 2000;
    options.seed = 17;
    options.require_convergence = true;
    options.optimizer_options = {{"x_tol", 1e-9}, {"f_tol", -1.0}};
    auto result = analysis.bootstrap([](size_t index, uint64_t) -> minion::MinionFunction {
        return [index](const auto& points, void*) {
            std::vector<double> values;
            for (const auto& point : points) {
                const double residual = point[0]-static_cast<double>(index);
                values.push_back(residual*residual);
            }
            return values;
        };
    }, options);
    // Fits at 0, 1, 2 have sample mean 1 and unbiased sample variance 1.
    if (!result.covariance_valid || result.n_used != 3) return false;
    return std::fabs(result.mean[0]-1) < 1e-6 &&
           std::fabs(result.covariance(0,0)-1) < 1e-6 &&
           std::fabs(result.std[0]-1) < 1e-6 &&
           std::fabs(result.ci_lower[0]-0.05) < 1e-6 &&
           std::fabs(result.ci_upper[0]-1.95) < 1e-6;
}

}  // namespace

int main() {
    const std::vector<CheckRunner::Check> checks = {
        {"C++ MinionResult and status stream operators", check_status_streaming},
        {"core algorithms solve Sphere within budget", check_sphere_benchmark},
        {"core algorithms solve Rosenbrock within budget", check_rosenbrock_benchmark},
        {"core algorithms run finite CEC2017 benchmark cases within budget", check_cec2017_benchmarks},
        {"C++ callback true stops optimization", check_callback_stop},
        {"LSHADE population reduction switch and adaptation", check_lshade_population_reduction},
        {"ARRDE normalized spatial restart and refine thresholds", check_arrde_spatial_phase_tolerances},
        {"Hessian uncertainty matches analytic correlated covariance", check_hessian_uncertainty},
        {"Bootstrap uncertainty matches empirical covariance and percentiles", check_bootstrap_uncertainty},
    };

    return CheckRunner().run(checks);
}
