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
    };
    return values;
}

std::map<std::string, minion::ConfigValue> settings_for(const std::string& algo) {
    auto settings = minion::DefaultSettings().getDefaultSettings(algo);
    settings["convergence_tol"] = 1e-8;
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
           result.succeeded();
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

}  // namespace

int main() {
    const std::vector<CheckRunner::Check> checks = {
        {"C++ MinionResult and status stream operators", check_status_streaming},
        {"core algorithms solve Sphere within budget", check_sphere_benchmark},
        {"core algorithms solve Rosenbrock within budget", check_rosenbrock_benchmark},
        {"core algorithms run finite CEC2017 benchmark cases within budget", check_cec2017_benchmarks},
        {"C++ callback true stops optimization", check_callback_stop},
    };

    return CheckRunner().run(checks);
}
