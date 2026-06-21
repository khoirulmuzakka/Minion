#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <minion.h>
#include <minion_cec.h>

namespace {

std::vector<double> sphere_batch(const std::vector<std::vector<double>>& X, void*) {
    std::vector<double> out(X.size(), 0.0);
    for (size_t i = 0; i < X.size(); ++i) {
        const auto& x = X[i];
        double f = 0.0;
        for (double value : x) {
            f += value * value;
        }
        out[i] = f;
    }
    return out;
}

std::vector<double> rosenbrock_batch(const std::vector<std::vector<double>>& X, void*) {
    std::vector<double> out(X.size(), 0.0);
    for (size_t i = 0; i < X.size(); ++i) {
        const auto& x = X[i];
        double f = 0.0;
        for (size_t j = 0; j + 1 < x.size(); ++j) {
            const double a = x[j + 1] - x[j] * x[j];
            const double b = 1.0 - x[j];
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

}  // namespace

int main() {
    const size_t dim = 5;
    std::vector<std::pair<double, double>> bounds(dim, {-5.0, 5.0});
    std::vector<double> x0(dim, 0.5);

    const std::vector<std::string> algorithms = {
        "DE", "LSHADE", "AGSK", "JADE", "j2020", "NLSHADE_RSP", "LSRTDE", "RDEX", "jSO",
        "IMODE", "ARRDE", "GWO_DE", "NelderMead", "ABC", "PSO", "SPSO2011", "DMSPSO",
        "LSHADE_cnEpSin", "CMAES", "RCMAES", "BIPOP_aCMAES", "DA", "L_BFGS_B", "L_BFGS"
    };

    const size_t maxevals = 4000;
    const size_t eval_slack = 100;

    int failed_checks = 0;
    int total_checks = 0;

    std::cout << "Sphere minimization using all Minion algorithms\n";
    std::cout << "dimension=" << dim << ", maxevals=" << maxevals << "\n\n";
    std::cout << std::left << std::setw(18) << "Algorithm" << std::right << std::setw(16) << "best_f" << std::setw(12)
              << "nfev" << '\n';
    std::cout << std::string(46, '-') << '\n';

    const std::map<std::string, double> sphere_upper = {
        {"DE", 1e-3}, {"LSHADE", 1e-3}, {"AGSK", 1e-3}, {"JADE", 1e-3}, {"j2020", 1e-3},
        {"NLSHADE_RSP", 1e-3}, {"LSRTDE", 1e-3}, {"RDEX", 1e-3}, {"jSO", 1e-3}, {"IMODE", 1e-3}, {"ARRDE", 1e-3},
        {"GWO_DE", 1e-3}, {"NelderMead", 1e-3}, {"ABC", 1e-3}, {"PSO", 1e-3}, {"SPSO2011", 1e-3},
        {"DMSPSO", 1e-3}, {"LSHADE_cnEpSin", 1e-3}, {"CMAES", 1e-3}, {"RCMAES", 1e-3},
        {"BIPOP_aCMAES", 1e-3}, {"DA", 1e-3}, {"L_BFGS_B", 1e-3}, {"L_BFGS", 1e-3},
    };

    for (const auto& algo : algorithms) {
        try {
            auto settings = minion::DefaultSettings().getDefaultSettings(algo);
            settings["convergence_tol"] = 1e-8;
            minion::Minimizer opt(sphere_batch, bounds, x0, nullptr, nullptr, algo, maxevals, 42, settings);
            minion::MinionResult res = opt.optimize();
            std::cout << std::left << std::setw(18) << algo << std::right << std::setw(16) << std::setprecision(8)
                      << std::scientific << res.fun << std::setw(12) << res.nfev << '\n';

            ++total_checks;
            const bool finite_ok = std::isfinite(res.fun);
            const bool eval_ok = res.nfev <= static_cast<int>(maxevals + eval_slack);
            const auto it = sphere_upper.find(algo);
            const double upper = (it == sphere_upper.end()) ? std::numeric_limits<double>::infinity() : it->second;
            const bool quality_ok = res.fun <= upper;
            if (!(finite_ok && eval_ok && quality_ok)) {
                ++failed_checks;
                std::cerr << "[FAIL][Sphere] " << algo
                          << " finite=" << finite_ok
                          << " nfev=" << res.nfev << " (limit " << (maxevals + eval_slack) << ")"
                          << " best_f=" << res.fun << " (limit " << upper << ")\n";
            }
        } catch (const std::exception& e) {
            std::cout << std::left << std::setw(18) << algo << "FAILED: " << e.what() << '\n';
            ++total_checks;
            ++failed_checks;
            std::cerr << "[FAIL][Sphere] " << algo << " exception: " << e.what() << "\n";
        }
    }

    std::cout << "\nRosenbrock minimization using all Minion algorithms\n";
    std::cout << "dimension=" << dim << ", maxevals=" << maxevals << "\n\n";
    std::cout << std::left << std::setw(18) << "Algorithm" << std::right << std::setw(16) << "best_f" << std::setw(12)
              << "nfev" << '\n';
    std::cout << std::string(46, '-') << '\n';

    const std::map<std::string, double> rosenbrock_upper = {
        {"DE", 10.0}, {"LSHADE", 10.0}, {"AGSK", 10.0}, {"JADE", 10.0}, {"j2020", 10.0},
        {"NLSHADE_RSP", 10.0}, {"LSRTDE", 10.0}, {"RDEX", 10.0}, {"jSO", 10.0}, {"IMODE", 10.0}, {"ARRDE", 10.0},
        {"GWO_DE", 10.0}, {"NelderMead", 10.0}, {"ABC", 10.0}, {"PSO", 10.0}, {"SPSO2011", 10.0},
        {"DMSPSO", 10.0}, {"LSHADE_cnEpSin", 10.0}, {"CMAES", 10.0}, {"RCMAES", 10.0},
        {"BIPOP_aCMAES", 10.0}, {"DA", 10.0}, {"L_BFGS_B", 10.0}, {"L_BFGS", 10.0},
    };

    for (const auto& algo : algorithms) {
        try {
            auto settings = minion::DefaultSettings().getDefaultSettings(algo);
            settings["convergence_tol"] = 1e-8;
            minion::Minimizer opt(rosenbrock_batch, bounds, x0, nullptr, nullptr, algo, maxevals, 42, settings);
            minion::MinionResult res = opt.optimize();
            std::cout << std::left << std::setw(18) << algo << std::right << std::setw(16) << std::setprecision(8)
                      << std::scientific << res.fun << std::setw(12) << res.nfev << '\n';

            ++total_checks;
            const bool finite_ok = std::isfinite(res.fun);
            const bool eval_ok = res.nfev <= static_cast<int>(maxevals + eval_slack);
            const auto it = rosenbrock_upper.find(algo);
            const double upper = (it == rosenbrock_upper.end()) ? std::numeric_limits<double>::infinity() : it->second;
            const bool quality_ok = res.fun <= upper;
            if (!(finite_ok && eval_ok && quality_ok)) {
                ++failed_checks;
                std::cerr << "[FAIL][Rosenbrock] " << algo
                          << " finite=" << finite_ok
                          << " nfev=" << res.nfev << " (limit " << (maxevals + eval_slack) << ")"
                          << " best_f=" << res.fun << " (limit " << upper << ")\n";
            }
        } catch (const std::exception& e) {
            std::cout << std::left << std::setw(18) << algo << "FAILED: " << e.what() << '\n';
            ++total_checks;
            ++failed_checks;
            std::cerr << "[FAIL][Rosenbrock] " << algo << " exception: " << e.what() << "\n";
        }
    }

    std::cout << "\nCEC2017 minimization (F1-F30, dimension=30)\n";
    const int cec_dimension = 10;
    const size_t cec_maxevals = 10000;
    const int cec_seed = 20250306;
    const int cec_function_start = 1;
    const int cec_function_end = 30;

    std::vector<std::pair<double, double>> cec_bounds(cec_dimension, {-100.0, 100.0});
    std::vector<double> cec_x0(cec_dimension, 0.0);
    const std::vector<std::string> cec_algorithms = algorithms;

    std::cout << "functions=F" << cec_function_start << "-F" << cec_function_end
              << ", maxevals=" << cec_maxevals
              << ", seed=" << cec_seed << "\n\n";
    for (int cec_function_number = cec_function_start; cec_function_number <= cec_function_end; ++cec_function_number) {
        minion::CEC2017Functions cec2017_fn(cec_function_number, cec_dimension);
        std::cout << "Function F" << cec_function_number << '\n';
        std::cout << std::left << std::setw(18) << "Algorithm" << std::right << std::setw(16) << "best_f" << std::setw(12)
                  << "nfev" << '\n';
        std::cout << std::string(46, '-') << '\n';

        for (const auto& algo : cec_algorithms) {
            try {
                auto settings = minion::DefaultSettings().getDefaultSettings(algo);
                settings["convergence_tol"] = 1e-8;
                minion::Minimizer cec_opt(cec2017_batch, cec_bounds, cec_x0, &cec2017_fn, nullptr, algo, cec_maxevals, cec_seed, settings);
                minion::MinionResult cec_res = cec_opt.optimize();
                std::cout << std::left << std::setw(18) << algo << std::right << std::setw(16) << std::setprecision(8)
                          << std::scientific << cec_res.fun << std::setw(12) << cec_res.nfev << '\n';

                ++total_checks;
                const bool finite_ok = std::isfinite(cec_res.fun);
                const bool eval_ok = cec_res.nfev <= static_cast<int>(cec_maxevals + eval_slack);
                if (!(finite_ok && eval_ok)) {
                    ++failed_checks;
                    std::cerr << "[FAIL][CEC2017 F" << cec_function_number << " D30] " << algo
                              << " finite=" << finite_ok
                              << " nfev=" << cec_res.nfev << " (limit " << (cec_maxevals + eval_slack) << ")"
                              << " best_f=" << cec_res.fun << '\n';
                }
            } catch (const std::exception& e) {
                std::cout << std::left << std::setw(18) << algo << "FAILED: " << e.what() << '\n';
                ++total_checks;
                ++failed_checks;
                std::cerr << "[FAIL][CEC2017 F" << cec_function_number << " D30] " << algo << " exception: " << e.what() << "\n";
            }
        }
        std::cout << '\n';
    }

    const int passed_checks = total_checks - failed_checks;
    std::cout << "\nTest summary: " << passed_checks << "/" << total_checks << " checks passed.\n";
    return (failed_checks == 0) ? 0 : 1;
}
