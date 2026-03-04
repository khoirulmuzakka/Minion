#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define MINION_ALGORITHMS_IMPLEMENTATION
#include <minion.h>

namespace {

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

}  // namespace

int main() {
    using namespace minion;

    MinionFunction func = rosenbrock_batch;

    const size_t dim = 5;
    std::vector<std::pair<double, double>> bounds(dim, {-5.0, 5.0});
    std::vector<double> x0(dim, 0.5);

    const std::vector<std::string> algorithms = {
        "DE", "LSHADE", "AGSK", "JADE", "j2020", "NLSHADE_RSP", "LSRTDE", "jSO",
        "IMODE", "ARRDE", "GWO_DE", "NelderMead", "ABC", "PSO", "SPSO2011", "DMSPSO",
        "LSHADE_cnEpSin", "CMAES", "RCMAES", "BIPOP_aCMAES", "DA", "L_BFGS_B", "L_BFGS"
    };

    const size_t maxevals = 4000;
    const double tol = 1e-8;

    std::cout << "Rosenbrock minimization using all Minion algorithms\n";
    std::cout << "dimension=" << dim << ", maxevals=" << maxevals << "\n\n";
    std::cout << std::left << std::setw(18) << "Algorithm" << std::right << std::setw(16) << "best_f" << std::setw(12)
              << "nfev" << '\n';
    std::cout << std::string(46, '-') << '\n';

    for (const auto& algo : algorithms) {
        try {
            Minimizer opt(func, bounds, x0, nullptr, nullptr, algo, tol, maxevals, 42);
            MinionResult res = opt.optimize();
            std::cout << std::left << std::setw(18) << algo << std::right << std::setw(16) << std::setprecision(8)
                      << std::scientific << res.fun << std::setw(12) << res.nfev << '\n';
        } catch (const std::exception& e) {
            std::cout << std::left << std::setw(18) << algo << "FAILED: " << e.what() << '\n';
        }
    }

    return 0;
}
