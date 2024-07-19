#include "powell.h"

Powell::Powell(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0,
               void* data, std::function<void(MinionResult*)> callback, double relTol, int maxevals, std::string boundStrategy, int seed)
    : MinimizerBase(func, bounds, x0, data, callback, relTol, maxevals, boundStrategy, seed) {}

MinionResult Powell::optimize() {
    std::vector<double> x = x0;
    size_t n = x.size();
    std::vector<std::vector<double>> directions(n, std::vector<double>(n, 0.0));

    // Initialize directions as identity matrix
    for (size_t i = 0; i < n; ++i) {
        directions[i][i] = 1.0;
    }

    double fval = func({x}, data)[0];
    size_t iter = 0;
    size_t nfev = 1;
    bool success = false;
    std::string message = "";

    while (nfev < maxevals) {
        std::vector<double> x_prev = x;
        double fval_prev = fval;
        std::vector<double> new_dir(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            std::vector<double> xi = x;  // Make a copy of x
            double fxi = line_minimization(x, directions[i], nfev);  // Perform line minimization

            // Update x using Powell's method formula
            for (size_t j = 0; j < n; ++j) {
                x[j] += directions[i][j] * (x[j] - xi[j]);
            }

            // Compute the new direction vector
            for (size_t j = 0; j < n; ++j) {
                new_dir[j] += x[j] - x_prev[j];
            }

            // Enforce bounds on x
            xtemp = {x};
            enforce_bounds(xtemp, bounds, boundStrategy);
            x = xtemp[0];

            // Update fval
            fval = func({x}, data)[0];
            ++nfev;
        }

        // Check convergence criteria
        if (std::abs(fval_prev - fval) < relTol * (std::abs(fval_prev) + std::abs(fval)) + relTol) {
            success = true;
            message = "Optimization converged.";
            break;
        }

        // Update directions matrix with the new direction
        directions.push_back(new_dir);
        directions.erase(directions.begin());

        ++iter;
    }

    if (!success) {
        message = "Maximum number of evaluations reached.";
    }

    return MinionResult(x, fval, iter, nfev, success, message);
}

double Powell::line_minimization(std::vector<double>& x, const std::vector<double>& direction, size_t& nfev) {
    std::vector<double> xi = x;  // Make a copy of x
    double alpha = 1.0;

    // Update x along the direction vector
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += alpha * direction[i];
    }

    // Enforce bounds on x
    xtemp = {x};
    enforce_bounds(xtemp, bounds, boundStrategy);
    x = xtemp[0];

    // Evaluate the objective function
    ++nfev;
    return func({x}, data)[0];
}
