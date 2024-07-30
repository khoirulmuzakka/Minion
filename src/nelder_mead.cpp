#include "nelder_mead.h"
#include <algorithm>

NelderMead::NelderMead(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0,
                                       void* data, std::function<void(MinionResult*)> callback, double tol, int maxevals, std::string boundStrategy, int seed)
    : MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, boundStrategy, seed) {}


MinionResult NelderMead::optimize() {
    try {
        size_t n = x0.size();
        std::vector<std::vector<double>> simplex(n + 1, std::vector<double>(n));

        // Initialize simplex around the initial point x0
        for (size_t i = 0; i < n; ++i) {
            simplex[i] = x0;
            if (x0[i] != 0) {
                simplex[i][i] += 0.05 * (bounds[i].second - bounds[i].first);
            } else {
                simplex[i][i] += 0.05;
            }
        }

        // Evaluate function values at the initial simplex points
        std::vector<double> fvals(n + 1);
        enforce_bounds(simplex, bounds, boundStrategy);
        fvals = func(simplex, data);

        size_t iter = 0;
        size_t nfev = n + 1;
        bool success = false;
        std::string message = "";

        while (nfev < maxevals) {
            // Sort simplex and fvals based on fvals
            std::vector<size_t> indices = argsort(fvals);
            std::vector<std::vector<double>> simplex_sorted(n + 1);
            std::vector<double> fvals_sorted(n + 1);
            for (size_t i = 0; i <= n; ++i) {
                simplex_sorted[i] = simplex[indices[i]];
                fvals_sorted[i] = fvals[indices[i]];
            }
            simplex = simplex_sorted;
            fvals = fvals_sorted;

            // Check convergence
            double frange = (*std::max_element(fvals.begin(), fvals.end()) - *std::min_element(fvals.begin(), fvals.end())) / calcMean(fvals);
            if (frange < stoppingTol) {
                success = true;
                message = "Optimization converged.";
                break;
            }

            // Calculate centroid
            std::vector<double> centroid(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    centroid[j] += simplex[i][j];
                }
            }
            for (size_t j = 0; j < n; ++j) {
                centroid[j] /= n;
            }

            // Reflection
            std::vector<double> xr = centroid;
            for (size_t j = 0; j < n; ++j) {
                xr[j] += 1.0 * (centroid[j] - simplex[n][j]);
            }
            xtemp = {xr};
            enforce_bounds(xtemp, bounds, boundStrategy);
            double fr = func(xtemp, data)[0];
            ++nfev;

            if (fr < fvals[n - 1]) {
                // Expansion
                if (fr < fvals[0]) {
                    std::vector<double> xe = centroid;
                    for (size_t j = 0; j < n; ++j) {
                        xe[j] += 2.0 * (xr[j] - centroid[j]);
                    }
                    double fe = func({xe}, data)[0];
                    ++nfev;
                    if (fe < fr) {
                        simplex[n] = xe;
                        fvals[n] = fe;
                    } else {
                        simplex[n] = xr;
                        fvals[n] = fr;
                    }
                } else {
                    simplex[n] = xr;
                    fvals[n] = fr;
                }
            } else {
                // Contraction
                std::vector<double> xc = centroid;
                for (size_t j = 0; j < n; ++j) {
                    xc[j] += 0.5 * (simplex[n][j] - centroid[j]);
                }
                xtemp = {xc};
                enforce_bounds(xtemp, bounds, boundStrategy);
                double fc = func(xtemp, data)[0];
                ++nfev;

                if (fc < fvals[n]) {
                    simplex[n] = xc;
                    fvals[n] = fc;
                } else {
                    // Reduction
                    std::vector<double> fvalTemp ; 
                    xtemp = std::vector<std::vector<double>>(n, std::vector<double>(n));
                    for (size_t i = 1; i <= n; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                            xtemp[i-1][j] = simplex[i][j];
                        }
                    }
                    enforce_bounds(simplex, bounds, boundStrategy );
                    enforce_bounds(xtemp, bounds, boundStrategy);
                    fvalTemp = func(xtemp, data);
                    for (size_t i=0; i<n; ++i){
                        fvals[i+1] = fvalTemp[i];
                    }
                    nfev += n;
                }
            }

            ++iter;
        }

        if (!success) {
            message = "Maximum number of evaluations reached.";
        }

        return MinionResult(simplex[0], fvals[0], iter, nfev, success, message);
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

