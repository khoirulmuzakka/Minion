#include "nelder_mead.h"
#include "default_options.h"

#include <algorithm>
#include <cmath>

namespace minion {

void NelderMead::initialize() {
    if (x0.empty()) {
        throw std::runtime_error("Nelder-Mead requires at least one initial guess.");
    }

    xinit = findBestPoint(x0);

    auto defaults = DefaultSettings().getDefaultSettings("NelderMead");
    for (const auto& entry : optionMap) {
        defaults[entry.first] = entry.second;
    }
    Options options(defaults);

    boundStrategy = options.get<std::string>("bound_strategy", "reflect-random");
    std::vector<std::string> supported = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(supported.begin(), supported.end(), boundStrategy) == supported.end()) {
        std::cerr << "Bound stategy '" << boundStrategy << "' is not recognized. 'reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    simplex_scale = clamp(options.get<double>("locality_factor", 0.05), 1e-10, 1.0);

    hasInitialized = true;
}

std::vector<std::vector<double>> NelderMead::build_simplex(const std::vector<double>& center) const {
    size_t n = center.size();
    std::vector<std::vector<double>> simplex(n + 1, center);
    if (n == 0) {
        return simplex;
    }

    const double nonzdelt = simplex_scale;
    const double zdelt = simplex_scale * 0.005; // ~0.00025 when simplex_scale=0.05

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> vertex = center;
        double step = nonzdelt * std::max(1.0, std::fabs(center[i]));
        if (step == 0.0) {
            step = zdelt;
        }
        vertex[i] += step;
        simplex[i + 1] = std::move(vertex);
    }

    enforce_bounds(simplex, bounds, boundStrategy);
    return simplex;
}

MinionResult NelderMead::optimize() {
    if (!hasInitialized) {
        initialize();
    }

    try {
        history.clear();

        size_t n = xinit.size();
        if (n == 0) {
            throw std::runtime_error("Nelder-Mead requires non-zero dimension.");
        }

        alpha = 1.0;
        gamma = 1.0 + 2.0 / static_cast<double>(n);
        rho   = 0.75 - 1.0 / (2.0 * static_cast<double>(n));
        sigma = 1.0 - rho;

        double xtol = stoppingTol;
        double ftol = stoppingTol;

        std::vector<std::vector<double>> simplex = build_simplex(xinit);
        std::vector<double> fvals = func(simplex, data);
        size_t nfev = simplex.size();

        bestIndex = findArgMin(fvals);
        best = simplex[bestIndex];
        fbest = fvals[bestIndex];

        size_t iter = 0;
        bool success = false;
        std::string message;

        auto push_history = [&](bool done) {
            minionResult = MinionResult(best, fbest, iter, nfev, done, message);
            history.push_back(minionResult);
            if (callback != nullptr) {
                callback(&minionResult);
            }
        };

        push_history(false);

        while (nfev < maxevals) {
            std::vector<size_t> order = argsort(fvals, true);
            std::vector<std::vector<double>> sortedSimplex(simplex.size());
            std::vector<double> sortedFvals(fvals.size());
            for (size_t i = 0; i < simplex.size(); ++i) {
                sortedSimplex[i] = simplex[order[i]];
                sortedFvals[i] = fvals[order[i]];
            }
            simplex.swap(sortedSimplex);
            fvals.swap(sortedFvals);

            best = simplex[0];
            fbest = fvals[0];
            bestIndex = 0;

            double maxXdiff = 0.0;
            double maxFdiff = 0.0;
            for (size_t i = 1; i < simplex.size(); ++i) {
                for (size_t d = 0; d < n; ++d) {
                    maxXdiff = std::max(maxXdiff, std::fabs(simplex[i][d] - best[d]));
                }
                maxFdiff = std::max(maxFdiff, std::fabs(fvals[i] - fbest));
            }

            if (maxXdiff <= xtol && maxFdiff <= ftol) {
                success = true;
                message = "Optimization converged.";
                break;
            }

            std::vector<double> centroid(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < n; ++d) {
                    centroid[d] += simplex[i][d];
                }
            }
            for (size_t d = 0; d < n; ++d) {
                centroid[d] /= static_cast<double>(n);
            }

            auto evaluate_point = [&](std::vector<double> point) {
                enforce_bounds(point, bounds, boundStrategy);
                std::vector<std::vector<double>> args = {point};
                double value = func(args, data)[0];
                ++nfev;
                return std::make_pair(std::move(point), value);
            };

            std::vector<double> xr(n);
            for (size_t d = 0; d < n; ++d) {
                xr[d] = centroid[d] + alpha * (centroid[d] - simplex[n][d]);
            }
            auto [xReflection, fReflection] = evaluate_point(std::move(xr));

            if (fReflection < fvals[0]) {
                std::vector<double> xe(n);
                for (size_t d = 0; d < n; ++d) {
                    xe[d] = centroid[d] + gamma * (xReflection[d] - centroid[d]);
                }
                auto [xExpansion, fExpansion] = evaluate_point(std::move(xe));
                if (fExpansion < fReflection) {
                    simplex[n] = std::move(xExpansion);
                    fvals[n] = fExpansion;
                } else {
                    simplex[n] = std::move(xReflection);
                    fvals[n] = fReflection;
                }
            } else if (fReflection < fvals[n - 1]) {
                simplex[n] = std::move(xReflection);
                fvals[n] = fReflection;
            } else {
                bool outside = fReflection < fvals[n];
                std::vector<double> xc(n);
                if (outside) {
                    for (size_t d = 0; d < n; ++d) {
                        xc[d] = centroid[d] + rho * (xReflection[d] - centroid[d]);
                    }
                } else {
                    for (size_t d = 0; d < n; ++d) {
                        xc[d] = centroid[d] + rho * (simplex[n][d] - centroid[d]);
                    }
                }
                auto [xContraction, fContraction] = evaluate_point(std::move(xc));

                if ((outside && fContraction <= fReflection) || (!outside && fContraction < fvals[n])) {
                    simplex[n] = std::move(xContraction);
                    fvals[n] = fContraction;
                } else {
                    for (size_t i = 1; i <= n; ++i) {
                        for (size_t d = 0; d < n; ++d) {
                            simplex[i][d] = simplex[0][d] + sigma * (simplex[i][d] - simplex[0][d]);
                        }
                    }
                    enforce_bounds(simplex, bounds, boundStrategy);
                    std::vector<std::vector<double>> reduced(simplex.begin() + 1, simplex.end());
                    auto shrinkVals = func(reduced, data);
                    nfev += shrinkVals.size();
                    for (size_t i = 0; i < shrinkVals.size(); ++i) {
                        fvals[i + 1] = shrinkVals[i];
                    }
                }
            }

            ++iter;
            push_history(false);

            if (nfev >= maxevals) {
                break;
            }
        }

        if (!success) {
            message = "Maximum number of evaluations reached.";
        }

        push_history(true);
        return history.back();
    } catch (const std::exception& ex) {
        throw std::runtime_error(ex.what());
    }
}

} // namespace minion
