#include "nelder_mead.h"
#include <algorithm>
#include "default_options.h"


namespace minion {

void NelderMead::initialize  (){
    auto defaultKey = DefaultSettings().getDefaultSettings("NelderMead");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }
    
    locality = options.get<double> ("locality_factor", 1.0);
    if (locality<0.0 || locality >1.0) locality = 0.5;
    hasInitialized = true;
};

MinionResult NelderMead::optimize() {
    try {
        if (!hasInitialized) initialize();
        size_t n = x0.size();
        std::vector<std::vector<double>> simplex(n + 1, std::vector<double>(n));

        std::vector<std::pair<double, double>> new_bounds = bounds; 
        if (!x0.empty()) {
            for (int i =0; i<bounds.size(); i++) {
                double dis_up = locality * fabs(bounds[i].second-x0[i]);
                double dis_down = locality * fabs(x0[i]-bounds[i].first);
                new_bounds[i] = { x0[i]-dis_down, x0[i]+dis_up };
            }
        }
        
        simplex= latin_hypercube_sampling(new_bounds, bounds.size()+1); 
        simplex[0] = x0;

        // Evaluate function values at the initial simplex points
        std::vector<double> fvals(n + 1);
        enforce_bounds(simplex, bounds, boundStrategy);
        fvals = func(simplex, data);
        bestIndex = findArgMin(fvals); 
        best = simplex[bestIndex];
        fbest = fvals[bestIndex];

        size_t iter = 0;
        size_t nfev = n + 1;
        bool success = false;
        std::string message = "";

        do {
            if (no_improve_counter>5000){
                simplex = latin_hypercube_sampling(bounds, bounds.size()+1);
                fvals = func(simplex, data); 
                simplex[0]= best;
                fvals[0] = fbest;
                nfev+=fvals.size();
            }   
            // Sort simplex and fvals based on fvals
            std::vector<size_t> indices = argsort(fvals, true);
            std::vector<std::vector<double>> simplex_sorted(n + 1);
            std::vector<double> fvals_sorted(n + 1);
            for (size_t i = 0; i <= n; ++i) {
                simplex_sorted[i] = simplex[indices[i]];
                fvals_sorted[i] = fvals[indices[i]];
            }
            simplex = simplex_sorted;
            fvals = fvals_sorted;

            if (fvals[0]<fbest) {
                fbest = fvals[0]; 
                best = simplex[0]; 
                no_improve_counter=0;
            } else {
                no_improve_counter++;
            };

            // Check convergence
            double frange =calcStdDev(fvals)/ calcMean(fvals);
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
            //std::cout << fvals[0] << "\n";
            ++iter;
        } while (nfev < maxevals);

        if (!success) {
            message = "Maximum number of evaluations reached.";
        }

        return MinionResult(simplex[0], fvals[0], iter, nfev, success, message);
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

}