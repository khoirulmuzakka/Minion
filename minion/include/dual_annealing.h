#ifndef DUAL_ANNEALING_H
#define DUAL_ANNEALING_H

#include "minimizer_base.h"
#include "default_options.h"
#include <cmath>  
#include <algorithm>
#include "nelder_mead.h"
#include "l_bfgs_b.h"

namespace minion {

/**
 * @class Dual Annealing
 * @brief A class for performing dual annealing algorithm. 
 * 
 * Reference : Tsallis C, Stariolo DA. Generalized Simulated Annealing. Physica A, 233, 395-406 (1996).
 * This class inherits from the MinimizerBase class.
 */
class Dual_Annealing : public MinimizerBase {
public:
    size_t Nevals = 0;
    double acceptance_par;
    double visit_par;
    double initial_temp;
    double restart_temp_ratio;
    double local_search_start;
    std::string local_min_algo;
    bool useLocalSearch = true;
    size_t max_no_improve;
    double func_noise_ratio = 1e-10;
    int der_N_points =3;

private : 
    size_t N_no_improve=0;
    std::vector<double> best_cand, current_cand; 
    double best_E=std::numeric_limits<double>::max(), current_E; 
    double temp_step;

    double factor2, factor3, factor4p, factor5, d1, factor6;
    double tail_limit = 1e+8;
    double pi = 3.14159265358979323846;

    void init(bool useX0=true);
    std::vector<double> visit_fn(double temperature, int dim);
    std::vector<double> generate_candidate(std::vector<double> cand, int j, double temp);
    void accept_reject (const std::vector<double>& cand, const double& Energy);
    void step (int iter, double temp);

public:
    /**
     * @brief Constructor for DA.
     * @param func The objective function to minimize.
     * @param bounds The bounds for the variables.
     * @param x0 The initial guesses for the solution.
     * @param data Additional data for the objective function.
     * @param callback Callback function for intermediate results.
     * @param tol The tolerance for stopping criteria.
     * @param maxevals The maximum number of evaluations.
     * @param seed The seed for random number generation.
     * @param options Option map that specifies further configurational settings for the algorithm.
     */
    Dual_Annealing(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
    ) :
        MinimizerBase(func, bounds, x0, data, callback, tol, maxevals, seed, options) {};

    /**
     * @brief Optimizes the objective function.
     * @return The result of the optimization.
     */
    MinionResult optimize() override;

    /**
     * @brief Initialize the algorithm given the input settings.
     */
    void initialize() override;
};

}

#endif
