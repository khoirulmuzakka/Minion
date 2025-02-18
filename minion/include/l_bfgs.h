#ifndef L_BFGS_H
#define L_BFGS_H

#include "minimizer_base.h"
#include "default_options.h"
#include <cmath>  
#include <algorithm>
#include "nelder_mead.h"
#include "LBFGS.h"
#include "exception.h"
#include "types.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace minion {



/**
 * @class L_BFGS
 * @brief A class for the L-BFGS optimization algorithm.
 * 
 * This class implements the Limited-memory BFGS unconstrained (L-BFGS) algorithm.
 * It inherits from the MinimizerBase class and provides methods for constrained optimization.
 */
class L_BFGS : public MinimizerBase {
public:
    size_t Nevals = 0;
    std::vector<double> best; 
    double f_best = std::numeric_limits<double>::max();
    int N_points= 1;
    double func_noise_ratio =1e-10;

private : 
    LBFGSpp::LBFGSSolver<double>* solver=nullptr;
    double last_f=1.0;
    double fin_diff_rel_step= sqrt(std::numeric_limits<double>::epsilon());

     /**
     * @brief Evaluates the function and its gradient.
     * @param x Current point.
     * @param grad Gradient vector (output parameter).
     * @return Function value at x.
     */
    double fun_and_grad(const VectorXd& x, VectorXd& grad);

public:
    /**
     * @brief Constructor for L-BFGS.
     * @param func The objective function to minimize.
     * @param bounds The bounds for the variables.
     * @param x0 The initial solution.
     * @param data Additional data for the objective function.
     * @param callback Callback function for intermediate results.
     * @param tol The tolerance for stopping criteria.
     * @param maxevals The maximum number of evaluations.
     * @param seed The seed for random number generation.
     * @param options Option map that specifies further configurational settings for the algorithm.
     */
    L_BFGS(
        MinionFunction func,
        const std::vector<double>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>()
    ) :
        MinimizerBase(func, x0, data, callback, tol, maxevals, seed, options) { };
    
    /**
     * @brief Destructor
     * 
     */
    ~L_BFGS(){  
        if (solver!=nullptr) delete solver;
    }

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
