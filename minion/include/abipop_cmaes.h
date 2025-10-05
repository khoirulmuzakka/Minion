#ifndef ABIPOP_CMAES_H
#define ABIPOP_CMAES_H

#include "minimizer_base.h"

#include <Eigen/Dense>
#include <limits>
#include <string>

namespace minion {

/**
 * @class ABIPOP_CMAES
 * @brief Adaptive BIPOP Covariance Matrix Adaptation Evolution Strategy implementation.
 */
class ABIPOP_CMAES : public MinimizerBase {
public:
    ABIPOP_CMAES(
        MinionFunction func,
        const std::vector<std::pair<double, double>>& bounds,
        const std::vector<std::vector<double>>& x0 = {},
        void* data = nullptr,
        std::function<void(MinionResult*)> callback = nullptr,
        double tol = 0.0001,
        size_t maxevals = 100000,
        int seed = -1,
        std::map<std::string, ConfigValue> options = std::map<std::string, ConfigValue>());

    void initialize() override;
    MinionResult optimize() override;

private:
    struct Parameter {
        size_t n_offsprings = 0;
        size_t n_offsprings_reserve = 0;
        size_t n_parents = 0;
        size_t n_parents_reserve = 0;
        size_t n_params = 0;
        size_t i_iteration = 0;
        size_t i_func_eval = 0;
        double n_mu_eff = 0.0;

        Eigen::MatrixXd x_offsprings;
        Eigen::MatrixXd x_parents_ranked;
        Eigen::MatrixXd z_offsprings;
        Eigen::MatrixXd y_offsprings;
        Eigen::MatrixXd y_offsprings_ranked;
        Eigen::VectorXd f_offsprings;
        Eigen::VectorXd w;
        Eigen::VectorXd w_var;
        Eigen::VectorXd y_mean;
        Eigen::VectorXd x_mean;
        Eigen::VectorXd x_mean_old;
        Eigen::VectorXd p_c;
        Eigen::VectorXd p_s;
        Eigen::VectorXd eigvals_C;
        Eigen::MatrixXd C;
        Eigen::MatrixXd C_invsqrt;
        Eigen::MatrixXd B;
        Eigen::MatrixXd D;
        std::vector<size_t> keys_offsprings;
        double c_c = 0.0;
        double c_s = 0.0;
        double c_1 = 0.0;
        double c_mu = 0.0;
        double d_s = 0.0;
        double chi = 0.0;
        double p_c_fact = 0.0;
        double p_s_fact = 0.0;
        double sigma = 0.0;
        bool h_sig = false;

        void reserve(size_t n_offsprings_reserve_, size_t n_parents_reserve_, size_t n_params_);
        void reinit(size_t n_offsprings_, size_t n_parents_, size_t n_params_, const Eigen::VectorXd& x_mean_, double sigma_);
    };

    std::vector<double> applyBounds(const std::vector<double>& candidate) const;
    void sampleOffsprings();
    size_t evaluatePopulation();
    void rankAndSort();
    void updateBest();
    void assignNewMean();
    void updateEvolutionPaths();
    void updateWeights();
    void updateCovarianceMatrix();
    void updateStepsize();
    void updateEigenDecomposition();
    void checkStoppingCriteria();
    void recordHistory(double relRange);
    std::vector<double> eigenToStd(const Eigen::VectorXd& vec) const;

    Parameter era;

    size_t dimension = 0;
    bool useBounds = false;
    std::string boundStrategy = "reflect-random";

    size_t lambda0 = 0;
    size_t mu0 = 0;
    size_t maxRestarts = 0;
    size_t maxIterations = 0;

    double sigma0 = 0.0;

    Eigen::VectorXd initialMean;

    std::vector<double> best;
    double best_fitness = std::numeric_limits<double>::infinity();
    size_t Nevals = 0;
    size_t globalGeneration = 0;

    std::vector<double> diversity;
    std::vector<double> currentFitness;

    bool support_tol = true;

    bool should_stop_run = false;
    bool should_stop_optimization = false;
};

}

#endif
