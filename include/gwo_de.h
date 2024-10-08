#ifndef GWO_DE_H
#define GWO_DE_H

#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include "utility.h"
#include "minimizer_base.h"

namespace minion {
/**
 * @class GWO_DE
 * @brief Combined Grey Wolf Optimizer with Differential Evolution algorithm.
 */
class GWO_DE : public MinimizerBase {
public:

    /**
     * @brief Constructor for the GWO_DE class.
     * @param func The objective function to be minimized.
     * @param bounds The bounds for each dimension.
     * @param x0 Initial guess for the solution (optional).
     * @param population_size The size of the population.
     * @param maxevals The maximum number of evaluations.
     * @param F Differential weight.
     * @param CR Crossover probability.
     * @param elimination_prob Probability of elimination for diversity.
     * @param relTol Relative tolerance for convergence.
     * @param boundStrategy Strategy for handling bounds.
     * @param seed Random seed for reproducibility.
     * @param data Additional data passed to the objective function (optional).
     * @param callback Callback function for intermediate results (optional).
     */
    GWO_DE(MinionFunction func,
           const std::vector<std::pair<double, double>>& bounds,
           const std::vector<double>& x0 = {},
           size_t population_size = 20,
           int maxevals = 1000,
           double F = 0.5,
           double CR = 0.7,
           double elimination_prob = 0.1,
           double relTol = 0.0001,
           std::string boundStrategy = "reflect-random",
           int seed = -1,
           void* data = nullptr,
           std::function<void(MinionResult*)> callback = nullptr);

    /**
     * @brief Performs the optimization process.
     * @return MinionResult containing the best found solution and associated information.
     */
    virtual MinionResult optimize() override;

public:
    double CR, F, elimination_prob=0.1;
    size_t dimension;
    double alpha_score;
    double beta_score;
    double delta_score;
    size_t eval_count;
    std::vector<double> alpha_pos;
    std::vector<double> beta_pos;
    std::vector<double> delta_pos;
    std::vector<std::vector<double>> population;
    std::vector<double> fitness;

    std::mt19937 rng;

    /**
     * @brief Initializes the population randomly within the given bounds.
     */
    void initialize_population();

    /**
     * @brief Evaluates the fitness of the entire population.
     */
    void evaluate_population();

    /**
     * @brief Updates the alpha, beta, and delta wolves based on the current population fitness.
     */
    void update_leaders();

    /**
     * @brief Updates the position of a given solution based on the GWO strategy.
     * @param X Current position of the solution.
     * @param A Attraction coefficient vector.
     * @param C Distance control coefficient vector.
     * @return New position of the solution.
     */
    std::vector<double> update_position(const std::vector<double>& X, const std::vector<double>& A, const std::vector<double>& C);

    /**
     * @brief Performs the Differential Evolution process to generate new solutions.
     * @return New population generated by Differential Evolution.
     */
    std::vector<std::vector<double>> differential_evolution();

    /**
     * @brief Performs the elimination process to introduce diversity into the population.
     */
    void eliminate();
};

}

#endif // GWO_DE_H
