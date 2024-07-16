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

class GWO_DE : public MinimizerBase {
public:
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

    void initialize_population();
    void evaluate_population();
    void update_leaders();
    std::vector<double> update_position(const std::vector<double>& X, const std::vector<double>& A, const std::vector<double>& C);
    std::vector<std::vector<double>> differential_evolution();
    void eliminate();
};

#endif // GWO_DE_H
