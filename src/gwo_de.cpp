#include "gwo_de.h"

GWO_DE::GWO_DE(MinionFunction func,
               const std::vector<std::pair<double, double>>& bounds,
               const std::vector<double>& x0,
               size_t population_size,
               int maxevals,
               double F,
               double CR,
               double elimination_prob,
               double relTol,
               std::string boundStrategy,
               int seed,
               void* data,
               std::function<void(MinionResult*)> callback)
    : MinimizerBase(func, bounds, x0, data, callback, relTol, maxevals, boundStrategy, seed),
      dimension(bounds.size()),
      alpha_score(std::numeric_limits<double>::infinity()),
      beta_score(std::numeric_limits<double>::infinity()),
      delta_score(std::numeric_limits<double>::infinity()),
      eval_count(0),
      alpha_pos(bounds.size(), 0.0),
      beta_pos(bounds.size(), 0.0),
      delta_pos(bounds.size(), 0.0),
      population(population_size, std::vector<double>(bounds.size(), 0.0)),
      fitness(population_size, 0.0),
      CR(CR), 
      F(F),
      elimination_prob(elimination_prob){
    if (!x0.empty()) {
        for (size_t i = 0; i < std::min(x0.size(), bounds.size()); ++i) {
            alpha_pos[i] = x0[i];
        }
    }
    initialize_population();
    evaluate_population();
    update_leaders();
}

MinionResult GWO_DE::optimize() {
    while (eval_count < maxevals) {
        double a = 2 - eval_count * (2.0 / maxevals);
        std::vector<std::vector<double>> A(population.size(), std::vector<double>(dimension));
        std::vector<std::vector<double>> C(population.size(), std::vector<double>(dimension));

        for (size_t i = 0; i < population.size(); ++i) {
            for (size_t j = 0; j < dimension; ++j) {
                A[i][j] = 2 * a * rand_gen(0.0, 1.0);
                C[i][j] = rand_gen(0.0, 1.0);
            }
        }

        for (size_t i = 0; i < population.size(); ++i) {
            population[i] = update_position(population[i], A[i], C[i]);
        }

        evaluate_population();
        auto de_population = differential_evolution();
        std::vector<double> de_fitness(de_population.size(), 0.0);

        de_fitness = func(de_population, data);
        eval_count = eval_count+de_population.size();

        std::vector<std::vector<double>> combined_population(population.size() + de_population.size(), std::vector<double>(dimension));
        std::vector<double> combined_fitness(population.size() + de_population.size());

        for (size_t i = 0; i < population.size(); ++i) {
            combined_population[i] = population[i];
            combined_fitness[i] = fitness[i];
        }

        for (size_t i = 0; i < de_population.size(); ++i) {
            combined_population[population.size() + i] = de_population[i];
            combined_fitness[population.size() + i] = de_fitness[i];
        }

        std::vector<size_t> indices = argsort(combined_fitness);
        std::vector<std::vector<double>> sorted_population(population.size(), std::vector<double>(dimension));
        std::vector<double> sorted_fitness(population.size(), 0.0);

        for (size_t i = 0; i < population.size(); ++i) {
            sorted_population[i] = combined_population[indices[i]];
            sorted_fitness[i] = combined_fitness[indices[i]];
        }

        for (size_t i = 0; i < population.size(); ++i) {
            population[i] = sorted_population[i];
            fitness[i] = sorted_fitness[i];
        }

        update_leaders();
        eliminate();
        evaluate_population();
        update_leaders();
    }

    MinionResult result;
    result.x = alpha_pos;
    result.fun = alpha_score;
    result.nit = eval_count;
    result.nfev = eval_count;
    result.success = true;
    result.message = "Optimization terminated successfully.";
    return result;
}

void GWO_DE::initialize_population() {
    for (size_t i = 0; i < population.size(); ++i) {
        for (size_t j = 0; j < dimension; ++j) {
            population[i][j] = rand_gen(bounds[j].first, bounds[j].second);
        }
    }
}

void GWO_DE::evaluate_population() {
    fitness = func(population, data);
    eval_count = eval_count+population.size();
}

void GWO_DE::update_leaders() {
    std::vector<size_t> sorted_indices = argsort(fitness);
    alpha_pos = population[sorted_indices[0]];
    alpha_score = fitness[sorted_indices[0]];
    beta_pos = population[sorted_indices[1]];
    beta_score = fitness[sorted_indices[1]];
    delta_pos = population[sorted_indices[2]];
    delta_score = fitness[sorted_indices[2]];
}

std::vector<double> GWO_DE::update_position(const std::vector<double>& X, const std::vector<double>& A, const std::vector<double>& C) {
    std::vector<double> D_alpha(dimension);
    std::vector<double> D_beta(dimension);
    std::vector<double> D_delta(dimension);

    for (size_t i = 0; i < dimension; ++i) {
        D_alpha[i] = C[0] * alpha_pos[i] - X[i];
        D_beta[i] = C[1] * beta_pos[i] - X[i];
        D_delta[i] = C[2] * delta_pos[i] - X[i];
    }

    std::vector<double> X1(dimension);
    std::vector<double> X2(dimension);
    std::vector<double> X3(dimension);

    for (size_t i = 0; i < dimension; ++i) {
        X1[i] = alpha_pos[i] - A[0] * D_alpha[i];
        X2[i] = beta_pos[i] - A[1] * D_beta[i];
        X3[i] = delta_pos[i] - A[2] * D_delta[i];
    }

    std::vector<double> new_X(dimension);
    for (size_t i = 0; i < dimension; ++i) {
        new_X[i] = (X1[i] + X2[i] + X3[i]) / 3;
    }

    return new_X;
}

std::vector<std::vector<double>> GWO_DE::differential_evolution() {
    std::vector<std::vector<double>> new_population(population.size(), std::vector<double>(dimension));

    for (size_t i = 0; i < population.size(); ++i) {
        std::vector<size_t> idxs(population.size());
        std::iota(idxs.begin(), idxs.end(), 0);
       std::shuffle(idxs.begin(), idxs.end(), get_rng());
        size_t r1 = idxs[0];
        size_t r2 = idxs[1];
        size_t r3 = idxs[2];

        for (size_t j = 0; j < dimension; ++j) {
            if (rand_gen(0.0, 1.0) < CR || j == rand_int(dimension)) {
                new_population[i][j] = population[r3][j] + F * (population[r1][j] - population[r2][j]);
            } else {
                new_population[i][j] = population[i][j];
            }
        }

        enforce_bounds(new_population, bounds, boundStrategy);
    }

    return new_population;
}

void GWO_DE::eliminate() {
    for (size_t i = 0; i < population.size(); ++i) {
        if (rand_gen(0.0, 1.0) < elimination_prob) {
            for (size_t j = 0; j < dimension; ++j) {
                population[i][j] = rand_gen(bounds[j].first, bounds[j].second);
            }
        }
    }
}
