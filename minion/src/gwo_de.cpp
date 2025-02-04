#include "gwo_de.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include "default_options.h"

namespace minion {

void GWO_DE::initialize  (){
    auto defaultKey = DefaultSettings().getDefaultSettings("GWO_DE");
    for (auto el : optionMap) defaultKey[el.first] = el.second;
    Options options(defaultKey);

    boundStrategy = options.get<std::string> ("bound_strategy", "reflect-random");
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    size_t population_size = options.get<int> ("population_size", 0); 
    if (population_size==0) population_size= 2*bounds.size();

    dimension = bounds.size(),
    alpha_score = std::numeric_limits<double>::infinity();
    beta_score = std::numeric_limits<double>::infinity();
    delta_score = std::numeric_limits<double>::infinity();
    eval_count = 0;
    alpha_pos = std::vector<double> (bounds.size(), 0.0);
    beta_pos= std::vector<double> (bounds.size(), 0.0);
    delta_pos = std::vector<double> (bounds.size(), 0.0);
    population = std::vector<std::vector<double>>(population_size, std::vector<double>(bounds.size(), 0.0));
    fitness = std::vector<double> (population_size, 0.0);
    CR = options.get<double>("crossover_rate", 0.7); 
    F= options.get<double>("mutation_rate", 0.5); 
    elimination_prob = options.get<double>("elimination_prob", 0.7); 
    // Initialize population
    initialize_population();
    // Evaluate initial population
    evaluate_population();
    // Update leaders
    update_leaders();
    hasInitialized=true;
}

MinionResult GWO_DE::optimize() {
    try {
        history.clear();
        if (!hasInitialized) initialize();
        int iter =0;
        while (eval_count < maxevals) {
            double a = 2.0 - eval_count * (2.0 / maxevals);
            std::vector<std::vector<double>> A(population.size(), std::vector<double>(dimension));
            std::vector<std::vector<double>> C(population.size(), std::vector<double>(dimension));

            // Update positions based on GWO
            for (size_t i = 0; i < population.size(); ++i) {
                for (size_t j = 0; j < dimension; ++j) {
                    A[i][j] = 2 * a * rand_gen(0.0, 1.0) - a;
                    C[i][j] = 2 * rand_gen(0.0, 1.0);
                }
                population[i] = update_position(population[i], A[i], C[i]);
            }

            // Evaluate updated population
            evaluate_population();

            // Differential evolution step
            auto de_population = differential_evolution();
            enforce_bounds(de_population, bounds, boundStrategy);
            auto de_fitness = func(de_population, data);
            eval_count += de_population.size();

            // Combine GWO and DE populations
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

            // Sort combined population by fitness
            auto indices = argsort(combined_fitness);
            for (size_t i = 0; i < population.size(); ++i) {
                population[i] = combined_population[indices[i]];
                fitness[i] = combined_fitness[indices[i]];
            }

            // Update leaders
            update_leaders();

            // Eliminate some population members
            //eliminate();

            // Evaluate population after elimination
            //evaluate_population();
            update_leaders();

            minionResult = MinionResult(population[0], fitness[0], iter, eval_count,  false, "");
            history.push_back(minionResult);
            if (callback != nullptr) callback(&minionResult);
            iter++;
        }

        MinionResult result;
        result.x = alpha_pos;
        result.fun = alpha_score;
        result.nit = eval_count;
        result.nfev = eval_count;
        result.success = true;
        result.message = "Optimization terminated successfully.";
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

void GWO_DE::initialize_population() {
    population = latin_hypercube_sampling(bounds, population.size());
}

void GWO_DE::evaluate_population() {
    enforce_bounds(population, bounds, boundStrategy);
    fitness = func(population, data);
    eval_count += population.size();
}

void GWO_DE::update_leaders() {
    auto sorted_indices = argsort(fitness);
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
        D_alpha[i] = fabs(C[i] * alpha_pos[i] - X[i]);
        D_beta[i] = fabs(C[i] * beta_pos[i] - X[i]);
        D_delta[i] = fabs(C[i] * delta_pos[i] - X[i]);
    }

    std::vector<double> X1(dimension);
    std::vector<double> X2(dimension);
    std::vector<double> X3(dimension);

    for (size_t i = 0; i < dimension; ++i) {
        X1[i] = alpha_pos[i] - A[i] * D_alpha[i];
        X2[i] = beta_pos[i] - A[i] * D_beta[i];
        X3[i] = delta_pos[i] - A[i] * D_delta[i];
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
        size_t randIndex = rand_int(dimension);
        for (size_t j = 0; j < dimension; ++j) {
            if (rand_gen(0.0, 1.0) < CR || j == randIndex) {
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

}