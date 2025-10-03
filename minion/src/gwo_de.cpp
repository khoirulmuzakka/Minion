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
    std::vector<std::string> all_boundStrategy = {"random", "reflect", "reflect-random", "clip", "periodic", "none"};
    if (std::find(all_boundStrategy.begin(), all_boundStrategy.end(), boundStrategy)== all_boundStrategy.end()) {
        std::cerr << "Bound stategy '"+ boundStrategy+"' is not recognized. 'Reflect-random' will be used.\n";
        boundStrategy = "reflect-random";
    }

    size_t population_size = options.get<int> ("population_size", 0); 
    if (population_size == 0) {
        population_size = std::max<size_t>(5 * bounds.size(), 20);
    }
    if (population_size < 4) {
        population_size = 4;
    }

    dimension = bounds.size();
    alpha_score = std::numeric_limits<double>::infinity();
    beta_score = std::numeric_limits<double>::infinity();
    delta_score = std::numeric_limits<double>::infinity();
    eval_count = 0;
    alpha_pos = std::vector<double> (dimension, 0.0);
    beta_pos  = std::vector<double> (dimension, 0.0);
    delta_pos = std::vector<double> (dimension, 0.0);
    population = std::vector<std::vector<double>>(population_size, std::vector<double>(dimension, 0.0));
    fitness = std::vector<double> (population_size, std::numeric_limits<double>::infinity());
    CR = options.get<double>("crossover_rate", 0.7); 
    F= options.get<double>("mutation_rate", 0.5); 
    elimination_prob = options.get<double>("elimination_prob", 0.7); 
    initialize_population();
    evaluate_population();
    update_leaders();
    hasInitialized=true;
}

MinionResult GWO_DE::optimize() {
    try {
        history.clear();
        if (!hasInitialized) initialize();
        size_t iter = 0;

        while (eval_count < maxevals) {
            double progress = maxevals > 0 ? static_cast<double>(eval_count) / static_cast<double>(maxevals) : 0.0;
            progress = std::clamp(progress, 0.0, 1.0);
            double a = 2.0 * (1.0 - progress);

            if (eval_count + population.size() > maxevals) {
                break;
            }
            auto gwo_candidates = generate_gwo_candidates(a);
            enforce_bounds(gwo_candidates, bounds, boundStrategy);
            auto gwo_fitness = func(gwo_candidates, data);
            eval_count += gwo_candidates.size();

            if (eval_count >= maxevals) {
                break;
            }
            auto de_candidates = differential_evolution();
            enforce_bounds(de_candidates, bounds, boundStrategy);
            auto de_fitness = func(de_candidates, data);
            eval_count += de_candidates.size();

            auto totalCandidates = population.size() + gwo_candidates.size() + de_candidates.size();
            std::vector<std::vector<double>> combined;
            std::vector<double> combinedFitness;
            combined.reserve(totalCandidates);
            combinedFitness.reserve(totalCandidates);

            combined.insert(combined.end(), population.begin(), population.end());
            combinedFitness.insert(combinedFitness.end(), fitness.begin(), fitness.end());

            combined.insert(combined.end(), gwo_candidates.begin(), gwo_candidates.end());
            combinedFitness.insert(combinedFitness.end(), gwo_fitness.begin(), gwo_fitness.end());

            combined.insert(combined.end(), de_candidates.begin(), de_candidates.end());
            combinedFitness.insert(combinedFitness.end(), de_fitness.begin(), de_fitness.end());

            auto indices = argsort(combinedFitness, true);
            std::vector<std::vector<double>> nextPopulation(population.size());
            std::vector<double> nextFitness(population.size());
            for (size_t i = 0; i < population.size(); ++i) {
                nextPopulation[i] = combined[indices[i]];
                nextFitness[i] = combinedFitness[indices[i]];
            }

            population.swap(nextPopulation);
            fitness.swap(nextFitness);

            update_leaders();

            minionResult = MinionResult(alpha_pos, alpha_score, iter + 1, eval_count, false, "");
            history.push_back(minionResult);
            if (callback != nullptr) {
                callback(&minionResult);
            }

            ++iter;
        }

        MinionResult result(alpha_pos, alpha_score, iter, eval_count, true, "Optimization terminated successfully.");
        history.push_back(result);
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

void GWO_DE::initialize_population() {
    population = latin_hypercube_sampling(bounds, population.size());
    if (!x0.empty()) {
        for (size_t i = 0; i < x0.size() && i < population.size(); ++i) {
            if (x0[i].size() == bounds.size()) {
                population[i] = x0[i];
            }
        }
    }
}

void GWO_DE::evaluate_population() {
    enforce_bounds(population, bounds, boundStrategy);
    fitness = func(population, data);
    eval_count += population.size();
}

void GWO_DE::update_leaders() {
    auto sorted_indices = argsort(fitness, true);
    if (sorted_indices.empty()) {
        return;
    }

    auto set_leader = [&](size_t targetIndex, std::vector<double>& pos, double& score) {
        if (targetIndex < sorted_indices.size()) {
            size_t idx = sorted_indices[targetIndex];
            pos = population[idx];
            score = fitness[idx];
        } else {
            pos = population[sorted_indices.front()];
            score = fitness[sorted_indices.front()];
        }
    };

    set_leader(0, alpha_pos, alpha_score);
    set_leader(1, beta_pos, beta_score);
    set_leader(2, delta_pos, delta_score);
}

std::vector<double> GWO_DE::update_position(const std::vector<double>& X,
                                            const std::vector<double>& A1, const std::vector<double>& C1,
                                            const std::vector<double>& A2, const std::vector<double>& C2,
                                            const std::vector<double>& A3, const std::vector<double>& C3) const {
    std::vector<double> newX(dimension, 0.0);
    for (size_t d = 0; d < dimension; ++d) {
        double D_alpha = std::fabs(C1[d] * alpha_pos[d] - X[d]);
        double D_beta  = std::fabs(C2[d] * beta_pos[d]  - X[d]);
        double D_delta = std::fabs(C3[d] * delta_pos[d] - X[d]);

        double X1 = alpha_pos[d] - A1[d] * D_alpha;
        double X2 = beta_pos[d]  - A2[d] * D_beta;
        double X3 = delta_pos[d] - A3[d] * D_delta;

        newX[d] = (X1 + X2 + X3) / 3.0;
    }
    return newX;
}

std::vector<std::vector<double>> GWO_DE::differential_evolution() const {
    std::vector<std::vector<double>> trials(population.size(), std::vector<double>(dimension));
    if (population.size() < 4) {
        return population;
    }

    for (size_t i = 0; i < population.size(); ++i) {
        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        indices.erase(indices.begin() + i);
        std::shuffle(indices.begin(), indices.end(), get_rng());

        size_t r1 = indices[0];
        size_t r2 = indices[1];
        size_t r3 = indices[2];
        size_t jrand = rand_int(dimension);
        for (size_t d = 0; d < dimension; ++d) {
            if (rand_gen(0.0, 1.0) < CR || d == jrand) {
                trials[i][d] = population[r1][d] + F * (population[r2][d] - population[r3][d]);
            } else {
                trials[i][d] = population[i][d];
            }
        }
    }

    return trials;
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

std::vector<std::vector<double>> GWO_DE::generate_gwo_candidates(double a) const {
    std::vector<std::vector<double>> candidates(population.size(), std::vector<double>(dimension, 0.0));
    for (size_t i = 0; i < population.size(); ++i) {
        std::vector<double> A1(dimension), A2(dimension), A3(dimension);
        std::vector<double> C1(dimension), C2(dimension), C3(dimension);
        for (size_t d = 0; d < dimension; ++d) {
            double r1 = rand_gen(0.0, 1.0);
            double r2 = rand_gen(0.0, 1.0);
            A1[d] = 2.0 * a * r1 - a;
            C1[d] = 2.0 * r2;

            r1 = rand_gen(0.0, 1.0);
            r2 = rand_gen(0.0, 1.0);
            A2[d] = 2.0 * a * r1 - a;
            C2[d] = 2.0 * r2;

            r1 = rand_gen(0.0, 1.0);
            r2 = rand_gen(0.0, 1.0);
            A3[d] = 2.0 * a * r1 - a;
            C3[d] = 2.0 * r2;
        }
        candidates[i] = update_position(population[i], A1, C1, A2, C2, A3, C3);
    }
    return candidates;
}

}
