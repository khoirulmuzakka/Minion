#include "de_base.h"
#include "utility.h"
#include <cmath> 


DE_Base::DE_Base(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, void* data, 
    const std::vector<double>& x0 , size_t population_size, size_t maxevals, std::string strategy , double relTol , size_t minPopSize,
    std::function<void(MinionResult*)> callback, std::string boundStrategy, int seed)
        : MinimizerBase(func, bounds, x0, data, callback, relTol, maxevals, boundStrategy, seed),
          original_popsize(population_size), popsize(population_size), minPopSize(minPopSize), 
          strategy(strategy), Nevals(0), rangeScale(1.0), use_clip(false) 
    {
        if (population_size < 10) {
            original_popsize = 10;
            popsize = 10;
        } ;

        double eta = log10(maxevals);
        if (population_size==0){
            original_popsize = static_cast<size_t> (2.0*eta*eta + bounds.size()/2.0);
            popsize = original_popsize;
        };
                
        popDecrease = minPopSize != original_popsize;
        max_no_improve = static_cast<size_t> (2.0*eta*eta + bounds.size());
        archiveSize = static_cast<size_t> (2.6*popsize);  // this is take from LSHADE value.
        if (minPopSize > original_popsize) throw std::invalid_argument("minPopSize must be smaller or equal to population_size.");
};

void DE_Base::_initialize_population() {
    history.clear();

    std::vector<double> lower_bounds, upper_bounds;
    for (const auto& bound : bounds) {
        lower_bounds.push_back(bound.first);
        upper_bounds.push_back(bound.second);
    }

    std::vector<std::pair<double, double>> bounds_copy = bounds;
    std::vector<double> midpoints, ranges, new_lower, new_upper;
    for (const auto& bound : bounds_copy) {
        midpoints.push_back((bound.first + bound.second) / 2.0);
        ranges.push_back(bound.second - bound.first);
    }
    for (size_t i = 0; i < ranges.size(); ++i) {
        double new_range = ranges[i] * rangeScale;
        new_lower.push_back(midpoints[i] - new_range / 2.0);
        new_upper.push_back(midpoints[i] + new_range / 2.0);
    }
    for (size_t i = 0; i < bounds_copy.size(); ++i) {
        bounds_copy[i].first = new_lower[i];
        bounds_copy[i].second = new_upper[i];
    }

    population = latin_hypercube_sampling(bounds_copy, popsize);
    if (!x0.empty()) population[0] = x0;

    fitness = func(population, data);
    for (auto& fit : fitness) {
        if (std::isnan(fit)) fit = 1e+100;
    }

    best_idx = static_cast<int>(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    best = population[best_idx];
    best_fitness = fitness[best_idx];
    Nevals += population.size();
    evalFrac = static_cast<double>(Nevals)/static_cast<double>(maxevals);
    history.push_back(new MinionResult(best, best_fitness, 0, popsize, false, "best initial fitness"));
}

void DE_Base::_disturb_population(std::vector<std::vector<double>>& pop){
    std::vector<size_t> sortedInd = argsort(fitness, false); 
    size_t Npop = popsize-5; //round(fitness.size()/2.0);
    if (Npop<2){Npop=2;};
    for (size_t i=0; i<Npop; ++i){
        std::vector<double> p = pop[sortedInd[i]];
        for (size_t j=0; j<p.size(); ++j){
            if (rand_gen()<0.1) { 
                //p[j] = evalFrac*p[j]+(1.0-evalFrac)*rand_gen(bounds[j].first, bounds[j].second) ;
                p[j] = evalFrac*p[j]+(1.0-evalFrac)*rand_gen(bounds[j].first, bounds[j].second) ;
            };
        }

        pop[sortedInd[i]] = p;
    };
}

std::vector<double> DE_Base::_mutate(size_t idx, std::string& mutation_strategy) {
    std::vector<int> available_indices(popsize), indices;
    size_t r1, r2, r3;
    std::iota(available_indices.begin(), available_indices.end(), 0);
    available_indices.erase(available_indices.begin() + idx);

    std::vector<double> mutant;

    if (mutation_strategy == "best1bin" || mutation_strategy == "best1exp") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = best;
        for (size_t i = 0; i < best.size(); ++i) {
            mutant[i] += F[idx] * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "rand1bin" || mutation_strategy == "rand1exp") {
        indices = random_choice<int>(available_indices, 3);
        r1  = indices[0];
        r2 = indices[1];
        r3 = indices[2];
        mutant = population[r1];
        for (size_t i = 0; i < population[r1].size(); ++i) {
            mutant[i] += F[idx] * (population[r2][i] - population[r3][i]);
        }
    } else if (mutation_strategy == "current_to_best1bin" || mutation_strategy == "current_to_best1exp") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += F[idx] * (best[i] - population[idx][i]) + F[idx] * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "current_to_pbest1bin" || mutation_strategy == "current_to_pbest1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + p);
        auto pbestind = random_choice(top_p_indices, 1).front();
        auto indices = random_choice(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += F[idx] * (population[pbestind][i] - population[idx][i]) + F[idx] * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "current_to_pbest_A1_1bin" || mutation_strategy == "current_to_pbest_A1_1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + p);
        auto pbestind = random_choice(top_p_indices, 1).front();

        std::vector<size_t> arch_ind(archive.size()+population.size()); 
        std::iota(arch_ind.begin(), arch_ind.end(), 0);
        auto indices = random_choice(available_indices, 1);
        auto indices2 = random_choice(arch_ind, 1);
        r1 = indices[0];
        r2 = indices2[0];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            if (r2 < archive.size()) {
                mutant[i] += F[idx] * (population[pbestind][i] - population[idx][i]) + F[idx] * (population[r1][i] - archive[r2][i]);
            } else {
                mutant[i] += F[idx] * (population[pbestind][i] - population[idx][i]) + F[idx] * (population[r1][i] - population[r2-archive.size()][i]);
            }
        }
    } else if (mutation_strategy == "current_to_pbest_A2_1bin" || mutation_strategy == "current_to_pbest_A2_1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + p);
        auto pbestind = random_choice(top_p_indices, 1).front();
        std::vector<size_t> arch_ind(archive.size()); 
        if (archive.size()<=2) {
            throw std::logic_error("Archive size is not sufficient");
        };
        std::iota(arch_ind.begin(), arch_ind.end(), 0);
        auto indices = random_choice(arch_ind, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += F[idx] * (population[pbestind][i] - population[idx][i]) + F[idx] * (archive[r1][i] - archive[r2][i]);
        }
    } else {
        throw std::invalid_argument("Unknown mutation strategy: " + mutation_strategy);
    }

    return mutant;
};

std::vector<double> DE_Base::_crossover_bin(const std::vector<double>& target, const std::vector<double>& mutant, double CR) {
    std::vector<double> trial(target.size());
    std::vector<bool> crossover_mask(target.size());
    std::generate(crossover_mask.begin(), crossover_mask.end(), [CR, this] { return rand_gen() < CR; });
    if (std::none_of(crossover_mask.begin(), crossover_mask.end(), [](bool v) { return v; })) {
        crossover_mask[rand_int(target.size())] = true;
    }
    for (size_t i = 0; i < target.size(); ++i) {
        trial[i] = crossover_mask[i] ? mutant[i] : target[i];
    }
    return trial;
};

std::vector<double> DE_Base::_crossover_exp(const std::vector<double>& target, const std::vector<double>& mutant, double CR) {
    std::vector<double> trial = target;
    size_t n = rand_int(target.size());
    int L = 0;
    while (rand_gen() < CR && L < target.size()) {
        trial[n] = mutant[n];
        n = (n + 1) % target.size();
        L++;
    }
    return trial;
}

std::vector<double> DE_Base::_crossover(const std::vector<double>& target, const std::vector<double>& mutant, double CR, std::string& mutation_strategy) {
    if (mutation_strategy.find("bin") != std::string::npos) {
        return _crossover_bin(target, mutant, CR);
    } else if (mutation_strategy.find("exp") != std::string::npos) {
        return _crossover_exp(target, mutant, CR);
    } else {
        throw std::invalid_argument("Unknown crossover strategy in mutation strategy: " + mutation_strategy);
    }
}

