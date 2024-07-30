#include "de.h"


std::vector<double> Differential_Evolution::mutate(size_t idx){
    std::vector<int> available_indices(population.size()), indices;
    size_t r1, r2, r3;
    std::iota(available_indices.begin(), available_indices.end(), 0);
    available_indices.erase(available_indices.begin() + idx);
    double Find = F[idx];
    size_t pind = p[idx];
    std::vector<double> mutant;
    auto best = population[findArgMin(fitness)];

    if (mutation_strategy == "best1bin" || mutation_strategy == "best1exp") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = best;
        for (size_t i = 0; i < best.size(); ++i) {
            mutant[i] += Find * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "rand1bin" || mutation_strategy == "rand1exp") {
        indices = random_choice<int>(available_indices, 3);
        r1  = indices[0];
        r2 = indices[1];
        r3 = indices[2];
        mutant = population[r1];
        for (size_t i = 0; i < population[r1].size(); ++i) {
            mutant[i] += Find * (population[r2][i] - population[r3][i]);
        }
    } else if (mutation_strategy == "current_to_best1bin" || mutation_strategy == "current_to_best1exp") {
        auto indices = random_choice<int>(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += Find * (best[i] - population[idx][i]) + Find * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "current_to_pbest1bin" || mutation_strategy == "current_to_pbest1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + pind);
        auto pbestind = random_choice(top_p_indices, 1).front();
        auto indices = random_choice(available_indices, 2);
        r1 = indices[0];
        r2 = indices[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - population[r2][i]);
        }
    } else if (mutation_strategy == "current_to_pbest_A1_1bin" || mutation_strategy == "current_to_pbest_A1_1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + pind);
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
                mutant[i] += Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - archive[r2][i]);
            } else {
                mutant[i] += Find * (population[pbestind][i] - population[idx][i]) + Find * (population[r1][i] - population[r2-archive.size()][i]);
            }
        }
    
    }else if (mutation_strategy == "current_to_pbest_A2_1bin" || mutation_strategy == "current_to_pbest_A2_1exp") {   
        auto sorted_indices = argsort(fitness, true);
        std::vector<size_t> top_p_indices(sorted_indices.begin(), sorted_indices.begin() + pind);
        auto pbestind = random_choice(top_p_indices, 1).front();

        std::vector<size_t> arch_ind(archive.size()+population.size()); 
        std::vector<std::vector<double>> combine = archive;
        for (auto el : population) combine.push_back(el);
        std::iota(arch_ind.begin(), arch_ind.end(), 0);
        auto indices = random_choice(available_indices, 1);
        auto indices2 = random_choice(arch_ind, 2);
        r1 = indices[0];
        r2 = indices2[0];
        r3= indices2[1];
        mutant = population[idx];
        for (size_t i = 0; i < population[idx].size(); ++i) {
            mutant[i] += Find * (population[pbestind][i] - combine[r3][i]) + Find * (population[r1][i] - combine[r2][i]);
        }
    } else {
        throw std::invalid_argument("Unknown mutation strategy: " + mutation_strategy);
    }
    return mutant;
};

std::vector<double> Differential_Evolution::_crossover_bin(const std::vector<double>& target, const std::vector<double>& mutant, double C) {
    std::vector<double> trial(target.size());
    std::vector<bool> crossover_mask(target.size());
    std::generate(crossover_mask.begin(), crossover_mask.end(), [C] { return rand_gen() < C; });
    if (std::none_of(crossover_mask.begin(), crossover_mask.end(), [](bool v) { return v; })) {
        crossover_mask[rand_int(target.size())] = true;
    }
    for (size_t i = 0; i < target.size(); ++i) {
        trial[i] = crossover_mask[i] ? mutant[i] : target[i];
    }
    return trial;
};

std::vector<double> Differential_Evolution::_crossover_exp(const std::vector<double>& target, const std::vector<double>& mutant, double C) {
    std::vector<double> trial = target;
    size_t n = rand_int(target.size());
    int L = 0;
    while (rand_gen() < C && L < target.size()) {
        trial[n] = mutant[n];
        n = (n + 1) % target.size();
        L++;
    }
    return trial;
}

std::vector<double> Differential_Evolution::crossover(const std::vector<double>& target, const std::vector<double>& mutant, double C) {
    if (mutation_strategy.find("bin") != std::string::npos) {
        return _crossover_bin(target, mutant, C);
    } else if (mutation_strategy.find("exp") != std::string::npos) {
        return _crossover_exp(target, mutant, C);
    } else {
        throw std::invalid_argument("Unknown crossover strategy in mutation strategy: " + mutation_strategy);
    }
}


void Differential_Evolution::init (){
    population = latin_hypercube_sampling(bounds, populationSize);
    fitness = func(population, data);
    size_t best_idx = findArgMin(fitness);
    best = population[best_idx];
    best_fitness = fitness[best_idx];
    Nevals++;
    history.push_back(MinionResult(best, best_fitness, 0, Nevals, false, ""));
};

bool Differential_Evolution::checkStopping(){
    double relRange = calcStdDev(fitness)/calcMean(fitness);
    diversity.push_back(relRange);
    bool stop = false;
    if (relRange <= stoppingTol) {
        stop= true;
    };
    return stop;
};

void Differential_Evolution::adaptParameters(){
    F= std::vector<double>(population.size(), 0.5);
    CR= std::vector<double>(population.size(), 0.5);
    p = std::vector<size_t>(population.size(), 1);
    mutation_strategy="current_to_pbest1bin";
    
    meanCR.push_back(calcMean(CR));
    meanF.push_back(calcMean(F));
    stdCR.push_back(calcStdDev(CR));
    stdF.push_back(calcStdDev(F));  
};

void Differential_Evolution::doDE_operation(std::vector<std::vector<double>>& trials){
    size_t popsize = population.size();
    std::vector<double> mutant;
    for (int i = 0; i < popsize; ++i) {
        mutant = mutate(i);
        trials[i] = crossover(population[i], mutant, CR[i]);
    }
};

MinionResult Differential_Evolution::optimize() {
    try {
        init();
        size_t iter=1;
        while(Nevals < maxevals) {
            adaptParameters();
            std::vector<std::vector<double>> trials(population.size(), std::vector<double>(population[0].size()));
            doDE_operation(trials);
            enforce_bounds(trials, bounds, boundStrategy);
            trial_fitness = func(trials, data);
            Nevals += trials.size();

            std::replace_if(trial_fitness.begin(), trial_fitness.end(), [](double f) { return std::isnan(f); }, 1e+100);
            fitness_before = fitness; 
            for (int i = 0; i < population.size(); ++i) {
                if (trial_fitness[i] < fitness[i]) { 
                    population[i] = trials[i];
                    fitness[i] = trial_fitness[i];
                } else  archive.push_back(trials[i]); 
            }

            size_t best_idx = findArgMin(fitness);
            if (fitness[best_idx] >= best_fitness) {no_improve_counter++;} else {no_improve_counter =0;};
            best = population[best_idx];
            best_fitness = fitness[best_idx];

            history.push_back(MinionResult(best, best_fitness, iter, Nevals, false, ""));
            iter++;
            if (checkStopping()) break;
        }  
        return history.back();  

    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}
